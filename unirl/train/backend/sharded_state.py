"""FSDP2-generic sharded-state helpers shared by every train backend."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List

import torch
from torch import nn
from torch.nn.parameter import Parameter

from unirl.distributed.local import local_view

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]


def gather_state_dict(model: nn.Module) -> StateDict:
    """Rank-0 DCP gather.  Returns full state on rank 0, empty on others."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    try:
        full = dict(get_model_state_dict(model, options=options))
    except TypeError:
        full = dict(get_model_state_dict(model))

    if _current_rank() != 0:
        return {}
    return _to_cpu_state_dict(full)


def load_model_state_dict(
    model: nn.Module,
    state_dict: StateDict,
    *,
    strict: bool = True,
    broadcast_from_rank0: bool = True,
) -> object:
    """Load a full state dict and reshard it into ``model``; returns torch's ``(missing_keys, unexpected_keys)``."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
        strict=strict,
    )
    try:
        return set_model_state_dict(model, state_dict, options=options)
    except TypeError:
        return set_model_state_dict(model, state_dict)


def gather_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Rank-0 DCP optimizer gather; never-updated AdamW params export as step-0 zero state."""
    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    full = _export_optimizer_state_dict(model, optimizer, options=options)
    return full if _current_rank() == 0 else {}


def gather_lora_state_dict(model: nn.Module, keep: Callable[[str], bool]) -> StateDict:
    """Gather the LoRA tensors whose keys pass ``keep``, preserving the model state-dict key format."""
    gathered: StateDict = {}
    for key, value in model.state_dict().items():
        if not keep(key):
            continue
        if isinstance(value, torch.Tensor) and value.is_meta:
            raise RuntimeError(f"gather_lora_state_dict: LoRA tensor {key!r} is still on meta")
        materialized = _materialize_checkpoint_tensor(value)
        if isinstance(materialized, torch.Tensor):
            materialized = materialized.detach().cpu()
        if _current_rank() == 0:
            gathered[key] = materialized
    if _current_rank() != 0:
        return {}
    return gathered


def load_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: StateDict,
    *,
    broadcast_from_rank0: bool = True,
) -> None:
    """Load a full optimizer state dict and reshard it into ``optimizer``."""
    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
    )
    _mark_omitted_adamw_state_fresh(optimizer, state_dict)
    _set_optimizer_state_dict(model, optimizer, state_dict, options=options)


def sharded_model_state_dict(model: nn.Module) -> StateDict:
    """Per-rank sharded model state for DCP."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    options = _build_state_dict_options(full_state_dict=False)
    try:
        return dict(get_model_state_dict(model, options=options))
    except TypeError:
        return dict(get_model_state_dict(model))


def sharded_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Per-rank sharded optimizer state for DCP; never-updated AdamW params export as step-0 zero state."""
    options = _build_state_dict_options(full_state_dict=False)
    return _export_optimizer_state_dict(model, optimizer, options=options)


def load_sharded_model_state_dict(model: nn.Module, state_dict: StateDict, *, strict: bool = True) -> None:
    """Load a per-rank sharded model state read by ``dcp.load`` in place."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _build_state_dict_options(full_state_dict=False, strict=strict)
    try:
        set_model_state_dict(model, state_dict, options=options)
    except TypeError:
        set_model_state_dict(model, state_dict)


def load_sharded_optimizer_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer, state_dict: StateDict
) -> None:
    """Load a per-rank sharded optimizer state read by ``dcp.load`` in place."""
    options = _build_state_dict_options(full_state_dict=False)
    _set_optimizer_state_dict(model, optimizer, state_dict, options=options)


@contextmanager
def fresh_adamw_state(optimizer: torch.optim.Optimizer) -> Iterator[None]:
    """Temporarily give never-updated AdamW params the step-0 zero state their first update would create."""
    added: List[Parameter] = []
    if isinstance(optimizer, torch.optim.AdamW):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.requires_grad and not optimizer.state.get(param):
                    optimizer.state[param] = _fresh_adamw_entry(param, group)
                    added.append(param)
    try:
        yield
    finally:
        for param in added:
            optimizer.state.pop(param, None)


def drop_meta_entries(state_dict: StateDict) -> StateDict:
    """Drop never-materialized (meta) entries from a sharded state dict."""
    kept: StateDict = {}
    for key, value in state_dict.items():
        local = local_view(value) if isinstance(value, torch.Tensor) else value
        if isinstance(local, torch.Tensor) and local.is_meta:
            continue
        kept[key] = value
    return kept


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: object) -> None:
    """Move optimizer state to ``device``; non-fused/capturable ``step`` stays on CPU as it is read via ``.item()``."""
    for group in optimizer.param_groups:
        step_device = device if group.get("capturable", False) or group.get("fused", False) else "cpu"
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(step_device if k == "step" else device)


def is_materialized(model: nn.Module) -> bool:
    return not any(p.is_meta for p in model.parameters())


def trainable_params(model: nn.Module) -> Iterator[Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def infer_device(model: nn.Module) -> torch.device:
    """First non-meta parameter's device, else current cuda, else cpu."""
    for param in model.parameters():
        if param.is_meta:
            continue
        return param.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _build_state_dict_options(**kwargs: object) -> object:
    """Construct ``StateDictOptions`` degrading gracefully on older torch."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    candidates = [
        dict(kwargs),
        {k: v for k, v in kwargs.items() if k != "strict"},
        {k: v for k, v in kwargs.items() if k not in {"strict", "broadcast_from_rank0"}},
        {k: v for k, v in kwargs.items() if k in {"full_state_dict", "cpu_offload"}},
        {},
    ]
    for candidate in candidates:
        try:
            return StateDictOptions(**candidate)
        except TypeError:
            continue
    return StateDictOptions()


def _maybe_dtensor_to_tensor(value: object) -> object:
    if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
        try:
            return value.full_tensor()
        except Exception:
            return value
    return value


def _materialize_checkpoint_tensor(value: object) -> object:
    if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
        device = getattr(value, "device", None)
        if (
            getattr(device, "type", None) == "cpu"
            and torch.cuda.is_available()
            and hasattr(value, "cuda")
            and callable(getattr(value, "cuda"))
        ):
            value = value.cuda()
    return _maybe_dtensor_to_tensor(value)


def _to_cpu_state_dict(state_dict: StateDict) -> StateDict:
    converted: StateDict = {}
    for key, value in state_dict.items():
        tensor_or_obj = _maybe_dtensor_to_tensor(value)
        if isinstance(tensor_or_obj, torch.Tensor):
            converted[key] = tensor_or_obj.detach().cpu()
        else:
            converted[key] = tensor_or_obj
    return converted


def _fresh_adamw_entry(param: Parameter, group: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """The state ``torch.optim.AdamW`` creates on a param's first update."""
    if group["capturable"] or group["fused"]:
        step = torch.zeros((), dtype=torch.float32, device=param.device)
    else:
        step = torch.tensor(0.0)
    entry = {"step": step, "exp_avg": torch.zeros_like(param), "exp_avg_sq": torch.zeros_like(param)}
    if group["amsgrad"]:
        entry["max_exp_avg_sq"] = torch.zeros_like(param)
    return entry


def _mark_omitted_adamw_state_fresh(optimizer: torch.optim.Optimizer, state_dict: StateDict) -> None:
    """Give params a sparse full checkpoint listed without state a step-0 entry, so the load drops them."""
    if not isinstance(optimizer, torch.optim.AdamW) or "state" not in state_dict:
        return
    for group in state_dict["param_groups"]:
        for name in group["params"]:
            state_dict["state"].setdefault(name, {"step": torch.tensor(0.0)})


def _drop_fresh_adamw_state(optimizer: torch.optim.Optimizer) -> None:
    """Return step-0 AdamW entries to lazy init; their moments are zero, so the next update is unchanged."""
    if not isinstance(optimizer, torch.optim.AdamW):
        return
    fresh = [param for param, entry in optimizer.state.items() if entry and float(entry["step"]) == 0]
    for param in fresh:
        del optimizer.state[param]


def _set_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: StateDict,
    *,
    options: object,
) -> None:
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    try:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=options)
    except TypeError:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict)
    _drop_fresh_adamw_state(optimizer)


def _export_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    options: object,
) -> StateDict:
    """Export a dense AdamW state without advancing any param's clock."""
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    with fresh_adamw_state(optimizer):
        return get_optimizer_state_dict(model, optimizer, options=options)


__all__ = [
    "StateDict",
    "gather_state_dict",
    "load_model_state_dict",
    "gather_optimizer_state_dict",
    "load_optimizer_state_dict",
    "sharded_model_state_dict",
    "sharded_optimizer_state_dict",
    "load_sharded_model_state_dict",
    "load_sharded_optimizer_state_dict",
    "fresh_adamw_state",
    "drop_meta_entries",
    "move_optimizer_state",
    "is_materialized",
    "trainable_params",
    "infer_device",
]
