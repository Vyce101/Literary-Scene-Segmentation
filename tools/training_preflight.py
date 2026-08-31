# Benchmarks one-GPU FP16 Qwen training feasibility with synthetic scene examples.

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tools.window_generation import (
    DEFAULT_PROMPT_PATH,
    Unit,
    chat_template_token_ids,
    render_user_message,
)


DEFAULT_MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"
DEFAULT_STEPS = 3
DEFAULT_WARMUP_STEPS = 1
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_SEED = 1729
IGNORE_INDEX = -100


class TrainingPreflightError(ValueError):
    """Raised when a preflight cannot build or execute a valid benchmark."""


@dataclass(frozen=True)
class TrainingExample:
    """One fully serialized synthetic supervised training example."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    user_message: str
    assistant_message: str
    target_unit_ids: tuple[str, ...]

    @property
    def serialized_token_count(self) -> int:
        return len(self.input_ids)


@dataclass(frozen=True)
class PreflightConfig:
    """Explicit parameters controlling one short feasibility run."""

    method: str
    sequence_length: int
    steps: int = DEFAULT_STEPS
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    model_name: str = DEFAULT_MODEL_NAME
    prompt_path: Path = DEFAULT_PROMPT_PATH
    learning_rate: float = DEFAULT_LEARNING_RATE
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: int = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    seed: int = DEFAULT_SEED


def _token_ids_from_encoded(value: Any) -> tuple[int, ...]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return tuple(int(token_id) for token_id in value)


def _plain_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is not None:
        return tuple(backend.encode(text, add_special_tokens=False).ids)
    return _token_ids_from_encoded(tokenizer.encode(text, add_special_tokens=False))


def _read_prompt(prompt_path: Path) -> str:
    try:
        return prompt_path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise TrainingPreflightError(f"cannot read frozen prompt {prompt_path}: {error}") from error


def _synthetic_units(unit_count: int) -> tuple[Unit, ...]:
    """Create original, non-copyrighted literary-like units with block breaks."""

    return tuple(
        Unit(
            f"{index:06d}",
            (
                f"Synthetic scene {index} begins at the rain-dark station. "
                "Mara studies the brass signal while Ivo keeps watch beside the "
                "locked archive door; their immediate goal remains to recover the "
                "missing map before the night train departs."
            ),
            1 + (index - 1) // 4,
        )
        for index in range(1, unit_count + 1)
    )


def _synthetic_messages(unit_count: int) -> tuple[str, str, tuple[str, ...]]:
    units = _synthetic_units(unit_count)
    target_start = max(1, unit_count // 2)
    target_end = min(unit_count - 1, target_start + 2)
    target = units[target_start : target_end + 1]
    past = units[:target_start]
    future = units[target_end + 1 :]
    user_message = render_user_message(past, target, future)
    assistant_message = json.dumps(
        {
            "boundaries_before": [target[-1].id],
            "document_boundaries_before": [],
        },
        separators=(",", ":"),
    )
    return user_message, assistant_message, tuple(unit.id for unit in target)


def serialize_training_example(
    tokenizer: Any,
    system_message: str,
    user_message: str,
    assistant_message: str,
) -> TrainingExample:
    """Serialize one example and mask every token before the assistant response."""

    prompt_messages = (
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    )
    full_messages = prompt_messages + (
        {"role": "assistant", "content": assistant_message},
    )
    prompt_ids = chat_template_token_ids(
        tokenizer, prompt_messages, add_generation_prompt=True
    )
    input_ids = chat_template_token_ids(
        tokenizer, full_messages, add_generation_prompt=False
    )
    response_ids = _plain_token_ids(tokenizer, assistant_message)
    response_start = len(prompt_ids)
    response_end = response_start + len(response_ids)
    if input_ids[:response_start] != prompt_ids:
        raise TrainingPreflightError(
            "the full Qwen serialization does not begin with its generation prompt"
        )
    if input_ids[response_start:response_end] != response_ids:
        raise TrainingPreflightError(
            "the tokenizer template does not expose a contiguous assistant response"
        )
    labels = (
        (IGNORE_INDEX,) * response_start
        + input_ids[response_start:response_end]
        + (IGNORE_INDEX,) * (len(input_ids) - response_end)
    )
    return TrainingExample(
        input_ids=input_ids,
        labels=labels,
        attention_mask=(1,) * len(input_ids),
        user_message=user_message,
        assistant_message=assistant_message,
        target_unit_ids=(),
    )


def build_synthetic_example(
    tokenizer: Any,
    system_message: str,
    requested_sequence_length: int,
    seed: int = DEFAULT_SEED,
) -> TrainingExample:
    """Find the largest whole-unit synthetic example that fits the requested length."""

    del seed  # Kept in the API so future synthetic variations remain reproducible.
    if requested_sequence_length < 1:
        raise TrainingPreflightError("sequence length must be positive")

    cache: dict[int, TrainingExample] = {}

    def candidate(unit_count: int) -> TrainingExample:
        if unit_count not in cache:
            user_message, assistant_message, target_ids = _synthetic_messages(unit_count)
            example = serialize_training_example(
                tokenizer, system_message, user_message, assistant_message
            )
            cache[unit_count] = TrainingExample(
                example.input_ids,
                example.labels,
                example.attention_mask,
                example.user_message,
                example.assistant_message,
                target_ids,
            )
        return cache[unit_count]

    lower = 3
    if candidate(lower).serialized_token_count > requested_sequence_length:
        raise TrainingPreflightError(
            "requested sequence length is too short for the synthetic example"
        )
    upper = lower
    while candidate(upper).serialized_token_count <= requested_sequence_length:
        lower = upper
        upper *= 2
    while upper - lower > 1:
        middle = (lower + upper) // 2
        if candidate(middle).serialized_token_count <= requested_sequence_length:
            lower = middle
        else:
            upper = middle
    return candidate(lower)


def _load_training_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise TrainingPreflightError(
            "training preflight requires torch, peft, and transformers in the project "
            "environment; run `uv sync`"
        ) from error
    return (
        torch,
        AutoTokenizer,
        _load_model_class(transformers),
        (LoraConfig, TaskType, get_peft_model),
    )


def _load_model_class(transformers_module: Any) -> Any:
    """Support the official multimodal auto class across Transformers releases."""

    model_class = getattr(transformers_module, "AutoModelForImageTextToText", None)
    if model_class is not None:
        return model_class
    model_class = getattr(transformers_module, "AutoModelForMultimodalLM", None)
    if model_class is not None:
        return model_class
    raise TrainingPreflightError(
        "installed Transformers does not provide a multimodal Qwen model auto class"
    )


def _load_model_and_tokenizer(config: PreflightConfig) -> tuple[Any, Any, Any, Any]:
    torch, auto_tokenizer_class, auto_model_class, peft_symbols = _load_training_dependencies()
    if not torch.cuda.is_available():
        raise TrainingPreflightError("CUDA is required; this preflight targets one NVIDIA T4")
    torch.cuda.set_device(0)
    tokenizer = auto_tokenizer_class.from_pretrained(config.model_name)
    model = auto_model_class.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.to(torch.device("cuda:0"))
    return torch, tokenizer, model, peft_symbols


def _configure_model(model: Any, config: PreflightConfig, peft_symbols: Any) -> Any:
    lora_config_class, task_type, get_peft_model = peft_symbols
    if config.method == "lora":
        lora_config = lora_config_class(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
            bias="none",
            task_type=task_type.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if not any(parameter.requires_grad for parameter in model.parameters()):
            raise TrainingPreflightError("LoRA configuration produced no trainable parameters")
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if config.method == "lora" and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    return model


def _parameter_report(model: Any) -> tuple[int, int, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total, (100.0 * trainable / total if total else 0.0)


def _run_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    example: TrainingExample,
    device: Any,
) -> float:
    input_ids = torch.tensor((example.input_ids,), dtype=torch.long, device=device)
    attention_mask = torch.tensor(
        (example.attention_mask,), dtype=torch.long, device=device
    )
    labels = torch.tensor((example.labels,), dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.float16):
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    if output.loss is None or not torch.isfinite(output.loss):
        raise TrainingPreflightError("model returned a non-finite training loss")
    scaler.scale(output.loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return float(output.loss.detach().item())


def run_preflight(config: PreflightConfig) -> dict[str, Any]:
    """Run a few real FP16 CUDA training steps and return JSON-safe metrics."""

    if config.method not in {"lora", "full"}:
        raise TrainingPreflightError("method must be lora or full")
    if config.warmup_steps < 0 or config.steps <= config.warmup_steps:
        raise TrainingPreflightError("steps must be greater than warmup-steps >= 0")
    if config.lora_rank < 1 or config.lora_alpha < 1:
        raise TrainingPreflightError("LoRA rank and alpha must be positive")
    if not 0 <= config.lora_dropout < 1:
        raise TrainingPreflightError("LoRA dropout must be in [0, 1)")

    torch, tokenizer, model, peft_symbols = _load_model_and_tokenizer(config)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    system_message = _read_prompt(config.prompt_path)
    example = build_synthetic_example(
        tokenizer, system_message, config.sequence_length, config.seed
    )
    device = torch.device("cuda:0")
    model = _configure_model(model, config, peft_symbols)
    trainable, total, trainable_percentage = _parameter_report(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    scaler = torch.cuda.amp.GradScaler()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    losses: list[float] = []
    try:
        for _ in range(config.steps):
            start = time.perf_counter()
            losses.append(_run_step(torch, model, optimizer, scaler, example, device))
            torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        torch.cuda.empty_cache()
        return _metric_report(
            config,
            example,
            trainable,
            total,
            trainable_percentage,
            durations[: config.warmup_steps],
            durations[config.warmup_steps :],
            torch.cuda.max_memory_allocated(device),
            torch.cuda.max_memory_reserved(device),
            status="cuda_oom",
        )
    return _metric_report(
        config,
        example,
        trainable,
        total,
        trainable_percentage,
        durations[: config.warmup_steps],
        durations[config.warmup_steps :],
        torch.cuda.max_memory_allocated(device),
        torch.cuda.max_memory_reserved(device),
        status="success",
        losses=losses,
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _metric_report(
    config: PreflightConfig,
    example: TrainingExample,
    trainable: int,
    total: int,
    trainable_percentage: float,
    warmup_durations: Sequence[float],
    measured_durations: Sequence[float],
    peak_allocated: int,
    peak_reserved: int,
    *,
    status: str,
    losses: Sequence[float] = (),
) -> dict[str, Any]:
    measured_step = _mean_or_none(measured_durations)
    return {
        "method": config.method,
        "model": config.model_name,
        "requested_sequence_length": config.sequence_length,
        "actual_serialized_token_count": example.serialized_token_count,
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
        "trainable_parameter_percentage": round(trainable_percentage, 6),
        "peak_cuda_allocated_memory_bytes": int(peak_allocated),
        "peak_cuda_reserved_memory_bytes": int(peak_reserved),
        "peak_cuda_allocated_memory_mib": round(peak_allocated / 2**20, 2),
        "peak_cuda_reserved_memory_mib": round(peak_reserved / 2**20, 2),
        "warmup_step_time_seconds": _mean_or_none(warmup_durations),
        "measured_step_time_seconds": measured_step,
        "tokens_per_second": (
            example.serialized_token_count / measured_step
            if measured_step
            else None
        ),
        "gradient_checkpointing": True,
        "precision": "float16",
        "device": "cuda:0",
        "batch_size": 1,
        "steps": config.steps,
        "warmup_steps": config.warmup_steps,
        "status": status,
        "success": status == "success",
        "losses": [round(loss, 6) for loss in losses],
        "synthetic_target_unit_ids": list(example.target_unit_ids),
        "lora": (
            {
                "rank": config.lora_rank,
                "alpha": config.lora_alpha,
                "dropout": config.lora_dropout,
                "target_modules": list(config.lora_target_modules),
            }
            if config.method == "lora"
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("lora", "full"), required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument(
        "--lora-target-modules",
        default=",".join(DEFAULT_LORA_TARGET_MODULES),
        help="comma-separated text projection suffixes for LoRA",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sequence_length < 1:
        raise SystemExit("--sequence-length must be positive")
    if args.steps < 1 or args.warmup_steps < 0:
        raise SystemExit("--steps must be positive and --warmup-steps cannot be negative")
    target_modules = tuple(
        item.strip() for item in args.lora_target_modules.split(",") if item.strip()
    )
    config = PreflightConfig(
        method=args.method,
        sequence_length=args.sequence_length,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        model_name=args.model,
        prompt_path=args.prompt_path,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=target_modules,
        seed=args.seed,
    )
    try:
        print(json.dumps(run_preflight(config), indent=2))
    except TrainingPreflightError as error:
        raise SystemExit(f"training preflight failed: {error}") from error


if __name__ == "__main__":
    main()
