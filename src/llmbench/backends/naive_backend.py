"""Reference backend: HuggingFace Transformers with optional static batching.

Two serving modes are supported:

``per_request``
    One forward pass per request. This is the "naive" deployment people
    actually start with: N concurrent users means N queued forward passes.

``static_batch``
    Prompts are padded to the longest sequence, one padded batch is executed,
    and every request finishes at the same time — the classic inefficiency
    that continuous batching fixes.

TTFT is measured for real, not approximated: a ``StoppingCriteria`` callback
is invoked after every decoding step, so the timestamp of the first callback
corresponds to the moment the first token was produced.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.logging import get_logger
from .base import Backend, BackendError, GenerationOutput, GenerationRequest

__all__ = ["NaiveBackend"]

log = get_logger("backends.naive")


def _require(module: str, hint: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise BackendError(
            f"The {module!r} package is required for the HuggingFace backend.\n"
            f"Install the GPU extras first:  pip install -r requirements-gpu.txt\n"
            f"({hint})"
        ) from exc


class NaiveBackend(Backend):
    """HuggingFace Transformers reference implementation."""

    name = "naive"
    supports_batch = True

    def __init__(self, run_config):
        super().__init__(run_config)
        mode = run_config.backend.effective_serving_mode
        self.serving_mode = "static_batch" if mode == "static_batch" else "per_request"
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._device = None
        self._stopping_cls = None
        self._criteria_list_cls = None

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self._loaded:
            return

        torch = _require("torch", "PyTorch is the inference runtime")
        transformers = _require("transformers", "AutoModelForCausalLM / AutoTokenizer")
        self._torch = torch

        from transformers import StoppingCriteria, StoppingCriteriaList

        self._stopping_cls = StoppingCriteria
        self._criteria_list_cls = StoppingCriteriaList

        model_path = self.model_config.resolve_path()
        tokenizer_path = self.model_config.resolve_tokenizer_path()

        log.info("Loading tokenizer from %s", tokenizer_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=self.model_config.trust_remote_code,
            local_files_only=True,
        )
        # Decoder-only batched generation requires left padding.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self._tokenizer = tokenizer

        log.info("Loading model from %s (dtype=%s)", model_path, self.model_config.dtype)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.model_config.trust_remote_code,
            "local_files_only": True,
        }
        quantization = (self.model_config.quantization or "").lower()
        if quantization in {"4bit", "8bit"}:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=quantization == "4bit",
                    load_in_8bit=quantization == "8bit",
                )
            except ImportError as exc:  # pragma: no cover
                raise BackendError("bitsandbytes is required for 4bit/8bit quantization") from exc
        else:
            dtype = getattr(torch, str(self.model_config.dtype), torch.float16)
            model_kwargs["torch_dtype"] = dtype

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=None if self.model_config.device == "cpu" else "auto",
            **model_kwargs,
        )
        if self.model_config.device == "cpu":
            model.to("cpu")
        model.eval()
        self._model = model
        self._device = next(model.parameters()).device

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            log.info("Model loaded. GPU memory allocated: %.0f MB", allocated)
        self._loaded = True

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        if self._torch is not None:
            try:
                import gc

                gc.collect()
                if self._torch.cuda.is_available():
                    self._torch.cuda.synchronize()
                    self._torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                pass

    # -- helpers ---------------------------------------------------------
    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:  # pragma: no cover - base models without a template
            return prompt

    def _sampling_kwargs(self, request: GenerationRequest) -> dict[str, Any]:
        greedy = request.temperature <= 0
        return {
            "max_new_tokens": int(request.max_new_tokens),
            "do_sample": not greedy,
            "temperature": None if greedy else request.temperature,
            "top_p": None if greedy else request.top_p,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

    def _make_timer(self):
        """StoppingCriteria that timestamps every decoding step."""
        torch = self._torch

        class _StepTimer(self._stopping_cls):  # type: ignore[misc, valid-type]
            def __init__(self) -> None:
                """Initialise an empty timestamp list."""
                super().__init__()
                self.times: list[float] = []

            def __call__(self, input_ids, scores, **kwargs):  # noqa: ANN001
                self.times.append(time.perf_counter())
                return torch.zeros(input_ids.shape[0], dtype=torch.bool)

        return _StepTimer()

    def _sync(self) -> None:
        if self._device is not None and self._device.type == "cuda":
            self._torch.cuda.synchronize()

    # -- execution -------------------------------------------------------
    def generate(self, request: GenerationRequest) -> GenerationOutput:
        if not self._loaded:
            self.load()

        text = self._format_prompt(request.prompt)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        input_tokens = int(inputs["input_ids"].shape[1])

        timer = self._make_timer()
        kwargs = self._sampling_kwargs(request)

        self._sync()
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                stopping_criteria=self._criteria_list_cls([timer]),
                **kwargs,
            )
        self._sync()
        t1 = time.perf_counter()

        generated = output_ids[0][input_tokens:]
        output_tokens = int(generated.shape[0])
        token_times = [round((ts - t0) * 1000.0, 4) for ts in timer.times]
        ttft_ms = token_times[0] if token_times else None

        return GenerationOutput(
            request_id=request.request_id,
            text=self._tokenizer.decode(generated, skip_special_tokens=True),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            e2e_ms=(t1 - t0) * 1000.0,
            token_times_ms=token_times,
            backend=self.name,
            serving_mode=self.serving_mode,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationOutput]:
        """Static batching: pad to the longest prompt, run one forward pass."""
        if not requests:
            return []
        if self.serving_mode != "static_batch" or len(requests) == 1:
            return [self.generate(r) for r in requests]
        if not self._loaded:
            self.load()

        texts = [self._format_prompt(r.prompt) for r in requests]
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_model_len,
        ).to(self._device)
        batch_input_len = int(inputs["input_ids"].shape[1])

        timer = self._make_timer()
        max_new = max(int(r.max_new_tokens) for r in requests)
        greedy = all(r.temperature <= 0 for r in requests)

        self._sync()
        t0 = time.perf_counter()
        with self._torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=not greedy,
                temperature=None if greedy else max(r.temperature for r in requests),
                top_p=None if greedy else max(r.top_p for r in requests),
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                stopping_criteria=self._criteria_list_cls([timer]),
            )
        self._sync()
        t1 = time.perf_counter()

        e2e_ms = (t1 - t0) * 1000.0
        token_times = [round((ts - t0) * 1000.0, 4) for ts in timer.times]
        ttft_ms = token_times[0] if token_times else None

        outputs: list[GenerationOutput] = []
        for idx, request in enumerate(requests):
            generated = output_ids[idx][batch_input_len:]
            real_tokens = int((generated != self._tokenizer.pad_token_id).sum().item())
            real_tokens = min(real_tokens, int(request.max_new_tokens))
            outputs.append(
                GenerationOutput(
                    request_id=request.request_id,
                    text=self._tokenizer.decode(generated, skip_special_tokens=True),
                    input_tokens=int(inputs["attention_mask"][idx].sum().item()),
                    output_tokens=real_tokens,
                    ttft_ms=ttft_ms,
                    e2e_ms=e2e_ms,
                    token_times_ms=token_times,
                    backend=self.name,
                    serving_mode=self.serving_mode,
                    extra={
                        "batch_size": len(requests),
                        "padded_prompt_tokens": batch_input_len,
                    },
                )
            )
        return outputs
