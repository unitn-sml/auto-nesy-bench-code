"""vLLM backend used for every LLM evaluated in the paper."""

import json
import logging
import os
import re
from importlib import metadata

from huggingface_hub import scan_cache_dir, snapshot_download

from fonesy.llms.base import BaseLLM, extract_thinking, strip_thinking
from fonesy.general_utils import log

# Short names of the evaluated checkpoints (``--model-path <shortcut>``).
_MODEL_SHORTCUTS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3-32b": "Qwen/Qwen3-32B",
    "qwen3-coder-next": "Qwen/Qwen3-Coder-Next",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "gemma3-27b": "google/gemma-3-27b-it",
    "gemma4-31b": "google/gemma-4-31B-it",
    "phi4-reasoning-plus": "microsoft/Phi-4-reasoning-plus",
    "deepseek-r1-llama-70b": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "olmo3-32b-think": "allenai/Olmo-3-32B-Think",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
}

_DEFAULT_MODEL = "Qwen/Qwen3-8B"

# Used for any sampling parameter that is neither passed on the command line
# nor set in _MODEL_CONFIGS.
_GENERIC_SAMPLING_DEFAULTS = {
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": None,
    "repetition_penalty": None,
}

_VLLM_ADVANCED_CONFIG_MIN_VERSION = (0, 23)

# Per-model generation (and engine) defaults.
_QWEN3_THINKING = {
    "do_sample": True,
    "thinking": True,
    "thinking_sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "non_thinking_sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
}
_MODEL_CONFIGS = {
    "Qwen/Qwen3-8B": _QWEN3_THINKING,
    "Qwen/Qwen3-32B": _QWEN3_THINKING,
    "Qwen/Qwen3-Coder-Next": {
        "do_sample": True,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        # Triton kernels avoid JIT hangs on Blackwell GPUs with CUDA < 12.9.
        "additional_config": {"gdn_prefill_backend": "triton"},
        "moe_backend": "triton",
        "linear_backend": "triton",
        "enable_flashinfer_autotune": False,
    },
    "mistralai/Mistral-Nemo-Instruct-2407": {
        "do_sample": False,
        "temperature": 0.3,
    },
    "google/gemma-3-27b-it": {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 64,
        "repetition_penalty": 1.3,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "thinking": True,
    },
    "openai/gpt-oss-120b": {
        "do_sample": False,
        "quantization": "fp8",
    },
    "openai/gpt-oss-20b": {
        "do_sample": False,
    },
}


def _resolve_from_cache(repo_id: str) -> str | None:
    """Return the local snapshot path for *repo_id* from the HF cache, or None."""
    cache_dir = os.environ.get(
        "HF_HUB_CACHE",
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "hub",
        ),
    )
    log(f"Offline cache lookup for '{repo_id}' in: {cache_dir}", "DEBUG")

    # Fast path: snapshot_download (works when refs/main exists in cache).
    try:
        path = snapshot_download(repo_id, local_files_only=True)
        log(f"snapshot_download resolved '{repo_id}' → {path}", "DEBUG")
        return path
    except Exception as exc:
        log(f"snapshot_download failed for '{repo_id}': {exc}", "DEBUG")

    # Slower path: scan_cache_dir (works even without refs/main).
    try:
        cache_info = scan_cache_dir()
        log(f"scan_cache_dir: {len(cache_info.repos)} repos in cache", "DEBUG")
        for repo in cache_info.repos:
            if repo.repo_id == repo_id and repo.revisions:
                latest = max(repo.revisions, key=lambda r: r.last_modified)
                path = str(latest.snapshot_path)
                log(f"scan_cache_dir resolved '{repo_id}' → {path}", "DEBUG")
                return path
    except Exception as exc:
        log(f"scan_cache_dir failed: {exc}", "DEBUG")

    # Last resort: direct filesystem lookup using the standard HF cache layout.
    dir_name = "models--" + repo_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, dir_name, "snapshots")
    log(f"Direct fs lookup: {snapshots_dir}", "DEBUG")
    if os.path.isdir(snapshots_dir):
        entries = sorted(os.listdir(snapshots_dir))
        if entries:
            path = os.path.join(snapshots_dir, entries[-1])
            log(f"Direct fs resolved '{repo_id}' → {path}", "DEBUG")
            return path

    log(f"'{repo_id}' not found in HF cache at {cache_dir}", "WARNING")
    return None


def _vllm_version(vllm_module) -> tuple[int, ...] | None:
    """Return vLLM's numeric version without importing it at module load time."""
    raw = getattr(vllm_module, "__version__", None)
    if raw is None:
        try:
            raw = metadata.version("vllm")
        except metadata.PackageNotFoundError:
            return None
    numbers = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(raw))
    if numbers is None:
        return None
    return tuple(int(part or 0) for part in numbers.groups())


def _supports_advanced_config(version: tuple[int, ...] | None) -> bool:
    return version is not None and version[:2] >= _VLLM_ADVANCED_CONFIG_MIN_VERSION


class VllmLLM(BaseLLM):
    """LLM backend backed by vLLM for high-throughput inference."""

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        max_new_tokens: int = 2048,
        temperature: float = None,
        top_p: float = None,
        top_k: int = None,
        do_sample: bool = None,
        repetition_penalty: float = None,
        think: bool = True,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: int = None,
        dtype: str = "auto",
        quantization: str = None,
        trust_remote_code: bool = False,
        enforce_eager: bool = False,
        hf_overrides: dict = None,
        additional_config: dict = None,
        moe_backend: str = None,
        linear_backend: str = None,
        enable_flashinfer_autotune: bool = None,
        **_ignored,
    ):
        model_path = _MODEL_SHORTCUTS.get(model_path, model_path)
        hf_model_id = model_path

        # With HF_HUB_OFFLINE=1, load from the local snapshot directly.
        load_path = model_path
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            resolved = _resolve_from_cache(model_path)
            if resolved:
                load_path = resolved

        # Fall back to per-model engine defaults when not explicitly set.
        _model_cfg = _MODEL_CONFIGS.get(hf_model_id, {})
        if quantization is None:
            quantization = _model_cfg.get("quantization")

        super().__init__(name=hf_model_id)

        self.model_path = hf_model_id  # used for _MODEL_CONFIGS lookups
        self._load_path = load_path  # absolute path passed to vLLM
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.think = think

        log(f"Loading vLLM model: {hf_model_id}", "INFO")

        # vllm is only imported when the backend is used.
        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("vLLM is not installed. Run: pip install vllm") from exc

        self._LLM = LLM
        self._SamplingParams = SamplingParams
        version = _vllm_version(vllm)
        supports_advanced = _supports_advanced_config(version)

        requested_advanced = {
            "additional_config": additional_config,
            "moe_backend": moe_backend,
            "linear_backend": linear_backend,
            "enable_flashinfer_autotune": enable_flashinfer_autotune,
        }
        unsupported = [
            name for name, value in requested_advanced.items() if value is not None
        ]
        if unsupported and not supports_advanced:
            version_label = getattr(vllm, "__version__", "unknown")
            options = ", ".join(unsupported)
            raise ValueError(
                f"{options} require vLLM >= 0.23.0; installed version is "
                f"{version_label}. Remove these options or upgrade vLLM."
            )

        init_kwargs = dict(
            model=self._load_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enforce_eager=enforce_eager,
        )
        if max_model_len is not None:
            init_kwargs["max_model_len"] = max_model_len
        if hf_overrides is not None:
            # Passed to the HF config, e.g. YaRN rope scaling for longer contexts.
            init_kwargs["hf_overrides"] = hf_overrides
        if quantization is not None:
            init_kwargs["quantization"] = quantization
        if supports_advanced:
            advanced_values = {
                "additional_config": (
                    additional_config
                    if additional_config is not None
                    else _model_cfg.get("additional_config")
                ),
                "moe_backend": (
                    moe_backend
                    if moe_backend is not None
                    else _model_cfg.get("moe_backend")
                ),
                "linear_backend": (
                    linear_backend
                    if linear_backend is not None
                    else _model_cfg.get("linear_backend")
                ),
                "enable_flashinfer_autotune": (
                    enable_flashinfer_autotune
                    if enable_flashinfer_autotune is not None
                    else _model_cfg.get("enable_flashinfer_autotune")
                ),
            }
            init_kwargs.update(
                {
                    name: value
                    for name, value in advanced_values.items()
                    if value is not None
                }
            )

        q_label = f", quant={quantization}" if quantization else ""
        print(
            f"Loading {hf_model_id} (tp={tensor_parallel_size}{q_label}) …",
            flush=True,
        )
        self._llm = LLM(**init_kwargs)
        print(f"Model ready.", flush=True)
        self._tokenizer = self._llm.get_tokenizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_prompt(self, prompt: str) -> str:
        cfg = _MODEL_CONFIGS.get(self.model_path, {})
        messages = [{"role": "user", "content": prompt.strip()}]
        extra = {}
        if cfg.get("thinking", False):
            extra["enable_thinking"] = bool(self.think)
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **extra
            )
        except TypeError as exc:
            if extra.get("enable_thinking") is False:
                raise RuntimeError(
                    f"The tokenizer for {self.model_path} cannot disable thinking. "
                    "Upgrade transformers/tokenizer files; generation was stopped "
                    "to avoid silently enabling reasoning."
                ) from exc
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _build_sampling_params(self, **overrides):
        cfg = _MODEL_CONFIGS.get(self.model_path, {})
        mode_cfg = cfg.get(
            "thinking_sampling" if self.think else "non_thinking_sampling", {}
        )

        def choose(name):
            if overrides.get(name) is not None:
                return overrides[name]
            constructor_value = getattr(self, name)
            if constructor_value is not None:
                return constructor_value
            if name in mode_cfg:
                return mode_cfg[name]
            if name in cfg:
                return cfg[name]
            return _GENERIC_SAMPLING_DEFAULTS[name]

        do_sample = choose("do_sample")

        if not do_sample:
            # vLLM greedy mode requires temperature=0.
            sp_kwargs = dict(
                max_tokens=overrides.get("max_new_tokens", self.max_new_tokens),
                temperature=0.0,
            )
        else:
            sp_kwargs = dict(
                max_tokens=overrides.get("max_new_tokens", self.max_new_tokens),
                temperature=choose("temperature"),
                top_p=choose("top_p"),
            )
            top_k = choose("top_k")
            if top_k is not None:
                sp_kwargs["top_k"] = top_k

        rep = choose("repetition_penalty")
        if rep is not None:
            sp_kwargs["repetition_penalty"] = rep

        return self._SamplingParams(**sp_kwargs)

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: list, **kwargs) -> list:
        """Generate for all prompts in a single vLLM call (true batching)."""
        if not prompts:
            return []
        log(
            f"vLLM batch generate(): {len(prompts)} prompt(s), "
            f"max_tokens={self.max_new_tokens}",
            "INFO",
        )
        formatted = [self._format_prompt(p) for p in prompts]
        sampling_params = self._build_sampling_params(**kwargs)
        outputs = self._llm.generate(formatted, sampling_params)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
            )
        think_logger = logging.getLogger("fonesy.think")
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            thinking = extract_thinking(text)
            if thinking:
                think_logger.info("Thinking:\n%s", thinking)
            results.append(strip_thinking(text))
        return results

    def __repr__(self):
        return f"VllmLLM(model={self.model_path})"


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


def configure_global_arguments(parser):
    shortcuts = sorted(_MODEL_SHORTCUTS.keys())
    parser.add_argument(
        "--model-path",
        default=_DEFAULT_MODEL,
        dest="model_path",
        help=(
            f"HuggingFace model ID or local path. "
            f"Shortcuts: {', '.join(shortcuts)}. "
            f"Default: {_DEFAULT_MODEL}"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        dest="max_new_tokens",
        help="Maximum number of generated tokens (default: 32768)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        dest="temperature",
        help=(
            "Sampling temperature. Defaults are model/mode-specific "
            "(Qwen3 thinking: 0.6; non-thinking: 0.7)."
        ),
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        dest="top_p",
        help=(
            "Nucleus sampling probability. Defaults are model/mode-specific "
            "(Qwen3 thinking: 0.95; non-thinking: 0.8)."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k",
        help="Top-k sampling. Defaults are model-specific (Qwen3: 20).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=None,
        dest="do_sample",
        help="Enable sampling (default).",
    )
    parser.add_argument(
        "--no-sample",
        action="store_false",
        dest="do_sample",
        help="Greedy decoding instead of sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        dest="repetition_penalty",
        help="Repetition penalty. Default: disabled.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=True,
        dest="think",
        help=(
            "Enable internal reasoning for models that support it; reasoning is "
            "logged and removed from the returned answer (default: on)."
        ),
    )
    parser.add_argument(
        "--no-think",
        action="store_false",
        dest="think",
        help="Disable internal reasoning during generation.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        dest="gpu_memory_utilization",
        help="Fraction of GPU memory vLLM may use. Default: 0.9.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="Number of GPUs for tensor parallelism. Default: 1.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.environ.get("VLLM_MAX_MODEL_LEN", 0)) or None,
        dest="max_model_len",
        help=(
            "Override the model's maximum context length. "
            "Also read from VLLM_MAX_MODEL_LEN env var. "
            "Reducing this (e.g. 16384) drastically speeds up the profiling "
            "warmup run on Blackwell GPUs with CUDA < 12.9."
        ),
    )
    parser.add_argument(
        "--hf-overrides",
        type=json.loads,
        default=None,
        dest="hf_overrides",
        help=(
            "Raw JSON dict of HF config overrides passed to vLLM, e.g. to enable "
            "YaRN rope scaling and extend a model's context past its native "
            "max-model-len: "
            '\'{"rope_scaling": {"rope_type": "yarn", "factor": 3.2, '
            '"original_max_position_embeddings": 40960}, '
            '"max_position_embeddings": 131072}\'. '
            "Combine with --max-model-len set to the new (post-scaling) length."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        dest="dtype",
        help="Weight dtype passed to vLLM (e.g. auto, bfloat16, float16). Default: auto.",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        dest="quantization",
        help="Quantization method (e.g. awq, gptq, fp8). Default: none.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        dest="trust_remote_code",
        help="Allow models with custom remote code.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        default=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        dest="enforce_eager",
        help=(
            "Disable CUDA graph compilation and run in eager mode. "
            "Use on Blackwell (SM 12.x) GPUs when the system CUDA toolkit "
            "is < 12.9 and CUTLASS JIT compilation hangs. "
            "Also activated by env var VLLM_ENFORCE_EAGER=1."
        ),
    )
    parser.add_argument(
        "--moe-backend",
        default=None,
        dest="moe_backend",
        help=(
            "Backend for MoE expert computation. Options: auto, triton, cutlass, deep_gemm, "
            "flashinfer_cutlass, emulation. Default: per-model config or vLLM auto. "
            "Use 'triton' on Blackwell GPUs with CUDA < 12.9 to avoid CUTLASS JIT hangs."
        ),
    )
    parser.add_argument(
        "--linear-backend",
        default=None,
        dest="linear_backend",
        help=(
            "Backend for linear (GEMM) layers. Options: auto, triton, cutlass, torch, "
            "flashinfer_cutlass, deep_gemm, emulation. Default: per-model config or vLLM auto. "
            "Use 'triton' on Blackwell GPUs with CUDA < 12.9 to avoid CUTLASS JIT hangs."
        ),
    )
    parser.add_argument(
        "--enable-flashinfer-autotune",
        action="store_true",
        default=None,
        dest="enable_flashinfer_autotune",
        help="Run FlashInfer autotuning during kernel warmup (vLLM default).",
    )
    parser.add_argument(
        "--no-enable-flashinfer-autotune",
        action="store_false",
        dest="enable_flashinfer_autotune",
        help=(
            "Disable FlashInfer autotuning. "
            "Use on Blackwell (SM 12.x) GPUs with CUDA < 12.9 where autotune JIT hangs."
        ),
    )


def configure_subparsers(subparsers):
    parser = subparsers.add_parser(
        "vllm",
        help="vLLM backend — fast PagedAttention inference for any HF model.",
    )
    configure_global_arguments(parser)
