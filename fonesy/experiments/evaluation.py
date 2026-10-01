"""``evaluate`` sub-command: LLM generation, self-correction and scoring."""

import csv as csv_module
import json
import os
import signal
from dataclasses import asdict, dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Optional

from fonesy import datasets
from fonesy.datasets.loaders import make_dataloader
from fonesy.general_utils import log
from fonesy.utils.factories import LLMFactory
from fonesy.statistics.statistics import Statistics, CorrectionStatus
from fonesy.statistics.metrics import Metrics
from fonesy.evaluation.dimacs import (
    verify_cnf, try_preamble_fix, try_clause_fix, classify_error_types,
    _ERROR_PATTERNS,
)
from fonesy.evaluation.solver import _DEFAULT_EPSILON, _DEFAULT_DELTA, _run_hard_timeout
from fonesy.evaluation.formula import CNFFormula


# Generation-length cap used for every model, benchmark and prompting mode.
DEFAULT_MAX_NEW_TOKENS = 32768


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_global_arguments(parser):
    """Configure global arguments that are shared across models and datasets."""
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for evaluation."
    )
    parser.add_argument(
        "--budget", type=int, default=5, help="Max self-correction attempts."
    )
    parser.add_argument(
        "--cnf-format",
        choices=["dimacs", "nat", "pysat", "cpmpy", "sympy"],
        default="dimacs",
        help="CNF output format expected from the LLM (dimacs, nat, pysat, cpmpy, or sympy).",
    )
    parser.add_argument(
        "--mc-workers", type=int, default=4,
        help=(
            "Background worker processes for model counting (see "
            "fonesy.evaluation.mc_pool.McWorkerPool), so ApproxMC/exact-SAT "
            "work never blocks the GPU-bound generation loop. 0 disables the "
            "pool and falls back to the fully synchronous pre-async behavior."
        ),
    )
    parser.add_argument(
        "--mc-harvest-timeout", type=int, default=600,
        help=(
            "Max seconds to wait for outstanding background model-counting "
            "jobs once generation has finished every sample, before giving "
            "up on whatever's still running (default 600s / 10 min)."
        ),
    )


def add_dataset_arguments(parser):
    """Add the shared evaluation flags and the ``<dataset> <llm>`` sub-commands."""
    configure_global_arguments(parser)
    datasets.configure_subparsers(parser.add_subparsers(dest="dataset"))


def configure_subparsers(subparsers):
    """Register the ``evaluate`` sub-command."""
    parser = subparsers.add_parser("evaluate", help="Evaluate an LLM on a benchmark")
    add_dataset_arguments(parser)
    parser.set_defaults(func=main)


# ---------------------------------------------------------------------------
# Self-correction helpers
# ---------------------------------------------------------------------------

_CORRECT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "correct.txt"
_NAT_CORRECT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "nat_correct.txt"
_PYSAT_CORRECT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "pysat_correct.txt"
_CPMPY_CORRECT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "cpmpy_correct.txt"
_SYMPY_CORRECT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "sympy_correct.txt"
_CORRECT_PROMPT_TEMPLATES: dict = {}  # per-path cache: Path → str


def _get_correct_prompt_template(prompt_path: Path = _CORRECT_PROMPT_PATH) -> str:
    if prompt_path not in _CORRECT_PROMPT_TEMPLATES:
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
        _CORRECT_PROMPT_TEMPLATES[prompt_path] = prompt_path.read_text()
    return _CORRECT_PROMPT_TEMPLATES[prompt_path]


def build_correction_prompt(
    problem: str,
    pred: str,
    error_message: str,
    status: CorrectionStatus,
    prompt_path: Path = _CORRECT_PROMPT_PATH,
    code_guidance: dict = None,
    variable_names=None,
    cnf_format: Optional[str] = None,
    allow_auxiliary_variables: bool = True,
):
    """Build the self-correction prompt, injecting all placeholders in one call."""
    template = _get_correct_prompt_template(prompt_path)
    if not allow_auxiliary_variables:
        template = "\n".join(
            (
                "- Do not use CardEnc, PBEnc, or any encoder that creates "
                "auxiliary variables; build the formula using only the listed "
                "primary variables"
                if "When using CardEnc or PBEnc" in line
                else line
            )
            for line in template.splitlines()
        )
    # The [ERRORS] block is only included when there is an error to report.
    error_str = (
        f"[ERRORS]\n{_enrich_errors_for_prompt(error_message, code_guidance=code_guidance)}"
        if status != CorrectionStatus.OK else ""
    )
    problem_text = problem.strip()
    if variable_names:
        mapping = "\n".join(
            f"  {index}: {name}" for index, name in enumerate(variable_names, start=1)
        )
        if cnf_format in {"dimacs", "pysat"}:
            if allow_auxiliary_variables:
                mapping_rule = (
                    "Use the displayed integer for each named primary variable; "
                    "the names are semantic labels, not output literals. Primary "
                    f"IDs are exactly 1..{len(variable_names)}. Any encoder-created "
                    "auxiliary must use a larger ID."
                )
            else:
                mapping_rule = (
                    "Use the displayed integer for each named primary variable; "
                    "the names are semantic labels, not output literals. Primary "
                    f"IDs are exactly 1..{len(variable_names)}. Do not introduce "
                    "auxiliary variables or use any other IDs."
                )
        else:
            mapping_rule = (
                "Use only these symbolic names. Each name always maps to the "
                "displayed DIMACS index, regardless of declaration or "
                "occurrence order."
            )
        problem_text += (
            "\n\n[REQUIRED VARIABLE MAPPING]\n"
            + mapping
            + "\n"
            + mapping_rule
        )
    return template.format(
        input=problem_text,
        prediction=pred.strip(),
        error=error_str,
    )


def parse_correction_response(response: str):
    """
    Parse the LLM correction response.

    Returns:
        (CorrectionStatus.OK,      None)      → CNF is correct
        (CorrectionStatus.FIXED,   response)  → CNF was fixed
        (CorrectionStatus.INVALID, None)      → unexpected output

    A response is treated as FIXED when either the ``[[FIXED]]`` marker is
    present *or* when the model directly provides a ``<cnf>…</cnf>`` block
    without the marker (observed with Llama-3.x models).
    """
    response = response.strip()
    if "[[OK]]" in response:
        return CorrectionStatus.OK, None
    if "[[FIXED]]" in response or ("<cnf>" in response and "</cnf>" in response):
        return CorrectionStatus.FIXED, response
    return CorrectionStatus.INVALID, None


# ---------------------------------------------------------------------------
# Logging helpers for CNF display
# ---------------------------------------------------------------------------

_CODE_EXPLANATIONS = {
    "no_cnf_tags":           "missing <cnf>…</cnf> wrapper",
    "invalid_header":        'first line must be "p cnf <vars> <clauses>"',
    "clause_no_zero":        "clause lines must end with 0",
    "empty_clause":          "empty clause (bare 0 line)",
    "invalid_content":       "non-integer value in header or clause",
    "missing_num_vars":      "variable count missing or unparseable in header",
    "clause_count_mismatch": "declared clause count doesn't match actual",
    "literal_out_of_range":  "literal exceeds declared variable count",
}

_CODE_GUIDANCE: dict = {
    "no_cnf_tags": (
        "Wrap your entire DIMACS output in <cnf>...</cnf> tags — "
        "the very first token must be <cnf> and the last must be </cnf>."
    ),
    "invalid_header": (
        'The first non-comment line must be exactly "p cnf <num_vars> <num_clauses>" '
        '(e.g. "p cnf 3 4"). Check spelling, spacing, and that no extra tokens appear.'
    ),
    "clause_no_zero": (
        'Append " 0" to the end of every clause line. '
        'The 0 is a mandatory clause terminator — e.g. "1 -2 3 0", not "1 -2 3".'
    ),
    "empty_clause": (
        'Remove any line that contains only "0". '
        "An empty clause (bare 0 line) is always unsatisfiable. "
        "Every clause must contain at least one literal before the 0 terminator."
    ),
    "invalid_content": (
        "Replace every non-integer token with a plain integer. "
        "Headers and clause lines must contain only integers — "
        "no variable names, commas, or other text."
    ),
    "missing_num_vars": (
        'Add "p cnf <num_vars> <num_clauses>" as the very first line. '
        "<num_vars> is the highest variable index you use anywhere; "
        "<num_clauses> is the total number of clause lines that follow."
    ),
    "clause_count_mismatch": (
        'Update the second number in the "p cnf X Y" header so Y equals the number '
        "of clause lines you actually wrote. Count every line ending with 0."
    ),
    "literal_out_of_range": (
        'Raise <num_vars> in the "p cnf" header to be at least as large as the '
        "highest variable index used in any clause, or replace the out-of-range literals."
    ),
}

_AUTO_FIX_DESCRIPTIONS = {
    "Clause terminator": "appended missing 0 to each clause line",
    "Preamble":          "corrected clause count in header",
    "Boolean variable name": (
        "converted positional boolvar names to name= keywords"
    ),
}


def _code_for_error_line(line: str) -> Optional[str]:
    """Return the error code matching *line*, or None if unrecognised."""
    for code, pattern in _ERROR_PATTERNS:
        if pattern in line:
            return code
    return None


def _enrich_errors_for_prompt(error_message: str, code_guidance: dict = None) -> str:
    """Augment raw [ERROR] lines with a short actionable hint for the LLM.

    The first occurrence of each error code is followed by a '-> Fix: ...' line.
    Subsequent occurrences keep their raw line (for line-number context) without
    repeating the guidance.
    """
    guidance_map = code_guidance if code_guidance is not None else _CODE_GUIDANCE
    lines = [l.strip() for l in error_message.splitlines() if l.strip()]
    if not lines:
        return error_message.strip()

    seen: set = set()
    parts: list = []
    for line in lines:
        code = _code_for_error_line(line)
        guidance = guidance_map.get(code, "") if code else ""
        if guidance and code not in seen:
            seen.add(code)
            parts.append(f"{line}\n  -> Fix: {guidance}")
        else:
            parts.append(line)
    return "\n".join(parts)


def _extract_cnf(text: str) -> str:
    """Return the content of the last <cnf>…</cnf> block, or the text stripped."""
    start = text.rfind("<cnf>")
    end = text.rfind("</cnf>")
    if start != -1 and end != -1 and end > start:
        return text[start + 5 : end].strip()
    return text.strip()


_DEFAULT_FIX_FNS = [
    (try_clause_fix, "Clause terminator"),
    (try_preamble_fix, "Preamble"),
]


def _try_auto_fix(
    pred: str,
    verify_cnf_fn,
    log_prefix: str = "│   ",
    fix_fns=None,
    logger=log,
) -> Optional[str]:
    """Run the sequential auto-fix chain on *pred*.

    *fix_fns* is a list of ``(fix_fn, label)`` pairs to try in order.
    Defaults to the DIMACS clause-terminator → preamble chain.

    Returns the corrected text when any fixer produces a valid formula,
    ``None`` otherwise.
    """
    pairs = fix_fns if fix_fns is not None else _DEFAULT_FIX_FNS
    for fix_fn, fix_label in pairs:
        fixed = fix_fn(pred, llm_response=True)
        if fixed is not None:
            _, _, fixed_status = verify_cnf_fn(fixed, True)
            if fixed_status == CorrectionStatus.OK:
                logger(
                    f"{log_prefix}auto-fix : {fix_label} → {_AUTO_FIX_DESCRIPTIONS.get(fix_label, 'applied')}",
                    "INFO",
                )
                logger(f"{log_prefix}fixed CNF:\n{_extract_cnf(fixed)}", "INFO")
                return fixed
            logger(f"{log_prefix}auto-fix : {fix_label} → no change (still invalid after fix)", "INFO")
        else:
            logger(f"{log_prefix}auto-fix : {fix_label} → not applicable", "INFO")
    return None


def _fmt_errors(error_message: str, codes: list, code_explanations: dict = None) -> str:
    """Plain-English summary of error codes."""
    expl_map = code_explanations if code_explanations is not None else _CODE_EXPLANATIONS
    descriptions = [expl_map.get(c, c) for c in codes]
    return "; ".join(descriptions) if descriptions else error_message.strip()


@dataclass
class _CorrectionRequest:
    """All per-sample inputs needed by the shared correction state machine."""

    problem: str
    pred: str
    verify_cnf_fn: object
    max_attempts: int = 3
    status_before: Optional[CorrectionStatus] = None
    classify_errors_fn: object = None
    correct_prompt_path: Optional[Path] = None
    fix_fns: object = None
    code_guidance: object = None
    code_explanations: object = None
    variable_names: object = None
    cnf_format: Optional[str] = None
    allow_auxiliary_variables: bool = True


@dataclass
class _CorrectionState:
    """Mutable progress for one request while correction rounds are batched."""

    request: _CorrectionRequest
    pred: str
    status_before: CorrectionStatus
    correction_used: CorrectionStatus = CorrectionStatus.INVALID
    status_response: CorrectionStatus = CorrectionStatus.INVALID
    attempts: list = field(default_factory=list)
    attempt: int = 0
    done: bool = False
    status_after: Optional[CorrectionStatus] = None
    log_messages: list = field(default_factory=list)

    def emit(self, message: str, level: str = "INFO") -> None:
        self.log_messages.append((message, level))

    def result(self):
        return (
            self.pred,
            self.status_before,
            self.status_after,
            self.correction_used,
            self.status_response,
            self.attempts,
        )


def _initialize_correction_state(
    request: _CorrectionRequest,
    *,
    buffer_logs: bool,
) -> _CorrectionState:
    """Verify and apply the pre-LLM auto-fix for one correction request."""
    if request.status_before is None:
        _, _, status_before = request.verify_cnf_fn(request.pred, True)
    else:
        status_before = request.status_before

    state = _CorrectionState(
        request=request,
        pred=request.pred,
        status_before=status_before,
    )
    logger = state.emit if buffer_logs else log

    if status_before == CorrectionStatus.OK:
        state.status_after = status_before
        state.status_response = status_before
        state.done = True
        return state

    fixed = _try_auto_fix(
        state.pred,
        request.verify_cnf_fn,
        log_prefix="│   ",
        fix_fns=request.fix_fns,
        logger=logger,
    )
    if fixed is not None:
        state.pred = fixed
        state.status_after = CorrectionStatus.OK
        state.correction_used = CorrectionStatus.AUTO_FIXED
        state.status_response = CorrectionStatus.AUTO_FIXED
        state.done = True
    return state


def _run_correction_rounds(
    requests,
    generate_responses,
    *,
    buffer_logs=False,
):
    """Run one shared correction state machine for singleton or batch callers.

    ``generate_responses`` accepts the ordered prompts for one active round
    and must return exactly one response per prompt.  No fallback or retry is
    attempted when generation fails or violates that contract.
    """
    states = [
        _initialize_correction_state(request, buffer_logs=buffer_logs)
        for request in requests
    ]

    while True:
        pending = []
        prompts = []
        for state in states:
            if state.done:
                continue
            request = state.request
            if state.attempt >= request.max_attempts:
                state.done = True
                continue

            # A prior FIXED response may already have produced a valid CNF.
            _, error_message, current_status = request.verify_cnf_fn(
                state.pred, True
            )
            if current_status == CorrectionStatus.OK:
                state.done = True
                continue

            classify_fn = (
                request.classify_errors_fn
                if request.classify_errors_fn is not None
                else classify_error_types
            )
            prompt_path = (
                request.correct_prompt_path
                if request.correct_prompt_path is not None
                else _CORRECT_PROMPT_PATH
            )
            codes = classify_fn(error_message)
            logger = state.emit if buffer_logs else log
            logger(
                f"│   correction {state.attempt + 1}/{request.max_attempts}",
                "INFO",
            )
            logger(f"│     current CNF  :\n{_extract_cnf(state.pred)}", "INFO")
            logger(
                "│     why invalid  : "
                + _fmt_errors(
                    error_message,
                    codes,
                    code_explanations=request.code_explanations,
                ),
                "INFO",
            )
            logger(f"│     codes        : {codes}", "DEBUG")
            logger(
                f"│     full errors  : {error_message.strip()[:300]}",
                "DEBUG",
            )

            correction_prompt = build_correction_prompt(
                request.problem,
                state.pred,
                error_message,
                current_status,
                prompt_path=prompt_path,
                code_guidance=request.code_guidance,
                variable_names=request.variable_names,
                cnf_format=request.cnf_format,
                allow_auxiliary_variables=request.allow_auxiliary_variables,
            )
            logger(
                f"│     correction prompt:\n{correction_prompt}",
                "DEBUG",
            )
            pending.append((state, error_message))
            prompts.append(correction_prompt)

        if not pending:
            break

        responses = list(generate_responses(prompts))
        if len(responses) != len(prompts):
            raise RuntimeError(
                "Correction generation returned "
                f"{len(responses)} responses for {len(prompts)} prompts"
            )

        for (state, error_message), response in zip(pending, responses):
            request = state.request
            logger = state.emit if buffer_logs else log
            logger(f"│     correction response:\n{response}", "DEBUG")
            status_response, corrected = parse_correction_response(response)
            state.status_response = status_response
            state.attempts.append({
                "attempt": state.attempt + 1,
                "errors": error_message.strip(),
                "raw_response": response,
                "status": status_response.name,
                "cnf": _extract_cnf(corrected) if corrected else None,
            })

            if status_response == CorrectionStatus.OK:
                logger(
                    "│     model verdict : OK — model says CNF is correct "
                    "(stopping loop)",
                    "INFO",
                )
                state.done = True
            elif status_response == CorrectionStatus.FIXED and corrected:
                state.pred = corrected
                state.correction_used = CorrectionStatus.FIXED
                logger("│     model verdict : FIXED", "INFO")
                logger(
                    f"│     proposed CNF  :\n{_extract_cnf(state.pred)}",
                    "INFO",
                )
                post_fixed = _try_auto_fix(
                    state.pred,
                    request.verify_cnf_fn,
                    log_prefix="│     ",
                    fix_fns=request.fix_fns,
                    logger=logger,
                )
                if post_fixed is not None:
                    state.pred = post_fixed
                    state.done = True
            else:
                logger(
                    "│     model verdict : unexpected response format "
                    "(not [[OK]] or [[FIXED]])",
                    "INFO",
                )

            state.attempt += 1

    # The model's [[OK]] verdict is not trusted: re-verify every state that
    # was not already settled by the initial verification or the auto-fix.
    for state in states:
        if state.status_after is None:
            _, _, state.status_after = state.request.verify_cnf_fn(
                state.pred, True
            )
    return states


def _self_correct_batch(requests, model):
    """Correct an ordered request batch with one backend call per round."""
    if not requests:
        return []

    states = _run_correction_rounds(
        requests,
        lambda prompts: model.generate_batch(prompts),
        buffer_logs=True,
    )
    return states


def self_correct(
    model,
    problem,
    pred,
    verify_cnf_fn,
    max_attempts=3,
    status_before=None,
    classify_errors_fn=None,
    correct_prompt_path=None,
    fix_fns=None,
    code_guidance=None,
    code_explanations=None,
    variable_names=None,
    cnf_format: Optional[str] = None,
    allow_auxiliary_variables: bool = True,
):
    """Run the self-correction loop.

    Before querying the LLM, the function attempts a cheap programmatic fix for
    clause-terminator and preamble-only errors.  If that succeeds the LLM is
    never called and ``correction_used`` is
    :attr:`~fonesy.statistics.statistics.CorrectionStatus.AUTO_FIXED`.

    Args:
        status_before: Pre-computed status from a prior ``verify_cnf_fn`` call.
            When provided the initial re-verification is skipped.

    Returns:
        (final_pred, status_before, status_after, correction_used, status_response, attempts)
        where correction_used is a CorrectionStatus enum value:
          - CorrectionStatus.AUTO_FIXED → preamble fixed programmatically
          - CorrectionStatus.FIXED      → at least one LLM correction was applied
          - CorrectionStatus.INVALID    → no correction was applied
        and attempts is a list of per-LLM-call dicts recording the errors sent,
        the raw model response, the parsed status, and the proposed CNF (if any).
    """
    request = _CorrectionRequest(
        problem=problem,
        pred=pred,
        verify_cnf_fn=verify_cnf_fn,
        max_attempts=max_attempts,
        status_before=status_before,
        classify_errors_fn=classify_errors_fn,
        correct_prompt_path=correct_prompt_path,
        fix_fns=fix_fns,
        code_guidance=code_guidance,
        code_explanations=code_explanations,
        variable_names=variable_names,
        cnf_format=cnf_format,
        allow_auxiliary_variables=allow_auxiliary_variables,
    )

    def _generate_single(prompts):
        return [model.generate(prompts[0])]

    return _run_correction_rounds(
        [request],
        _generate_single,
        buffer_logs=False,
    )[0].result()


# ---------------------------------------------------------------------------
# Crash-safe per-sample JSONL sink
# ---------------------------------------------------------------------------


def _write_sample_record(fh, record: dict) -> None:
    """Append one JSON record and flush it to disk (survives crashes)."""
    fh.write(json.dumps(record, default=str) + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except (AttributeError, OSError):
        pass  # in-memory writers have no file descriptor


def _write_resolved_corrections(fh, pending_sample_records: dict, resolved: list) -> None:
    """Append the resolved copy of samples written with ``outcome_pending``.

    *resolved* is the ``[(sample_key, SampleOutcome), ...]`` returned by
    ``Metrics.harvest_ready``/``harvest``. The file is append-only: readers keep
    the last record of each ``(batch_idx, sample_idx)``.
    """
    if fh is None:
        return
    for sample_key, outcome in resolved:
        record = pending_sample_records.pop(sample_key, None)
        if record is None:
            continue  # already resolved synchronously, or an unrecognized key
        record = dict(record)  # don't mutate the dict eval_dataloader still holds
        record["outcome"] = asdict(outcome)
        record.pop("outcome_pending", None)
        _write_sample_record(fh, record)


def _cnf_artifact_paths(cnf_dir: Path, sample_id: str):
    cnf_path = cnf_dir / f"{sample_id}.cnf"
    return cnf_path, cnf_path.with_suffix(cnf_path.suffix + ".meta.json")


def _remove_cnf_artifacts(cnf_path: Path, metadata_path: Path) -> None:
    """Remove accepted and partial artifacts for one sample."""
    for path in (
        cnf_path,
        metadata_path,
        Path(f"{cnf_path}.tmp.{os.getpid()}"),
        Path(f"{metadata_path}.tmp.{os.getpid()}"),
    ):
        path.unlink(missing_ok=True)


def _write_cnf_artifacts(cnf_path: Path, metadata_path: Path, cnf_text: str, metadata: dict) -> None:
    """Write ``<id>.cnf`` and its ``.meta.json`` sidecar atomically.

    The sidecar is published first and the CNF last, so ``compile`` never
    sees a CNF without its metadata.
    """
    cnf_path.parent.mkdir(parents=True, exist_ok=True)
    cnf_tmp = Path(f"{cnf_path}.tmp.{os.getpid()}")
    metadata_tmp = Path(f"{metadata_path}.tmp.{os.getpid()}")
    cnf_path.unlink(missing_ok=True)
    try:
        cnf_tmp.write_text(cnf_text, encoding="utf-8")
        metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        os.replace(metadata_tmp, metadata_path)
        os.replace(cnf_tmp, cnf_path)
    finally:
        cnf_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)


def _clear_experiment_cnf_artifacts(output_dir: Path, experiment_name: str) -> None:
    """Delete the CNFs saved by a previous run of the same experiment."""
    cnf_dir = output_dir / "cnf" / experiment_name
    if not cnf_dir.is_dir():
        return
    stale_paths = set(cnf_dir.glob("*.cnf"))
    stale_paths.update(cnf_dir.glob("*.cnf.meta.json"))
    stale_paths.update(cnf_dir.glob("*.cnf.tmp.*"))
    stale_paths.update(cnf_dir.glob("*.cnf.meta.json.tmp.*"))
    for path in stale_paths:
        if path.is_file():
            path.unlink()


def _dataset_variable_spec(dataset, sample_id):
    """Return the :class:`VariableSpec` of *sample_id* (``None`` if unavailable)."""
    getter = getattr(dataset, "get_variable_spec", None)
    return getter(sample_id) if getter is not None else None


# Verifiers that parse or execute model-generated content (NAT to CNF
# conversion, execution of PySAT/CPMpy/SymPy programs) run in a subprocess
# that is killed after this many seconds; the prediction is then invalid.
_VERIFY_TIMEOUT_FORMATS = frozenset({"nat", "pysat", "cpmpy", "sympy"})
_VERIFY_HARD_TIMEOUT_SECONDS = 120


def _bind_sample_verifier(verify_fn, cnf_format: str, variable_spec, primary_num_vars):
    """Bind authoritative per-sample mapping/count metadata to a verifier."""
    if cnf_format in {"nat", "cpmpy", "sympy"}:
        names = getattr(variable_spec, "variable_names", None)
        if names is None:
            raise ValueError(
                f"{cnf_format} evaluation requires an authoritative "
                "VariableSpec; the dataset did not provide one"
            )
        if primary_num_vars is not None:
            names = names[:primary_num_vars]
        bound = partial(verify_fn, variable_names=names)
    elif cnf_format in {"dimacs", "pysat"} and primary_num_vars is not None:
        bound = partial(verify_fn, primary_num_vars=primary_num_vars)
    else:
        bound = verify_fn

    if cnf_format in _VERIFY_TIMEOUT_FORMATS:
        _bound_inner = bound

        def bound(text, llm_response=True):
            try:
                return _run_hard_timeout(
                    _bound_inner,
                    (text, llm_response),
                    timeout=_VERIFY_HARD_TIMEOUT_SECONDS,
                    label=f"{cnf_format} verification",
                )
            except TimeoutError:
                error_message = (
                    "[ERROR]: Verification timed out after "
                    f"{_VERIFY_HARD_TIMEOUT_SECONDS}s — likely a pathological "
                    "prediction triggering unbounded parsing/execution "
                    "(treated as INVALID, not a crash)\n"
                )
                return (
                    CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=text),
                    error_message,
                    CorrectionStatus.INVALID,
                )
            except RuntimeError as exc:
                error_message = f"[ERROR]: Verification subprocess failed: {exc}\n"
                return (
                    CNFFormula(num_vars=None, num_clauses=None, clauses=[], raw=text),
                    error_message,
                    CorrectionStatus.INVALID,
                )

    # Predictions are programs that may be non-deterministic: verify each
    # candidate once and reuse the result for correction, scoring and saving.
    cache = {}

    @wraps(verify_fn)
    def memoized(text, llm_response=True):
        key = (text, bool(llm_response))
        if key not in cache:
            cache[key] = bound(text, llm_response)
        return cache[key]

    return memoized


# ---------------------------------------------------------------------------
# Unified evaluation loop
# ---------------------------------------------------------------------------


def eval_dataloader(
    model,
    dataloader,
    metrics,
    statistics,
    max_attempts,
    output_dir=None,
    experiment_name=None,
    verify_cnf_fn=None,
    classify_errors_fn=None,
    fix_fns=None,
    correct_prompt_path=None,
    code_guidance=None,
    code_explanations=None,
    samples_output_h=None,
    cnf_format="dimacs",
    mc_harvest_timeout=600,
):
    """Generate, self-correct and score every sample of *dataloader*.

    Each prediction is verified with the format-specific verifier, repaired
    (auto-fix, then up to *max_attempts* LLM correction rounds) and scored
    against the ground-truth DIMACS by model counting (see
    :class:`fonesy.statistics.metrics.Metrics`). With a background model-counting
    pool (``Metrics(mc_workers>0)``) scoring never blocks generation: finished
    jobs are collected after every batch and the remaining ones are awaited for
    at most *mc_harvest_timeout* seconds at the end.

    Returns the list of per-sample ``{input, target, prediction, attempts}``.
    """
    _verify_fn = verify_cnf_fn if verify_cnf_fn is not None else verify_cnf
    _classify_fn = classify_errors_fn if classify_errors_fn is not None else classify_error_types

    # "program" datasets (auto-nesy-bench) return the oracle path as target.
    dataset_type = getattr(dataloader.dataset, "dataset_type", "dimacs")
    allow_auxiliary_variables = getattr(
        dataloader.dataset, "allow_auxiliary_variables", True
    )
    total_batches = len(dataloader)
    results = []

    # sample_key -> samples.jsonl record whose model count is still running.
    pending_sample_records: dict = {}

    for batch_idx, batch in enumerate(dataloader, start=1):
        inputs, targets, formulas, ids = batch

        n = len(inputs)
        sample_ids = [
            Path(tgt).stem if dataset_type == "program" else problem_id
            for tgt, problem_id in zip(targets, ids)
        ]
        variable_specs = [
            _dataset_variable_spec(dataloader.dataset, sample_id)
            for sample_id in sample_ids
        ]
        primary_counts = [getattr(spec, "primary_num_vars", None) for spec in variable_specs]
        sample_verify_fns = [
            _bind_sample_verifier(_verify_fn, cnf_format, spec, primary_num_vars)
            for spec, primary_num_vars in zip(variable_specs, primary_counts)
        ]
        log(f"[EVAL] {'═' * 10} Batch {batch_idx}/{total_batches}  ({n} samples) {'═' * 10}", "INFO")

        # 1. Generate the whole batch at once.
        preds = list(model.generate_batch(list(inputs)))
        preds_initial = list(preds)  # snapshot before any corrections (needed for preamble_err below)
        log(f"[EVAL] Batch generation: {n} prompts → done", "INFO")

        # 2. Verify every prediction.
        _verifications = [
            verify_fn(pred, True)
            for pred, verify_fn in zip(preds, sample_verify_fns)
        ]
        statuses_before = [v[2] for v in _verifications]
        errors_before = [v[1] for v in _verifications]

        # 3. Self-correct the invalid ones, one batched LLM call per round.
        correction_requests = []
        for i, (formula, pred) in enumerate(zip(formulas, preds)):
            variable_names = getattr(
                variable_specs[i], "variable_names", None
            )
            if variable_names is not None and primary_counts[i] is not None:
                variable_names = variable_names[:primary_counts[i]]
            correction_requests.append(_CorrectionRequest(
                problem=formula,
                pred=pred,
                verify_cnf_fn=sample_verify_fns[i],
                max_attempts=max_attempts,
                status_before=statuses_before[i],
                classify_errors_fn=_classify_fn,
                correct_prompt_path=correct_prompt_path,
                fix_fns=fix_fns,
                code_guidance=code_guidance,
                code_explanations=code_explanations,
                variable_names=variable_names,
                cnf_format=cnf_format,
                allow_auxiliary_variables=allow_auxiliary_variables,
            ))
        correction_states = _self_correct_batch(
            correction_requests, model
        )

        # 4. Score, save and log every sample in order.
        batch_ok = batch_auto = batch_invalid = 0
        for i, (inp, tgt, formula, problem_id) in enumerate(
            zip(inputs, targets, formulas, sample_ids)
        ):
            status_before = statuses_before[i]
            err_before = errors_before[i]

            sample_id = problem_id
            verify_i = sample_verify_fns[i]
            variable_spec = variable_specs[i]
            variable_names = getattr(variable_spec, "variable_names", None)
            if variable_names is not None and primary_counts[i] is not None:
                variable_names = variable_names[:primary_counts[i]]

            log(f"[EVAL] ┌── [{i + 1}/{n}] id={sample_id}", "INFO")
            log(f"[EVAL] │   prompt:\n{inp}", "DEBUG")
            log(f"[EVAL] │   raw output:\n{preds_initial[i]}", "DEBUG")

            codes0 = _classify_fn(err_before)
            if status_before == CorrectionStatus.OK:
                log(f"[EVAL] │   initial CNF  : OK", "INFO")
                log(f"[EVAL] │   CNF          :\n{_extract_cnf(preds_initial[i])}", "INFO")
            else:
                log(f"[EVAL] │   initial CNF  : INVALID", "INFO")
                log(f"[EVAL] │   CNF          :\n{_extract_cnf(preds_initial[i])}", "INFO")
                log(f"[EVAL] │   why invalid  : {_fmt_errors(err_before, codes0, code_explanations=code_explanations)}", "INFO")

            state = correction_states[i]
            (
                preds[i],
                before,
                after,
                corrected,
                status_response,
                attempts,
            ) = state.result()
            for message, level in state.log_messages:
                log(message, level)

            statistics.update(before, after, corrected, status_response)

            # The original error of a fixed prediction still counts as a syntax error.
            preamble_err = None
            initial_err = None
            if corrected == CorrectionStatus.AUTO_FIXED:
                preamble_err = err_before
            elif status_before == CorrectionStatus.INVALID and after == CorrectionStatus.OK:
                initial_err = err_before

            if status_before == CorrectionStatus.OK and preds[i] == preds_initial[i]:
                final_cnf, final_error, final_status = _verifications[i]
            else:
                final_cnf, final_error, final_status = verify_i(preds[i], True)
            if final_status == CorrectionStatus.INVALID:
                final_cnf = None

            target_dimacs_str = None
            outcome = None  # SampleOutcome

            sample_key = (batch_idx, i)

            if dataset_type == "program":
                target_dimacs_str = Path(tgt).with_suffix(".dimacs").read_text()
            else:
                target_dimacs_str = tgt
            outcome = metrics.submit(
                preds[i], target_dimacs_str, verify_i,
                preamble_fix_error=preamble_err, initial_error=initial_err,
                problem_id=sample_id, sample_key=sample_key,
                verified_cnf=final_cnf,
                verification_error=final_error,
                already_verified=True,
                primary_num_vars=primary_counts[i],
            )

            # Save valid CNFs for the `compile` sub-command.
            if output_dir is not None:
                cnf_dir = output_dir / "cnf"
                if experiment_name:
                    cnf_dir = cnf_dir / experiment_name
                cnf_path, metadata_path = _cnf_artifact_paths(cnf_dir, sample_id)
                if final_status == CorrectionStatus.OK and final_cnf is not None:
                    metadata = {
                        "mapping_version": 1,
                        "variable_names": list(variable_names) if variable_names else None,
                        "primary_num_vars": primary_counts[i],
                        "auxiliary_num_vars": getattr(
                            final_cnf, "auxiliary_num_vars", 0
                        ),
                    }
                    _write_cnf_artifacts(
                        cnf_path,
                        metadata_path,
                        final_cnf.raw,
                        metadata,
                    )
                else:
                    # Drop the CNF of a previous run for a now-invalid sample.
                    _remove_cnf_artifacts(cnf_path, metadata_path)

            results.append({"input": inp, "target": tgt, "prediction": preds[i], "attempts": attempts})

            if samples_output_h is not None:
                record = {
                    "experiment_name": experiment_name,
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                    "dataset_type": dataset_type,
                    "cnf_format": cnf_format,
                    "id": sample_id,
                    "correction_status_before": before.name,
                    "correction_status_after": after.name,
                    "correction_used": corrected.name,
                    "status_response": status_response.name,
                    "target_cnf": target_dimacs_str,
                    "predicted_cnf": _extract_cnf(preds[i]),
                    "mapping_version": 1,
                    "variable_names": list(variable_names) if variable_names else None,
                    "primary_num_vars": primary_counts[i],
                    "auxiliary_num_vars": (
                        getattr(final_cnf, "auxiliary_num_vars", None)
                        if final_cnf is not None else None
                    ),
                    "predicted_dimacs": final_cnf.raw if final_cnf is not None else None,
                    "verification_status": final_status.name,
                    "verification_error": final_error or None,
                }
                if outcome is not None:
                    record["outcome"] = asdict(outcome)
                else:
                    # Model counting still running: rewritten when harvested.
                    record["outcome_pending"] = True
                    pending_sample_records[sample_key] = record
                _write_sample_record(samples_output_h, record)

            if target_dimacs_str is not None:
                final_dimacs = final_cnf.raw if final_cnf else _extract_cnf(preds[i])
                log(f"[EVAL] │   final CNF    :\n{final_dimacs}", "INFO")
                log(f"[EVAL] │   target CNF   :\n{target_dimacs_str}", "INFO")

            if status_before == CorrectionStatus.OK:
                log(f"[EVAL] └── OK (no correction needed)", "INFO")
                batch_ok += 1
            elif after == CorrectionStatus.OK:
                if corrected == CorrectionStatus.AUTO_FIXED:
                    log(f"[EVAL] └── INVALID → OK  (AUTO_FIXED)", "INFO")
                    batch_auto += 1
                else:
                    log(f"[EVAL] └── INVALID → OK  (FIXED by model)", "INFO")
                    batch_ok += 1
            else:
                log(f"[EVAL] └── INVALID → INVALID  (no fix found)", "INFO")
                batch_invalid += 1

        log(
            f"[EVAL] {'═' * 4} Batch {batch_idx}/{total_batches} summary: "
            f"{batch_ok} OK  {batch_auto} AUTO_FIXED  {batch_invalid} INVALID {'═' * 4}",
            "INFO",
        )

        # Collect the model counts that are already done (non-blocking).
        ready = metrics.harvest_ready()
        if ready:
            log(f"[EVAL] background MC: {len(ready)} sample(s) reconciled", "DEBUG")
            _write_resolved_corrections(samples_output_h, pending_sample_records, ready)

    # Wait (at most mc_harvest_timeout seconds) for the remaining model counts.
    resolved = metrics.harvest(timeout=mc_harvest_timeout)
    if resolved:
        log(f"[EVAL] background MC: {len(resolved)} sample(s) reconciled at harvest", "INFO")
        _write_resolved_corrections(samples_output_h, pending_sample_records, resolved)

    return results


# ---------------------------------------------------------------------------
# Top-level evaluation dispatcher
# ---------------------------------------------------------------------------


def resolve_format_handlers(cnf_format: str):
    """Return the verifier, error classifier, auto-fixers and correction prompt of a format.

    Returns a dict with keys ``verify_fn``, ``classify_fn``, ``error_codes``,
    ``fix_fns``, ``correct_path``, ``code_expl`` and ``code_guid``.
    """
    if cnf_format == "nat":
        from fonesy.evaluation import nat as nat_mod
        return {
            "verify_fn": nat_mod.verify_nat_cnf,
            "classify_fn": nat_mod.classify_nat_error_types,
            "error_codes": nat_mod.NAT_ERROR_TYPE_CODES,
            "fix_fns": [],
            "correct_path": _NAT_CORRECT_PROMPT_PATH,
            "code_expl": nat_mod._NAT_CODE_EXPLANATIONS,
            "code_guid": nat_mod._NAT_CODE_GUIDANCE,
        }
    if cnf_format == "pysat":
        from fonesy.evaluation import pysat_eval as pysat_mod
        return {
            "verify_fn": pysat_mod.verify_pysat_cnf,
            "classify_fn": pysat_mod.classify_pysat_error_types,
            "error_codes": pysat_mod.PYSAT_ERROR_TYPE_CODES,
            "fix_fns": [(pysat_mod.try_import_fix, "Import fix")],
            "correct_path": _PYSAT_CORRECT_PROMPT_PATH,
            "code_expl": pysat_mod._PYSAT_CODE_EXPLANATIONS,
            "code_guid": pysat_mod._PYSAT_CODE_GUIDANCE,
        }
    if cnf_format == "cpmpy":
        from fonesy.evaluation import cpmpy_eval as cpmpy_mod
        return {
            "verify_fn": cpmpy_mod.verify_cpmpy_cnf,
            "classify_fn": cpmpy_mod.classify_cpmpy_error_types,
            "error_codes": cpmpy_mod.CPMPY_ERROR_TYPE_CODES,
            "fix_fns": [
                (cpmpy_mod.try_import_fix, "Import fix"),
                (cpmpy_mod.try_boolvar_name_fix, "Boolean variable name"),
            ],
            "correct_path": _CPMPY_CORRECT_PROMPT_PATH,
            "code_expl": cpmpy_mod._CPMPY_CODE_EXPLANATIONS,
            "code_guid": cpmpy_mod._CPMPY_CODE_GUIDANCE,
        }
    if cnf_format == "sympy":
        from fonesy.evaluation import sympy_eval as sympy_mod
        return {
            "verify_fn": sympy_mod.verify_sympy_cnf,
            "classify_fn": sympy_mod.classify_sympy_error_types,
            "error_codes": sympy_mod.SYMPY_ERROR_TYPE_CODES,
            "fix_fns": [(sympy_mod.try_import_fix, "Import fix")],
            "correct_path": _SYMPY_CORRECT_PROMPT_PATH,
            "code_expl": sympy_mod._SYMPY_CODE_EXPLANATIONS,
            "code_guid": sympy_mod._SYMPY_CODE_GUIDANCE,
        }
    return {
        "verify_fn": verify_cnf,
        "classify_fn": classify_error_types,
        "error_codes": None,   # Metrics defaults to DIMACS ERROR_TYPE_CODES
        "fix_fns": None,       # eval_dataloader defaults to DIMACS chain
        "correct_path": None,  # self_correct defaults to _CORRECT_PROMPT_PATH
        "code_expl": None,     # _fmt_errors defaults to _CODE_EXPLANATIONS
        "code_guid": None,     # _enrich_errors_for_prompt defaults to _CODE_GUIDANCE
    }


def _install_shutdown_signal_handlers(metrics: Metrics):
    """On SIGTERM/SIGINT, shut the model-counting pool down, then re-raise.

    Workers are also killed by the kernel if this process dies
    (see :mod:`fonesy.evaluation.procutil`); this only makes it prompt.
    Returns a function that restores the previous handlers.
    """
    previous = {}

    def _handler(signum, frame):
        metrics.shutdown()
        signal.signal(signum, previous[signum])
        os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, _handler)

    def _restore():
        for sig, handler in previous.items():
            signal.signal(sig, handler)

    return _restore


def evaluation(model, test_dl, args, experiment_name=None, samples_output_h=None):
    """Run evaluation on test_dl."""
    cnf_format = getattr(args, "cnf_format", "dimacs")

    handlers = resolve_format_handlers(cnf_format)
    _verify_fn = handlers["verify_fn"]
    _classify_fn = handlers["classify_fn"]
    _error_codes = handlers["error_codes"]
    _fix_fns = handlers["fix_fns"]
    _correct_path = handlers["correct_path"]
    _code_expl = handlers["code_expl"]
    _code_guid = handlers["code_guid"]

    metrics = Metrics(
        error_type_codes=_error_codes, classify_errors_fn=_classify_fn,
        mc_workers=getattr(args, "mc_workers", 0),
    )
    statistics = Statistics()

    log("Model evaluation...", "INFO")

    output_dir = getattr(args, "output_dir_path", None)

    if (
        output_dir is not None
        and experiment_name
        and not getattr(args, "dry_run", False)
    ):
        _clear_experiment_cnf_artifacts(output_dir, experiment_name)

    restore_signal_handlers = _install_shutdown_signal_handlers(metrics)
    try:
        results = eval_dataloader(
            model,
            test_dl,
            metrics,
            statistics,
            args.budget,
            output_dir=output_dir,
            experiment_name=experiment_name,
            verify_cnf_fn=_verify_fn,
            classify_errors_fn=_classify_fn,
            fix_fns=_fix_fns,
            correct_prompt_path=_correct_path,
            code_guidance=_code_guid,
            code_explanations=_code_expl,
            samples_output_h=samples_output_h,
            cnf_format=cnf_format,
            mc_harvest_timeout=getattr(args, "mc_harvest_timeout", 600),
        )
    finally:
        restore_signal_handlers()
        metrics.shutdown()

    log("Model evaluation done!", "INFO")
    return metrics, results, statistics


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def write_metrics_csv(csv_path, experiment_name, args, metrics, statistics):
    """Write a flat CSV row with run parameters + aggregated metrics."""
    m = metrics.compute()
    s = statistics.summary()

    row = {
        "experiment": experiment_name,
        "model": getattr(args, "llm", "N/A"),
        "model_path": getattr(args, "model", getattr(args, "model_path", "N/A")),
        "dataset": getattr(args, "dataset", "N/A"),
        "cnf_format": getattr(args, "cnf_format", "dimacs"),
        "seed": getattr(args, "seed", "N/A"),
        "budget": getattr(args, "budget", "N/A"),
        "batch_size": getattr(args, "batch_size", "N/A"),
        "device": getattr(args, "device", "N/A"),
        "total": m["total"],
        "syntax_error_rate": m["syntax_error_rate"],
        "fixable_syntax_rate": m["fixable_syntax_rate"],
        "semantic_error_rate": m["semantic_error_rate"],
        "semantic_unknown_count": m.get("semantic_unknown_count", 0),
        "semantic_unknown_rate": m.get("semantic_unknown_rate", 0),
        "equal_rate": m["equal_rate"],
        "avg_formula_length_pred": m["avg_formula_length"]["pred"],
        "avg_formula_length_target": m["avg_formula_length"]["target"],
        "avg_num_clauses_pred": m["avg_num_clauses"]["pred"],
        "avg_num_clauses_target": m["avg_num_clauses"]["target"],
        "avg_num_vars_pred": m["avg_num_vars"]["pred"],
        "avg_num_vars_target": m["avg_num_vars"]["target"],
        "avg_model_count_pred": m["avg_model_count"]["pred"],
        "avg_model_count_target": m["avg_model_count"]["target"],
        "frac_approx_model_count_pred": m.get("frac_approx_model_count_pred"),
        "frac_approx_model_count_target": m.get("frac_approx_model_count_target"),
        "tseitin_conversion_count": m.get("tseitin_conversion_count", 0),
        "tseitin_conversion_rate": m.get("tseitin_conversion_rate", 0),
        "avg_precision": m.get("avg_precision"),
        **{
            f"precision_{pid}": val
            for pid, val in m.get("per_task_precision", {}).items()
        },
        "avg_recall": m.get("avg_recall"),
        **{
            f"recall_{pid}": val
            for pid, val in m.get("per_task_recall", {}).items()
        },
        "avg_f1": m.get("avg_f1"),
        **{
            f"f1_{pid}": val
            for pid, val in m.get("per_task_f1", {}).items()
        },
        "avg_accuracy": m.get("avg_accuracy"),
        **{
            f"accuracy_{pid}": val
            for pid, val in m.get("per_task_accuracy", {}).items()
        },
        "avg_mc_target_xor_pred": m.get("avg_mc_target_xor_pred"),
        "frac_approx_tp": m.get("frac_approx_tp"),
        "frac_approx_fp": m.get("frac_approx_fp"),
        "frac_approx_fn": m.get("frac_approx_fn"),
        "frac_approx_tn": m.get("frac_approx_tn"),
        "frac_approx_xor": m.get("frac_approx_xor"),
        "mc_epsilon": _DEFAULT_EPSILON,
        "mc_delta": _DEFAULT_DELTA,
        **{
            f"error_count_{code}": m["error_type_counts"].get(code, 0)
            for code in m["error_type_counts"]
        },
        **s,
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    log(f"Metrics CSV written to {csv_path}", "INFO")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(experiment_name, results_output_h, stats_output_h, args, device):
    """Load the dataset and the LLM, run the evaluation and write the outputs."""
    log("Loading the dataset...", "INFO")
    test_dl = make_dataloader(datasets.get_dataset(args), batch_size=args.batch_size)

    log("Loading the model...", "INFO")
    model_path = getattr(args, "model", None) or getattr(args, "model_path", None)
    # The remaining CLI arguments configure the LLM backend.
    extra = {
        k: v for k, v in vars(args).items()
        if k not in ("model_path", "model", "samples_output_h")
    }
    extra.setdefault("cnf_format", getattr(args, "cnf_format", "dimacs"))
    if extra.get("max_new_tokens") is None:
        extra["max_new_tokens"] = DEFAULT_MAX_NEW_TOKENS
    model = LLMFactory.get_llm(args.llm, model_path=model_path, **extra)

    metrics, results, statistics = evaluation(
        model, test_dl, args, experiment_name=experiment_name,
        samples_output_h=getattr(args, "samples_output_h", None),
    )

    result_data = {"results": results, "metrics": metrics.compute()}

    log(f"Statistics: {statistics.summary()}", "INFO")
    log(f"Metrics:    {metrics.compute()}", "INFO")

    # default=str: averages of huge model counts may be decimal.Decimal.
    json.dump(result_data, results_output_h, indent=4, default=str)
    json.dump(statistics.summary(), stats_output_h, indent=4, default=str)

    if not getattr(args, "dry_run", False):
        csv_path = args.output_dir_path / (experiment_name + ".metrics.csv")
        write_metrics_csv(csv_path, experiment_name, args, metrics, statistics)

    log("Results written successfully.", "INFO")

