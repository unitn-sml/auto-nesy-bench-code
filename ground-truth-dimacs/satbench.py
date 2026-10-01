#!/usr/bin/env python3
"""Download SATBench and build the ``satbench`` benchmark (satisfiable instances).

Downloads ``LLM4Code/SATBench`` at a pinned revision, keeps the 1,041
satisfiable instances without quality issues (condition count different from
the clause count, or recovered formula different from the reference one) and
writes ``fonesy/data/satbench/dataset.jsonl``. Variable ``x(i, j, ...)`` becomes
the one-based row-major DIMACS id; every target is checked against the source
clauses and proven satisfiable with PySAT.

Usage::

    python ground-truth-dimacs/satbench.py {download,build,all} [--data-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Sequence
from urllib.request import Request, urlopen


SOURCE_REPO = "LLM4Code/SATBench"
SOURCE_REVISION = "186740e5fb7c0fede11d13f3fbcf7d7d92d70dc9"
SOURCE_FILENAME = "SATBench-problems.jsonl"
SOURCE_SHA256 = "d32ee8ca8ccee4ee3dcb322e174d4cbe5ffebbd1b76dcdb702d397afd34294b5"
SOURCE_LICENSE = "apache-2.0"
SOURCE_RECORD_COUNT = 2_100
SAT_RECORD_COUNT = 1_041
PINNED_CLAUSE_COUNT_BUCKETS = {
    **{count: 70 for count in (4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 40)},
    30: 69,
    35: 69,
    45: 64,
    50: 69,
}
SOURCE_URL = (
    f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILENAME}?download=true"
)

SCHEMA_VERSION = 1
MAPPING_VERSION = 1
QUALITY_FLAGS = (
    "condition_count_mismatch",
    "recovered_formula_structural_mismatch",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "fonesy" / "data" / "satbench"

_VARIABLE_RE = re.compile(r"x\s*\(\s*([^()]*)\)")
_EQUIVALENT_RE = re.compile(r"\[\s*EQUIVALENT\s*\]", re.IGNORECASE)
_CONDITION_ORDINAL_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


class SATBenchFormatError(ValueError):
    """Raised when pinned SATBench source data violates the build contract."""


def _is_int(value: object) -> bool:
    """Return whether *value* is an integer but not a Boolean."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validated_dims(dims: object, *, context: str = "dims") -> tuple[int, ...]:
    if not isinstance(dims, list) or not dims:
        raise SATBenchFormatError(f"{context} must be a non-empty list of integers")
    if any(not _is_int(dim) or dim <= 0 for dim in dims):
        raise SATBenchFormatError(f"{context} must contain only positive integers")
    return tuple(dims)


def encode_coordinate(indices: Sequence[int], dims: Sequence[int]) -> int:
    """Encode one coordinate as a one-based native row-major DIMACS ID."""
    if len(indices) != len(dims):
        raise SATBenchFormatError(
            f"coordinate arity {len(indices)} does not match dims arity {len(dims)}"
        )
    flat = 0
    for axis, (index, size) in enumerate(zip(indices, dims)):
        if not _is_int(index) or not 0 <= index < size:
            raise SATBenchFormatError(
                f"coordinate index {index!r} is out of range for axis {axis} "
                f"with size {size}"
            )
        flat = flat * size + index
    return flat + 1


def _coordinate_source_name(indices: Sequence[int]) -> str:
    if len(indices) == 1:
        return f"x({indices[0]},)"
    return f"x({', '.join(str(index) for index in indices)})"


def _coordinate_alias(indices: Sequence[int]) -> str:
    return "x_" + "_".join(str(index) for index in indices)


def build_variable_names(dims: Sequence[int]) -> tuple[list[str], list[str]]:
    """Return full native coordinate names and portable aliases in ID order."""
    coordinates = itertools.product(*(range(size) for size in dims))
    source_names: list[str] = []
    variable_names: list[str] = []
    for expected_id, coordinate in enumerate(coordinates, start=1):
        actual_id = encode_coordinate(coordinate, dims)
        if actual_id != expected_id:
            raise AssertionError("row-major coordinate enumeration is inconsistent")
        source_names.append(_coordinate_source_name(coordinate))
        variable_names.append(_coordinate_alias(coordinate))
    return source_names, variable_names


class _ReadableParser:
    """Small strict parser for SATBench's Unicode readable-CNF grammar."""

    def __init__(self, text: str, dims: Sequence[int]):
        self.text = text
        self.dims = dims
        self.pos = 0

    def error(self, message: str) -> SATBenchFormatError:
        excerpt = self.text[max(0, self.pos - 15) : self.pos + 20]
        return SATBenchFormatError(
            f"{message} at character {self.pos}; near {excerpt!r}"
        )

    def skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos].isspace():
            self.pos += 1

    def consume(self, token: str) -> bool:
        self.skip_space()
        if self.text.startswith(token, self.pos):
            self.pos += len(token)
            return True
        return False

    def expect(self, token: str) -> None:
        if not self.consume(token):
            raise self.error(f"expected {token!r}")

    def literal(self) -> int:
        self.skip_space()
        is_negative = self.consume("¬")
        self.skip_space()
        match = _VARIABLE_RE.match(self.text, self.pos)
        if match is None:
            raise self.error("expected a variable such as x(0,) or x(0, 1)")
        self.pos = match.end()

        body = match.group(1).strip()
        if not body:
            raise self.error("variable coordinate cannot be empty")
        has_trailing_comma = body.endswith(",")
        if has_trailing_comma:
            body = body[:-1].rstrip()
        parts = [part.strip() for part in body.split(",")]
        if any(not part or not part.isdecimal() for part in parts):
            raise self.error("variable coordinates must be unsigned decimal integers")
        if has_trailing_comma and len(parts) != 1:
            raise self.error(
                "only one-dimensional tuple coordinates may end in a comma"
            )

        indices = tuple(int(part) for part in parts)
        if len(indices) != len(self.dims):
            raise self.error(
                f"coordinate arity {len(indices)} does not match dims arity "
                f"{len(self.dims)}"
            )
        variable = encode_coordinate(indices, self.dims)
        return -variable if is_negative else variable

    def clause(self) -> list[int]:
        self.expect("(")
        self.skip_space()
        if self.pos < len(self.text) and self.text[self.pos] == ")":
            raise self.error("empty clauses are not allowed")
        literals = [self.literal()]
        while self.consume("∨"):
            literals.append(self.literal())
        self.expect(")")
        return literals

    def formula(self) -> list[list[int]]:
        self.skip_space()
        if self.pos == len(self.text):
            raise self.error("readable CNF cannot be empty")
        clauses = [self.clause()]
        while self.consume("∧"):
            clauses.append(self.clause())
        self.skip_space()
        if self.pos != len(self.text):
            raise self.error("unexpected trailing input or non-CNF operator")
        return clauses


def parse_readable_cnf(readable: object, dims: object) -> list[list[int]]:
    """Parse a complete SATBench readable CNF into native DIMACS clauses.

    The accepted grammar contains only parenthesized clauses, ``¬`` negation,
    ``∨`` disjunction, ``∧`` conjunction, and coordinate variables.  Coordinate
    arity and bounds are checked against *dims*; unparsed text is never ignored.
    """
    if not isinstance(readable, str) or not readable.strip():
        raise SATBenchFormatError("readable must be a non-empty string")
    validated_dims = _validated_dims(dims)
    return _ReadableParser(readable, validated_dims).formula()


def _canonical_clause_multiset(
    clauses: Iterable[Sequence[int]],
) -> Counter[tuple[int, ...]]:
    return Counter(tuple(sorted(clause)) for clause in clauses)


def _validated_raw_clauses(
    clauses: object,
    *,
    num_vars: int,
    context: str,
) -> list[list[int]]:
    if not isinstance(clauses, list) or not clauses:
        raise SATBenchFormatError(f"{context} clauses must be a non-empty list")
    validated: list[list[int]] = []
    for clause_index, clause in enumerate(clauses):
        if not isinstance(clause, list) or not clause:
            raise SATBenchFormatError(
                f"{context} clause {clause_index} must be a non-empty list"
            )
        if any(not _is_int(literal) for literal in clause):
            raise SATBenchFormatError(
                f"{context} clause {clause_index} contains a non-integer literal"
            )
        if any(literal == 0 or abs(literal) > num_vars for literal in clause):
            raise SATBenchFormatError(
                f"{context} clause {clause_index} contains a zero or "
                "out-of-range literal"
            )
        validated.append(list(clause))
    return validated


def clauses_to_dimacs(
    clauses: Sequence[Sequence[int]], *, num_vars: int
) -> str:
    lines = [f"p cnf {num_vars} {len(clauses)}"]
    lines.extend(
        " ".join(str(literal) for literal in clause) + " 0" for clause in clauses
    )
    return "\n".join(lines)


def _verify_target(
    target: str,
    *,
    expected_clauses: Sequence[Sequence[int]],
    num_vars: int,
    context: str,
) -> None:
    """Reparse one final target and prove that it is satisfiable.

    Imports are deliberately local: parser/mapping helpers and ``download``
    remain usable in a minimal Python installation, while dataset builds fail
    with a clear dependency message if the project's normal ``python-sat``
    dependency is absent.
    """
    # Make ``fonesy`` importable when run as a script.
    repository_path = str(REPO_ROOT)
    if repository_path not in sys.path:
        sys.path.insert(0, repository_path)
    try:
        from fonesy.evaluation.dimacs import verify_cnf
        from pysat.solvers import Solver
    except ImportError as exc:
        raise RuntimeError(
            "building SATBench requires FoNeSy's runtime dependencies, including "
            "python-sat; install requirements.txt before running the build"
        ) from exc

    formula, error_message, _status = verify_cnf(target, llm_response=False)
    if formula is None or error_message:
        details = error_message.strip() or "the verifier returned no formula"
        raise SATBenchFormatError(
            f"{context} rendered target failed final DIMACS verification: {details}"
        )
    expected = [list(clause) for clause in expected_clauses]
    if (
        formula.num_vars != num_vars
        or formula.num_clauses != len(expected)
        or formula.clauses != expected
    ):
        raise SATBenchFormatError(
            f"{context} rendered target changed during final DIMACS verification"
        )

    try:
        with Solver(bootstrap_with=formula.clauses) as solver:
            satisfiable = solver.solve()
    except Exception as exc:
        raise RuntimeError(f"{context} PySAT solver check failed to run") from exc
    if not satisfiable:
        raise SATBenchFormatError(
            f"{context} rendered target is not satisfiable according to PySAT"
        )


def _required_string(row: dict, field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SATBenchFormatError(
            f"{context} field {field!r} must be a non-empty string"
        )
    return value


def _normalize_conditions(conditions: Sequence[str]) -> list[str]:
    """Remove SATBench's matching list marker from each condition.

    Prompt rendering owns the visible ordering.  Only a marker matching the
    condition's actual one-based position is removed, so leading numeric text
    that is part of a condition is preserved.
    """
    normalized: list[str] = []
    for expected_ordinal, condition in enumerate(conditions, start=1):
        condition = condition.strip()
        match = _CONDITION_ORDINAL_RE.fullmatch(condition)
        if match is not None and int(match.group(1)) == expected_ordinal:
            condition = match.group(2).strip()
        normalized.append(condition)
    return normalized


def build_record(row: object, source_index: int) -> dict:
    """Validate and convert one satisfiable source row."""
    context = f"source record {source_index}"
    if not isinstance(row, dict):
        raise SATBenchFormatError(f"{context} must be a JSON object")
    if row.get("satisfiable") is not True:
        raise SATBenchFormatError(f"{context} is not explicitly satisfiable")

    dims = _validated_dims(row.get("dims"), context=f"{context} dims")
    num_vars = row.get("num_vars")
    num_clauses = row.get("num_clauses")
    if not _is_int(num_vars) or num_vars <= 0:
        raise SATBenchFormatError(f"{context} num_vars must be a positive integer")
    if not _is_int(num_clauses) or num_clauses <= 0:
        raise SATBenchFormatError(f"{context} num_clauses must be a positive integer")
    expected_num_vars = 1
    for size in dims:
        expected_num_vars *= size
    if num_vars != expected_num_vars:
        raise SATBenchFormatError(
            f"{context} num_vars={num_vars} does not equal "
            f"prod(dims)={expected_num_vars}"
        )

    scenario = _required_string(row, "scenario", context=context)
    variable_mapping = _required_string(row, "variable_mapping", context=context)
    readable = _required_string(row, "readable", context=context)
    recovered_formula = _required_string(row, "recovered_formula", context=context)
    equivalence_verdict = _required_string(
        row, "formula_equivalence_check", context=context
    )
    if _EQUIVALENT_RE.fullmatch(equivalence_verdict.strip()) is None:
        raise SATBenchFormatError(
            f"{context} formula_equivalence_check is not [EQUIVALENT]"
        )

    conditions = row.get("conditions")
    if not isinstance(conditions, list) or any(
        not isinstance(condition, str) or not condition.strip()
        for condition in conditions
    ):
        raise SATBenchFormatError(
            f"{context} conditions must be a list of non-empty strings"
        )
    conditions = _normalize_conditions(conditions)

    readable_clauses = parse_readable_cnf(readable, list(dims))
    raw_clauses = _validated_raw_clauses(
        row.get("clauses"), num_vars=num_vars, context=context
    )
    if len(readable_clauses) != num_clauses:
        raise SATBenchFormatError(
            f"{context} readable formula has {len(readable_clauses)} clauses, "
            f"not num_clauses={num_clauses}"
        )
    if len(raw_clauses) != num_clauses:
        raise SATBenchFormatError(
            f"{context} raw clauses has length {len(raw_clauses)}, "
            f"not num_clauses={num_clauses}"
        )
    if _canonical_clause_multiset(readable_clauses) != _canonical_clause_multiset(
        raw_clauses
    ):
        raise SATBenchFormatError(
            f"{context} readable CNF does not match the raw clause multiset"
        )

    quality_flags: list[str] = []
    if len(conditions) != num_clauses:
        quality_flags.append("condition_count_mismatch")

    recovered_clauses = parse_readable_cnf(recovered_formula, list(dims))
    if _canonical_clause_multiset(recovered_clauses) != _canonical_clause_multiset(
        readable_clauses
    ):
        quality_flags.append("recovered_formula_structural_mismatch")

    source_names, variable_names = build_variable_names(dims)
    if len(source_names) != num_vars or len(variable_names) != num_vars:
        raise AssertionError("full native variable mapping has the wrong size")

    target = clauses_to_dimacs(readable_clauses, num_vars=num_vars)
    _verify_target(
        target,
        expected_clauses=readable_clauses,
        num_vars=num_vars,
        context=context,
    )

    return {
        "id": f"satbench_{source_index:04d}",
        "source_index": source_index,
        "dims": list(dims),
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "scenario": scenario,
        "variable_mapping": variable_mapping,
        "conditions": conditions,
        "source_names": source_names,
        "variable_names": variable_names,
        "target": target,
        "quality_flags": quality_flags,
    }


def prepare_records(
    source_records: Sequence[object],
    *,
    expected_source_record_count: int | None = None,
    expected_record_count: int | None = None,
) -> list[dict]:
    """Filter and convert clean SAT source rows while retaining original indices."""
    if (
        expected_source_record_count is not None
        and len(source_records) != expected_source_record_count
    ):
        raise SATBenchFormatError(
            f"source has {len(source_records)} records; expected "
            f"{expected_source_record_count}"
        )

    records: list[dict] = []
    for source_index, row in enumerate(source_records):
        if not isinstance(row, dict):
            raise SATBenchFormatError(
                f"source record {source_index} must be a JSON object"
            )
        satisfiable = row.get("satisfiable")
        if not isinstance(satisfiable, bool):
            raise SATBenchFormatError(
                f"source record {source_index} satisfiable must be a Boolean"
            )
        if satisfiable:
            record = build_record(row, source_index)
            if not record["quality_flags"]:
                records.append(record)

    if expected_record_count is not None and len(records) != expected_record_count:
        raise SATBenchFormatError(
            f"SAT/quality filters produced {len(records)} records; expected "
            f"{expected_record_count}"
        )
    if any(
        record["id"] != f"satbench_{record['source_index']:04d}"
        for record in records
    ):
        raise AssertionError("SATBench IDs are not stable source-index IDs")
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[object]:
    records: list[object] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise SATBenchFormatError(
                    f"{path} contains a blank line at source line {line_number}"
                )
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SATBenchFormatError(
                    f"invalid JSON in {path} at source line {line_number}: {exc}"
                ) from exc
    return records


def _metadata(
    records: Sequence[dict], *, source_record_count: int, source_sha256: str
) -> dict:
    clause_count_buckets = Counter(record["num_clauses"] for record in records)
    quality_flag_counts = {
        flag: sum(flag in record["quality_flags"] for record in records)
        for flag in QUALITY_FLAGS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "mapping_version": MAPPING_VERSION,
        "source": {
            "repo": SOURCE_REPO,
            "revision": SOURCE_REVISION,
            "filename": SOURCE_FILENAME,
            "sha256": source_sha256,
            "license": SOURCE_LICENSE,
        },
        "filters": {"satisfiable": True, "quality_flags": []},
        "source_record_count": source_record_count,
        "record_count": len(records),
        "clause_count_buckets": {
            str(count): clause_count_buckets[count]
            for count in sorted(clause_count_buckets)
        },
        "quality_flag_counts": quality_flag_counts,
    }


def _validate_pinned_distribution(records: Sequence[dict]) -> None:
    actual = Counter(record["num_clauses"] for record in records)
    if actual != Counter(PINNED_CLAUSE_COUNT_BUCKETS):
        raise SATBenchFormatError(
            "pinned SAT clause-count distribution drifted: expected "
            f"{PINNED_CLAUSE_COUNT_BUCKETS}, got {dict(sorted(actual.items()))}"
        )


def _stage_text(destination: Path, content: str) -> Path:
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.fchmod(fd, 0o644)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_backup(path: Path) -> Path | None:
    """Create a same-directory hard-link backup without moving *path*."""
    if not path.is_file():
        return None
    fd, backup_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".bak", dir=path.parent
    )
    os.close(fd)
    backup = Path(backup_name)
    backup.unlink()
    os.link(path, backup)
    return backup


def _atomic_publish(data_dir: Path, dataset_text: str, metadata_text: str) -> None:
    """Publish a consistent dataset/metadata pair, rolling back on failure."""
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "dataset.jsonl"
    metadata_path = data_dir / "metadata.json"
    dataset_tmp: Path | None = None
    metadata_tmp: Path | None = None
    dataset_backup: Path | None = None
    metadata_backup: Path | None = None
    dataset_published = False
    metadata_published = False
    rollback_failed = False
    try:
        dataset_tmp = _stage_text(dataset_path, dataset_text)
        metadata_tmp = _stage_text(metadata_path, metadata_text)
        dataset_backup = _stage_backup(dataset_path)
        metadata_backup = _stage_backup(metadata_path)
        os.replace(dataset_tmp, dataset_path)
        dataset_tmp = None
        dataset_published = True
        os.replace(metadata_tmp, metadata_path)
        metadata_tmp = None
        metadata_published = True
    except BaseException as publish_error:
        try:
            if dataset_published:
                if dataset_backup is None:
                    dataset_path.unlink(missing_ok=True)
                else:
                    os.replace(dataset_backup, dataset_path)
                    dataset_backup = None
            if metadata_published:
                if metadata_backup is None:
                    metadata_path.unlink(missing_ok=True)
                else:
                    os.replace(metadata_backup, metadata_path)
                    metadata_backup = None
        except BaseException as rollback_error:
            rollback_failed = True
            raise RuntimeError(
                "SATBench artifact publication failed and could not be rolled back"
            ) from rollback_error
        raise publish_error
    finally:
        if dataset_tmp is not None:
            dataset_tmp.unlink(missing_ok=True)
        if metadata_tmp is not None:
            metadata_tmp.unlink(missing_ok=True)
        if dataset_backup is not None and not rollback_failed:
            dataset_backup.unlink(missing_ok=True)
        if metadata_backup is not None and not rollback_failed:
            metadata_backup.unlink(missing_ok=True)


def build_satbench_dataset(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    expected_sha256: str | None = SOURCE_SHA256,
    expected_source_record_count: int | None = SOURCE_RECORD_COUNT,
    expected_record_count: int | None = SAT_RECORD_COUNT,
) -> int:
    """Build deterministic ``dataset.jsonl`` and ``metadata.json`` artifacts."""
    data_dir = Path(data_dir)
    source_path = data_dir / "raw" / SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"SATBench source file not found: {source_path}\n"
            "Run 'python ground-truth-dimacs/satbench.py download' first."
        )
    actual_sha256 = _sha256(source_path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise SATBenchFormatError(
            f"SATBench source SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )

    source_records = _load_jsonl(source_path)
    records = prepare_records(
        source_records,
        expected_source_record_count=expected_source_record_count,
        expected_record_count=expected_record_count,
    )
    if (
        expected_sha256 == SOURCE_SHA256
        and expected_source_record_count == SOURCE_RECORD_COUNT
        and expected_record_count == SAT_RECORD_COUNT
    ):
        _validate_pinned_distribution(records)
    metadata = _metadata(
        records,
        source_record_count=len(source_records),
        source_sha256=actual_sha256,
    )
    dataset_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"

    # No destination is touched until the complete source has passed all
    # parsing, mapping, count, and clause-multiset checks.
    _atomic_publish(data_dir, dataset_text, metadata_text)
    print(f"Built {len(records)} SATBench SAT records in {data_dir}")
    return len(records)


def download_source(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    url: str = SOURCE_URL,
    expected_sha256: str = SOURCE_SHA256,
    opener: Callable[..., BinaryIO] | None = None,
) -> Path:
    """Atomically download and checksum the pinned SATBench JSONL source."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    destination = raw_dir / SOURCE_FILENAME
    if destination.is_file() and _sha256(destination) == expected_sha256:
        destination.chmod(0o644)
        print(f"Pinned SATBench source already present at {destination}")
        return destination

    open_url = opener if opener is not None else urlopen
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{SOURCE_FILENAME}.", suffix=".tmp", dir=raw_dir
    )
    os.fchmod(fd, 0o644)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        request = Request(url, headers={"User-Agent": "fonesy-satbench-builder/1"})
        # Own the descriptor before opening the network response so an opener
        # failure cannot leak it.
        with os.fdopen(fd, "wb") as output:
            with open_url(request, timeout=60) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise SATBenchFormatError(
                f"downloaded SATBench SHA-256 mismatch: expected {expected_sha256}, "
                f"got {actual_sha256}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"Downloaded pinned SATBench source to {destination}")
    return destination


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and build the pinned SAT-only SATBench dataset."
    )
    parser.add_argument("command", choices=("download", "build", "all"))
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"SATBench data directory (default: {DEFAULT_DATA_DIR})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command in {"download", "all"}:
        download_source(args.data_dir)
    if args.command in {"build", "all"}:
        build_satbench_dataset(args.data_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
