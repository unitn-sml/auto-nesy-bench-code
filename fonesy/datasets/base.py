"""
Abstract dataset interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import keyword
from types import MappingProxyType
from typing import Mapping, Optional, Sequence


_NAT_OPERATOR_NAMES = frozenset({"AND", "OR", "NOT"})
_FORMAT_HELPER_NAMES = frozenset(
    {
        # PySAT program helpers / conventional output object.
        "CNF", "IDPool", "CardEnc", "PBEnc", "sys", "pysat", "cnf", "encoded",
        # CPMpy program helpers.
        "cp", "cpmpy", "boolvar", "constraints", "model", "solver",
        # SymPy program helpers and conventional intermediate names.
        "Symbol", "symbols", "sympy", "to_cnf", "And", "Or", "Not",
        "expr", "cnf_expr", "DimacsMapping", "DimacsFormula",
        "to_dimacs_formula", "_dimacs_literal", "_EXPECTED_VARIABLE_NAMES",
        "_EXPECTED_VARIABLE_NAME_SET", "_ORIGINAL_SYMBOL_NEW",
        "_validated_symbol_new",
        # Additional injected CPMpy/SymPy module globals.
        "CPM_pysat", "get_variables", "json",
        # Injected verifier entry point (all symbolic Python formats).
        "_print_dimacs",
    }
)


def _is_portable_variable_name(name: str) -> bool:
    """Return whether *name* is safe in every supported symbolic format.

    NAT accepts ASCII identifiers, while CPMpy and SymPy responses are Python
    programs.  Using the intersection of those syntaxes gives every format the
    same stable names and avoids format-dependent variable mappings.
    """
    return (
        name.isascii()
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and name not in _NAT_OPERATOR_NAMES
        and name not in _FORMAT_HELPER_NAMES
        # CPMpy names anonymous Boolean transformation variables BV0, BV1,
        # ...; reserving that namespace prevents a primary from aliasing one.
        and not (name.startswith("BV") and name[2:].isdigit())
    )


@dataclass(frozen=True)
class VariableSpec:
    """Immutable ordered variable contract for one dataset sample.

    ``source_names`` are the semantic labels supplied by the target metadata.
    ``variable_names`` are the exact names exposed to symbolic generators and
    verifiers.  Position is authoritative in both tuples: index ``i`` maps to
    DIMACS variable ``i + 1``.
    """

    source_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    mapping_version: int = 1
    # Number of primary variables when the target CNF itself has trailing
    # auxiliary variables (dcp-bench's N_PRIMARY_VARS); None = all primary.
    primary_num_vars_override: Optional[int] = None

    def __post_init__(self) -> None:
        source_names = tuple(self.source_names)
        variable_names = tuple(self.variable_names)
        object.__setattr__(self, "source_names", source_names)
        object.__setattr__(self, "variable_names", variable_names)
        if not source_names:
            raise ValueError("VariableSpec requires at least one variable")
        if len(source_names) != len(variable_names):
            raise ValueError("VariableSpec source and output names must have equal length")
        for label, names in (
            ("source", source_names),
            ("output", variable_names),
        ):
            if any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError(f"VariableSpec {label} names must be nonempty strings")
        if len(set(source_names)) != len(source_names):
            raise ValueError("VariableSpec source names must be unique")
        if len(set(variable_names)) != len(variable_names):
            raise ValueError("VariableSpec output names must be unique")
        if not all(_is_portable_variable_name(name) for name in variable_names):
            raise ValueError("VariableSpec output names must be portable identifiers")
        if self.mapping_version != 1:
            raise ValueError("Unsupported VariableSpec mapping_version")
        override = self.primary_num_vars_override
        if override is not None:
            if isinstance(override, bool) or not isinstance(override, int):
                raise ValueError("VariableSpec primary_num_vars_override must be an integer")
            if not (1 <= override <= len(variable_names)):
                raise ValueError(
                    "VariableSpec primary_num_vars_override must be between 1 and "
                    f"{len(variable_names)} (got {override})"
                )

    @classmethod
    def from_source_names(
        cls, source_names: Sequence[str], primary_num_vars: Optional[int] = None
    ) -> "VariableSpec":
        """Create a spec, aliasing the whole sample if any name is unsafe."""
        names = tuple(source_names)
        if not names or any(
            not isinstance(name, str) or not name.strip() for name in names
        ):
            raise ValueError("VariableSpec source names must be nonempty strings")
        output_names = (
            names
            if all(_is_portable_variable_name(name) for name in names)
            else tuple(f"x{i}" for i in range(1, len(names) + 1))
        )
        return cls(
            source_names=names,
            variable_names=output_names,
            primary_num_vars_override=primary_num_vars,
        )

    @property
    def primary_num_vars(self) -> int:
        if self.primary_num_vars_override is not None:
            return self.primary_num_vars_override
        return len(self.variable_names)

    @property
    def uses_aliases(self) -> bool:
        return self.source_names != self.variable_names

    @property
    def name_to_dimacs(self) -> Mapping[str, int]:
        """Read-only exact symbolic-name to DIMACS-index mapping."""
        return MappingProxyType(
            {name: index for index, name in enumerate(self.variable_names, start=1)}
        )


class BaseDataset(ABC):
    """
    Abstract dataset class.

    All datasets should implement:
    - load()
    - optionally: splits (train/val/test)
    """

    def __init__(self):
        self.data = None
        self._variable_specs = {}

    @abstractmethod
    def load(self):
        """
        Load dataset into memory.

        Returns:
            list or structured dataset
        """
        pass

    def get_data(self):
        """
        Returns loaded data (lazy loading).
        """
        if self.data is None:
            self.data = self.load()
        return self.data

    def __len__(self):
        return len(self.get_data())

    def __getitem__(self, idx):
        return self.get_data()[idx]

    def get_splits(self):
        """
        Override if dataset supports splits.

        Returns:
            (train, val, test)
        """
        data = self.get_data()
        return data, None, None

    def get_variable_spec(self, sample_id: str) -> VariableSpec | None:
        """Return ordered variable metadata for *sample_id*, when available."""
        return getattr(self, "_variable_specs", {}).get(str(sample_id))

    def get_variable_names(self, sample_id: str) -> tuple[str, ...] | None:
        """Return exact symbolic names in DIMACS order for *sample_id*."""
        spec = self.get_variable_spec(sample_id)
        return spec.variable_names if spec is not None else None

    def __repr__(self):
        return f"{self.__class__.__name__}(size={len(self)})"
