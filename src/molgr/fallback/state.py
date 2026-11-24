"""State containers and explicit state-machine helpers for fallback.

The v2 pipeline keeps chemical mutations explicit: each stage mutates an `omol`
through a small state machine, while frozen `ReconstructionState` snapshots are
hashable and can be cached directly by pure helper functions.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from openbabel import openbabel as ob
from openbabel import pybel

from molgr.fallback.utils.dataclasses import MetalAtomPosition, MetalSite
from molgr.fallback.utils.tools import typed_lru_cache


if TYPE_CHECKING:
    from molgr.config import MolGRConfig
    from molgr.fallback.utils.force_field import (
        ForceFieldScoreKey,
        MetalStateKey,
    )


T = TypeVar("T")
ReconstructionScoreProfile = Literal["organic_core", "full"]
MetalCandidateScoreProfile = Literal["combined"]
CombinedOmolBuilder = Callable[[pybel.Molecule, Sequence[MetalAtomPosition]], pybel.Molecule]
_DEFAULT_RECONSTRUCTION_STATE_CACHE_MAXSIZE = 4096
TRACE_NODE_METADATA_KEY = "_omol_trace_node_id"
_TRACE_METADATA_KEYS = (
    "neighbor_radical_actions",
    "neighbor_radical_resolution",
    "neighbor_radical_charge_separation_discrepancy",
    "positive_atom_idx",
    "recovery_tier",
    "recovery_strategy",
    "resonance_seed_index",
    "resonance_index",
    "resonance_raw_index",
    "resonance_normalization",
)
_OMOL_TRACE_OBSERVER: ContextVar[Optional[Callable[[Any, str, dict[str, Any]], None]]] = ContextVar(
    "molgr_omol_trace_observer",
    default=None,
)
_OMOL_DERIVED_METADATA_KEYS = (
    "force_field_energy",
    "force_field_score_key",
    "organic_core_score",
    "score",
)
_CANDIDATE_DERIVED_METADATA_KEYS = (
    "force_field_energy",
    "score",
)


def _invalidate_omol_derived_metadata(metadata: Dict[str, Any]) -> None:
    for key in _OMOL_DERIVED_METADATA_KEYS:
        metadata.pop(key, None)


def _invalidate_candidate_derived_state(
    metadata: Dict[str, Any],
    key_cache: Dict[str, Any],
) -> None:
    for key in _CANDIDATE_DERIVED_METADATA_KEYS:
        metadata.pop(key, None)
    key_cache.pop("force_field_candidate_key", None)
    key_cache.pop("combined_score", None)
    key_cache.pop("force_field_energy", None)
    key_cache.pop("combined_omol", None)


@dataclass(eq=False)
class ReconstructionState:
    """Frozen no-metal reconstruction state plus score/key caches."""

    omol: pybel.Molecule
    given_charge: int
    total_charge: int
    total_radical_electrons: int
    phase_history: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    omol_revision: int = 0

    def _cache_identity(self) -> tuple[int, int, int, int, int]:
        return (
            id(self.omol),
            int(self.given_charge),
            int(self.total_charge),
            int(self.total_radical_electrons),
            int(self.omol_revision),
        )

    def __hash__(self) -> int:
        return hash(self._cache_identity())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReconstructionState):
            return NotImplemented
        return self._cache_identity() == other._cache_identity()

    def get_cached_revision_value(
        self,
        cache_name: str,
        builder: Callable[[], T],
    ) -> T:
        return cast(T, _cached_reconstruction_revision_value(self, cache_name, builder))

    def get_cached_omol_value(
        self,
        cache_name: str,
        builder: Callable[[pybel.Molecule], T],
    ) -> T:
        return cast(T, _cached_reconstruction_omol_value(self, cache_name, builder))

    def force_field_score_key(self) -> ForceFieldScoreKey:
        """Return the cached key for the no-metal force-field evaluation."""

        from molgr.fallback.utils.force_field import _build_score_key

        score_key = self.get_cached_omol_value("force_field_score_key", _build_score_key)
        self.metadata["force_field_score_key"] = score_key
        return cast(Any, score_key)

    def organic_core_score(self, *, config: MolGRConfig | None = None) -> float:
        """Score the metal-free reconstruction with fixed organic UFF scoring."""

        from molgr.fallback.utils.force_field import (
            OmolForceFieldContext,
            organic_force_field_evaluation,
        )

        score_key = self.force_field_score_key()
        del config
        evaluation = organic_force_field_evaluation(
            OmolForceFieldContext(self.omol, score_key=score_key)
        )
        score = evaluation.energy_kj_mol
        self.metadata["force_field_energy"] = score
        self.metadata["organic_core_score"] = score
        self.metadata["score"] = score
        self.metadata["force_field_score_key"] = score_key
        return float(score)

    def full_score(self, *, config: MolGRConfig | None = None) -> float:
        """Score the complete no-metal reconstruction state."""

        if config is None:
            score = self.organic_core_score()
        else:
            score = self.organic_core_score(config=config)
        self.metadata["score"] = score
        return float(score)

    def score(
        self,
        profile: ReconstructionScoreProfile = "full",
        *,
        config: MolGRConfig | None = None,
    ) -> float:
        if profile == "organic_core":
            if config is None:
                return self.organic_core_score()
            return self.organic_core_score(config=config)
        if profile == "full":
            if config is None:
                return self.full_score()
            return self.full_score(config=config)
        raise ValueError(f"Unsupported ReconstructionState score profile: {profile!r}")


class OmolTraceRecorder:
    """Collect exact state-machine phase snapshots without patching the pipeline."""

    def __init__(self) -> None:
        self.records: dict[int, dict[str, Any]] = {}
        self._next_snapshot_id = 0

    @staticmethod
    def _copy_omol(omol: pybel.Molecule) -> pybel.Molecule:
        return pybel.Molecule(ob.OBMol(cast(ob.OBMol, omol.OBMol)))

    def record_initial(
        self,
        omol: pybel.Molecule,
        phase: str,
        *,
        parent_id: int = -1,
    ) -> int:
        """Register a parser/seed snapshot before a state machine is created."""

        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        self.records[snapshot_id] = {
            "phase": phase,
            "parent_id": parent_id,
            "omol": self._copy_omol(omol),
            "given_charge": 0,
            "total_charge": 0,
            "total_radical_electrons": 0,
            "event": {"kind": "initial"},
            "metadata": {},
        }
        return snapshot_id

    def __call__(self, machine: Any, phase: str, event: dict[str, Any]) -> None:
        parent_id = int(machine.metadata.get(TRACE_NODE_METADATA_KEY, -1))
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1
        machine.metadata[TRACE_NODE_METADATA_KEY] = snapshot_id
        self.records[snapshot_id] = {
            "phase": phase,
            "parent_id": parent_id,
            "omol": self._copy_omol(machine.omol),
            "given_charge": int(machine.given_charge),
            "total_charge": int(machine.total_charge),
            "total_radical_electrons": int(machine.total_radical_electrons),
            "event": dict(event),
            "metadata": {
                key: machine.metadata[key]
                for key in _TRACE_METADATA_KEYS
                if key in machine.metadata
            },
        }


@contextmanager
def trace_omol_state_machine(recorder: OmolTraceRecorder) -> Iterator[OmolTraceRecorder]:
    """Enable automatic trace-node registration for the current execution context."""

    token = _OMOL_TRACE_OBSERVER.set(recorder)
    try:
        yield recorder
    finally:
        _OMOL_TRACE_OBSERVER.reset(token)


@typed_lru_cache(maxsize=_DEFAULT_RECONSTRUCTION_STATE_CACHE_MAXSIZE, typed=True)
def _cached_reconstruction_revision_value(
    state: ReconstructionState,
    cache_name: str,
    builder: Callable[[], Any],
) -> Any:
    del cache_name
    return builder()


@typed_lru_cache(maxsize=_DEFAULT_RECONSTRUCTION_STATE_CACHE_MAXSIZE, typed=True)
def _cached_reconstruction_omol_value(
    state: ReconstructionState,
    cache_name: str,
    builder: Callable[[pybel.Molecule], Any],
) -> Any:
    del cache_name
    return builder(state.omol)


class OmolStateMachine:
    """Mutable helper for staged `omol` transformations with cache invalidation."""

    def __init__(
        self,
        omol: pybel.Molecule,
        given_charge: int = 0,
        *,
        total_charge: int = 0,
        total_radical_electrons: int = 0,
        phase_history: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        key_cache: Optional[Dict[str, Any]] = None,
        omol_revision: int = 0,
    ) -> None:
        self.omol = omol
        self.given_charge = given_charge
        self.total_charge = total_charge
        self.total_radical_electrons = total_radical_electrons
        self.phase_history = list(phase_history)
        self.metadata = {} if metadata is None else dict(metadata)
        self.key_cache = {} if key_cache is None else dict(key_cache)
        self.omol_revision = omol_revision

    @classmethod
    def from_reconstruction_state(cls, state: ReconstructionState) -> OmolStateMachine:
        return cls(
            state.omol,
            state.given_charge,
            total_charge=state.total_charge,
            total_radical_electrons=state.total_radical_electrons,
            phase_history=state.phase_history,
            metadata=state.metadata,
            omol_revision=state.omol_revision,
        )

    def get_cached_omol_value(
        self,
        cache_name: str,
        builder: Callable[[pybel.Molecule], T],
    ) -> T:
        cached = self.key_cache.get(cache_name)
        if cached is not None:
            cached_revision, cached_value = cast(Tuple[int, T], cached)
            if cached_revision == self.omol_revision:
                return cached_value
        value = builder(self.omol)
        self.key_cache[cache_name] = (self.omol_revision, value)
        return value

    def _notify_trace(self, phase: Optional[str], **event: Any) -> None:
        if phase is None:
            return
        observer = _OMOL_TRACE_OBSERVER.get()
        if observer is not None:
            observer(self, phase, event)

    def trace_checkpoint(self, phase: str, **event: Any) -> None:
        """Emit a trace-only decision node without changing production phase history."""

        self._notify_trace(phase, kind="checkpoint", **event)

    def run_omol_stage(
        self,
        phase: Optional[str],
        stage: Callable[..., Tuple[pybel.Molecule, bool]],
        *args: object,
    ) -> bool:
        """Run an `omol -> omol` stage and invalidate derived metadata on mutation."""

        self.omol, hit = stage(self.omol, *args)
        if hit:
            self.omol_revision += 1
            _invalidate_omol_derived_metadata(self.metadata)
        if phase is not None:
            self.phase_history.append(phase)
        self._notify_trace(
            phase,
            kind="omol_stage",
            stage=getattr(stage, "__name__", type(stage).__name__),
            hit=bool(hit),
        )
        return hit

    def run_omol_charge_stage(
        self,
        phase: Optional[str],
        stage: Callable[..., Tuple[pybel.Molecule, int, bool]],
        *args: object,
    ) -> bool:
        """Run an `omol, charge -> omol, charge` stage with mutation tracking."""

        previous_charge = self.given_charge
        self.omol, self.given_charge, hit = stage(self.omol, self.given_charge, *args)
        if hit:
            self.omol_revision += 1
            _invalidate_omol_derived_metadata(self.metadata)
        if phase is not None:
            self.phase_history.append(phase)
        self._notify_trace(
            phase,
            kind="omol_charge_stage",
            stage=getattr(stage, "__name__", type(stage).__name__),
            hit=bool(hit),
            charge_before=int(previous_charge),
            charge_after=int(self.given_charge),
        )
        return hit

    def set_given_charge(self, phase: Optional[str], given_charge: int) -> None:
        previous_charge = self.given_charge
        self.given_charge = given_charge
        if phase is not None:
            self.phase_history.append(phase)
        self._notify_trace(
            phase,
            kind="set_given_charge",
            charge_before=int(previous_charge),
            charge_after=int(given_charge),
        )

    def annotate(self, phase: Optional[str], **metadata: Any) -> None:
        if phase is not None:
            self.phase_history.append(phase)
        if metadata:
            self.metadata.update(metadata)
        self._notify_trace(
            phase,
            kind="annotation",
            metadata_keys=tuple(sorted(metadata)),
        )

    def branch(
        self,
        phase: Optional[str],
        *,
        omol: Optional[pybel.Molecule] = None,
        given_charge: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OmolStateMachine:
        """Fork a new mutable machine from the current point in the phase history."""

        next_machine = OmolStateMachine(
            self.omol if omol is None else omol,
            self.given_charge if given_charge is None else given_charge,
            total_charge=self.total_charge,
            total_radical_electrons=self.total_radical_electrons,
            phase_history=self.phase_history,
            metadata=self.metadata,
            key_cache=self.key_cache if omol is None else None,
            omol_revision=self.omol_revision if omol is None else self.omol_revision + 1,
        )
        if omol is not None:
            # A replacement molecule may carry perception from its parent
            # topology. Match the native branch boundary before scoring it.
            next_machine.omol.OBMol.SetHybridizationPerceived(False)
            next_machine.omol.OBMol.SetAtomTypesPerceived(False)
            _invalidate_omol_derived_metadata(next_machine.metadata)
        if phase is not None:
            next_machine.phase_history.append(phase)
        if metadata:
            next_machine.metadata.update(metadata)
        next_machine._notify_trace(phase, kind="branch")
        return next_machine

    def freeze(
        self,
        *,
        total_charge: Optional[int] = None,
        total_radical_electrons: Optional[int] = None,
    ) -> ReconstructionState:
        return ReconstructionState(
            omol=self.omol,
            given_charge=self.given_charge,
            total_charge=self.total_charge if total_charge is None else total_charge,
            total_radical_electrons=(
                self.total_radical_electrons
                if total_radical_electrons is None
                else total_radical_electrons
            ),
            phase_history=tuple(self.phase_history),
            metadata=dict(self.metadata),
            omol_revision=self.omol_revision,
        )

    def freeze_like(self, state: ReconstructionState) -> ReconstructionState:
        return self.freeze(
            total_charge=state.total_charge,
            total_radical_electrons=state.total_radical_electrons,
        )


@dataclass
class MetalPreparationState:
    """Input with metals removed plus the candidate metal-state options."""

    no_metal_xyz_block: str
    available_valence_radical_states: Tuple[Tuple[MetalAtomPosition, ...], ...]
    total_charge: int
    total_radical_electrons: int
    phase_history: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    metal_sites: Tuple[MetalSite, ...] = ()


@dataclass
class MetalCandidateState:
    """A metal assignment tied to one no-metal reconstruction target bucket."""

    metal_states: Tuple[MetalAtomPosition, ...]
    no_metal_charge_target: int
    no_metal_radical_target: int
    phase_history: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)
    no_metal_state: Optional[ReconstructionState] = None
    combined_omol: Optional[pybel.Molecule] = None
    score: Optional[float] = None
    key_cache: Dict[str, Any] = field(default_factory=dict)

    def get_cached_candidate_value(
        self,
        cache_name: str,
        dependency_key: object,
        builder: Callable[[], T],
    ) -> T:
        cached = self.key_cache.get(cache_name)
        if cached is not None:
            cached_dependency, cached_value = cast(Tuple[object, T], cached)
            if cached_dependency == dependency_key:
                return cached_value

        value = builder()
        self.key_cache[cache_name] = (dependency_key, value)
        return value

    def metal_state_key(self) -> MetalStateKey:
        """Return the cached score key for the selected metal assignment."""

        cached = self.key_cache.get("metal_state_key")
        if cached is not None:
            return cached

        from molgr.fallback.utils.force_field import build_metal_state_key

        metal_key = build_metal_state_key(self.metal_states)
        self.key_cache["metal_state_key"] = metal_key
        return metal_key

    def combined_score_key(self) -> Tuple[ForceFieldScoreKey, MetalStateKey]:
        """Return the cache key for the combined no-metal + metal selection score."""

        no_metal_state = self.no_metal_state
        if no_metal_state is None:
            raise ValueError("MetalCandidateState requires no_metal_state before scoring")

        dependency_key = (id(no_metal_state), no_metal_state.omol_revision)
        return self.get_cached_candidate_value(
            "force_field_candidate_key",
            dependency_key,
            lambda: (no_metal_state.force_field_score_key(), self.metal_state_key()),
        )

    def combined_omol_dependency_key(self) -> Tuple[int, int, MetalStateKey]:
        no_metal_state = self.no_metal_state
        if no_metal_state is None:
            raise ValueError("MetalCandidateState requires no_metal_state before materialization")
        return id(no_metal_state), no_metal_state.omol_revision, self.metal_state_key()

    def materialize_combined_omol(self, combiner: CombinedOmolBuilder) -> pybel.Molecule:
        """Build and cache the final molecule with metals reinserted."""

        no_metal_state = self.no_metal_state
        if no_metal_state is None:
            raise ValueError("MetalCandidateState requires no_metal_state before materialization")

        dependency_key = self.combined_omol_dependency_key()
        cached = self.key_cache.get("combined_omol")
        if cached is not None:
            cached_dependency, cached_omol = cast(Tuple[object, pybel.Molecule], cached)
            if cached_dependency == dependency_key:
                self.combined_omol = cached_omol
                return cached_omol

        combined_omol = combiner(no_metal_state.omol, self.metal_states)
        self.key_cache["combined_omol"] = (dependency_key, combined_omol)
        self.combined_omol = combined_omol
        return combined_omol

    def combined_score(self, *, config: MolGRConfig | None = None) -> float:
        """Score the candidate using only the shared organic force-field energy."""

        no_metal_state = self.no_metal_state
        if no_metal_state is None:
            raise ValueError("MetalCandidateState requires no_metal_state before scoring")

        score_key = self.combined_score_key()
        cached = self.key_cache.get("combined_score")
        if cached is not None:
            cached_key, cached_score = cast(Tuple[object, float], cached)
            if cached_key == score_key:
                self.score = float(cached_score)
                self.metadata["score"] = float(cached_score)
                return float(cached_score)

        if config is None:
            organic_score = no_metal_state.score("organic_core")
        else:
            organic_score = no_metal_state.score("organic_core", config=config)
        score_value = organic_score
        self.metadata["force_field_energy"] = organic_score
        self.key_cache["combined_score"] = (score_key, score_value)
        self.metadata["score"] = score_value
        self.score = score_value
        return score_value

    def evaluate_score(
        self,
        profile: MetalCandidateScoreProfile = "combined",
        *,
        config: MolGRConfig | None = None,
    ) -> float:
        if profile == "combined":
            return self.combined_score(config=config)
        raise ValueError(f"Unsupported MetalCandidateState score profile: {profile!r}")


class MetalCandidateStateMachine:
    """Mutable helper for metal-candidate bookkeeping and cache invalidation."""

    def __init__(
        self,
        metal_states: Sequence[MetalAtomPosition],
        no_metal_charge_target: int,
        no_metal_radical_target: int,
        *,
        phase_history: Sequence[str] = (),
        metadata: Optional[Dict[str, Any]] = None,
        no_metal_state: Optional[ReconstructionState] = None,
        combined_omol: Optional[pybel.Molecule] = None,
        score: Optional[float] = None,
        key_cache: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.metal_states = tuple(metal_states)
        self.no_metal_charge_target = no_metal_charge_target
        self.no_metal_radical_target = no_metal_radical_target
        self.phase_history = list(phase_history)
        self.metadata = {} if metadata is None else dict(metadata)
        self.no_metal_state = no_metal_state
        self.combined_omol = combined_omol
        self.score = score
        self.key_cache = {} if key_cache is None else dict(key_cache)

    @classmethod
    def from_candidate_state(cls, state: MetalCandidateState) -> MetalCandidateStateMachine:
        return cls(
            state.metal_states,
            state.no_metal_charge_target,
            state.no_metal_radical_target,
            phase_history=state.phase_history,
            metadata=state.metadata,
            no_metal_state=state.no_metal_state,
            combined_omol=state.combined_omol,
            score=state.score,
            key_cache=state.key_cache,
        )

    def annotate(self, phase: Optional[str], **metadata: Any) -> None:
        if phase is not None:
            self.phase_history.append(phase)
        if metadata:
            self.metadata.update(metadata)

    def set_no_metal_state(
        self,
        phase: Optional[str],
        no_metal_state: Optional[ReconstructionState],
    ) -> None:
        if self.no_metal_state is not no_metal_state:
            _invalidate_candidate_derived_state(self.metadata, self.key_cache)
            self.combined_omol = None
            self.score = None
        self.no_metal_state = no_metal_state
        if phase is not None:
            self.phase_history.append(phase)

    def freeze(self) -> MetalCandidateState:
        return MetalCandidateState(
            metal_states=self.metal_states,
            no_metal_charge_target=self.no_metal_charge_target,
            no_metal_radical_target=self.no_metal_radical_target,
            phase_history=tuple(self.phase_history),
            metadata=dict(self.metadata),
            no_metal_state=self.no_metal_state,
            combined_omol=self.combined_omol,
            score=self.score,
            key_cache=dict(self.key_cache),
        )


def make_metal_candidate_state(
    base_phase_history: Sequence[str],
    metal_states: Sequence[MetalAtomPosition],
    no_metal_charge_target: int,
    no_metal_radical_target: int,
    *,
    combination_index: int,
) -> MetalCandidateState:
    """Create one bucketed metal candidate with the standard phase annotations."""

    machine = MetalCandidateStateMachine(
        metal_states,
        no_metal_charge_target,
        no_metal_radical_target,
        phase_history=base_phase_history,
        metadata={"combination_index": combination_index},
    )
    machine.annotate("enumerate_metal_combination")
    machine.annotate("reconstruct_no_metal_candidate")
    return machine.freeze()


__all__ = [
    "MetalCandidateStateMachine",
    "MetalCandidateState",
    "MetalPreparationState",
    "OmolStateMachine",
    "OmolTraceRecorder",
    "ReconstructionState",
    "TRACE_NODE_METADATA_KEY",
    "trace_omol_state_machine",
    "make_metal_candidate_state",
]
