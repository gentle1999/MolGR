from molgr._native_types import NativeEquivalenceResult as _NativeEquivalenceResult

from .utils import MoleculeData

def compare_mapped_stereo(left: MoleculeData, right: MoleculeData, mapping: list[int]) -> int:
    """Compare a zero-based atom bijection. Return 1 for agreement, 0 for conflict or -1 for incomplete stereo."""

def evaluate(
    left: MoleculeData,
    right: MoleculeData,
    use_chirality: bool = False,
    max_resonance: int = 5000,
    ignore_metal_coordination_bonds: bool = True,
    max_mappings: int = 10000,
    max_mapping_nodes: int = 100000,
    max_resonance_steps: int = 24,
) -> _NativeEquivalenceResult:
    """Compare explicit-hydrogen graphs within the search limits and return a decision, mapping and resonance evidence."""
