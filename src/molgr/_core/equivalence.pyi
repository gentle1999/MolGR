from typing import Any

from .utils import MoleculeData

def compare_mapped_stereo(left: MoleculeData, right: MoleculeData, mapping: list[int]) -> int: ...
def evaluate(
    left: MoleculeData,
    right: MoleculeData,
    use_chirality: bool = False,
    max_resonance: int = 1000,
    ignore_metal_coordination_bonds: bool = True,
    max_mappings: int = 10000,
    max_mapping_nodes: int = 100000,
    max_resonance_steps: int = 24,
) -> dict[str, Any]: ...
