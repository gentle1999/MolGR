"""Dataset-independent native graph comparison and explicit source policies."""

from __future__ import annotations

import hashlib
import math


DEFAULT_POLICY = {
    "use_chirality": False,
    "ignore_metal_coordination_bonds": True,
    "max_resonance": 5000,
    "comparison_timeout_seconds": 5.0,
}


def comparison_policy(source, timeout=None):
    """Resolve a prepared source's policy, respecting publisher-pinned settings."""
    policy = dict(source["comparison_policy"])
    if set(policy) != set(DEFAULT_POLICY):
        raise ValueError("Invalid comparison policy fields; prepare the case store again")
    for key in ("use_chirality", "ignore_metal_coordination_bonds"):
        if type(policy[key]) is not bool:
            raise ValueError(f"Invalid comparison policy: {key}")
    budget = policy["max_resonance"]
    if type(budget) is not int or budget < 1:
        raise ValueError("Invalid comparison policy: max_resonance")
    if timeout is not None:
        if (
            source.get("comparison_policy_locked")
            and timeout != policy["comparison_timeout_seconds"]
        ):
            raise ValueError("Source pins the comparison policy; its timeout cannot be overridden")
        policy["comparison_timeout_seconds"] = timeout
    value = policy["comparison_timeout_seconds"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("Invalid comparison policy: comparison_timeout_seconds")
    return policy


def native_reference(data, digest):
    """Load the stored native graph without a lossy SMILES round trip."""
    from rdkit import Chem

    if data is None or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Native reference hash mismatch")
    mol = Chem.Mol(data)
    if mol is None:
        raise ValueError("Invalid native reference graph")
    return mol


def compare_output(
    reference,
    output,
    *,
    max_resonance=5000,
    use_chirality=False,
    ignore_metal_coordination_bonds=True,
):
    """Compare the returned graph, independently of API status or display SMILES.

    Each source supplies one reference. Local alternatives cannot replace it.
    """
    from molgr.utils.equivalence import evaluate_equivalence

    if reference["reference_status"] != "valid" or output is None or output.rdkit_mol is None:
        return {}
    try:
        info = evaluate_equivalence(
            reference["ground_truth_rdmol"],
            output.rdkit_mol,
            use_chirality=use_chirality,
            max_resonance=max_resonance,
            ignore_metal_coordination_bonds=ignore_metal_coordination_bonds,
        )
        decision, reason = info.decision.value, info.reason
    except Exception as exc:
        decision, reason = "inconclusive", repr(exc)
    return {"decision": decision, "comparison_reason": reason}
