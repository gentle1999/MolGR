"""Initialize the chemistry libraries in the same order as the public MolGR API."""

from importlib.util import find_spec


# Loading pybel before RDKit Chem can crash during collection on Linux wheels.
# Keep individual geometry tests runnable regardless of their collection order.
if find_spec("rdkit") is not None:
    from rdkit import Chem  # noqa: F401
