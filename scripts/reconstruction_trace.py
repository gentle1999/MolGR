#!/usr/bin/env python3
"""Trace MolGR reconstruction for arbitrary XYZ/electronic-state inputs.

This script intentionally uses the Python fallback internals. The public C++
backend does not expose every intermediate metal candidate, while the fallback
path is the semantic reference used for parity.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import math
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, cast


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openbabel import openbabel as ob
from openbabel import pybel
from rdkit import Chem, RDLogger

from molgr.config import CONFIG, MolGRConfig
from molgr.fallback.pipeline import reconstruct_without_metals
from molgr.fallback.stages.preprocess import validate_omol
from molgr.fallback.state import (
    TRACE_NODE_METADATA_KEY,
    MetalCandidateState,
    OmolTraceRecorder,
    ReconstructionState,
    trace_omol_state_machine,
)
from molgr.fallback.utils import consts
from molgr.fallback.utils import metal_radical_inference as metal_radical_inference_module
from molgr.fallback.utils.dataclasses import MetalAtomPosition
from molgr.fallback.utils.electrons import (
    get_lone_pair_count,
    get_unpaired_electron_count,
    has_unresolved_two_electron_center,
)
from molgr.fallback.utils.metals import preparation, scoring, search
from molgr.fallback.utils.no_metals import preparation as no_metal_preparation
from molgr.fallback.utils.no_metals import selection as no_metal_selection
from molgr.utils.converter import pybel_to_rdmol
from molgr.utils.equivalence import check_equivalence
from molgr.utils.post_process import make_dative_bond


RDLogger.DisableLog("rdApp.*")  # type: ignore[arg-type]


@dataclasses.dataclass(frozen=True)
class TraceInputCase:
    """One coordinate/electronic-state input to trace."""

    id: str
    xyz_block: str
    total_charge: int
    total_radical_electrons: int
    xyz_path: Path | None = None
    xyz_source: str = ""
    fixture_kind: str = ""
    fixture_structure_file: str = ""
    expected_smiles: str = ""
    expected_smiles_options: tuple[str, ...] = ()
    reference_smiles: str = ""


@dataclasses.dataclass
class DofRenderContext:
    """Runtime state for optional rdkit-dof image rendering."""

    image_dir: Path
    display_base_dir: Path | None
    image_format: str = "svg"
    defer_images: bool = False
    max_images: int | None = 1000
    image_size: tuple[int, int] = (360, 300)
    grid_sub_img_size: tuple[int, int] = (320, 260)
    grid_mols_per_row: int = 3
    grid_max_mols: int = 24
    image_count: int = 0
    skipped_count: int = 0
    errors: list[dict[str, Any]] = dataclasses.field(default_factory=list)

    @property
    def use_svg(self) -> bool:
        return self.image_format == "svg"


_SCORE_DETAIL_PREFIXES = (
    "analysis_",
    "force_field_",
    "metal_",
    "organic_",
    "passes_",
    "selection_",
)
_SCORE_DETAIL_KEYS = {
    "combination_index",
    "score",
}
_IMPORTANT_SCORE_KEYS = (
    "metal_discordance_count",
    "metal_discordance_structural_count",
    "metal_discordance_conjugated_atom_deficit_count",
    "metal_discordance_max_conjugated_atom_count",
    "metal_discordance_conjugated_bond_deficit_count",
    "metal_discordance_max_conjugated_bond_count",
    "metal_discordance_aromatic_atom_deficit_count",
    "metal_discordance_max_aromatic_atom_count",
    "metal_discordance_aromatic_stability_deficit",
    "metal_discordance_max_aromatic_stability_score",
    "metal_discordance_aromatic_ring_deficit_count",
    "metal_discordance_max_aromatic_ring_count",
    "organic_aromatic_stability_score",
    "organic_aromatic_ring_count",
    "organic_aromatic_atom_count",
    "organic_conjugated_atom_count",
    "organic_conjugated_bond_count",
    "organic_max_conjugated_component_size",
    "organic_hyperconjugative_donor_count",
    "organic_hyperconjugation_score",
    "organic_hyperconjugation_max_score",
    "organic_hyperconjugation_deficit",
    "metal_discordance_inner_visible_diradical_count",
    "metal_discordance_excess_visible_singlet_two_electron_center_count",
    "metal_discordance_bent_cumulated_ring_allene_count",
    "metal_discordance_outer_or_invisible_adjacent_double_charge_count",
    "metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count",
    "metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count",
    "metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count",
    "metal_discordance_inner_visible_adjacent_carbanion_pair_count",
    "metal_discordance_inner_visible_conjugated_carbanion_pair_count",
    "metal_discordance_inner_visible_same_sign_charge_count",
    "metal_discordance_negative_metal_count",
    "metal_discordance_negative_metal_penalty",
    "metal_discordance_zero_valent_metals_with_organic_cation_count",
    "metal_discordance_unsaturated_organic_cation_count",
    "metal_discordance_negative_metal_outer_sphere_cation_exception",
    "metal_discordance_negative_metal_positive_metal_counterion_exception",
    "passes_metal_discordance_filter",
    "score",
    "force_field_energy",
    "organic_charge_localization_penalty",
    "organic_charge_localization_component_cancellation",
    "organic_charge_localization_polarity_inversion_penalty",
    "organic_charge_localization_reference_penalty",
    "organic_charge_localization_selection_margin",
    "organic_charge_localization_margin_difference",
    "organic_charge_localization_margin_exceeded",
    "organic_charge_localization_reference_metal_valence_max_delta",
    "organic_charge_localization_metal_valence_jump_exceeded",
    "organic_radical_localization_penalty",
    "selection_key",
    "analysis_selection_key_all_candidates",
)

_METAL_SELECTION_KEY_FIELDS = (
    (
        "metal_discordance_count",
        "金属失谐总数",
        "第一阶段只保留生产层中该值最小的候选",
    ),
    (
        "organic_charge_localization_margin_exceeded",
        "电荷局域化超出 margin",
        "相对同一最低金属失谐层的最低局域化分数高出至少 selection margin 时为 1",
    ),
    (
        "metal_discordance_conjugated_atom_deficit_count",
        "共轭原子亏损",
        "相对同批候选最佳共轭原子数的亏损",
    ),
    (
        "metal_discordance_conjugated_bond_deficit_count",
        "共轭键亏损",
        "相对同批候选最佳共轭键数的亏损",
    ),
    (
        "metal_discordance_aromatic_atom_deficit_count",
        "芳香原子亏损",
        "相对同批候选最佳芳香原子数的亏损",
    ),
    (
        "metal_discordance_aromatic_ring_deficit_count",
        "芳香环亏损",
        "相对同批候选最佳芳香环数的亏损",
    ),
    (
        "metal_discordance_aromatic_stability_deficit",
        "芳香稳定性亏损",
        "相对同批候选最佳芳香稳定性分数的亏损",
    ),
    (
        "organic_radical_localization_penalty",
        "自由基局域化惩罚",
        "有机部分自由基位于不利原子环境的惩罚",
    ),
    (
        "organic_hyperconjugation_deficit",
        "超共轭分数亏损",
        "前述指标相同时，相对同批候选最高超共轭分数的亏损",
    ),
    ("score", "候选总分", "当前为组合结构的力场分数"),
    ("combination_index", "组合序号", "此前各项完全相同时的稳定排序"),
)

_RESONANCE_SELECTION_KEY_FIELDS = (
    (
        "organic_formal_charge_absolute_sum",
        "形式电荷绝对值和",
        "min",
        "优先保留电荷分离程度更低的共振候选",
    ),
    (
        "organic_aromatic_atom_count",
        "芳香原子数",
        "max",
        "优先保留更多芳香原子的共振候选",
    ),
    (
        "organic_aromatic_ring_count",
        "芳香环数",
        "max",
        "芳香原子数相同时优先保留更多芳香环",
    ),
    (
        "organic_aromatic_stability_score",
        "芳香稳定性分数",
        "max",
        "按环大小和芳香覆盖计算的稳定性指标",
    ),
    (
        "organic_adjusted_max_conjugated_component_size",
        "校正后最大共轭组分",
        "max",
        "最大共轭组分减去形式电荷绝对值和的一半",
    ),
    (
        "organic_adjusted_conjugated_atom_count",
        "校正后共轭原子数",
        "max",
        "共轭原子数减去形式电荷绝对值和的一半",
    ),
    (
        "organic_adjusted_conjugated_bond_count",
        "校正后共轭键数",
        "max",
        "共轭键数减去形式电荷绝对值和的一半",
    ),
    (
        "organic_excess_radical_labels",
        "超额自由基标记",
        "min",
        "显式自由基标记超过全局自由基目标的数量",
    ),
    (
        "organic_hyperconjugation_score",
        "超共轭分数",
        "max",
        "饱和中性 sp3 碳向相邻 π/缺电子中心供给的 C-H σ 键数",
    ),
    (
        "score",
        "有机力场分数",
        "min",
        "前述电子拓扑指标完全相同时才比较力场能量",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xyz",
        dest="xyz_paths",
        action="append",
        type=Path,
        default=[],
        help="XYZ file to trace. May be repeated.",
    )
    parser.add_argument(
        "--xyz-block",
        dest="xyz_blocks",
        action="append",
        default=[],
        help="Inline XYZ block to trace. May be repeated.",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read one XYZ block from stdin.",
    )
    parser.add_argument(
        "--id",
        dest="ids",
        action="append",
        default=[],
        help=(
            "Optional case id/label. May be repeated or comma-separated. Labels are assigned "
            "to --xyz, --xyz-block, and --stdin inputs in order."
        ),
    )
    parser.add_argument(
        "--case-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON case file. The file may contain one object or a list of objects with "
            "id, xyz_path or xyz_block, total_charge/charge, and total_radical_electrons or "
            "spin_multiplicity."
        ),
    )
    parser.add_argument(
        "--review-fixtures-manifest",
        type=Path,
        default=None,
        help=(
            "Load trace inputs directly from a reviewed fixture manifest. SDF coordinates and "
            "XYZ electronic states are resolved relative to the manifest."
        ),
    )
    parser.add_argument(
        "--fixture-id",
        dest="fixture_ids",
        action="append",
        default=[],
        help=(
            "Optional reviewed fixture id. May be repeated or comma-separated. "
            "Defaults to all manifest fixtures."
        ),
    )
    parser.add_argument(
        "--total-charge",
        "--charge",
        dest="total_charge",
        type=int,
        default=0,
        help="Total molecular charge for direct XYZ inputs. Default: 0.",
    )
    radical_group = parser.add_mutually_exclusive_group()
    radical_group.add_argument(
        "--total-radical-electrons",
        type=int,
        default=None,
        help="Total radical-electron count for direct XYZ inputs. Default: 0.",
    )
    radical_group.add_argument(
        "--spin-multiplicity",
        type=int,
        default=None,
        help="Spin multiplicity for direct XYZ inputs. Converted to radical electrons as M - 1.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--format",
        choices=("auto", "json", "html"),
        default="auto",
        help=(
            "Output format. In auto mode, .json writes JSON; every other output path and stdout "
            "write an HTML report."
        ),
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation when --format json is used. Use 0 for compact JSON.",
    )
    parser.add_argument(
        "--no-score-all-candidates",
        action="store_true",
        help=(
            "Do not compute analysis metal-candidate metrics for candidates filtered out by "
            "discordance. Production metadata is still reported."
        ),
    )
    parser.add_argument(
        "--render-dof-images",
        action="store_true",
        help=(
            "Render reconstructed molecule graphs with rdkit-dof. SVG output is embedded in "
            "the report data; non-SVG output is written as image files."
        ),
    )
    parser.add_argument(
        "--dof-image-dir",
        type=Path,
        default=None,
        help=(
            "Directory for rdkit-dof images. Defaults to <report>.dof-images for --out, or "
            "molgr_trace_dof_images when writing to stdout."
        ),
    )
    parser.add_argument(
        "--dof-image-format",
        choices=("svg", "png"),
        default="svg",
        help="rdkit-dof image format. Default: svg.",
    )
    parser.add_argument(
        "--dof-max-images",
        type=int,
        default=1000,
        help="Maximum number of individual rdkit-dof images to write. Default: 1000.",
    )
    parser.add_argument(
        "--dof-image-size",
        default="360x300",
        help="Single-molecule rdkit-dof image size as WIDTHxHEIGHT. Default: 360x300.",
    )
    parser.add_argument(
        "--dof-grid-max-mols",
        type=int,
        default=24,
        help="Maximum molecules per comparison grid image. Default: 24.",
    )
    parser.add_argument(
        "--dof-grid-mols-per-row",
        type=int,
        default=3,
        help="Molecules per row for rdkit-dof comparison grids. Default: 3.",
    )
    parser.add_argument(
        "--dof-grid-sub-img-size",
        default="320x260",
        help="Grid sub-image size as WIDTHxHEIGHT. Default: 320x260.",
    )
    args = parser.parse_args()
    if args.total_radical_electrons is not None and args.total_radical_electrons < 0:
        parser.error("--total-radical-electrons must be >= 0")
    if args.spin_multiplicity is not None and args.spin_multiplicity < 1:
        parser.error("--spin-multiplicity must be >= 1")
    if args.dof_max_images < 0:
        parser.error("--dof-max-images must be >= 0")
    if args.dof_grid_max_mols < 0:
        parser.error("--dof-grid-max-mols must be >= 0")
    if args.dof_grid_mols_per_row < 1:
        parser.error("--dof-grid-mols-per-row must be >= 1")
    try:
        args.dof_image_size = _parse_size(args.dof_image_size)
        args.dof_grid_sub_img_size = _parse_size(args.dof_grid_sub_img_size)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _parse_size(raw_size: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*[xX,]\s*(\d+)\s*", raw_size)
    if match is None:
        raise ValueError(f"invalid size {raw_size!r}; expected WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


def split_repeated_values(raw_values: Sequence[str]) -> list[str]:
    values: list[str] = []
    for raw_value in raw_values:
        values.extend(item.strip() for item in raw_value.split(",") if item.strip())
    return values


def _radicals_from_spin_multiplicity(spin_multiplicity: int) -> int:
    if spin_multiplicity < 1:
        raise ValueError("spin_multiplicity must be >= 1")
    return spin_multiplicity - 1


def _direct_total_radicals(args: argparse.Namespace) -> int:
    if args.total_radical_electrons is not None:
        return int(args.total_radical_electrons)
    if args.spin_multiplicity is not None:
        return _radicals_from_spin_multiplicity(int(args.spin_multiplicity))
    return 0


def _default_dof_image_dir(args: argparse.Namespace) -> Path:
    if args.dof_image_dir is not None:
        return args.dof_image_dir
    if args.out is not None:
        return args.out.with_suffix(args.out.suffix + ".dof-images")
    return Path("molgr_trace_dof_images")


def _make_dof_render_context(args: argparse.Namespace) -> DofRenderContext | None:
    if not args.render_dof_images and _resolve_output_format(args) != "html":
        return None
    return DofRenderContext(
        image_dir=_default_dof_image_dir(args),
        display_base_dir=args.out.parent if args.out is not None else None,
        image_format=args.dof_image_format,
        max_images=int(args.dof_max_images),
        image_size=args.dof_image_size,
        grid_sub_img_size=args.dof_grid_sub_img_size,
        grid_mols_per_row=int(args.dof_grid_mols_per_row),
        grid_max_mols=int(args.dof_grid_max_mols),
    )


def dof_rendering_summary(render_context: DofRenderContext) -> dict[str, Any]:
    if render_context.defer_images:
        storage = "deferred_sdf"
    else:
        storage = "embedded" if render_context.use_svg else "files"
    return {
        "image_dir": ""
        if render_context.defer_images or render_context.use_svg
        else str(render_context.image_dir),
        "storage": storage,
        "format": render_context.image_format,
        "image_count": render_context.image_count,
        "skipped_count": render_context.skipped_count,
        "max_images": render_context.max_images,
        "errors": render_context.errors,
    }


def _safe_filename_part(value: Any) -> str:
    text = str(value).strip().replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = text.strip("._")
    return text or "item"


def _display_image_path(path: Path, render_context: DofRenderContext) -> str:
    if render_context.display_base_dir is None:
        return str(path)
    try:
        return str(path.relative_to(render_context.display_base_dir))
    except ValueError:
        return str(path)


def _dof_image_record(
    path: Path,
    *,
    render_context: DofRenderContext,
    label: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "path": str(path),
        "display_path": _display_image_path(path, render_context),
        "format": render_context.image_format,
    }


def _inline_dof_svg_record(
    svg_fragment: str,
    *,
    render_context: DofRenderContext,
    label: str,
    kind: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "format": render_context.image_format,
        "svg_fragment": svg_fragment,
    }


def _rdmol_sdf_block(mol: Chem.Mol) -> str:
    block = Chem.MolToMolBlock(mol, includeStereo=True, kekulize=False, forceV3000=True)
    if not block.endswith("\n"):
        block += "\n"
    return block + "$$$$\n"


def _safe_candidate_smiles(mol: Chem.Mol) -> str:
    """Serialize a candidate without forcing an invalid aromatic Kekule form."""

    clone = Chem.RemoveHs(Chem.Mol(mol), sanitize=False)
    try:
        return Chem.MolToSmiles(clone, canonical=True, isomericSmiles=True)
    except Chem.KekulizeException:
        # Candidate bond edits can leave aromatic flags inconsistent.  Clear
        # only those flags for display; explicit bond orders remain intact.
        rw_mol = Chem.RWMol(clone)
        for atom in rw_mol.GetAtoms():
            atom.SetIsAromatic(False)
        for bond in rw_mol.GetBonds():
            bond.SetIsAromatic(False)
        fallback = rw_mol.GetMol()
        fallback.UpdatePropertyCache(strict=False)
        return Chem.MolToSmiles(fallback, canonical=True, isomericSmiles=True)


def _deferred_dof_record(
    mols: Sequence[Chem.Mol],
    *,
    render_context: DofRenderContext,
    label: str,
    kind: str,
    render_type: str,
    legends: Sequence[str],
    duration: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "label": label,
        "format": "sdf",
        "render_type": render_type,
        "legends": list(legends),
        "sdfs": [_rdmol_sdf_block(mol) for mol in mols],
    }
    if render_type == "single":
        record["sdf"] = record.pop("sdfs")[0]
        record["size"] = list(render_context.image_size)
    elif render_type == "grid":
        record["mols_per_row"] = render_context.grid_mols_per_row
        record["sub_image_size"] = list(render_context.grid_sub_img_size)
    else:
        record["size"] = list(render_context.image_size)
        record["duration"] = duration
        record["animation"] = True
        record["frame_count"] = len(mols)
    return record


def _svg_fragment_from_dof_image(image: Any) -> str:
    if isinstance(image, str):
        svg_text = image
    elif hasattr(image, "data"):
        svg_text = str(image.data)
    elif hasattr(image, "_repr_svg_"):
        svg_text = str(image._repr_svg_())
    else:
        svg_text = str(image)
    svg_start = svg_text.find("<svg")
    if svg_start >= 0:
        svg_text = svg_text[svg_start:]
    return svg_text


def _reserve_dof_render_slot(
    *,
    render_context: DofRenderContext,
    label: str,
    kind: str,
    animation: bool = False,
) -> int | dict[str, Any]:
    if render_context.max_images is None or render_context.image_count < render_context.max_images:
        index = render_context.image_count
        render_context.image_count += 1
        return index
    render_context.skipped_count += 1
    skipped = {
        "kind": kind,
        "label": label,
        "status": "skipped_limit",
        "max_images": render_context.max_images,
    }
    if animation:
        skipped["animation"] = True
    return skipped


def _render_dof_molecule(
    omol: pybel.Molecule,
    *,
    render_context: DofRenderContext | None,
    case_id: str,
    label: str,
    kind: str,
) -> dict[str, Any] | None:
    if render_context is None:
        return None
    slot = _reserve_dof_render_slot(render_context=render_context, label=label, kind=kind)
    if isinstance(slot, dict):
        return slot

    file_stem = (
        f"{_safe_filename_part(case_id)}__{slot:04d}"
        f"__{_safe_filename_part(kind)}__{_safe_filename_part(label)}"
    )
    try:
        rdmol = pybel_to_rdmol(omol, sanitize=False, kekulize=False)
        if render_context.defer_images:
            return _deferred_dof_record(
                [rdmol],
                render_context=render_context,
                label=label,
                kind=kind,
                render_type="single",
                legends=[label],
            )
        sdf = _rdmol_sdf_block(rdmol)
        from rdkit_dof import MolToDofImage

        if render_context.use_svg:
            image = MolToDofImage(
                rdmol,
                size=render_context.image_size,
                legend=label,
                use_svg=True,
                return_image=False,
            )
            record = _inline_dof_svg_record(
                _svg_fragment_from_dof_image(image),
                render_context=render_context,
                label=label,
                kind=kind,
            )
            record["sdf"] = sdf
            return record
        path = render_context.image_dir / f"{file_stem}.{render_context.image_format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        MolToDofImage(
            rdmol,
            size=render_context.image_size,
            legend=label,
            use_svg=False,
            return_image=False,
            filename=str(path),
        )
        record = _dof_image_record(
            path,
            render_context=render_context,
            label=label,
            kind=kind,
        )
        record["sdf"] = sdf
        return record
    except Exception as exc:
        error = {
            "kind": kind,
            "label": label,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        render_context.errors.append(error)
        return error


def _render_dof_grid(
    items: Iterable[tuple[pybel.Molecule, str]],
    *,
    render_context: DofRenderContext | None,
    case_id: str,
    label: str,
    kind: str,
) -> dict[str, Any] | None:
    if render_context is None or render_context.grid_max_mols == 0:
        return None
    slot = _reserve_dof_render_slot(render_context=render_context, label=label, kind=kind)
    if isinstance(slot, dict):
        return slot
    selected_items = list(items)[: render_context.grid_max_mols]
    if not selected_items:
        render_context.image_count -= 1
        return None

    file_stem = (
        f"{_safe_filename_part(case_id)}__{slot:04d}__grid__{_safe_filename_part(kind)}"
        f"__{_safe_filename_part(label)}"
    )
    try:
        mols = [
            pybel_to_rdmol(omol, sanitize=False, kekulize=False) for omol, _legend in selected_items
        ]
        legends = [legend for _omol, legend in selected_items]
        if render_context.defer_images:
            return _deferred_dof_record(
                mols,
                render_context=render_context,
                label=label,
                kind=kind,
                render_type="grid",
                legends=legends,
            )
        from rdkit_dof import MolsToGridDofImage

        if render_context.use_svg:
            image = MolsToGridDofImage(
                mols,
                molsPerRow=render_context.grid_mols_per_row,
                subImgSize=render_context.grid_sub_img_size,
                legends=legends,
                use_svg=True,
                return_image=False,
            )
            record = _inline_dof_svg_record(
                _svg_fragment_from_dof_image(image),
                render_context=render_context,
                label=label,
                kind=kind,
            )
            record["sdfs"] = [_rdmol_sdf_block(mol) for mol in mols]
            return record
        path = render_context.image_dir / f"{file_stem}.{render_context.image_format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        MolsToGridDofImage(
            mols,
            molsPerRow=render_context.grid_mols_per_row,
            subImgSize=render_context.grid_sub_img_size,
            legends=legends,
            use_svg=False,
            return_image=False,
            filename=str(path),
        )
        record = _dof_image_record(
            path,
            render_context=render_context,
            label=label,
            kind=kind,
        )
        record["sdfs"] = [_rdmol_sdf_block(mol) for mol in mols]
        return record
    except Exception as exc:
        error = {
            "kind": kind,
            "label": label,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        render_context.errors.append(error)
        return error


def _copy_omol(omol: pybel.Molecule) -> pybel.Molecule:
    return pybel.Molecule(ob.OBMol(cast(ob.OBMol, omol.OBMol)))


def _render_dof_animation(
    items: Iterable[tuple[pybel.Molecule, str]],
    *,
    render_context: DofRenderContext | None,
    case_id: str,
    label: str,
    kind: str,
    duration: int = 650,
) -> dict[str, Any] | None:
    if render_context is None:
        return None
    slot = _reserve_dof_render_slot(
        render_context=render_context,
        label=label,
        kind=kind,
        animation=True,
    )
    if isinstance(slot, dict):
        return slot
    selected_items = list(items)
    if not selected_items:
        render_context.image_count -= 1
        return None

    file_stem = (
        f"{_safe_filename_part(case_id)}__{slot:04d}"
        f"__animation__{_safe_filename_part(kind)}__{_safe_filename_part(label)}"
    )
    try:
        mols = [
            pybel_to_rdmol(omol, sanitize=False, kekulize=False) for omol, _legend in selected_items
        ]
        legends = [legend for _omol, legend in selected_items]
        if render_context.defer_images:
            return _deferred_dof_record(
                mols,
                render_context=render_context,
                label=label,
                kind=kind,
                render_type="animation",
                legends=legends,
                duration=duration,
            )
        from rdkit_dof import MolsToDofSvgAnimation

        image = MolsToDofSvgAnimation(
            mols,
            size=render_context.image_size,
            legends=legends,
            duration=duration,
            loop=0,
            return_image=False,
        )
        svg_fragment = _svg_fragment_from_dof_image(image)
        if render_context.use_svg:
            record = _inline_dof_svg_record(
                svg_fragment,
                render_context=render_context,
                label=label,
                kind=kind,
            )
        else:
            path = render_context.image_dir / f"{file_stem}.svg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(svg_fragment, encoding="utf-8")
            record = _dof_image_record(
                path,
                render_context=render_context,
                label=label,
                kind=kind,
            )
            record["format"] = "svg"
        record.update({"animation": True, "frame_count": len(selected_items)})
        record["sdfs"] = [_rdmol_sdf_block(mol) for mol in mols]
        return record
    except Exception as exc:
        error = {
            "kind": kind,
            "label": label,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "animation": True,
        }
        render_context.errors.append(error)
        return error


def _require_int_field(
    raw_case: dict[str, Any], keys: Sequence[str], *, default: int | None
) -> int:
    for key in keys:
        value = raw_case.get(key)
        if value is not None and value != "":
            return int(value)
    if default is not None:
        return default
    raise ValueError(f"JSON case is missing one of: {', '.join(keys)}")


def _json_case_to_input_case(
    raw_case: dict[str, Any],
    *,
    base_dir: Path,
    fallback_index: int,
) -> TraceInputCase:
    case_id = str(raw_case.get("id") or raw_case.get("label") or f"case_{fallback_index}")
    total_charge = _require_int_field(raw_case, ("total_charge", "charge"), default=0)
    if raw_case.get("total_radical_electrons") is not None or raw_case.get("radicals") is not None:
        total_radicals = _require_int_field(
            raw_case,
            ("total_radical_electrons", "radicals"),
            default=None,
        )
    elif raw_case.get("spin_multiplicity") is not None:
        total_radicals = _radicals_from_spin_multiplicity(int(raw_case["spin_multiplicity"]))
    else:
        total_radicals = 0

    xyz_block = raw_case.get("xyz_block") or raw_case.get("xyz")
    xyz_path: Path | None = None
    xyz_source = "json_xyz_block"
    if xyz_block is None:
        raw_path = raw_case.get("xyz_path") or raw_case.get("path")
        if raw_path is None:
            raise ValueError(f"JSON case {case_id!r} is missing xyz_block or xyz_path")
        xyz_path = Path(str(raw_path))
        if not xyz_path.is_absolute():
            xyz_path = base_dir / xyz_path
        xyz_block = xyz_path.read_text(encoding="utf-8")
        xyz_source = "json_xyz_path"

    return TraceInputCase(
        id=case_id,
        xyz_block=str(xyz_block),
        total_charge=total_charge,
        total_radical_electrons=total_radicals,
        xyz_path=xyz_path,
        xyz_source=xyz_source,
    )


def _load_json_cases(path: Path) -> list[TraceInputCase]:
    raw_payload = json.loads(path.read_text(encoding="utf-8"))
    raw_cases = raw_payload if isinstance(raw_payload, list) else [raw_payload]
    cases: list[TraceInputCase] = []
    for case_index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"JSON case #{case_index} must be an object")
        cases.append(
            _json_case_to_input_case(
                raw_case,
                base_dir=path.parent,
                fallback_index=case_index,
            )
        )
    return cases


_APPROVED_REVIEW_FIXTURE_KINDS = {
    "accepted_both",
    "approved_graph",
    "manual_reference",
    "reference_graph",
}


def _review_fixture_xyz_block(path: Path) -> str:
    if path.suffix.lower() != ".sdf":
        return path.read_text(encoding="utf-8")
    expected = next(
        (
            mol
            for mol in Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
            if mol is not None
        ),
        None,
    )
    if expected is None:
        raise ValueError(f"review fixture SDF is unreadable: {path}")
    if expected.GetNumConformers() != 1:
        raise ValueError(f"review fixture SDF must contain one conformer: {path}")
    return Chem.MolToXYZBlock(expected)


def load_review_fixture_cases(
    manifest_path: Path,
    fixture_ids: Sequence[str] = (),
) -> list[TraceInputCase]:
    """Load trace inputs from the current reviewed fixture manifest."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_records = payload.get("fixtures") if isinstance(payload, dict) else None
    if not isinstance(raw_records, list):
        raise ValueError(f"invalid reviewed fixture manifest: {manifest_path}")

    requested_ids = set(fixture_ids)
    found_ids: set[str] = set()
    cases: list[TraceInputCase] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise ValueError(f"invalid reviewed fixture record: {raw_record!r}")
        case_id = str(raw_record.get("case_id") or "").strip()
        if not case_id:
            raise ValueError("reviewed fixture record is missing case_id")
        if requested_ids and case_id not in requested_ids:
            continue
        if case_id in found_ids:
            raise ValueError(f"duplicate reviewed fixture id: {case_id}")
        found_ids.add(case_id)

        structure_file = str(raw_record.get("structure_file") or "").strip()
        if not structure_file:
            raise ValueError(f"reviewed fixture {case_id} is missing structure_file")
        structure_path = manifest_path.parent / structure_file
        if not structure_path.is_file():
            raise FileNotFoundError(structure_path)
        fixture_kind = str(raw_record.get("kind") or "").strip()
        expected_smiles = (
            str(raw_record.get("approved_smiles") or "").strip()
            if fixture_kind in _APPROVED_REVIEW_FIXTURE_KINDS
            else ""
        )
        raw_accepted_smiles = raw_record.get("accepted_smiles")
        expected_smiles_options = (
            tuple(
                str(value).strip()
                for value in raw_accepted_smiles
                if isinstance(value, str) and value.strip()
            )
            if isinstance(raw_accepted_smiles, list)
            else ()
        )
        if not expected_smiles_options and expected_smiles:
            expected_smiles_options = (expected_smiles,)
        cases.append(
            TraceInputCase(
                id=case_id,
                xyz_block=_review_fixture_xyz_block(structure_path),
                total_charge=int(raw_record.get("total_charge") or 0),
                total_radical_electrons=int(raw_record.get("total_radical_electrons") or 0),
                xyz_path=structure_path,
                xyz_source="review_fixture",
                fixture_kind=fixture_kind,
                fixture_structure_file=structure_file,
                expected_smiles=expected_smiles,
                expected_smiles_options=expected_smiles_options,
            )
        )

    missing_ids = requested_ids - found_ids
    if missing_ids:
        raise ValueError("reviewed fixture ids not found: " + ", ".join(sorted(missing_ids)))
    if not cases:
        raise ValueError(f"reviewed fixture manifest selected no cases: {manifest_path}")
    return cases


def _collect_input_cases(args: argparse.Namespace) -> list[TraceInputCase]:
    cases: list[TraceInputCase] = []
    if args.case_json is not None:
        cases.extend(_load_json_cases(args.case_json))
    fixture_ids = split_repeated_values(args.fixture_ids)
    if args.review_fixtures_manifest is not None:
        cases.extend(load_review_fixture_cases(args.review_fixtures_manifest, fixture_ids))
    elif fixture_ids:
        raise ValueError("--fixture-id requires --review-fixtures-manifest")

    direct_total_radicals = _direct_total_radicals(args)
    labels = split_repeated_values(args.ids)
    label_index = 0

    def next_label(default: str) -> str:
        nonlocal label_index
        if label_index >= len(labels):
            return default
        label = labels[label_index]
        label_index += 1
        return label

    direct_input_index = 1
    for xyz_path in args.xyz_paths:
        cases.append(
            TraceInputCase(
                id=next_label(xyz_path.stem),
                xyz_block=xyz_path.read_text(encoding="utf-8"),
                total_charge=int(args.total_charge),
                total_radical_electrons=direct_total_radicals,
                xyz_path=xyz_path,
                xyz_source="xyz_path",
            )
        )
        direct_input_index += 1

    for xyz_block in args.xyz_blocks:
        cases.append(
            TraceInputCase(
                id=next_label(f"inline_{direct_input_index}"),
                xyz_block=str(xyz_block),
                total_charge=int(args.total_charge),
                total_radical_electrons=direct_total_radicals,
                xyz_source="xyz_block",
            )
        )
        direct_input_index += 1

    if args.stdin:
        cases.append(
            TraceInputCase(
                id=next_label("stdin"),
                xyz_block=sys.stdin.read(),
                total_charge=int(args.total_charge),
                total_radical_electrons=direct_total_radicals,
                xyz_source="stdin",
            )
        )

    if label_index < len(labels):
        raise ValueError("more --id labels were provided than direct coordinate inputs")
    if not cases:
        raise ValueError(
            "provide at least one of --xyz, --xyz-block, --stdin, --case-json, "
            "or --review-fixtures-manifest"
        )
    return cases


def _first_smiles_token(raw_smiles: str) -> str:
    tokens = raw_smiles.strip().split()
    return tokens[0] if tokens else ""


def _write_omol_smiles(omol: pybel.Molecule, output_format: str) -> str:
    try:
        return _first_smiles_token(cast(str, omol.write(output_format)))
    except (OSError, ValueError, RuntimeError):
        return ""


def _omol_smiles_detail(omol: pybel.Molecule) -> dict[str, str]:
    return {
        "smiles": _write_omol_smiles(omol, "smi"),
        "canonical_smiles": _write_omol_smiles(omol, "can"),
    }


def _omol_charge_radical_summary(omol: pybel.Molecule) -> dict[str, Any]:
    obmol = cast(ob.OBMol, omol.OBMol)
    formal_charge = 0
    spin_multiplicity_sum = 0
    spin_multiplicity_singlet_sum = 0
    atom_charge_counts: dict[str, int] = {}
    atom_spin_counts: dict[str, int] = {}
    atom_lone_pair_counts: dict[str, int] = {}
    unresolved_center_counts: dict[str, int] = {}
    lone_pair_sum = 0
    for atom_iter in ob.OBMolAtomIter(obmol):
        atom = cast(ob.OBAtom, atom_iter)
        charge = int(atom.GetFormalCharge())
        spin = get_unpaired_electron_count(atom)
        lone_pairs = get_lone_pair_count(atom)
        formal_charge += charge
        spin_multiplicity_sum += spin
        spin_multiplicity_singlet_sum += spin % 2
        lone_pair_sum += lone_pairs
        if charge != 0:
            atom_charge_counts[f"{atom.GetAtomicNum()}:{charge:+d}"] = (
                atom_charge_counts.get(f"{atom.GetAtomicNum()}:{charge:+d}", 0) + 1
            )
        if spin != 0:
            atom_spin_counts[f"{atom.GetAtomicNum()}:r{spin}"] = (
                atom_spin_counts.get(f"{atom.GetAtomicNum()}:r{spin}", 0) + 1
            )
        if lone_pairs != 0:
            atom_lone_pair_counts[f"{atom.GetAtomicNum()}:lp{lone_pairs}"] = (
                atom_lone_pair_counts.get(f"{atom.GetAtomicNum()}:lp{lone_pairs}", 0) + 1
            )
        if has_unresolved_two_electron_center(atom):
            unresolved_center_counts[str(atom.GetAtomicNum())] = (
                unresolved_center_counts.get(str(atom.GetAtomicNum()), 0) + 1
            )

    return {
        "atom_count": int(obmol.NumAtoms()),
        "bond_count": int(obmol.NumBonds()),
        "formal_charge_sum": formal_charge,
        "spin_multiplicity_sum": spin_multiplicity_sum,
        "spin_multiplicity_singlet_sum": spin_multiplicity_singlet_sum,
        "lone_pair_sum": lone_pair_sum,
        "charged_atom_counts": atom_charge_counts,
        "radical_atom_counts": atom_spin_counts,
        "lone_pair_atom_counts": atom_lone_pair_counts,
        "unresolved_two_electron_center_counts": unresolved_center_counts,
    }


def _omol_state_snapshot(
    omol: pybel.Molecule,
    *,
    given_charge: int,
    target_charge: int,
    target_radical_electrons: int,
) -> dict[str, Any]:
    summary = _omol_charge_radical_summary(omol)
    summary.update(_omol_smiles_detail(omol))
    summary["given_charge"] = int(given_charge)
    summary["valid_for_target"] = validate_omol(
        omol,
        target_charge,
        target_radical_electrons,
    )
    return summary


def _metal_state_to_dict(
    metal_state: MetalAtomPosition,
    *,
    field_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prior_valences = consts.METAL_VALENCE_AVAILABLE_PRIOR.get(metal_state.symbol, [])
    minor_valences = consts.METAL_VALENCE_AVAILABLE_MINOR.get(metal_state.symbol, [])
    if metal_state.valence in prior_valences:
        prior_class = "prior"
        prior_penalty = 0.0
    elif metal_state.valence in minor_valences:
        prior_class = "minor"
        prior_penalty = 10.0
    else:
        prior_class = "other"
        prior_penalty = 20.0
    nonpositive_penalty = (
        10.0 * max(abs(metal_state.valence), 1) if metal_state.valence <= 0 else 0.0
    )
    result = {
        "idx": int(metal_state.idx),
        "symbol": metal_state.symbol,
        "element_idx": int(metal_state.element_idx),
        "valence": int(metal_state.valence),
        "radical_num": int(metal_state.radical_num),
        "valence_prior_class": prior_class,
        "valence_prior_penalty": prior_penalty,
        "nonpositive_valence_penalty": nonpositive_penalty,
        "assignment_penalty": prior_penalty + nonpositive_penalty,
        "position": [
            float(metal_state.position_x),
            float(metal_state.position_y),
            float(metal_state.position_z),
        ],
    }
    if field_analysis is not None:
        result["ligand_field"] = field_analysis
    return result


def _metal_field_analysis_by_state(
    xyz_block: str,
    available_states: Sequence[Sequence[MetalAtomPosition]],
    *,
    config: MolGRConfig | None,
) -> dict[tuple[int, int], dict[str, Any]]:
    resolved_config = MolGRConfig() if config is None else config
    field_config = resolved_config.metal_radical_inference
    omol = pybel.readstring("xyz", xyz_block)
    analyses: dict[tuple[int, int], dict[str, Any]] = {}
    for state_options in available_states:
        for state in state_options:
            key = (int(state.idx), int(state.valence))
            if key in analyses:
                continue
            atom = cast(ob.OBAtom, omol.OBMol.GetAtom(int(state.idx)))
            inference = metal_radical_inference_module.infer_metal_radical_state(
                atom,
                int(state.valence),
                config=resolved_config,
            )
            donors = metal_radical_inference_module._collect_coordination_environment(
                atom,
                metal_radical_config=field_config,
                metal_scoring_config=resolved_config.metal_scoring,
            )
            analyses[key] = {
                "coordination_number": int(inference.coordination_number),
                "geometry": inference.geometry,
                "field_score": float(inference.field_score),
                "field_strength": inference.field_strength,
                "weak_field_threshold": float(field_config.weak_field_threshold),
                "strong_field_threshold": float(field_config.strong_field_threshold),
                "ambiguity_margin": float(field_config.field_ambiguity_margin),
                "effective_d_electrons": int(inference.effective_d_electrons),
                "radical_options_preferred_first": list(inference.radical_counts),
                "donors": [
                    {
                        "idx": int(donor.atom_idx),
                        "symbol": ob.GetSymbol(int(donor.atomic_num)),
                        "atomic_num": int(donor.atomic_num),
                        "distance_angstrom": float(donor.distance_angstrom),
                        "base_field_strength": float(
                            metal_radical_inference_module._DONOR_FIELD_STRENGTH.get(
                                int(donor.atomic_num),
                                0.60,
                            )
                        ),
                    }
                    for donor in donors
                ],
            }
    return analyses


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(cast(Any, value)))
    if isinstance(value, dict):
        return {str(_jsonable(key)): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    return repr(value)


def _format_float(value: float) -> str:
    if not math.isfinite(value):
        return _jsonable(value)
    if abs(value) >= 1000.0 or (0.0 < abs(value) < 0.001):
        return f"{value:.6e}"
    return f"{value:.6g}"


def _format_value(value: Any) -> str:
    value = _jsonable(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            return "[" + ", ".join(_format_value(item) for item in value) + "]"
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _is_rendered_dof_image(image: Any) -> bool:
    return (
        isinstance(image, dict)
        and not image.get("status")
        and bool(image.get("svg_fragment") or image.get("display_path") or image.get("path"))
    )


def _dof_image_path_text(image: Any) -> str:
    if not isinstance(image, dict):
        return ""
    if image.get("svg_fragment"):
        return "embedded svg"
    return image.get("display_path") or image.get("path") or image.get("status", "")


def _resolve_output_format(args: argparse.Namespace) -> str:
    if args.format != "auto":
        return cast(str, args.format)
    if args.out is not None and args.out.suffix.lower() == ".json":
        return "json"
    return "html"


def _score_detail_value(candidate: dict[str, Any], key: str, default: Any = "") -> Any:
    details = cast(Dict[str, Any], candidate.get("score_details", {}))
    metadata = cast(Dict[str, Any], candidate.get("metadata", {}))
    if key in details:
        return details[key]
    return metadata.get(key, default)


def _short_hash(value: Any) -> str:
    return hashlib.sha1(repr(value).encode("utf-8")).hexdigest()[:12]


def _metal_state_label(metal_state: dict[str, Any]) -> str:
    valence = int(metal_state.get("valence", 0))
    valence_label = f"{valence:+d}"
    return (
        f"#{metal_state.get('idx', '')} {metal_state.get('symbol', '')}"
        f"({valence_label}, r{metal_state.get('radical_num', 0)})"
    )


def _metal_states_label(metal_states: Sequence[dict[str, Any]]) -> str:
    return "; ".join(_metal_state_label(metal_state) for metal_state in metal_states)


def _html_escape(value: Any) -> str:
    return html.escape(_format_value(value), quote=True)


def _html_json(value: Any) -> str:
    return html.escape(
        json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False), quote=False
    )


def _html_script_json(value: Any) -> str:
    return (
        json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _read_svg_fragment(image: Any) -> str:
    if not _is_rendered_dof_image(image):
        return ""
    if isinstance(image.get("svg_fragment"), str):
        return cast(str, image["svg_fragment"])
    path = Path(str(image.get("path") or image.get("display_path")))
    image_format = str(image.get("format") or path.suffix.lstrip(".")).lower()
    if image_format != "svg":
        return ""
    try:
        svg_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        svg_text = path.read_text(encoding="iso-8859-1")
    except OSError:
        return ""
    svg_start = svg_text.find("<svg")
    if svg_start >= 0:
        svg_text = svg_text[svg_start:]
    return svg_text


def _with_inline_dof_svgs(value: Any) -> Any:
    if isinstance(value, dict):
        copied = {key: _with_inline_dof_svgs(item) for key, item in value.items()}
        if _is_rendered_dof_image(copied) and "svg_fragment" not in copied:
            svg_fragment = _read_svg_fragment(copied)
            if svg_fragment:
                copied["svg_fragment"] = svg_fragment
        return copied
    if isinstance(value, list):
        return [_with_inline_dof_svgs(item) for item in value]
    return value


_DEFERRED_DOF_PAYLOAD_KEYS = {
    "render_type",
    "sdf",
    "sdfs",
    "legends",
    "size",
    "mols_per_row",
    "sub_image_size",
    "duration",
}


def _extract_deferred_dof_payloads(value: Any) -> tuple[Any, dict[str, dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}

    def extract(item: Any) -> Any:
        if isinstance(item, dict):
            if item.get("render_type") and (item.get("sdf") or item.get("sdfs")):
                payload = {
                    key: _jsonable(item[key]) for key in _DEFERRED_DOF_PAYLOAD_KEYS if key in item
                }
                digest_source = json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                render_id = "dof-" + hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:20]
                payloads.setdefault(render_id, payload)
                copied = {
                    key: extract(child)
                    for key, child in item.items()
                    if key not in _DEFERRED_DOF_PAYLOAD_KEYS
                }
                copied["render_id"] = render_id
                copied["render_type"] = item["render_type"]
                return copied
            return {key: extract(child) for key, child in item.items()}
        if isinstance(item, list):
            return [extract(child) for child in item]
        return item

    return extract(value), payloads


def _html_metric_grid(items: Sequence[tuple[str, Any]]) -> str:
    cells = []
    for label, value in items:
        if value in ("", None, [], {}):
            continue
        cells.append(
            '<div class="metric">'
            f'<span class="metric-label">{_html_escape(label)}</span>'
            f'<span class="metric-value">{_html_escape(value)}</span>'
            "</div>"
        )
    return '<div class="metrics">' + "".join(cells) + "</div>" if cells else ""


def _html_table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], *, class_name: str = ""
) -> str:
    if not rows:
        return '<p class="empty-inline">无</p>'
    class_attr = f' class="{html.escape(class_name)}"' if class_name else ""
    head = "".join(f"<th>{_html_escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>" + "".join(f"<td>{_html_escape(cell)}</td>" for cell in row) + "</tr>"
        )
    return f"<table{class_attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _html_kv_table(items: Sequence[tuple[str, Any]], *, class_name: str = "") -> str:
    rows = [(key, value) for key, value in items if value not in ("", None, [], {})]
    return _html_table(("字段", "值"), rows, class_name=class_name)


def _html_details(title: str, body: str, *, open_: bool = False, class_name: str = "") -> str:
    class_attr = f' class="{html.escape(class_name)}"' if class_name else ""
    open_attr = " open" if open_ else ""
    return (
        f"<details{class_attr}{open_attr}><summary>{_html_escape(title)}</summary>{body}</details>"
    )


def _html_json_block(value: Any, *, class_name: str = "json-block") -> str:
    return f'<pre class="{html.escape(class_name)}">{_html_json(value)}</pre>'


def _html_score_table(mapping: Any) -> str:
    if not isinstance(mapping, dict) or not mapping:
        return '<p class="empty-inline">无</p>'
    keys = [key for key in _IMPORTANT_SCORE_KEYS if key in mapping]
    keys.extend(sorted(key for key in mapping if key not in set(keys)))
    return _html_table(
        ("分数项", "值"), [(key, mapping[key]) for key in keys], class_name="score-table"
    )


def _candidate_title(candidate: dict[str, Any]) -> str:
    if candidate.get("trace_item_title"):
        return str(candidate["trace_item_title"])
    if "search_layer_index" in candidate or "combination_index" in candidate:
        selected = "selected " if candidate.get("selected") else ""
        return (
            f"{selected}L{candidate.get('search_layer_index', '')}/"
            f"C{candidate.get('combination_index', '')}"
        ).strip()
    if "resonance_index" in candidate:
        return f"resonance {candidate.get('resonance_index', '')}"
    return str(candidate.get("label") or candidate.get("kind") or "item")


def _collect_html_image_items(output: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    def add_item(
        *,
        case: dict[str, Any],
        title: str,
        image: Any,
        group: str,
        selected: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(image, dict):
            return
        svg = _read_svg_fragment(image)
        if not svg:
            return
        item_metadata = {} if metadata is None else metadata
        items.append(
            {
                "id": f"item-{len(items)}",
                "case_id": case.get("id", ""),
                "title": title,
                "group": group,
                "selected": selected,
                "svg": svg,
                "image": image,
                "metadata": _jsonable(item_metadata),
                "score": item_metadata.get("score", ""),
                "layer": item_metadata.get("search_layer_index", ""),
                "combination": item_metadata.get("combination_index", ""),
                "smiles": item_metadata.get("canonical_smiles")
                or item_metadata.get("smiles")
                or item_metadata.get("organic_smiles", ""),
            }
        )

    for raw_case in cast(List[Any], output.get("cases", [])):
        if not isinstance(raw_case, dict):
            continue
        case = raw_case
        no_metal_trace = cast(Dict[str, Any], case.get("no_metal_trace", {}))
        selected = cast(Dict[str, Any], no_metal_trace.get("selected_candidate", {}))
        selected_state = cast(Dict[str, Any], selected.get("state", {}))
        add_item(
            case=case,
            title="selected no-metal candidate",
            image=selected_state.get("dof_image"),
            group="selected",
            selected=True,
            metadata=selected,
        )
        add_item(
            case=case,
            title="metal candidate grid",
            image=case.get("dof_candidate_grid"),
            group="grid",
            metadata={
                "candidate_count": case.get("candidate_count", ""),
                "production_candidate_count": case.get("production_candidate_count", ""),
                "selected_candidate": case.get("selected_candidate"),
            },
        )
        for candidate in cast(List[Any], case.get("candidates", [])):
            if not isinstance(candidate, dict):
                continue
            add_item(
                case=case,
                title=_candidate_title(candidate),
                image=candidate.get("dof_image"),
                group="candidate",
                selected=bool(candidate.get("selected")),
                metadata=candidate,
            )
        search_summary = cast(Dict[str, Any], case.get("search", {}))
        for layer in cast(List[Any], search_summary.get("layer_summaries", [])):
            if not isinstance(layer, dict):
                continue
            for bucket in cast(List[Any], layer.get("target_buckets", [])):
                if not isinstance(bucket, dict):
                    continue
                target = cast(Dict[str, Any], bucket.get("target", {}))
                add_item(
                    case=case,
                    title=(
                        f"organic target L{layer.get('layer_index', '')} "
                        f"Q={target.get('no_metal_charge', '')} "
                        f"R={target.get('no_metal_radical_electrons', '')}"
                    ),
                    image=bucket.get("dof_image"),
                    group="organic",
                    metadata=bucket,
                )
    return items


def _html_no_metal_trace(no_metal_trace: dict[str, Any]) -> str:
    if not no_metal_trace:
        return '<p class="empty-inline">无</p>'

    target = cast(Dict[str, Any], no_metal_trace.get("target", {}))
    selected = cast(Dict[str, Any], no_metal_trace.get("selected_candidate", {}))
    selected_state = cast(Dict[str, Any], selected.get("state", {}))
    resonance = cast(Dict[str, Any], no_metal_trace.get("resonance", {}))
    actions = cast(List[Any], no_metal_trace.get("neighbor_radical_actions", []))
    sections = [
        _html_kv_table(
            (
                ("状态", no_metal_trace.get("status", "")),
                ("目标电荷", target.get("total_charge", "")),
                ("目标自由基电子", target.get("total_radical_electrons", "")),
                ("邻位自由基动作", actions),
                ("恢复层", no_metal_trace.get("recovery_tier", 0)),
                ("共振种子序号", resonance.get("seed_index", "")),
                ("共振序号", resonance.get("resonance_index", "")),
                ("raw 共振序号", resonance.get("raw_index", "")),
                ("归一化方式", resonance.get("normalization", "")),
                ("选中 canonical SMILES", selected_state.get("canonical_smiles", "")),
                ("选中分数", selected.get("score", "")),
                ("选中选择键", selected.get("organic_topology_selection_key", "")),
            )
        )
    ]

    step_rows = [
        (step.get("step_index", ""), step.get("phase", ""), step.get("kind", ""))
        for step in cast(List[Any], no_metal_trace.get("pipeline_steps", []))
        if isinstance(step, dict)
    ]
    sections.append(
        _html_details(
            f"生产管线阶段 ({len(step_rows)})",
            _html_table(("步骤", "阶段", "类型"), step_rows),
            open_=True,
        )
    )
    trace_node_rows = []
    for node in cast(List[Any], no_metal_trace.get("trace_nodes", [])):
        if not isinstance(node, dict):
            continue
        event = cast(Dict[str, Any], node.get("event", {}))
        metadata = cast(Dict[str, Any], node.get("metadata", {}))
        state = cast(Dict[str, Any], node.get("state", {}))
        trace_node_rows.append(
            (
                node.get("global_node_index", ""),
                node.get("global_node_locator", ""),
                node.get("global_tree_parent_index", ""),
                node.get("node_id", ""),
                node.get("tree_parent_id", ""),
                node.get("tree_depth", ""),
                node.get("selected_path", False),
                node.get("phase", ""),
                node.get("kind", ""),
                event.get("stage", event.get("kind", "")),
                event.get("hit", ""),
                metadata.get("resonance_seed_index", ""),
                metadata.get("resonance_raw_index", ""),
                metadata.get("resonance_normalization", ""),
                state.get("given_charge", ""),
                state.get("canonical_smiles", state.get("smiles", "")),
                _dof_image_path_text(node.get("dof_image")),
            )
        )
    sections.append(
        _html_details(
            f"完整状态机分支树 ({len(trace_node_rows)})",
            _html_table(
                (
                    "全局索引",
                    "定位符",
                    "全局树父节点",
                    "节点",
                    "树父节点",
                    "层级",
                    "选中路径",
                    "阶段",
                    "类型",
                    "函数/事件",
                    "命中",
                    "共振种子",
                    "raw 共振",
                    "规范化",
                    "剩余电荷",
                    "canonical SMILES",
                    "DOF 图像",
                ),
                trace_node_rows,
            ),
            open_=True,
        )
    )
    if selected:
        sections.append(_html_details("选中候选完整 JSON", _html_json_block(selected)))
    if resonance:
        sections.append(_html_details("共振选择元数据", _html_json_block(resonance)))
    sections.append(_html_details("完整 no-metal trace JSON", _html_json_block(no_metal_trace)))
    return "".join(sections)


def _html_candidate_details(candidate: dict[str, Any]) -> str:
    organic_part = cast(Dict[str, Any], candidate.get("organic_part", {}))
    metal_rows = []
    for metal_state in cast(List[Any], candidate.get("metal_states", [])):
        if isinstance(metal_state, dict):
            metal_rows.append(
                (
                    metal_state.get("idx", ""),
                    metal_state.get("symbol", ""),
                    metal_state.get("valence", ""),
                    metal_state.get("radical_num", ""),
                    metal_state.get("position", ""),
                )
            )
    body = _html_kv_table(
        (
            ("全局索引", candidate.get("global_node_index", "")),
            ("定位符", candidate.get("global_node_locator", "")),
            ("候选序号", candidate.get("candidate_index", "")),
            ("候选 identity", candidate.get("candidate_identity", "")),
            ("选中", candidate.get("selected", "")),
            ("层", candidate.get("search_layer_index", "")),
            ("组合", candidate.get("combination_index", "")),
            ("生产层", candidate.get("in_production_selection_layer", "")),
            ("候选总电荷", candidate.get("candidate_total_charge", "")),
            ("目标", candidate.get("target", "")),
            ("有机 canonical SMILES", organic_part.get("canonical_smiles", "")),
            ("分数", candidate.get("score", "")),
            ("DOF 图像", _dof_image_path_text(candidate.get("dof_image"))),
        )
    )
    body += "<h4>金属状态</h4>" + _html_table(
        ("原子序号", "元素", "价态", "自由基数", "坐标"), metal_rows
    )
    body += "<h4>分数构成</h4>" + _html_score_table(candidate.get("score_details", {}))
    production_metadata = cast(Dict[str, Any], candidate.get("production_metadata", {}))
    if production_metadata:
        body += _html_details(
            "生产选择 metadata",
            _html_score_table(production_metadata)
            + _html_details("完整 production metadata JSON", _html_json_block(production_metadata)),
        )
    body += _html_details(
        "score_details JSON", _html_json_block(candidate.get("score_details", {}))
    )
    body += _html_details("organic_part JSON", _html_json_block(candidate.get("organic_part", {})))
    body += _html_details("target JSON", _html_json_block(candidate.get("target", {})))
    body += _html_details(
        "phase_history JSON", _html_json_block(candidate.get("phase_history", []))
    )
    body += _html_details(
        "完整 candidate metadata", _html_json_block(candidate.get("metadata", {}))
    )
    body += _html_details(
        "no_metal_state",
        _html_json_block(candidate.get("no_metal_state", {})),
    )
    body += _html_details("完整 candidate JSON", _html_json_block(candidate))
    return body


def _html_case_trace(case: dict[str, Any]) -> str:
    case_id = str(case.get("id", "unknown"))
    if case.get("status") == "error":
        return (
            f'<section class="case-trace" id="case-{html.escape(_safe_filename_part(case_id))}">'
            f"<h2>{_html_escape(case_id)}</h2>"
            + _html_kv_table((("状态", "error"), ("错误", case.get("error", ""))))
            + "</section>"
        )
    base_state = cast(Dict[str, Any], case.get("base_state", {}))
    selected = cast(Dict[str, Any], case.get("selected_candidate", {}))
    review_fixture = cast(Dict[str, Any], case.get("review_fixture", {}))
    summary = _html_kv_table(
        (
            ("id", case.get("id", "")),
            ("CSV 行号", case.get("row_index", "")),
            ("状态", case.get("status", "")),
            ("重建类型", case.get("trace_kind", "")),
            ("总电荷", case.get("charge", "")),
            ("总自由基电子数", case.get("total_radical_electrons", "")),
            ("自旋多重度", case.get("spin_multiplicity", "")),
            ("自旋来源", case.get("spin_source", "")),
            ("XYZ 路径", case.get("xyz_path", "")),
            ("XYZ 来源", case.get("xyz_source", "")),
            ("review fixture 类型", review_fixture.get("kind", "")),
            ("review fixture 等价", review_fixture.get("equivalent", "")),
            ("review fixture 比较原因", review_fixture.get("equivalence_reason", "")),
            ("参考 SMILES", case.get("reference_smiles", "")),
            ("金属原子数", base_state.get("metal_atom_count", "")),
            (
                "生产选择层",
                cast(Dict[str, Any], case.get("search", {})).get("selected_layer_index", ""),
            ),
            ("生产候选数", case.get("production_candidate_count", "")),
            ("全部已评分候选数", case.get("candidate_count", "")),
            (
                "全局节点范围",
                (
                    f"{case.get('global_node_index_start')}..{case.get('global_node_index_end')}"
                    if case.get("global_node_index_start") is not None
                    else ""
                ),
            ),
            ("报告全局节点数", case.get("report_global_node_count", "")),
            ("选中候选", selected.get("combination_index", "")),
            ("耗时秒", case.get("elapsed_seconds", "")),
        )
    )
    sections = [
        f'<section class="case-trace" id="case-{html.escape(_safe_filename_part(case_id))}">',
        f"<h2>{_html_escape(case_id)}</h2>",
        _html_details("基本信息", summary, open_=True),
        _html_details("基础状态完整 JSON", _html_json_block(base_state)),
    ]
    if review_fixture:
        sections.append(
            _html_details("review fixture 同步检查", _html_json_block(review_fixture), open_=True)
        )
    if case.get("trace_kind") == "no_metal":
        sections.append(
            _html_details(
                "无金属重建过程",
                _html_no_metal_trace(cast(Dict[str, Any], case.get("no_metal_trace", {}))),
                open_=True,
            )
        )
    else:
        state_rows = []
        for site_index, state_options in enumerate(
            cast(List[Any], base_state.get("available_metal_states_by_site", []))
        ):
            if not isinstance(state_options, list):
                continue
            for option_index, metal_state in enumerate(state_options):
                if isinstance(metal_state, dict):
                    state_rows.append(
                        (
                            site_index,
                            option_index,
                            metal_state.get("idx", ""),
                            metal_state.get("symbol", ""),
                            metal_state.get("valence", ""),
                            metal_state.get("radical_num", ""),
                            metal_state.get("position", ""),
                        )
                    )
        sections.append(
            _html_details(
                "金属位点候选",
                _html_table(
                    ("位点", "选项", "原子序号", "元素", "价态", "自由基数", "坐标"),
                    state_rows,
                ),
            )
        )
        search_summary = cast(Dict[str, Any], case.get("search", {}))
        layer_rows = []
        layer_detail_blocks = []
        target_rows = []
        no_metal_sections = []
        for layer in cast(List[Any], search_summary.get("layer_summaries", [])):
            if not isinstance(layer, dict):
                continue
            layer_detail_blocks.append(
                _html_details(
                    f"Layer {layer.get('layer_index', '')} 完整 JSON",
                    _html_json_block(layer),
                )
            )
            layer_rows.append(
                (
                    layer.get("layer_index", ""),
                    layer.get("status", ""),
                    layer.get("production_selected_layer", ""),
                    layer.get("state_group_count", ""),
                    layer.get("state_options_per_group", ""),
                    layer.get("target_bucket_count", ""),
                    layer.get("candidate_count", ""),
                    layer.get("prepared_candidate_count", ""),
                    bool(layer.get("analysis_score_context")),
                )
            )
            for bucket in cast(List[Any], layer.get("target_buckets", [])):
                if not isinstance(bucket, dict):
                    continue
                target = cast(Dict[str, Any], bucket.get("target", {}))
                organic_part = cast(Dict[str, Any], bucket.get("organic_part", {}))
                target_rows.append(
                    (
                        layer.get("layer_index", ""),
                        target.get("no_metal_charge", ""),
                        target.get("no_metal_radical_electrons", ""),
                        bucket.get("status", ""),
                        bucket.get("candidate_count", ""),
                        bucket.get("prepared_candidate_count", ""),
                        organic_part.get("canonical_smiles", organic_part.get("smiles", "")),
                        bucket.get("no_metal_score", ""),
                        _dof_image_path_text(bucket.get("dof_image")),
                    )
                )
                if isinstance(bucket.get("no_metal_trace"), dict):
                    no_metal_sections.append(
                        _html_details(
                            (
                                f"L{layer.get('layer_index', '')} "
                                f"Q={target.get('no_metal_charge', '')} "
                                f"R={target.get('no_metal_radical_electrons', '')}"
                            ),
                            _html_no_metal_trace(cast(Dict[str, Any], bucket["no_metal_trace"])),
                        )
                    )
        sections.append(
            _html_details(
                "搜索层",
                _html_table(
                    (
                        "层",
                        "状态",
                        "生产选择层",
                        "金属组数",
                        "每组候选数",
                        "目标桶数",
                        "枚举候选数",
                        "已评分候选数",
                        "analysis score context",
                    ),
                    layer_rows,
                ),
                open_=True,
            )
        )
        if layer_detail_blocks:
            sections.append(_html_details("搜索层完整 trace", "".join(layer_detail_blocks)))
        sections.append(
            _html_details(
                "有机目标桶",
                _html_table(
                    (
                        "层",
                        "有机目标电荷",
                        "有机自由基电子",
                        "状态",
                        "候选数",
                        "已评分数",
                        "有机部分 canonical SMILES",
                        "有机力场分",
                        "DOF 图像",
                    ),
                    target_rows,
                ),
                open_=True,
            )
        )
        sections.append(
            _html_details("分析和评分上下文", _html_json_block(case.get("analysis", {})))
        )
        if no_metal_sections:
            sections.append(_html_details("无金属重建过程", "".join(no_metal_sections)))
        candidate_rows = []
        for candidate in cast(List[Any], case.get("candidates", [])):
            if not isinstance(candidate, dict):
                continue
            organic_part = cast(Dict[str, Any], candidate.get("organic_part", {}))
            candidate_rows.append(
                (
                    candidate.get("global_node_index", ""),
                    candidate.get("global_node_locator", ""),
                    "是" if candidate.get("selected") else "",
                    candidate.get("search_layer_index", ""),
                    candidate.get("combination_index", ""),
                    "是" if candidate.get("in_production_selection_layer") else "",
                    _metal_states_label(
                        cast(List[Dict[str, Any]], candidate.get("metal_states", []))
                    ),
                    organic_part.get("canonical_smiles", organic_part.get("smiles", "")),
                    candidate.get("candidate_total_charge", ""),
                    cast(Dict[str, Any], candidate.get("target", {})).get("no_metal_charge", ""),
                    cast(Dict[str, Any], candidate.get("target", {})).get(
                        "no_metal_radical_electrons", ""
                    ),
                    *[
                        _score_detail_value(candidate, score_key, "")
                        for score_key in _IMPORTANT_SCORE_KEYS
                    ],
                    _dof_image_path_text(candidate.get("dof_image")),
                )
            )
        sections.append(
            _html_details(
                "候选总览",
                _html_table(
                    (
                        "全局索引",
                        "定位符",
                        "选中",
                        "层",
                        "组合",
                        "生产层",
                        "金属状态",
                        "有机 canonical SMILES",
                        "候选总电荷",
                        "有机电荷",
                        "有机自由基",
                        *_IMPORTANT_SCORE_KEYS,
                        "DOF 图像",
                    ),
                    candidate_rows,
                    class_name="score-overview",
                ),
                open_=True,
            )
        )
        detail_blocks = []
        for candidate in cast(List[Any], case.get("candidates", [])):
            if isinstance(candidate, dict):
                detail_blocks.append(
                    _html_details(_candidate_title(candidate), _html_candidate_details(candidate))
                )
        sections.append(_html_details("候选分数和 metadata", "".join(detail_blocks), open_=True))
    sections.append(_html_details("完整 case JSON", _html_json_block(case)))
    sections.append("</section>")
    return "".join(sections)


def _render_html_trace_sections(output: dict[str, Any]) -> str:
    sections = []
    for case in cast(List[Any], output.get("cases", [])):
        if isinstance(case, dict):
            sections.append(_html_case_trace(case))
    return "".join(sections) if sections else '<div class="empty">No cases.</div>'


def _render_html_report(output: dict[str, Any]) -> str:
    items = _collect_html_image_items(output)
    trace_sections = _render_html_trace_sections(output)
    input_summary = cast(Dict[str, Any], output.get("input", {}))
    dof_rendering = cast(Dict[str, Any], output.get("dof_rendering", {}))
    metrics = _html_metric_grid(
        (
            ("source", input_summary.get("source", "")),
            ("cases", output.get("case_count", "")),
            ("ids", ", ".join(str(item) for item in cast(List[Any], input_summary.get("ids", [])))),
            ("storage", dof_rendering.get("storage", "")),
            ("images", dof_rendering.get("image_count", "")),
            ("skipped", dof_rendering.get("skipped_count", "")),
            ("errors", len(cast(List[Any], dof_rendering.get("errors", [])))),
        )
    )
    case_options = sorted({str(item.get("case_id", "")) for item in items if item.get("case_id")})
    group_options = sorted({str(item.get("group", "")) for item in items if item.get("group")})
    nav_cards: list[str] = []
    detail_cards: list[str] = []
    for index, item in enumerate(items):
        item_id = str(item["id"])
        selected_class = " is-selected" if item.get("selected") else ""
        active_class = " is-active" if index == 0 else ""
        metadata_json = _html_json(item.get("metadata", {}))
        nav_cards.append(
            f'<button class="trace-card{selected_class}{active_class}" type="button" '
            f'data-target="{html.escape(item_id)}" data-case="{_html_escape(item.get("case_id", ""))}" '
            f'data-group="{_html_escape(item.get("group", ""))}" '
            f'data-search="{_html_escape(" ".join(str(item.get(k, "")) for k in ("case_id", "title", "group", "smiles", "score")))}">'
            f'<span class="card-title">{_html_escape(item.get("title", ""))}</span>'
            f'<span class="card-meta">{_html_escape(item.get("case_id", ""))} · {_html_escape(item.get("group", ""))}</span>'
            "</button>"
        )
        detail_cards.append(
            f'<article id="{html.escape(item_id)}" class="detail-card{active_class}" '
            f'data-case="{_html_escape(item.get("case_id", ""))}" data-group="{_html_escape(item.get("group", ""))}">'
            '<div class="detail-head">'
            f"<div><h2>{_html_escape(item.get('title', ''))}</h2>"
            f"<p>{_html_escape(item.get('case_id', ''))} · {_html_escape(item.get('group', ''))}</p></div>"
            f'<span class="badge{" selected" if item.get("selected") else ""}">'
            f"{'selected' if item.get('selected') else _html_escape(item.get('group', ''))}</span>"
            "</div>"
            '<div class="svg-wrap">'
            f"{item.get('svg', '')}"
            "</div>"
            f'<pre class="metadata">{metadata_json}</pre>'
            "</article>"
        )

    cases_json = _html_json(output)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MolGR Trace</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#fff; --line:#d8dee8; --text:#172033; --muted:#667085; --accent:#2563eb; --selected:#b45309; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; min-width:1000px; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:16px 20px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:10; }}
    h1 {{ margin:0 0 10px; font-size:20px; }}
    .metrics {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .metric {{ border:1px solid var(--line); border-radius:6px; background:#fafbfc; padding:6px 9px; min-width:96px; }}
    .metric-label {{ display:block; font-size:11px; color:var(--muted); text-transform:uppercase; }}
    .metric-value {{ display:block; font-size:13px; font-weight:700; overflow-wrap:anywhere; }}
    main {{ display:grid; grid-template-columns:340px minmax(0,1fr); gap:14px; padding:14px; min-height:calc(100vh - 86px); }}
    aside {{ min-width:0; border:1px solid var(--line); background:var(--panel); border-radius:8px; padding:10px; align-self:stretch; position:relative; display:flex; flex-direction:column; gap:10px; }}
    .filters {{ display:grid; gap:8px; }}
    .filters input, .filters select {{ width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; background:#fff; }}
    .toggle {{ display:flex; align-items:center; gap:8px; font-size:13px; color:var(--muted); }}
    .list {{ overflow:auto; display:grid; gap:6px; padding-right:2px; }}
    .trace-card {{ width:100%; border:1px solid var(--line); border-radius:7px; background:#fff; text-align:left; padding:8px; cursor:pointer; }}
    .trace-card:hover, .trace-card.is-active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(37,99,235,.14); }}
    .trace-card.is-selected {{ border-color:var(--selected); background:#fff7ed; }}
    .card-title {{ display:block; font-weight:800; font-size:13px; overflow-wrap:anywhere; }}
    .card-meta {{ display:block; margin-top:3px; color:var(--muted); font-size:12px; }}
    .content {{ min-width:0; display:grid; gap:14px; }}
    .section-title {{ margin:4px 0 0; font-size:15px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }}
    .detail-card {{ display:none; border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:14px; min-width:0; }}
    .detail-card.is-active {{ display:block; }}
    .detail-head {{ display:flex; justify-content:space-between; gap:10px; align-items:flex-start; margin-bottom:12px; }}
    h2 {{ margin:0; font-size:18px; }}
    .detail-head p {{ margin:4px 0 0; color:var(--muted); }}
    .badge {{ border:1px solid var(--line); border-radius:999px; padding:4px 8px; font-size:12px; color:var(--muted); white-space:nowrap; }}
    .badge.selected {{ color:#92400e; border-color:#f59e0b; background:#fffbeb; }}
    .svg-wrap {{ width:100%; overflow:auto; border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; }}
    .svg-wrap svg {{ display:block; max-width:100%; height:auto; margin:auto; }}
    .metadata {{ margin:12px 0 0; padding:10px; border:1px solid var(--line); border-radius:8px; background:#0f172a; color:#e5e7eb; overflow:auto; max-height:300px; font-size:12px; }}
    .trace-section-wrap {{ display:grid; gap:14px; }}
    .case-trace {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:14px; min-width:0; }}
    .case-trace > h2 {{ margin-bottom:12px; }}
    details {{ border:1px solid var(--line); border-radius:8px; background:#fff; margin:10px 0; padding:0; overflow:hidden; }}
    details > summary {{ cursor:pointer; padding:9px 11px; font-weight:800; background:#f8fafc; border-bottom:1px solid var(--line); }}
    details:not([open]) > summary {{ border-bottom:0; }}
    details > *:not(summary) {{ margin:10px; max-width:calc(100% - 20px); overflow-x:auto; }}
    h4 {{ margin:14px 10px 6px; font-size:13px; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; margin:10px 0; }}
    th, td {{ border-bottom:1px solid #e5e7eb; padding:6px 7px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }}
    th {{ color:#334155; background:#f8fafc; position:sticky; top:0; z-index:1; }}
    .score-overview {{ min-width:2400px; }}
    .json-block {{ margin:10px; padding:10px; border:1px solid var(--line); border-radius:8px; background:#0f172a; color:#e5e7eb; overflow:auto; max-height:480px; font-size:12px; }}
    .empty-inline {{ margin:10px; color:var(--muted); }}
    .empty {{ border:1px dashed var(--line); border-radius:8px; padding:24px; text-align:center; color:var(--muted); background:#fff; }}
    @media (max-width:900px) {{
      main {{ grid-template-columns:300px minmax(0,1fr); gap:8px; padding:8px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>MolGR Trace</h1>
    {metrics}
  </header>
  <main>
    <aside>
      <div class="filters">
        <input id="search" type="search" placeholder="Filter case, title, group, score...">
        <select id="case-filter"><option value="">All cases</option>{"".join(f'<option value="{html.escape(case_id)}">{html.escape(case_id)}</option>' for case_id in case_options)}</select>
        <select id="group-filter"><option value="">All groups</option>{"".join(f'<option value="{html.escape(group)}">{html.escape(group)}</option>' for group in group_options)}</select>
        <label class="toggle"><input id="selected-only" type="checkbox"> Selected only</label>
      </div>
      <div id="list" class="list">{"".join(nav_cards) if nav_cards else '<div class="empty">No DOF images were rendered.</div>'}</div>
    </aside>
    <section class="content">
      <h2 class="section-title">DOF images</h2>
      {"".join(detail_cards) if detail_cards else '<div class="empty">No DOF images were rendered. Increase --dof-max-images if needed.</div>'}
      <h2 class="section-title">Full trace</h2>
      <div class="trace-section-wrap">{trace_sections}</div>
    </section>
  </main>
  <script id="trace-json" type="application/json">{cases_json}</script>
  <script>
    (() => {{
      const cards = Array.from(document.querySelectorAll(".trace-card"));
      const details = Array.from(document.querySelectorAll(".detail-card"));
      const search = document.getElementById("search");
      const caseFilter = document.getElementById("case-filter");
      const groupFilter = document.getElementById("group-filter");
      const selectedOnly = document.getElementById("selected-only");

      function activate(id) {{
        cards.forEach(card => card.classList.toggle("is-active", card.dataset.target === id));
        details.forEach(detail => detail.classList.toggle("is-active", detail.id === id));
      }}

      function applyFilters() {{
        const q = (search.value || "").toLowerCase();
        const caseValue = caseFilter.value || "";
        const groupValue = groupFilter.value || "";
        let firstVisible = null;
        cards.forEach(card => {{
          const matchesSearch = !q || (card.dataset.search || "").toLowerCase().includes(q);
          const matchesCase = !caseValue || card.dataset.case === caseValue;
          const matchesGroup = !groupValue || card.dataset.group === groupValue;
          const matchesSelected = !selectedOnly.checked || card.classList.contains("is-selected");
          const visible = matchesSearch && matchesCase && matchesGroup && matchesSelected;
          card.hidden = !visible;
          if (visible && !firstVisible) firstVisible = card;
        }});
        if (firstVisible && !cards.some(card => !card.hidden && card.classList.contains("is-active"))) {{
          activate(firstVisible.dataset.target);
        }}
      }}

      cards.forEach(card => card.addEventListener("click", () => activate(card.dataset.target)));
      [search, caseFilter, groupFilter, selectedOnly].forEach(el => el && el.addEventListener("input", applyFilters));
      applyFilters();
    }})();
  </script>
</body>
</html>
"""


def _render_html_browser_report(output: dict[str, Any]) -> str:
    dof_rendering = output.get("dof_rendering")
    external_images = isinstance(dof_rendering, dict) and dof_rendering.get("storage") == "files"
    inline_output = output if external_images else _with_inline_dof_svgs(output)
    cases_json = _html_script_json(inline_output)
    return (
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MolGR Trace</title>
  <style>
    :root { color-scheme: light; --bg:#f6f7f9; --panel:#fff; --line:#d8dee8; --text:#172033; --muted:#667085; --accent:#2563eb; --selected:#b45309; }
    * { box-sizing: border-box; }
    html { width:100%; overflow-x:auto; }
    body { margin:0; width:100%; min-width:1000px; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
    header { padding:14px 18px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:20; }
    .trace-header { display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .language-toggle { min-width:72px; border:1px solid var(--line); border-radius:6px; padding:7px 10px; background:#fff; color:var(--text); font:inherit; font-size:12px; cursor:pointer; }
    .language-toggle:hover { border-color:var(--accent); }
    h1 { margin:0; font-size:20px; }
    h2 { margin:0 0 10px; font-size:15px; }
    .global-info { display:grid; grid-template-columns:minmax(240px,.8fr) minmax(220px,.65fr) minmax(360px,1.55fr); gap:12px; padding:12px; }
    .global-card { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:12px; min-width:0; max-width:100%; overflow:auto; }
    main { width:100%; max-width:100%; display:grid; grid-template-columns:380px minmax(0,1fr); align-items:stretch; gap:12px; padding:0 12px 12px; min-height:0; }
    aside { border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:10px; align-self:start; position:relative; display:flex; flex-direction:column; gap:10px; min-width:0; min-height:0; overflow:hidden; }
    #tree-search { width:100%; border:1px solid var(--line); border-radius:6px; padding:8px; font:inherit; background:#fff; }
    .tree { flex:1 1 0; height:0; min-height:0; overflow:auto; overscroll-behavior:contain; scrollbar-gutter:stable; }
    .tree ol { list-style:none; margin:0; padding-left:14px; border-left:1px solid #e5e7eb; }
    .tree > ol { padding-left:0; border-left:0; }
    .tree li { margin:4px 0; }
    .tree-row { display:flex; align-items:flex-start; gap:4px; min-width:0; }
    .tree-toggle, .tree-toggle-spacer { flex:0 0 24px; width:24px; min-width:24px; height:24px; margin-top:2px; }
    .tree-toggle { border:1px solid transparent; border-radius:6px; background:transparent; padding:0; cursor:pointer; color:var(--muted); display:inline-flex; align-items:center; justify-content:center; }
    .tree-toggle::before { content:"▸"; font-size:12px; line-height:1; transition:transform .14s ease; }
    .tree-toggle:hover, .tree-toggle:focus-visible { border-color:var(--line); background:#f8fafc; outline:none; }
    .tree li.is-expanded > .tree-row > .tree-toggle::before { transform:rotate(90deg); }
    .tree-children { list-style:none; margin:4px 0 0 12px; padding-left:14px; border-left:1px solid #e5e7eb; }
    .tree li.is-collapsed > .tree-children { display:none; }
    .tree-node { flex:1 1 auto; width:auto; min-width:0; border:1px solid transparent; border-radius:6px; background:transparent; padding:7px 8px; text-align:left; cursor:pointer; color:var(--text); }
    .tree-node:hover, .tree-node.is-active { border-color:var(--accent); background:#eff6ff; }
    .tree-node.is-selected { border-color:var(--selected); background:#fff7ed; }
    .tree-label { display:block; font-weight:800; font-size:13px; overflow-wrap:anywhere; }
    .tree-meta { display:block; margin-top:2px; color:var(--muted); font-size:11px; }
    .content, .node-panel, .panel-info, .panel-info > *, .node-panel > *, .resonance-image-comparison, .resonance-image-group, .discordance-breakdown, .discordance-comparison { min-width:0; max-width:100%; }
    .content { align-self:start; min-height:0; overflow:auto; overscroll-behavior:contain; scrollbar-gutter:stable; }
    .node-panel { display:none; border:1px solid var(--line); border-radius:8px; background:var(--panel); padding:14px; min-width:0; }
    .node-panel.is-active { display:block; }
    .panel-head { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:flex-start; gap:10px; margin-bottom:12px; }
    .panel-head h2 { margin:0; font-size:18px; }
    .panel-head p { margin:4px 0 0; color:var(--muted); }
    .badge { border:1px solid var(--line); border-radius:999px; padding:4px 8px; font-size:12px; color:var(--muted); white-space:nowrap; }
    .badge.selected { color:#92400e; border-color:#f59e0b; background:#fffbeb; }
    .image-box { border:1px solid var(--line); border-radius:8px; background:#fff; padding:10px; overflow:auto; }
    .image-box.is-zoomable { cursor:zoom-in; }
    .image-box.is-zoomable:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
    .image-box img { display:block; max-width:100%; height:auto; margin:auto; }
    .image-box svg { display:block; max-width:100%; height:auto; margin:auto; }
    .dof-visual-row { display:grid; grid-template-columns:minmax(0,1fr) minmax(280px,1fr); gap:12px; align-items:stretch; }
    .dof-visual-row > .image-box, .dof-3dmol { min-width:0; min-height:280px; height:100%; }
    .dof-3dmol { border:1px solid var(--line); border-radius:8px; background:#fff; overflow:hidden; position:relative; }
    .dof-3dmol .dof-3dmol-viewer { width:100%; height:100%; min-height:280px; }
    .dof-3dmol-empty { display:grid; place-items:center; height:100%; min-height:280px; padding:20px; color:var(--muted); text-align:center; }
    .resonance-image-comparison { display:grid; gap:12px; }
    .resonance-image-group { min-width:0; }
    .resonance-image-group h3 { margin:0 0 6px; font-size:14px; }
    .related-resonance-title { margin:16px 0 8px; font-size:16px; }
    .image-lightbox { width:min(96vw,1500px); max-width:none; height:min(94vh,1100px); max-height:none; padding:0; border:0; border-radius:8px; background:transparent; overflow:hidden; }
    .image-lightbox::backdrop { background:rgb(18 27 33 / 78%); }
    .image-lightbox-shell { width:100%; height:100%; display:grid; grid-template-rows:auto minmax(0,1fr); border:1px solid #b9c4ca; border-radius:8px; background:#fff; overflow:hidden; }
    .image-lightbox-header { min-height:48px; padding:8px 10px 8px 14px; display:flex; align-items:center; justify-content:space-between; gap:12px; border-bottom:1px solid var(--line); }
    .image-lightbox-header h2 { margin:0; font-size:16px; }
    .image-lightbox-close { width:34px; height:34px; padding:0; border:1px solid var(--line); border-radius:6px; background:#fff; font-size:24px; line-height:1; cursor:pointer; }
    .image-lightbox-content { min-width:0; min-height:0; padding:16px; display:grid; place-items:center; overflow:auto; background:#f5f7f8; }
    .image-lightbox-content svg { display:block; width:clamp(900px,90vw,1400px); max-width:none; height:auto; background:#fff; }
    .image-lightbox-content img { display:block; max-width:none; height:auto; }
    .image-empty { border:1px dashed var(--line); border-radius:8px; padding:28px; text-align:center; color:var(--muted); background:#fff; }
    .panel-info { margin-top:12px; display:grid; gap:10px; }
    details { width:100%; min-width:0; max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; margin:0; overflow:hidden; }
    details > summary { cursor:pointer; padding:9px 11px; font-weight:800; background:#f8fafc; border-bottom:1px solid var(--line); }
    details:not([open]) > summary { border-bottom:0; }
    table { width:calc(100% - 20px); max-width:calc(100% - 20px); table-layout:fixed; border-collapse:collapse; font-size:12px; margin:10px; }
    th, td { border-bottom:1px solid #e5e7eb; padding:6px 7px; text-align:left; vertical-align:top; overflow-wrap:anywhere; }
    th { color:#334155; background:#f8fafc; }
    .wide-table { width:100%; min-width:0; max-width:100%; overflow-x:auto; overflow-y:hidden; overscroll-behavior-inline:contain; }
    .wide-table table { width:max-content; min-width:calc(100% - 20px); max-width:none; table-layout:auto; }
    .selection-result-selected td { background:#ecfdf3; font-weight:700; }
    .selection-result-error td { background:#fef2f2; color:#991b1b; }
    pre { width:calc(100% - 20px); max-width:calc(100% - 20px); margin:10px; padding:10px; border:1px solid var(--line); border-radius:8px; background:#0f172a; color:#e5e7eb; overflow:auto; max-height:520px; font-size:12px; }
    .empty { border:1px dashed var(--line); border-radius:8px; padding:24px; text-align:center; color:var(--muted); background:#fff; }
    @media (max-width:1400px) {
      .global-info { grid-template-columns:repeat(2,minmax(0,1fr)); }
      .global-info .global-card:last-child { grid-column:1 / -1; }
      main { grid-template-columns:minmax(280px,320px) minmax(0,1fr); }
    }
    @media (max-width:1100px) {
      header { position:static; }
      .global-info { grid-template-columns:1fr; }
      .global-info .global-card:last-child { grid-column:auto; }
      main { grid-template-columns:300px minmax(0,1fr); min-height:0; }
      .tree { overscroll-behavior:contain; }
      .node-panel { scroll-margin-top:12px; }
    }
    @media (max-width:720px) {
      header { padding:10px 12px; }
      .global-info { gap:8px; padding:8px; }
      .global-card { padding:9px; }
      main { grid-template-columns:300px minmax(0,1fr); gap:8px; padding:0 8px 8px; }
      aside { padding:8px; }
      .tree ol { padding-left:8px; }
      .tree-children { margin-left:7px; padding-left:8px; }
      .tree-toggle, .tree-toggle-spacer { flex-basis:22px; width:22px; min-width:22px; }
      .tree-node { padding:6px; }
      .node-panel { padding:10px; }
      .panel-head h2 { font-size:16px; }
      .panel-head .badge { white-space:normal; }
      .image-box { padding:6px; }
      .panel-info { margin-top:8px; gap:8px; }
      details > summary { padding:8px; }
      table { width:calc(100% - 12px); margin:6px; max-width:calc(100% - 12px); }
      th, td { padding:5px 6px; }
      .wide-table table { width:max-content; min-width:max-content; max-width:none; }
      pre { width:calc(100% - 12px); max-width:calc(100% - 12px); margin:6px; max-height:50vh; }
      .image-lightbox { width:calc(100vw - 12px); height:calc(100dvh - 12px); }
      .image-lightbox-content { padding:8px; place-items:start; }
      .image-lightbox-content svg { width:900px; }
      .image-lightbox-header { padding-left:10px; }
    }
  </style>
</head>
<body>
  <header class="trace-header">
    <h1>MolGR Trace</h1>
    <button id="language-toggle" class="language-toggle" type="button">English</button>
  </header>
  <section id="global-info" class="global-info"></section>
  <main>
    <aside>
      <input id="tree-search" type="search" placeholder="筛选 Trace 节点...">
      <nav id="tree" class="tree"></nav>
    </aside>
    <section id="detail" class="content"></section>
  </main>
  <dialog id="image-lightbox" class="image-lightbox" aria-labelledby="image-lightbox-title">
    <section class="image-lightbox-shell">
      <div class="image-lightbox-header">
        <h2 id="image-lightbox-title">DOF 图像</h2>
        <button id="image-lightbox-close" class="image-lightbox-close" type="button" aria-label="关闭放大图片" title="关闭">×</button>
      </div>
      <div id="image-lightbox-content" class="image-lightbox-content"></div>
    </section>
  </dialog>
  <script id="trace-json" type="application/json">"""
        + cases_json
        + r"""</script>
  <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
  <script>
    (() => {
      const trace = JSON.parse(document.getElementById("trace-json").textContent);
      const dofPayloads = trace.dof_payloads || {};
      const scorePrefixes = ["analysis_", "force_field_", "metal_", "organic_", "passes_", "selection_"];
      const scoreKeys = new Set(["combination_index", "score"]);
      const nodes = [];
      const nodeById = new Map();
      const roots = [];
      let columnHeightFrame = 0;
      let language = localStorage.getItem("moleculeReviewLanguage") === "en" ? "en" : "zh";

      const traceUi = {
        zh: {
          languageToggle: "English",
          searchPlaceholder: "筛选 Trace 节点...",
          dofImage: "DOF 图像",
          close: "关闭",
          closeZoomedImage: "关闭放大图片",
          expandNode: "折叠或展开节点",
          noDofSdf: "该 DOF 图没有可用的 SDF 坐标",
          threeDmolUnavailable: "3Dmol.js 未加载",
          renderOnOpen: "打开节点后动态渲染",
          noRenderableDof: "此节点没有可渲染的 DOF 图像",
          rendering: "渲染中...",
          renderFailed: "DOF 渲染失败",
          noCandidateSdf: "候选没有可用的原始 SDF",
          selected: "selected",
          yes: "是",
          no: "否",
        },
        en: {
          languageToggle: "中文",
          searchPlaceholder: "Filter trace nodes...",
          dofImage: "DOF image",
          close: "Close",
          closeZoomedImage: "Close enlarged image",
          expandNode: "Collapse or expand node",
          noDofSdf: "No SDF coordinates are available for this DOF image",
          threeDmolUnavailable: "3Dmol.js is not loaded",
          renderOnOpen: "Rendered dynamically when the node is opened",
          noRenderableDof: "This node has no renderable DOF image",
          rendering: "Rendering...",
          renderFailed: "DOF rendering failed",
          noCandidateSdf: "The candidate has no source SDF",
          selected: "selected",
          yes: "Yes",
          no: "No",
        },
      };

      const traceEnglishText = new Map(Object.entries({
        "输入": "Input",
        "来源": "Source",
        "XYZ 目录": "XYZ directory",
        "默认总电荷": "Default total charge",
        "默认总自由基电子数": "Default radical electrons",
        "自旋来源": "Spin source",
        "样本数": "Cases",
        "全局节点数": "Global nodes",
        "DOF 渲染": "DOF rendering",
        "存储": "Storage",
        "图片目录": "Image directory",
        "格式": "Format",
        "图片数": "Images",
        "跳过数": "Skipped",
        "最大图片数": "Maximum images",
        "错误数": "Errors",
        "状态": "Status",
        "类型": "Type",
        "电荷": "Charge",
        "自由基": "Radicals",
        "金属数": "Metals",
        "生产层": "Production layer",
        "生产候选": "Production candidates",
        "全部候选": "All candidates",
        "耗时秒": "Elapsed seconds",
        "字段": "Field",
        "值": "Value",
        "无": "None",
        "摘要": "Summary",
        "完整 JSON": "Full JSON",
        "分数构成": "Score components",
        "决策流程": "Decision flow",
        "阶段": "Stage",
        "规则": "Rule",
        "规则 / 结果": "Rule / result",
        "优先级": "Priority",
        "名称": "Name",
        "方向": "Direction",
        "含义": "Meaning",
        "候选": "Candidate",
        "候选数": "Candidates",
        "选中候选": "Selected candidate",
        "结论": "Decision",
        "首个决定字段": "First decisive field",
        "候选值": "Candidate value",
        "选中值": "Selected value",
        "选择字段顺序": "Selection field order",
        "候选选择结论": "Candidate selection decisions",
        "共振候选逐指标横向对比": "Resonance candidate comparison by criterion",
        "共振候选": "Resonance candidate",
        "共振候选对比": "Resonance candidate comparison",
        "共振候选多图": "Resonance candidate grid",
        "该金属价态对应有机目标的共振候选": "Resonance candidates for this metal-state organic target",
        "金属候选失谐分解对比": "Metal candidate discordance comparison",
        "金属候选对比中的失谐分解": "Discordance breakdown in metal candidate comparison",
        "候选选择中的失谐分解": "Discordance breakdown in candidate selection",
        "硬失谐分解": "Hard discordance breakdown",
        "失谐诊断子计数与例外": "Discordance diagnostic subcounts and exceptions",
        "失谐并列后的电子态排序": "Electronic-state ordering after discordance ties",
        "失谐例外与诊断": "Discordance exceptions and diagnostics",
        "分项": "Component",
        "数值": "Value",
        "作用": "Role",
        "诊断项": "Diagnostic",
        "说明": "Description",
        "排序项": "Ordering field",
        "失谐分项": "Discordance component",
        "硬失谐分": "Hard discordance score",
        "硬失谐分项之和": "Sum of hard discordance components",
        "最终硬失谐总分": "Final hard discordance score",
        "核对": "Check",
        "后续排序，不计入失谐总分": "Follow-up ordering; excluded from discordance total",
        "诊断：用于计算 0.5 倍惩罚": "Diagnostic: used for the 0.5x penalty",
        "诊断：清零负价金属惩罚": "Diagnostic: clears the negative-metal penalty",
        "内圈可见双自由基": "Inner visible diradical",
        "外圈/不可见邻位双电荷": "Outer/invisible adjacent double charges",
        "内圈可见相邻带电碳对": "Inner visible adjacent charged-carbon pair",
        "内圈可见共轭带电碳对": "Inner visible conjugated charged-carbon pair",
        "内圈可见同号电荷": "Inner visible same-sign charges",
        "负价金属惩罚": "Negative-metal penalty",
        "全零价金属与有机阳离子": "All-zero-valent metals with organic cation",
        "金属配合物中的欠饱和有机阳离子": "Unsaturated organic cation in metal complex",
        "重复片段净电荷失谐": "Repeated-component net-charge discordance",
        "可见多齿碳环还原断裂pi": "Visible haptic arene reduction breaks pi bonding",
        "配位几何失谐": "Coordination-geometry discordance",
        "负价金属绝对价态总数": "Total absolute valence of negative metals",
        "外圈质子例外": "Outer-sphere proton exception",
        "正价金属对离子例外": "Positive-metal counterion exception",
        "相对金属同号邻位双电荷": "Adjacent same-sign double charges relative to metal",
        "相对金属异号邻位双电荷": "Adjacent opposite-sign double charges relative to metal",
        "金属符号未知邻位双电荷": "Adjacent double charges with unknown metal sign",
        "共轭原子亏损": "Conjugated-atom deficit",
        "共轭键亏损": "Conjugated-bond deficit",
        "芳香原子亏损": "Aromatic-atom deficit",
        "芳香环亏损": "Aromatic-ring deficit",
        "芳香稳定性亏损": "Aromatic-stability deficit",
        "超共轭分数": "Hyperconjugation score",
        "超共轭分数亏损": "Hyperconjugation-score deficit",
        "低优先级后续排序的原始分数": "Raw score for lower-priority follow-up ordering",
        "电荷局域化 margin 之后、力场之前比较": "Compared after charge-localization margin and before force field",
        "最终选中路径动画": "Final selected-path animation",
        "无金属最终路径动画": "No-metal final-path animation",
        "含金属最终路径动画": "Metal-containing final-path animation",
        "帧数": "Frames",
        "动画": "Animation",
        "路径": "Path",
        "目标电荷": "Target charge",
        "目标自由基电子": "Target radical electrons",
        "选中分数": "Selected score",
        "选中选择键": "Selected selection key",
        "全局索引": "Global index",
        "定位符": "Locator",
        "全局树父节点": "Global tree parent",
        "父节点": "Parent node",
        "事件": "Event",
        "命中": "Hit",
        "共振种子": "Resonance seed",
        "raw 共振": "Raw resonance",
        "规范化": "Normalization",
        "无金属完整 trace": "Full no-metal trace",
        "节点数": "Nodes",
        "trace 节点": "Trace node",
        "选中 trace 节点": "Selected trace node",
        "种子": "Seed",
        "种子序号": "Seed index",
        "共振序号": "Resonance index",
        "raw 共振序号": "Raw resonance index",
        "raw 序号": "Raw index",
        "力场分": "Force-field score",
        "图排序键摘要": "Graph-order key summary",
        "最终共振身份": "Final resonance identity",
        "无金属共振元数据": "No-metal resonance metadata",
        "归一化方式": "Normalization method",
        "选中 no-metal 候选": "Selected no-metal candidate",
        "无金属候选": "No-metal candidate",
        "分数": "Score",
        "选择键": "Selection key",
        "无金属重建": "No-metal reconstruction",
        "选中无金属重建": "Selected no-metal reconstruction",
        "对应无金属重建": "Corresponding no-metal reconstruction",
        "邻位自由基动作": "Neighbor-radical actions",
        "恢复层": "Recovery tier",
        "选中 canonical SMILES": "Selected canonical SMILES",
        "形式电荷": "Formal charge",
        "自由基和": "Radical sum",
        "自由基奇偶和": "Radical parity sum",
        "活性孤对电子数": "Active lone-pair electrons",
        "剩余电荷预算": "Remaining charge budget",
        "匹配目标": "Matches target",
        "带电原子": "Charged atoms",
        "自由基原子": "Radical atoms",
        "活性孤对原子": "Active lone-pair atoms",
        "未决二电子中心": "Unresolved two-electron centers",
        "CSV 行号": "CSV row",
        "重建类型": "Reconstruction type",
        "总电荷": "Total charge",
        "总自由基电子数": "Total radical electrons",
        "自旋多重度": "Spin multiplicity",
        "XYZ 路径": "XYZ path",
        "XYZ 来源": "XYZ source",
        "review fixture 类型": "Review fixture type",
        "review fixture 等价": "Review fixture equivalent",
        "review fixture 比较原因": "Review fixture comparison reason",
        "参考 SMILES": "Reference SMILES",
        "金属原子数": "Metal atoms",
        "生产选择层": "Production selection layer",
        "生产候选数": "Production candidates",
        "全部已评分候选数": "All scored candidates",
        "review fixture 同步检查": "Review fixture synchronization check",
        "fixture 类型": "Fixture type",
        "结构文件": "Structure file",
        "等价": "Equivalent",
        "方法": "Method",
        "原因": "Reason",
        "氧化加成前后体检查": "Oxidative-addition pre/post check",
        "匹配": "Matched",
        "建议结论": "Recommended decision",
        "参考等价的 +/-2 价态候选数": "Reference-equivalent +/-2 valence candidates",
        "基础状态": "Base state",
        "含金属重建": "Metal-containing reconstruction",
        "金属价态候选": "Metal-valence candidates",
        "金属电子态候选": "Metal electronic-state candidate",
        "金属搜索层": "Metal search layer",
        "有机目标桶": "Organic target bucket",
        "分析和评分上下文": "Analysis and scoring context",
        "金属位点数": "Metal sites",
        "选择算法": "Selection algorithm",
        "最小金属失谐总数": "Minimum metal discordance count",
        "选中组合": "Selected combination",
        "选中金属状态": "Selected metal states",
        "选中字典序键": "Selected lexicographic key",
        "选中": "Selected",
        "组合": "Combination",
        "候选总电荷": "Candidate total charge",
        "有机目标电荷": "Organic target charge",
        "有机自由基电子": "Organic radical electrons",
        "有机 canonical SMILES": "Organic canonical SMILES",
        "有机力场分": "Organic force-field score",
        "金属状态": "Metal states",
        "金属失谐总数": "Metal discordance count",
        "通过失谐过滤": "Passes discordance filter",
        "最终选择键": "Final selection key",
        "对应共振候选数": "Corresponding resonance candidates",
        "选中共振候选": "Selected resonance candidate",
        "层": "Layer",
        "已评分数": "Scored",
        "金属组数": "Metal groups",
        "每组候选数": "Candidates per group",
        "目标桶数": "Target buckets",
        "枚举候选数": "Enumerated candidates",
        "已评分候选数": "Scored candidates",
        "金属价态先验与 assignment penalty": "Metal-valence priors and assignment penalty",
        "位点": "Site",
        "选项": "Option",
        "金属": "Metal",
        "价态": "Valence",
        "先验类别": "Prior class",
        "价态先验惩罚": "Valence-prior penalty",
        "非正价态惩罚": "Nonpositive-valence penalty",
        "总 assignment penalty": "Total assignment penalty",
        "构型": "Geometry",
        "配位数": "Coordination number",
        "场强分数": "Field score",
        "场强类别": "Field class",
        "弱场阈值": "Weak-field threshold",
        "强场阈值": "Strong-field threshold",
        "模糊 margin": "Ambiguity margin",
        "自旋选项（首选在前）": "Spin options (preferred first)",
        "donor 依据": "Donor evidence",
        "最终 selection_key 字段顺序": "Final selection_key field order",
        "失谐": "Discordance",
        "通过过滤": "Passes filter",
        "生产层候选逐项对比": "Production-layer candidate comparison",
        "1. 候选范围": "1. Candidate scope",
        "2. 化学指标": "2. Chemical criteria",
        "3. 图排序": "3. Graph ordering",
        "1. 价态先验": "1. Valence priors",
        "2. 生产层": "2. Production layer",
        "3. 失谐过滤": "3. Discordance filter",
        "4. 最终比较": "4. Final comparison",
        "只比较通过 process_resonance、去重、全局电荷/自由基验证和力场评分的候选": "Compare only candidates that pass process_resonance, deduplication, global charge/radical validation, and force-field scoring",
        "按下表九项指标依次做字典序比较；芳香与共轭优先，弱超共轭靠后，力场分最后": "Compare the nine criteria below lexicographically; aromaticity and conjugation come first, weak hyperconjugation later, and force-field score last",
        "九项指标完全相同时，按原子、显式电子标签和键表做稳定排序": "When all nine criteria tie, use stable ordering by atoms, explicit electron labels, and bond table",
        "按 prior/minor/other 和非正价态惩罚计算 assignment rank；用于搜索顺序、分层和每个 target 的剪枝": "Compute assignment rank from prior/minor/other and nonpositive-valence penalties for search order, layering, and per-target pruning",
        "对通过过滤的候选按下表字段从上到下做字典序最小化": "Lexicographically minimize passing candidates using the fields below in order",
        "assignment rank 是否进入最终键": "Assignment rank participates in final key",
        "电荷局域化超出 margin": "Charge localization exceeds margin",
        "自由基局域化惩罚": "Radical-localization penalty",
        "候选总分": "Candidate total score",
        "组合序号": "Combination index",
        "第一阶段只保留生产层中该值最小的候选": "First retain only production-layer candidates with the minimum value",
        "相对同一最低金属失谐层的最低局域化分数高出至少 selection margin 时为 1": "1 when localization exceeds the minimum score in the same lowest-discordance layer by at least the selection margin",
        "相对同批候选最佳共轭原子数的亏损": "Deficit relative to the best conjugated-atom count in the batch",
        "相对同批候选最佳共轭键数的亏损": "Deficit relative to the best conjugated-bond count in the batch",
        "相对同批候选最佳芳香原子数的亏损": "Deficit relative to the best aromatic-atom count in the batch",
        "相对同批候选最佳芳香环数的亏损": "Deficit relative to the best aromatic-ring count in the batch",
        "相对同批候选最佳芳香稳定性分数的亏损": "Deficit relative to the best aromatic-stability score in the batch",
        "有机部分自由基位于不利原子环境的惩罚": "Penalty for organic radicals in unfavorable atomic environments",
        "前述指标相同时，相对同批候选最高超共轭分数的亏损": "When prior criteria tie, deficit relative to the highest hyperconjugation score in the batch",
        "当前为组合结构的力场分数": "Current force-field score of the combined structure",
        "此前各项完全相同时的稳定排序": "Stable ordering when all previous fields tie",
        "形式电荷绝对值和": "Absolute formal-charge sum",
        "芳香原子数": "Aromatic atoms",
        "芳香环数": "Aromatic rings",
        "芳香稳定性分数": "Aromatic-stability score",
        "校正后最大共轭组分": "Adjusted largest conjugated component",
        "校正后共轭原子数": "Adjusted conjugated atoms",
        "校正后共轭键数": "Adjusted conjugated bonds",
        "超额自由基标记": "Excess radical labels",
        "有机力场分数": "Organic force-field score",
        "优先保留电荷分离程度更低的共振候选": "Prefer resonance candidates with less charge separation",
        "优先保留更多芳香原子的共振候选": "Prefer resonance candidates with more aromatic atoms",
        "芳香原子数相同时优先保留更多芳香环": "When aromatic-atom counts tie, prefer more aromatic rings",
        "按环大小和芳香覆盖计算的稳定性指标": "Stability metric based on ring size and aromatic coverage",
        "最大共轭组分减去形式电荷绝对值和的一半": "Largest conjugated component minus half the absolute formal-charge sum",
        "共轭原子数减去形式电荷绝对值和的一半": "Conjugated atoms minus half the absolute formal-charge sum",
        "共轭键数减去形式电荷绝对值和的一半": "Conjugated bonds minus half the absolute formal-charge sum",
        "显式自由基标记超过全局自由基目标的数量": "Explicit radical labels exceeding the global radical target",
        "饱和中性 sp3 碳向相邻 π/缺电子中心供给的 C-H σ 键数": "C-H sigma bonds donated from saturated neutral sp3 carbon to adjacent pi or electron-deficient centers",
        "前述电子拓扑指标完全相同时才比较力场能量": "Compare force-field energy only when preceding electronic-topology criteria tie",
        "选中：字典序键最小": "Selected: minimum lexicographic key",
        "淘汰：金属失谐总数不是最小值": "Rejected: metal discordance count is not minimal",
        "淘汰：首个不同的字典序字段更大": "Rejected: first differing lexicographic field is larger",
        "仅分析：不在第一个成功评分层": "Analysis only: not in the first successfully scored layer",
        "异常：通过过滤但没有 selection_key": "Error: passed filter without selection_key",
        "未选中：selection_key 完全相同": "Not selected: identical selection_key",
        "异常：该键优于被选候选": "Error: key is better than selected candidate",
        "选中：选择键字典序最小": "Selected: minimum selection key",
        "淘汰：首个不同的化学指标更差": "Rejected: first differing chemical criterion is worse",
        "淘汰：化学指标相同，显式电子态图排序靠后": "Rejected: chemical criteria tie and explicit-electron graph orders later",
        "未选中：选择键和图排序键完全相同": "Not selected: identical selection and graph-order keys"
      }));

      function ui(key) {
        return (traceUi[language] || traceUi.zh)[key] || key;
      }

      function localizeText(value) {
        const text = String(value ?? "");
        if (language !== "en") return text;
        if (traceEnglishText.has(text)) return traceEnglishText.get(text);
        let match = text.match(/^完整状态机分支树 \((\d+)\)$/);
        if (match) return `Full state-machine branch tree (${match[1]})`;
        match = text.match(/^共振候选对比 \((\d+)\)$/);
        if (match) return `Resonance candidate comparison (${match[1]})`;
        match = text.match(/^共振候选多图(?: (\d+)\/(\d+))?$/);
        if (match) return match[1]
          ? `Resonance candidate grid ${match[1]}/${match[2]}`
          : "Resonance candidate grid";
        match = text.match(/^状态机 (.+)$/);
        if (match) return `State machine ${localizeText(match[1])}`;
        match = text.match(/^只从第一个存在可评分候选的 Layer (.+) 中做最终选择$/);
        if (match) return `Make the final selection only from the first layer with scorable candidates: Layer ${match[1]}`;
        match = text.match(/^只保留 metal_discordance_count = (.+) 的候选$/);
        if (match) return `Keep only candidates with metal_discordance_count = ${match[1]}`;
        match = text.match(/^(.+) \(([^()]*)\)$/);
        if (match && traceEnglishText.has(match[1])) {
          return `${traceEnglishText.get(match[1])} (${match[2]})`;
        }
        return text;
      }

      function applyStaticLanguage() {
        document.documentElement.lang = language === "en" ? "en" : "zh-CN";
        document.getElementById("language-toggle").textContent = ui("languageToggle");
        document.getElementById("tree-search").placeholder = ui("searchPlaceholder");
        const close = document.getElementById("image-lightbox-close");
        close.setAttribute("aria-label", ui("closeZoomedImage"));
        close.title = ui("close");
        document.getElementById("image-lightbox-title").textContent = ui("dofImage");
      }

      function syncMainColumnHeights() {
        const main = document.querySelector("main");
        const sidebar = main && main.querySelector("aside");
        const content = document.getElementById("detail");
        if (!main || !sidebar || !content) return;

        const viewportHeight = window.visualViewport?.height || window.innerHeight;
        const header = document.querySelector("header");
        const stickyHeaderHeight = header && getComputedStyle(header).position === "sticky"
          ? header.getBoundingClientRect().height
          : 0;
        const availableHeight = Math.max(320, viewportHeight - stickyHeaderHeight - 24);
        const height = `${Math.floor(availableHeight)}px`;
        main.style.height = height;
        sidebar.style.height = height;
        content.style.height = height;
      }

      function scheduleMainColumnHeightSync() {
        if (columnHeightFrame) cancelAnimationFrame(columnHeightFrame);
        columnHeightFrame = requestAnimationFrame(() => {
          columnHeightFrame = 0;
          syncMainColumnHeights();
        });
      }

      function hasValue(value) {
        return value !== undefined && value !== null && value !== "" &&
          !(Array.isArray(value) && value.length === 0) &&
          !(typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0);
      }

      function fmt(value) {
        if (value === undefined || value === null) return "";
        if (typeof value === "boolean") return value ? ui("yes") : ui("no");
        if (typeof value === "number") return Number.isFinite(value) ? String(Number.parseFloat(value.toPrecision(6))) : String(value);
        if (typeof value === "string") return localizeText(value);
        return JSON.stringify(value);
      }

      function imagePath(image) {
        if (!image || typeof image !== "object" || image.status) return "";
        return image.display_path || image.path || "";
      }

      function imageSvg(image) {
        if (!image || typeof image !== "object" || image.status) return "";
        return image.svg_fragment || "";
      }

      function deferredDofPayload(image) {
        if (!image || typeof image !== "object" || !image.render_id) return null;
        return dofPayloads[image.render_id] || null;
      }

      function dofSdfs(image) {
        if (!image || typeof image !== "object") return [];
        const payload = deferredDofPayload(image) || image;
        if (typeof payload.sdf === "string" && payload.sdf) return [payload.sdf];
        return Array.isArray(payload.sdfs) ? payload.sdfs.filter(sdf => typeof sdf === "string" && sdf) : [];
      }

      function syncDofVisualHeight(row) {
        if (!row) return;
        const graph = row.querySelector(".image-box");
        const viewer = row.querySelector(".dof-3dmol");
        if (!graph || !viewer) return;
        const height = Math.max(280, Math.ceil(graph.getBoundingClientRect().height));
        viewer.style.height = `${height}px`;
        const canvas = viewer.querySelector(".dof-3dmol-viewer");
        if (canvas) canvas.style.height = `${height}px`;
        const instance = viewer._molgrViewer;
        if (instance && typeof instance.resize === "function") {
          instance.resize();
          instance.render();
        }
      }

      function renderDofMol3d(box, image) {
        const sdfs = dofSdfs(image);
        if (!sdfs.length) {
          box.classList.add("dof-3dmol-empty");
          box.textContent = ui("noDofSdf");
          return;
        }
        if (!window.$3Dmol) {
          box.classList.add("dof-3dmol-empty");
          box.textContent = ui("threeDmolUnavailable");
          return;
        }
        const canvas = document.createElement("div");
        canvas.className = "dof-3dmol-viewer";
        box.appendChild(canvas);
        const viewer = window.$3Dmol.createViewer(canvas, {backgroundColor: "white"});
        sdfs.forEach(sdf => {
          const model = viewer.addModel(sdf, "sdf");
          model.setStyle({}, {stick: {radius: 0.14}, sphere: {scale: 0.25}});
        });
        viewer.zoomTo();
        viewer.render();
        box._molgrViewer = viewer;
      }

      function setImageZoomState(box, label) {
        const image = box && box.querySelector("svg, img");
        if (!box) return;
        box.classList.toggle("is-zoomable", Boolean(image));
        if (!image) {
          box.removeAttribute("role");
          box.removeAttribute("tabindex");
          box.removeAttribute("aria-label");
          delete box.dataset.zoomLabel;
          return;
        }
        box.setAttribute("role", "button");
        box.tabIndex = 0;
        const localizedLabel = localizeText(label || ui("dofImage"));
        box.setAttribute("aria-label", language === "en" ? `Enlarge ${localizedLabel}` : `放大${localizedLabel}`);
        box.dataset.zoomLabel = localizedLabel;
      }

      function openImageLightbox(box, label) {
        const image = box && box.querySelector("svg, img");
        if (!image) return;
        const dialog = document.getElementById("image-lightbox");
        document.getElementById("image-lightbox-title").textContent = localizeText(label || box.dataset.zoomLabel || ui("dofImage"));
        document.getElementById("image-lightbox-content").replaceChildren(image.cloneNode(true));
        if (!dialog.open) dialog.showModal();
      }

      function closeImageLightbox() {
        const dialog = document.getElementById("image-lightbox");
        if (dialog.open) dialog.close();
      }

      function stateImage(value) {
        return value && value.state && value.state.dof_image ? value.state.dof_image : null;
      }

      function scoreDetails(metadata) {
        if (!metadata || typeof metadata !== "object") return {};
        if (metadata.score_details && typeof metadata.score_details === "object") return metadata.score_details;
        const source = metadata.metadata && typeof metadata.metadata === "object" ? metadata.metadata : metadata;
        const result = {};
        Object.entries(source).forEach(([key, value]) => {
          if (scoreKeys.has(key) || scorePrefixes.some(prefix => key.startsWith(prefix))) result[key] = value;
        });
        return result;
      }

      const discordanceComponents = [
        ["metal_discordance_inner_visible_diradical_count", "内圈可见双自由基", "硬失谐分"],
        ["metal_discordance_outer_or_invisible_adjacent_double_charge_count", "外圈/不可见邻位双电荷", "硬失谐分"],
        ["metal_discordance_inner_visible_adjacent_carbanion_pair_count", "内圈可见相邻带电碳对", "硬失谐分"],
        ["metal_discordance_inner_visible_conjugated_carbanion_pair_count", "内圈可见共轭带电碳对", "硬失谐分"],
        ["metal_discordance_inner_visible_same_sign_charge_count", "内圈可见同号电荷", "硬失谐分"],
        ["metal_discordance_negative_metal_penalty", "负价金属惩罚", "硬失谐分"],
        ["metal_discordance_zero_valent_metals_with_organic_cation_count", "全零价金属与有机阳离子", "硬失谐分"],
        ["metal_discordance_unsaturated_organic_cation_count", "金属配合物中的欠饱和有机阳离子", "硬失谐分"],
        ["metal_discordance_repeated_component_charge_asymmetry_count", "重复片段净电荷失谐", "硬失谐分"],
        ["metal_discordance_haptic_arene_reduction_count", "可见多齿碳环还原断裂pi", "硬失谐分"],
        ["metal_discordance_coordination_geometry_count", "配位几何失谐", "硬失谐分"],
      ];
      const discordanceDiagnosticFields = [
        ["metal_discordance_negative_metal_count", "负价金属绝对价态总数", "诊断：用于计算 0.5 倍惩罚"],
        ["metal_discordance_negative_metal_outer_sphere_cation_exception", "外圈质子例外", "诊断：清零负价金属惩罚"],
        ["metal_discordance_negative_metal_positive_metal_counterion_exception", "正价金属对离子例外", "诊断：清零负价金属惩罚"],
        ["metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count", "相对金属同号邻位双电荷", "诊断子计数"],
        ["metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count", "相对金属异号邻位双电荷", "诊断子计数"],
        ["metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count", "金属符号未知邻位双电荷", "诊断子计数"],
      ];
      const discordanceFollowupFields = [
        ["metal_discordance_conjugated_atom_deficit_count", "共轭原子亏损", "后续排序，不计入失谐总分"],
        ["metal_discordance_conjugated_bond_deficit_count", "共轭键亏损", "后续排序，不计入失谐总分"],
        ["metal_discordance_aromatic_atom_deficit_count", "芳香原子亏损", "后续排序，不计入失谐总分"],
        ["metal_discordance_aromatic_ring_deficit_count", "芳香环亏损", "后续排序，不计入失谐总分"],
        ["metal_discordance_aromatic_stability_deficit", "芳香稳定性亏损", "后续排序，不计入失谐总分"],
        ["organic_hyperconjugation_score", "超共轭分数", "低优先级后续排序的原始分数"],
        ["organic_hyperconjugation_deficit", "超共轭分数亏损", "电荷局域化 margin 之后、力场之前比较"],
      ];

      function numericValue(details, key) {
        const value = details && details[key];
        return typeof value === "number" && Number.isFinite(value) ? value : 0;
      }

      function discordanceBreakdown(details) {
        if (!details || typeof details !== "object" || !("metal_discordance_count" in details)) return null;
        const rows = discordanceComponents.map(([key, label, role]) => [label, numericValue(details, key), role, key]);
        const componentSum = discordanceComponents.reduce((sum, [key]) => sum + numericValue(details, key), 0);
        rows.push(["硬失谐分项之和", componentSum, "核对", "component_sum"]);
        rows.push(["metal_discordance_count", details.metal_discordance_count, "最终硬失谐总分", "metal_discordance_count"]);
        const diagnosticRows = discordanceDiagnosticFields
          .filter(([key]) => key in details)
          .map(([key, label, role]) => [label, details[key], role, key]);
        const followupRows = discordanceFollowupFields
          .filter(([key]) => key in details)
          .map(([key, label, role]) => [label, details[key], role, key]);
        return {
          rows,
          diagnosticRows,
          followupRows,
          componentSum,
          reported: details.metal_discordance_count,
        };
      }

      function renderDiscordanceBreakdown(scoreDetailData) {
        const breakdown = discordanceBreakdown(scoreDetailData);
        if (!breakdown) return null;
        const root = document.createElement("section");
        root.className = "discordance-breakdown";
        root.appendChild(details("硬失谐分解", table(breakdown.rows.map(([label, value, role]) => [label, value, role]), ["分项", "数值", "作用"]), true));
        if (breakdown.diagnosticRows.length) {
          root.appendChild(details("失谐诊断子计数与例外", table(breakdown.diagnosticRows.map(([label, value, role]) => [label, value, role]), ["诊断项", "数值", "说明"]), true));
        }
        if (breakdown.followupRows.length) {
          root.appendChild(details("失谐并列后的电子态排序", table(breakdown.followupRows.map(([label, value, role]) => [label, value, role]), ["排序项", "数值", "说明"]), true));
        }
        return root;
      }

      function renderDiscordanceComparison(entries, title = "金属候选失谐分解对比") {
        if (!Array.isArray(entries) || !entries.length) return null;
        const columns = entries.map(entry => {
          const states = entry.metal_states || [];
          const stateLabel = metalStatesLabel(states);
          return `${entry.selected ? "selected " : ""}C${entry.combination_index ?? ""} ${stateLabel}`;
        });
        const rows = discordanceComponents.map(([key, label]) => [
          label,
          ...entries.map(entry => numericValue(entry.discordance_details || entry.score_details || {}, key)),
        ]);
        rows.push(["硬失谐分项之和", ...entries.map(entry => {
          const details = entry.discordance_details || entry.score_details || {};
          return discordanceComponents.reduce((sum, [key]) => sum + numericValue(details, key), 0);
        })]);
        rows.push(["metal_discordance_count", ...entries.map(entry => numericValue(entry.discordance_details || entry.score_details || {}, "metal_discordance_count"))]);
        const root = document.createElement("section");
        root.className = "discordance-comparison";
        root.appendChild(details(title, wideTable(rows, ["失谐分项", ...columns]), true));
        const diagnostics = entries.map(entry => {
          const details = entry.discordance_details || entry.score_details || {};
          return [
            entry.metal_states ? metalStatesLabel(entry.metal_states) : `C${entry.combination_index ?? ""}`,
            details.metal_discordance_negative_metal_count,
            details.metal_discordance_negative_metal_penalty,
            details.metal_discordance_negative_metal_outer_sphere_cation_exception,
            details.metal_discordance_negative_metal_positive_metal_counterion_exception,
            details.metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count,
            details.metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count,
            details.metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count,
          ];
        });
        root.appendChild(details("失谐例外与诊断", wideTable(diagnostics, ["候选", "负价金属数", "负价惩罚", "外圈质子例外", "正价金属例外", "同号邻位双电荷", "异号邻位双电荷", "未知符号双电荷"]), false));
        return root;
      }

      function makeNode({
        label,
        kind,
        caseId,
        summary = [],
        metadata = {},
        image = null,
        resonanceBasis = null,
        selected = false,
        children = [],
      }) {
        const globalIndex = metadata && Number.isInteger(metadata.global_node_index)
          ? metadata.global_node_index : null;
        const globalLocator = metadata && metadata.global_node_locator
          ? metadata.global_node_locator : "";
        const node = {
          id: `node-${nodes.length}`,
          globalIndex,
          globalLocator,
          label,
          kind,
          caseId,
          summary,
          metadata,
          image,
          resonanceBasis,
          selected,
          children,
          collapsed: undefined,
          parent: null,
          treeLi: null,
          treeToggle: null,
          treeChildren: null,
          scoreDetails: scoreDetails(metadata),
        };
        node.search = [
          caseId,
          label,
          localizeText(label),
          kind,
          localizeText(kind),
          globalLocator,
          fmt(summary),
          fmt(node.scoreDetails),
          resonanceBasis ? "共振候选对比" : "",
        ].join(" ");
        nodes.push(node);
        nodeById.set(node.id, node);
        return node;
      }

      function stateSummary(state) {
        state = state || {};
        return [
          ["canonical SMILES", state.canonical_smiles || state.smiles],
          ["形式电荷", state.formal_charge_sum],
          ["自由基和", state.spin_multiplicity_sum],
          ["自由基奇偶和", state.spin_multiplicity_singlet_sum],
          ["活性孤对电子数", state.lone_pair_sum],
          ["剩余电荷预算", state.given_charge],
          ["匹配目标", state.valid_for_target],
          ["带电原子", state.charged_atom_counts],
          ["自由基原子", state.radical_atom_counts],
          ["活性孤对原子", state.lone_pair_atom_counts],
          ["未决二电子中心", state.unresolved_two_electron_center_counts],
        ];
      }

      function metalStatesLabel(states) {
        return (states || []).map(state => {
          const valence = Number(state.valence || 0);
          const sign = valence >= 0 ? `+${valence}` : String(valence);
          return `#${state.idx || ""} ${state.symbol || ""}(${sign}, r${state.radical_num || 0})`;
        }).join("; ");
      }

      function buildSelectedPathAnimation(caseId, traceData, selected, kind) {
        const animation = traceData && traceData.selected_path_animation;
        if (!animation) return null;
        return makeNode({
          label: "最终选中路径动画",
          kind,
          caseId,
          summary: [
            ["状态", animation.status || "rendered"],
            ["帧数", animation.frame_count],
            ["动画", animation.animation],
            ["路径", animation.display_path || animation.path],
            ["目标电荷", traceData.target && traceData.target.total_charge],
            ["目标自由基电子", traceData.target && traceData.target.total_radical_electrons],
            ["选中分数", selected && selected.score],
            ["选中选择键", selected && selected.organic_topology_selection_key],
          ],
          metadata: animation,
          image: animation,
          selected: true,
        });
      }

      function selectedNoMetalTrace(item) {
        const selected = item && item.selected_candidate;
        const selectedTarget = selected && selected.target;
        const selectedLayer = item && item.search && item.search.selected_layer_index;
        const layers = item && item.search && item.search.layer_summaries || [];
        if (!selectedTarget) return null;
        const selectedCharge = String(selectedTarget.no_metal_charge);
        const selectedRadicals = String(selectedTarget.no_metal_radical_electrons);
        const orderedLayers = [
          ...layers.filter(layer => String(layer.layer_index) === String(selectedLayer)),
          ...layers.filter(layer => String(layer.layer_index) !== String(selectedLayer)),
        ];
        for (const layer of orderedLayers) {
          for (const bucket of layer.target_buckets || []) {
            const target = bucket.target || {};
            if (
              String(target.no_metal_charge) === selectedCharge &&
              String(target.no_metal_radical_electrons) === selectedRadicals &&
              bucket.no_metal_trace
            ) {
              return bucket.no_metal_trace;
            }
          }
        }
        return null;
      }

      function buildNoMetal(caseId, label, traceData) {
        traceData = traceData || {};
        const target = traceData.target || {};
        const selected = traceData.selected_candidate || {};
        const selectedState = selected.state || {};
        const resonance = traceData.resonance || {};
        const resonanceBasis = traceData.resonance_selection_basis || {};
        const children = [];
        const selectedPathAnimation = buildSelectedPathAnimation(
          caseId,
          traceData,
          selected,
          "无金属最终路径动画",
        );
        if (selectedPathAnimation) children.push(selectedPathAnimation);

        const allTraceNodes = traceData.trace_nodes || [];
        const traceNodeMap = new Map();
        allTraceNodes.forEach(node => {
          const event = node.event || {};
          const metadata = node.metadata || {};
          const state = node.state || {};
          traceNodeMap.set(String(node.node_id), makeNode({
            label: `${node.global_node_locator || `#${node.global_node_index ?? node.node_id}`}. ${node.phase}`,
            kind: `状态机 ${node.kind || "stage"}`,
            caseId,
            selected: Boolean(node.selected_path),
            summary: [
              ["全局索引", node.global_node_index],
              ["定位符", node.global_node_locator],
              ["全局树父节点", node.global_tree_parent_index],
              ["父节点", node.parent_id],
              ["阶段", node.phase],
              ["事件", event.stage || event.kind || ""],
              ["命中", event.hit],
              ["共振种子", metadata.resonance_seed_index],
              ["raw 共振", metadata.resonance_raw_index],
              ["规范化", metadata.resonance_normalization],
              ["canonical SMILES", state.canonical_smiles || state.smiles],
            ],
            metadata: node,
            image: node.dof_image,
          }));
        });
        const traceRoots = [];
        allTraceNodes.forEach(node => {
          const current = traceNodeMap.get(String(node.node_id));
          const parent = traceNodeMap.get(String(node.tree_parent_id));
          if (parent && parent !== current) parent.children.push(current);
          else if (current) traceRoots.push(current);
        });
        if (traceRoots.length) {
          children.push(makeNode({
            label: `完整状态机分支树 (${allTraceNodes.length})`,
            kind: "无金属完整 trace",
            caseId,
            summary: [["节点数", allTraceNodes.length]],
            metadata: { trace_nodes: allTraceNodes },
            children: traceRoots,
          }));
        }

        if (Array.isArray(resonanceBasis.candidates) && resonanceBasis.candidates.length) {
          const comparisonChildren = [];
          resonanceBasis.candidates.forEach(candidate => {
            const candidateState = candidate.state || {};
            const values = candidate.selection_values || {};
            comparisonChildren.push(makeNode({
              label: resonanceCandidateLabel(candidate),
              kind: "共振候选",
              caseId,
              selected: Boolean(candidate.selected),
              summary: [
                ["trace 节点", candidate.global_node_locator || candidate.trace_node_id],
                ["种子序号", candidate.resonance_seed_index],
                ["共振序号", candidate.resonance_index],
                ["raw 共振序号", candidate.resonance_raw_index],
                ["结论", resonanceSelectionDecisionLabel(candidate.decision)],
                ["首个决定字段", candidate.decisive_field],
                ...Object.entries(values),
                ["图排序键摘要", candidate.graph_tie_break_hash],
                ...stateSummary(candidateState),
              ],
              metadata: candidate,
              image: candidate.dof_image,
            }));
          });
          children.push(makeNode({
            label: `共振候选对比 (${resonanceBasis.candidate_count})`,
            kind: "共振候选对比",
            caseId,
            selected: true,
            summary: [
              ["候选数", resonanceBasis.candidate_count],
              ["选中候选", resonanceBasis.selected_candidate_index],
              ["选中 trace 节点", resonanceBasis.selected_global_node_locator || resonanceBasis.selected_trace_node_id],
              ["选中选择键", resonanceBasis.selected_selection_key],
            ],
            metadata: resonanceBasis,
            children: comparisonChildren,
          }));
        }

        if (Object.keys(resonance).length) {
          children.push(makeNode({
            label: "最终共振身份",
            kind: "无金属共振元数据",
            caseId,
            summary: [
              ["种子序号", resonance.seed_index],
              ["共振序号", resonance.resonance_index],
              ["raw 共振序号", resonance.raw_index],
              ["归一化方式", resonance.normalization],
            ],
            metadata: resonance,
          }));
        }

        if (Object.keys(selected).length) {
          children.push(makeNode({
            label: "选中 no-metal 候选",
            kind: "无金属候选",
            caseId,
            summary: [
              ["分数", selected.score],
              ["选择键", selected.organic_topology_selection_key],
              ...stateSummary(selectedState),
            ],
            metadata: selected,
            image: selectedState.dof_image,
            selected: true,
          }));
        }

        return makeNode({
          label,
          kind: "无金属重建",
          caseId,
          summary: [
            ["状态", traceData.status],
            ["目标电荷", target.total_charge],
            ["目标自由基电子", target.total_radical_electrons],
            ["邻位自由基动作", traceData.neighbor_radical_actions],
            ["恢复层", traceData.recovery_tier],
            ["归一化方式", resonance.normalization],
            ["选中 canonical SMILES", selectedState.canonical_smiles],
            ["选中分数", selected.score],
            ["选中选择键", selected.organic_topology_selection_key],
          ],
          metadata: traceData,
          image: selectedState.dof_image,
          children,
        });
      }

      function buildTree() {
        (trace.cases || []).forEach(item => {
          const caseId = String(item.id || "unknown");
          const base = item.base_state || {};
          const selected = item.selected_candidate || {};
          const children = [];
          const caseNode = makeNode({
            label: caseId,
            kind: "case",
            caseId,
            summary: [
              ["id", item.id],
              ["CSV 行号", item.row_index],
              ["状态", item.status],
              ["重建类型", item.trace_kind],
              ["总电荷", item.charge],
              ["总自由基电子数", item.total_radical_electrons],
              ["自旋多重度", item.spin_multiplicity],
              ["自旋来源", item.spin_source],
              ["XYZ 路径", item.xyz_path],
              ["XYZ 来源", item.xyz_source],
              ["review fixture 类型", item.review_fixture && item.review_fixture.kind],
              ["review fixture 等价", item.review_fixture && item.review_fixture.equivalent],
              ["review fixture 比较原因", item.review_fixture && item.review_fixture.equivalence_reason],
              ["参考 SMILES", item.reference_smiles],
              ["金属原子数", base.metal_atom_count],
              ["生产选择层", item.search && item.search.selected_layer_index],
              ["生产候选数", item.production_candidate_count],
              ["全部已评分候选数", item.candidate_count],
              ["选中候选", selected.combination_index],
              ["耗时秒", item.elapsed_seconds],
            ],
            metadata: item,
            children,
          });
          roots.push(caseNode);
          if (item.review_fixture) {
            children.push(makeNode({
              label: "review fixture 同步检查",
              kind: "fixture equivalence",
              caseId,
              summary: [
                ["fixture 类型", item.review_fixture.kind],
                ["结构文件", item.review_fixture.structure_file],
                ["等价", item.review_fixture.equivalent],
                ["方法", item.review_fixture.equivalence_method],
                ["原因", item.review_fixture.equivalence_reason],
              ],
              metadata: item.review_fixture,
            }));
          }
          if (item.oxidative_addition_reference_match) {
            const match = item.oxidative_addition_reference_match;
            children.push(makeNode({
              label: "氧化加成前后体检查",
              kind: "氧化加成前后体检查",
              caseId,
              summary: [
                ["匹配", match.matched],
                ["建议结论", match.recommended_review_status],
                ["原因", match.reason],
                ["参考等价的 +/-2 价态候选数", (match.matching_candidates || []).length],
              ],
              metadata: match,
            }));
          }
          if (item.trace_kind === "no_metal") {
            children.push(buildNoMetal(caseId, "无金属重建", item.no_metal_trace || {}));
            return;
          }
          const candidates = (item.candidates || []).filter(candidate => candidate && typeof candidate === "object");
          const selectionBasis = item.metal_selection_basis || {};
          const selectedPathAnimation = buildSelectedPathAnimation(
            caseId,
            item,
            selected,
            "含金属最终路径动画",
          );
          if (selectedPathAnimation) children.push(selectedPathAnimation);
          const selectedOrganicTrace = selectedNoMetalTrace(item);
          if (selectedOrganicTrace) {
            children.push(buildNoMetal(caseId, "选中无金属重建", selectedOrganicTrace));
          }
          children.push(makeNode({
            label: "基础状态",
            kind: "含金属重建",
            caseId,
            summary: [
              ["金属原子数", base.metal_atom_count],
              ["金属位点数", (base.available_metal_states_by_site || []).length],
              ["phase_history", base.phase_history],
            ],
            metadata: base,
          }));
          if (Object.keys(selectionBasis).length || item.dof_candidate_grid || candidates.length) {
            const mergedSelectionBasis = Object.keys(selectionBasis).length ? {
              ...selectionBasis,
              available_metal_states_by_site: base.available_metal_states_by_site || [],
            } : null;
            children.push(makeNode({
              label: "金属价态候选",
              kind: "金属价态候选",
              caseId,
              summary: [
                ["选择算法", selectionBasis.algorithm],
                ["生产选择层", selectionBasis.selected_layer_index],
                ["候选数", item.candidate_count],
                ["生产候选数", item.production_candidate_count],
                ["最小金属失谐总数", selectionBasis.minimum_metal_discordance_count],
                ["选中组合", selectionBasis.selected_combination_index],
                ["选中金属状态", metalStatesLabel(selectionBasis.selected_metal_states)],
                ["选中字典序键", selectionBasis.selected_selection_key],
              ],
              metadata: {
                candidate_count: item.candidate_count,
                production_candidate_count: item.production_candidate_count,
                selected_candidate: item.selected_candidate,
                selection_basis: mergedSelectionBasis,
                discordance_candidates: candidates.filter(candidate => candidate.in_production_selection_layer),
              },
              image: item.dof_candidate_grid,
              selected: true,
            }));
          }
          ((item.search && item.search.layer_summaries) || []).forEach(layer => {
            const layerIndex = layer.layer_index;
            const layerChildren = [];
            (layer.target_buckets || []).forEach(bucket => {
              const target = bucket.target || {};
              const organic = bucket.organic_part || {};
              const bucketChildren = [];
              if (bucket.no_metal_trace) {
                bucketChildren.push(buildNoMetal(caseId, "对应无金属重建", bucket.no_metal_trace));
              }
              candidates
                .filter(candidate => (
                  String(candidate.search_layer_index) === String(layerIndex)
                  && String((candidate.target || {}).no_metal_charge) === String(target.no_metal_charge)
                  && String((candidate.target || {}).no_metal_radical_electrons) === String(target.no_metal_radical_electrons)
                ))
                .forEach(candidate => {
                  const candidateOrganic = candidate.organic_part || {};
                  const candidateTarget = candidate.target || {};
                  const candidateResonanceBasis = bucket.no_metal_trace?.resonance_selection_basis || null;
                  const title = `${candidate.global_node_locator || `#${candidate.global_node_index ?? ""}`} ${candidate.selected ? "selected " : ""}L${candidate.search_layer_index}/C${candidate.combination_index}`;
                  bucketChildren.push(makeNode({
                    label: title,
                    kind: "金属电子态候选",
                    caseId,
                    summary: [
                      ["全局索引", candidate.global_node_index],
                      ["定位符", candidate.global_node_locator],
                      ["选中", candidate.selected],
                      ["组合", candidate.combination_index],
                      ["生产层", candidate.in_production_selection_layer],
                      ["候选总电荷", candidate.candidate_total_charge],
                      ["有机目标电荷", candidateTarget.no_metal_charge],
                      ["有机自由基电子", candidateTarget.no_metal_radical_electrons],
                      ["有机 canonical SMILES", candidateOrganic.canonical_smiles || candidateOrganic.smiles],
                      ["分数", candidate.score],
                      ["金属状态", metalStatesLabel(candidate.metal_states)],
                      ["assignment rank", (candidate.score_details || {}).metal_assignment_rank],
                      ["金属失谐总数", (candidate.score_details || {}).metal_discordance_count],
                      ["通过失谐过滤", (candidate.score_details || {}).passes_metal_discordance_filter],
                      ["最终选择键", (candidate.score_details || {}).selection_key],
                      ["对应共振候选数", candidateResonanceBasis?.candidate_count],
                      ["选中共振候选", candidateResonanceBasis?.selected_candidate_index],
                    ],
                    metadata: candidate,
                    image: candidate.dof_image,
                    resonanceBasis: candidateResonanceBasis,
                    selected: Boolean(candidate.selected),
                  }));
                });
              layerChildren.push(makeNode({
                label: `Target Q=${target.no_metal_charge} R=${target.no_metal_radical_electrons}`,
                kind: "有机目标桶",
                caseId,
                summary: [
                  ["层", layerIndex],
                  ["有机目标电荷", target.no_metal_charge],
                  ["有机自由基电子", target.no_metal_radical_electrons],
                  ["状态", bucket.status],
                  ["候选数", bucket.candidate_count],
                  ["已评分数", bucket.prepared_candidate_count],
                  ["有机 canonical SMILES", organic.canonical_smiles || organic.smiles],
                  ["有机力场分", bucket.no_metal_score],
                ],
                metadata: bucket,
                image: bucket.dof_image,
                children: bucketChildren,
              }));
            });
            children.push(makeNode({
              label: `Layer ${layerIndex}`,
              kind: "金属搜索层",
              caseId,
              summary: [
                ["状态", layer.status],
                ["生产选择层", layer.production_selected_layer],
                ["金属组数", layer.state_group_count],
                ["每组候选数", layer.state_options_per_group],
                ["目标桶数", layer.target_bucket_count],
                ["枚举候选数", layer.candidate_count],
                ["已评分候选数", layer.prepared_candidate_count],
                ["analysis score context", Boolean(layer.analysis_score_context)],
              ],
              metadata: layer,
              children: layerChildren,
            }));
          });
          if (item.analysis) {
            children.push(makeNode({
              label: "分析和评分上下文",
              kind: "analysis",
              caseId,
              summary: [
                ["score_all_candidates", item.analysis.score_all_candidates],
                ["note", item.analysis.note],
              ],
              metadata: item.analysis,
            }));
          }
        });
      }

      function table(rows, headers = ["字段", "值"]) {
        const tableEl = document.createElement("table");
        const thead = tableEl.createTHead();
        const headRow = thead.insertRow();
        headers.forEach(header => {
          const th = document.createElement("th");
          th.textContent = localizeText(header);
          headRow.appendChild(th);
        });
        const tbody = tableEl.createTBody();
        rows.filter(([_, value]) => hasValue(value)).forEach(row => {
          const tr = tbody.insertRow();
          row.forEach(cell => {
            const td = tr.insertCell();
            td.textContent = fmt(cell);
          });
        });
        if (!tbody.rows.length) {
          const tr = tbody.insertRow();
          const td = tr.insertCell();
          td.colSpan = headers.length;
          td.textContent = localizeText("无");
        }
        return tableEl;
      }

      function wideTable(rows, headers) {
        const wrapper = document.createElement("div");
        wrapper.className = "wide-table";
        wrapper.appendChild(table(rows, headers));
        return wrapper;
      }

      function metalSelectionDecisionLabel(decision) {
        const labels = {
          selected_lexicographic_minimum: "选中：字典序键最小",
          rejected_by_discordance_filter: "淘汰：金属失谐总数不是最小值",
          rejected_by_lexicographic_key: "淘汰：首个不同的字典序字段更大",
          analysis_only_later_layer: "仅分析：不在第一个成功评分层",
          missing_selection_key: "异常：通过过滤但没有 selection_key",
          selection_key_tie_not_selected: "未选中：selection_key 完全相同",
          selection_inconsistency_candidate_beats_selected: "异常：该键优于被选候选",
        };
        return labels[decision] || decision;
      }

      function resonanceCandidateLabel(candidate) {
        const selected = candidate && candidate.selected ? "selected " : "";
        return `${selected}R${candidate?.resonance_raw_index ?? ""} / C${candidate?.candidate_index ?? ""}`;
      }

      function resonanceSelectionDecisionLabel(decision) {
        const labels = {
          selected_lexicographic_minimum: "选中：选择键字典序最小",
          rejected_by_selection_key: "淘汰：首个不同的化学指标更差",
          rejected_by_graph_tie_break: "淘汰：化学指标相同，显式电子态图排序靠后",
          selection_key_tie_not_selected: "未选中：选择键和图排序键完全相同",
        };
        return labels[decision] || decision;
      }

      function renderResonanceSelectionBasis(basis) {
        const root = document.createElement("section");
        const candidates = basis.candidates || [];
        const criteria = basis.selection_key_fields || [];
        const graphTieBreak = basis.graph_tie_break || {};
        root.appendChild(details("决策流程", table([
          ["1. 候选范围", "只比较通过 process_resonance、去重、全局电荷/自由基验证和力场评分的候选"],
          ["2. 化学指标", "按下表九项指标依次做字典序比较；芳香与共轭优先，弱超共轭靠后，力场分最后"],
          ["3. 图排序", "九项指标完全相同时，按原子、显式电子标签和键表做稳定排序"],
          ["候选数", basis.candidate_count],
          ["选中候选", `C${basis.selected_candidate_index ?? ""}`],
          ["选中 trace 节点", basis.selected_global_node_locator || basis.selected_trace_node_id],
        ], ["阶段", "规则 / 结果"]), true));

        const criteriaRows = criteria.map(field => [
          field.priority,
          field.key,
          field.label,
          field.direction,
          field.description,
        ]);
        if (Object.keys(graphTieBreak).length) {
          criteriaRows.push([
            graphTieBreak.priority,
            graphTieBreak.key,
            graphTieBreak.label,
            graphTieBreak.direction,
            graphTieBreak.description,
          ]);
        }
        root.appendChild(details("选择字段顺序", wideTable(criteriaRows, [
          "优先级", "字段", "名称", "方向", "含义",
        ]), true));

        const fieldByKey = new Map(criteria.map(field => [field.key, field]));
        const resultTable = table(candidates.map(candidate => {
          const decisive = fieldByKey.get(candidate.decisive_field);
          return [
            resonanceCandidateLabel(candidate),
            candidate.global_node_locator || candidate.trace_node_id,
            candidate.resonance_seed_index,
            candidate.resonance_index,
            candidate.resonance_raw_index,
            candidate.state?.canonical_smiles || candidate.state?.smiles,
            candidate.score,
            resonanceSelectionDecisionLabel(candidate.decision),
            decisive ? `${decisive.label} (${decisive.key})` : candidate.decisive_field,
            candidate.candidate_value,
            candidate.selected_value,
          ];
        }), [
          "候选", "trace 节点", "种子", "共振序号", "raw 序号", "canonical SMILES", "力场分",
          "结论", "首个决定字段", "候选值", "选中值",
        ]);
        candidates.forEach((candidate, index) => {
          const row = resultTable.tBodies[0] && resultTable.tBodies[0].rows[index];
          if (row && candidate.selected) row.classList.add("selection-result-selected");
        });
        const resultWrapper = document.createElement("div");
        resultWrapper.className = "wide-table";
        resultWrapper.appendChild(resultTable);
        root.appendChild(details("候选选择结论", resultWrapper, true));

        const comparisonHeaders = [
          "优先级",
          "字段",
          "名称",
          "方向",
          ...candidates.map(resonanceCandidateLabel),
        ];
        const comparisonRows = criteria.map(field => [
          field.priority,
          field.key,
          field.label,
          field.direction,
          ...candidates.map(candidate => candidate.selection_values?.[field.key]),
        ]);
        comparisonRows.push([
          graphTieBreak.priority,
          graphTieBreak.key,
          graphTieBreak.label,
          graphTieBreak.direction,
          ...candidates.map(candidate => candidate.graph_tie_break_hash),
        ]);
        root.appendChild(details("共振候选逐指标横向对比", wideTable(
          comparisonRows,
          comparisonHeaders,
        ), true));
        return root;
      }

      function renderMetalSelectionBasis(basis) {
        const root = document.createElement("section");
        const rankPolicy = basis.assignment_rank_policy || {};
        root.appendChild(details("决策流程", table([
          ["1. 价态先验", "按 prior/minor/other 和非正价态惩罚计算 assignment rank；用于搜索顺序、分层和每个 target 的剪枝"],
          ["2. 生产层", `只从第一个存在可评分候选的 Layer ${fmt(basis.selected_layer_index)} 中做最终选择`],
          ["3. 失谐过滤", `只保留 metal_discordance_count = ${fmt(basis.minimum_metal_discordance_count)} 的候选`],
          ["4. 最终比较", "对通过过滤的候选按下表字段从上到下做字典序最小化"],
          ["assignment rank 是否进入最终键", rankPolicy.participates_in_final_selection_key],
        ], ["阶段", "规则"]), true));

        const stateRows = [];
        (basis.available_metal_states_by_site || []).forEach((options, siteIndex) => {
          (options || []).forEach((state, optionIndex) => {
            stateRows.push([
              siteIndex,
              optionIndex,
              `#${state.idx} ${state.symbol}`,
              state.valence,
              state.radical_num,
              state.valence_prior_class,
              state.valence_prior_penalty,
              state.nonpositive_valence_penalty,
              state.assignment_penalty,
              state.ligand_field?.geometry,
              state.ligand_field?.coordination_number,
              state.ligand_field?.field_score,
              state.ligand_field?.field_strength,
              state.ligand_field?.weak_field_threshold,
              state.ligand_field?.strong_field_threshold,
              state.ligand_field?.ambiguity_margin,
              (state.ligand_field?.radical_options_preferred_first || []).join(", "),
              (state.ligand_field?.donors || []).map(donor =>
                `${donor.symbol}#${donor.idx} ${Number(donor.distance_angstrom).toFixed(3)}Å / ${donor.base_field_strength}`
              ).join("; "),
            ]);
          });
        });
        root.appendChild(details("金属价态先验与 assignment penalty", wideTable(stateRows, [
          "位点", "选项", "金属", "价态", "自由基", "先验类别", "价态先验惩罚", "非正价态惩罚", "总 assignment penalty",
          "构型", "配位数", "场强分数", "场强类别", "弱场阈值", "强场阈值", "模糊 margin", "自旋选项（首选在前）", "donor 依据",
        ]), true));

        const criteria = basis.selection_key_fields || [];
        root.appendChild(details("最终 selection_key 字段顺序", table(criteria.map(field => [
          field.priority,
          field.key,
          field.label,
          field.direction,
          field.description,
        ]), ["优先级", "字段", "名称", "方向", "含义"]), true));

        const decisions = basis.candidate_decisions || [];
        const fieldByKey = new Map(criteria.map(field => [field.key, field]));
        const resultTable = table(decisions.map(candidate => {
          const decisive = fieldByKey.get(candidate.decisive_field);
          return [
            `C${candidate.combination_index}`,
            metalStatesLabel(candidate.metal_states),
            candidate.search_layer_index,
            candidate.in_production_selection_layer,
            candidate.metal_assignment_rank,
            candidate.selection_values && candidate.selection_values.metal_discordance_count,
            candidate.passes_metal_discordance_filter,
            metalSelectionDecisionLabel(candidate.decision),
            decisive ? `${decisive.label} (${decisive.key})` : "",
            candidate.candidate_value,
            candidate.selected_value,
          ];
        }), [
          "候选", "金属状态", "Layer", "生产层", "assignment rank", "失谐", "通过过滤", "结论", "首个决定字段", "候选值", "选中值",
        ]);
        decisions.forEach((candidate, index) => {
          const row = resultTable.tBodies[0] && resultTable.tBodies[0].rows[index];
          if (!row) return;
          if (candidate.selected) row.classList.add("selection-result-selected");
          if (String(candidate.decision).startsWith("selection_inconsistency") || candidate.decision === "missing_selection_key") {
            row.classList.add("selection-result-error");
          }
        });
        const resultWrapper = document.createElement("div");
        resultWrapper.className = "wide-table";
        resultWrapper.appendChild(resultTable);
        root.appendChild(details("候选选择结论", resultWrapper, true));
        const discordanceComparison = renderDiscordanceComparison(decisions, "候选选择中的失谐分解");
        if (discordanceComparison) root.appendChild(discordanceComparison);

        const production = decisions.filter(candidate => candidate.in_production_selection_layer);
        const comparisonHeaders = [
          "优先级",
          "字段",
          "名称",
          ...production.map(candidate => `${candidate.selected ? "selected " : ""}C${candidate.combination_index} ${metalStatesLabel(candidate.metal_states)}`),
        ];
        const comparisonRows = criteria.map(field => [
          field.priority,
          field.key,
          field.label,
          ...production.map(candidate => candidate.selection_values && candidate.selection_values[field.key]),
        ]);
        root.appendChild(details("生产层候选逐项对比", wideTable(comparisonRows, comparisonHeaders), true));
        return root;
      }

      function details(title, content, open = false) {
        const el = document.createElement("details");
        if (open) el.open = true;
        const summary = document.createElement("summary");
        summary.textContent = localizeText(title);
        el.appendChild(summary);
        el.appendChild(content);
        return el;
      }

      function lazyDetails(title, value) {
        const placeholder = document.createElement("div");
        const el = details(title, placeholder);
        let loaded = false;
        el.addEventListener("toggle", () => {
          if (!el.open || loaded) return;
          placeholder.replaceWith(pre(value));
          loaded = true;
        });
        return el;
      }

      function pre(value) {
        const el = document.createElement("pre");
        el.textContent = JSON.stringify(value || {}, null, 2);
        return el;
      }

      function renderGlobal() {
        const root = document.getElementById("global-info");
        root.replaceChildren();
        const input = trace.input || {};
        const dof = trace.dof_rendering || {};
        const inputCard = document.createElement("section");
        inputCard.className = "global-card";
        inputCard.innerHTML = `<h2>${localizeText("输入")}</h2>`;
        inputCard.appendChild(table([
          ["来源", input.source],
          ["CSV", input.csv],
          ["XYZ 目录", input.xyz_dir],
          ["id", input.ids],
          ["默认总电荷", input.total_charge],
          ["默认总自由基电子数", input.total_radical_electrons],
          ["自旋来源", input.spin_source],
          ["样本数", trace.case_count],
          ["全局节点数", trace.global_node_count],
        ]));
        const dofCard = document.createElement("section");
        dofCard.className = "global-card";
        dofCard.innerHTML = `<h2>${localizeText("DOF 渲染")}</h2>`;
        dofCard.appendChild(table([
          ["存储", dof.storage],
          ["图片目录", dof.image_dir],
          ["格式", dof.format],
          ["图片数", dof.image_count],
          ["跳过数", dof.skipped_count],
          ["最大图片数", dof.max_images],
          ["错误数", (dof.errors || []).length],
        ]));
        const caseCard = document.createElement("section");
        caseCard.className = "global-card";
        caseCard.innerHTML = "<h2>Cases</h2>";
        const caseRows = (trace.cases || []).map(item => [
          item.id,
          item.status,
          item.trace_kind,
          item.charge,
          item.total_radical_electrons,
          item.base_state && item.base_state.metal_atom_count,
          item.search && item.search.selected_layer_index,
          item.production_candidate_count,
          item.candidate_count,
          item.elapsed_seconds,
        ]);
        caseCard.appendChild(table(caseRows.map(row => ["", row]), ["", ""]));
        const caseTable = caseCard.querySelector("table");
        caseTable.innerHTML = "";
        const head = caseTable.createTHead().insertRow();
        ["id", "状态", "类型", "电荷", "自由基", "金属数", "生产层", "生产候选", "全部候选", "耗时秒"].forEach(text => {
          const th = document.createElement("th");
          th.textContent = localizeText(text);
          head.appendChild(th);
        });
        const body = caseTable.createTBody();
        caseRows.forEach(row => {
          const tr = body.insertRow();
          row.forEach(cell => {
            const td = tr.insertCell();
            td.textContent = fmt(cell);
          });
        });
        root.append(inputCard, dofCard, caseCard);
      }

      function syncTreeNode(node, forcedOpen = false) {
        const hasChildren = node.children.length > 0;
        const isOpen = hasChildren && (!node.collapsed || forcedOpen);
        if (node.treeLi) {
          node.treeLi.classList.toggle("is-expanded", hasChildren && isOpen);
          node.treeLi.classList.toggle("is-collapsed", hasChildren && !isOpen);
        }
        if (node.treeToggle) {
          node.treeToggle.setAttribute("aria-expanded", hasChildren ? String(isOpen) : "false");
          node.treeToggle.disabled = !hasChildren;
        }
        if (node.treeChildren) {
          node.treeChildren.hidden = hasChildren && !isOpen;
        }
      }

      function renderTreeNode(node, depth = 0, parent = null) {
        const li = document.createElement("li");
        node.parent = parent;
        if (node.children.length && typeof node.collapsed !== "boolean") {
          const compactTree = window.matchMedia("(max-width: 720px)").matches;
          node.collapsed = depth >= (compactTree ? 1 : 2);
        }
        const row = document.createElement("div");
        row.className = "tree-row";
        if (node.children.length) {
          const toggle = document.createElement("button");
          toggle.type = "button";
          toggle.className = "tree-toggle";
          toggle.setAttribute("aria-label", ui("expandNode"));
          toggle.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();
            node.collapsed = !node.collapsed;
            syncTreeNode(node);
            applyTreeFilter();
          });
          row.appendChild(toggle);
          node.treeToggle = toggle;
        } else {
          const spacer = document.createElement("span");
          spacer.className = "tree-toggle-spacer";
          spacer.setAttribute("aria-hidden", "true");
          row.appendChild(spacer);
        }
        const button = document.createElement("button");
        button.type = "button";
        button.className = `tree-node${node.selected ? " is-selected" : ""}`;
        button.dataset.target = node.id;
        button.dataset.search = node.search || "";
        const label = document.createElement("span");
        label.className = "tree-label";
        label.textContent = localizeText(node.label);
        const meta = document.createElement("span");
        meta.className = "tree-meta";
        meta.textContent = localizeText(node.kind);
        button.append(label, meta);
        button.addEventListener("click", () => activate(node.id, {reveal: true}));
        row.appendChild(button);
        li.appendChild(row);
        if (node.children.length) {
          const ol = document.createElement("ol");
          ol.className = "tree-children";
          node.children.forEach(child => ol.appendChild(renderTreeNode(child, depth + 1, node)));
          li.appendChild(ol);
          node.treeChildren = ol;
        }
        node.treeLi = li;
        syncTreeNode(node);
        return li;
      }

      function resonanceCandidateChunks(basis) {
        const candidates = Array.isArray(basis?.candidates) ? basis.candidates : [];
        const configured = Number(basis?.image_comparison?.max_molecules_per_grid);
        const chunkSize = Number.isInteger(configured) && configured > 0 ? configured : 24;
        const chunks = [];
        for (let start = 0; start < candidates.length; start += chunkSize) {
          chunks.push(candidates.slice(start, start + chunkSize));
        }
        return chunks;
      }

      function renderResonanceImageComparison(basis) {
        const root = document.createElement("section");
        root.className = "resonance-image-comparison";
        const chunks = resonanceCandidateChunks(basis);
        chunks.forEach((chunk, index) => {
          const group = document.createElement("section");
          group.className = "resonance-image-group";
          const heading = document.createElement("h3");
          heading.textContent = localizeText(chunks.length > 1
            ? `共振候选多图 ${index + 1}/${chunks.length}`
            : "共振候选多图");
          const row = document.createElement("div");
          row.className = "dof-visual-row";
          const box = document.createElement("div");
          box.className = "image-box";
          box.dataset.resonanceGridIndex = String(index);
          box.textContent = ui("renderOnOpen");
          row.appendChild(box);
          const molBox = document.createElement("div");
          molBox.className = "dof-3dmol";
          renderDofMol3d(molBox, {
            sdfs: chunk.flatMap(candidate => dofSdfs(candidate.dof_image)),
          });
          row.appendChild(molBox);
          group.append(heading, row);
          window.requestAnimationFrame(() => syncDofVisualHeight(row));
          root.appendChild(group);
        });
        return root;
      }

      function renderPanel(node) {
        const panel = document.createElement("article");
        panel.id = node.id;
        panel.className = "node-panel is-active";
        const head = document.createElement("div");
        head.className = "panel-head";
        const titleBox = document.createElement("div");
        const title = document.createElement("h2");
        title.textContent = localizeText(node.label);
        const meta = document.createElement("p");
        meta.textContent = `${node.caseId} · ${localizeText(node.kind)}`;
        titleBox.append(title, meta);
        head.appendChild(titleBox);
        if (node.selected) {
          const badge = document.createElement("span");
          badge.className = "badge selected";
          badge.textContent = ui("selected");
          head.appendChild(badge);
        }
        panel.appendChild(head);
        if (node.kind === "共振候选对比") {
          panel.appendChild(renderResonanceImageComparison(node.metadata));
        } else {
          const visualRow = document.createElement("div");
          visualRow.className = "dof-visual-row";
          let graphBox = null;
          const svg = imageSvg(node.image);
          const imgPath = imagePath(node.image);
          const deferredPayload = deferredDofPayload(node.image);
          if (svg) {
            const box = document.createElement("div");
            box.className = "image-box";
            box.innerHTML = svg;
            setImageZoomState(box, node.label);
            graphBox = box;
          } else if (imgPath) {
            const box = document.createElement("div");
            box.className = "image-box";
            const img = document.createElement("img");
            img.dataset.src = imgPath;
            img.loading = "lazy";
            img.alt = localizeText(node.label);
            box.appendChild(img);
            graphBox = box;
          } else if (deferredPayload) {
            const box = document.createElement("div");
            box.className = "image-box";
            box.dataset.deferredDof = node.image.render_id;
            graphBox = box;
          } else {
            const empty = document.createElement("div");
            empty.className = "image-empty";
            empty.textContent = ui("noRenderableDof");
            graphBox = empty;
          }
          visualRow.appendChild(graphBox);
          const molBox = document.createElement("div");
          molBox.className = "dof-3dmol";
          renderDofMol3d(molBox, node.image);
          visualRow.appendChild(molBox);
          panel.appendChild(visualRow);
          window.requestAnimationFrame(() => syncDofVisualHeight(visualRow));
        }
        if (node.kind === "金属电子态候选" && node.resonanceBasis) {
          const heading = document.createElement("h3");
          heading.className = "related-resonance-title";
          heading.textContent = localizeText("该金属价态对应有机目标的共振候选");
          panel.append(heading, renderResonanceImageComparison(node.resonanceBasis));
        }
        const info = document.createElement("section");
        info.className = "panel-info";
        info.appendChild(details("摘要", table(node.summary), true));
        if (node.kind === "金属价态候选" && node.metadata.selection_basis) {
          info.appendChild(renderMetalSelectionBasis(node.metadata.selection_basis));
        }
        if (node.kind === "共振候选对比") {
          info.appendChild(renderResonanceSelectionBasis(node.metadata));
        }
        if (node.kind === "金属电子态候选" && node.resonanceBasis) {
          info.appendChild(renderResonanceSelectionBasis(node.resonanceBasis));
        }
        if (
          node.kind === "金属价态候选"
          && !node.metadata.selection_basis
          && Array.isArray(node.metadata.discordance_candidates)
        ) {
          const comparisonEntries = node.metadata.discordance_candidates.map(candidate => ({
            ...candidate,
            discordance_details: candidate.score_details || {},
          }));
          const comparison = renderDiscordanceComparison(comparisonEntries, "金属候选对比中的失谐分解");
          if (comparison) info.appendChild(comparison);
        }
        const discordancePanel = renderDiscordanceBreakdown(node.scoreDetails);
        if (discordancePanel) info.appendChild(discordancePanel);
        if (Object.keys(node.scoreDetails || {}).length) {
          info.appendChild(details("分数构成", table(Object.entries(node.scoreDetails)), true));
        }
        info.appendChild(lazyDetails("完整 JSON", node.metadata));
        panel.appendChild(info);
        return panel;
      }

      async function renderDeferredDof(node, panel) {
        const payload = deferredDofPayload(node.image);
        const box = panel && panel.querySelector("[data-deferred-dof]");
        if (!payload || !box || box.dataset.state === "loading" || box.dataset.state === "loaded") return;
        box.dataset.state = "loading";
        box.textContent = ui("rendering");
        try {
          const response = await fetch("/api/render-dof", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
          });
          const body = await response.text();
          if (!response.ok) throw new Error(body || `HTTP ${response.status}`);
          box.innerHTML = body;
          box.dataset.state = "loaded";
          setImageZoomState(box, node.label);
        } catch (error) {
          box.textContent = `${ui("renderFailed")}: ${error.message || error}`;
          box.dataset.state = "error";
        } finally {
          syncDofVisualHeight(box.closest(".dof-visual-row"));
          scheduleMainColumnHeightSync();
        }
      }

      async function renderDeferredResonanceGrids(node, panel) {
        if (!panel) return;
        const basis = node.kind === "共振候选对比" ? node.metadata : node.resonanceBasis;
        if (!basis || !Array.isArray(basis.candidates)) return;
        const imageConfig = basis.image_comparison || {};
        const chunks = resonanceCandidateChunks(basis);
        for (let index = 0; index < chunks.length; index += 1) {
          const box = panel.querySelector(`[data-resonance-grid-index="${index}"]`);
          if (!box || box.dataset.state === "loaded") continue;
          const items = chunks[index].map(candidate => ({
            candidate,
            payload: deferredDofPayload(candidate.dof_image),
          })).filter(item => item.payload?.render_type === "single" && item.payload.sdf);
          if (!items.length) {
            box.textContent = ui("noCandidateSdf");
            box.dataset.state = "error";
            scheduleMainColumnHeightSync();
            continue;
          }
          box.dataset.state = "loading";
          box.textContent = ui("rendering");
          try {
            const response = await fetch("/api/render-dof", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                render_type: "grid",
                sdfs: items.map(item => item.payload.sdf),
                legends: items.map(item => (
                  `${item.candidate.selected ? "selected " : ""}` +
                  `R${item.candidate.resonance_raw_index ?? ""} ` +
                  `score=${fmt(item.candidate.score)}`
                )),
                mols_per_row: imageConfig.molecules_per_row || 3,
                sub_image_size: imageConfig.sub_image_size || [320, 260],
              }),
            });
            const body = await response.text();
            if (!response.ok) throw new Error(body || `HTTP ${response.status}`);
            box.innerHTML = body;
            box.dataset.state = "loaded";
            setImageZoomState(box, `共振候选多图 ${index + 1}/${chunks.length}`);
          } catch (error) {
            box.textContent = `${ui("renderFailed")}: ${error.message || error}`;
            box.dataset.state = "error";
          } finally {
            scheduleMainColumnHeightSync();
          }
        }
      }

      function activate(id, {reveal = false} = {}) {
        const node = nodeById.get(id);
        let current = node ? node.parent : null;
        while (current) {
          if (current.collapsed) {
            current.collapsed = false;
            syncTreeNode(current);
          }
          current = current.parent;
        }
        document.querySelectorAll(".tree-node").forEach(button => {
          button.classList.toggle("is-active", button.dataset.target === id);
        });
        const detailRoot = document.getElementById("detail");
        detailRoot.replaceChildren(renderPanel(node));
        const activePanel = detailRoot.firstElementChild;
        if (activePanel) {
          activePanel.querySelectorAll("img[data-src]").forEach(img => {
            img.src = img.dataset.src;
            delete img.dataset.src;
            setImageZoomState(img.closest(".image-box"), node.label);
            img.addEventListener("load", () => syncDofVisualHeight(img.closest(".dof-visual-row")), {once: true});
          });
          void renderDeferredDof(node, activePanel);
          void renderDeferredResonanceGrids(node, activePanel);
          if (reveal && window.matchMedia("(max-width: 1100px)").matches) {
            window.requestAnimationFrame(() => {
              activePanel.scrollIntoView({behavior: "smooth", block: "start"});
            });
          }
        }
        applyTreeFilter();
        scheduleMainColumnHeightSync();
      }

      function applyTreeFilter() {
        const q = (document.getElementById("tree-search").value || "").trim().toLowerCase();

        function walk(node) {
          const hasChildren = node.children.length > 0;
          const selfMatches = !q || (node.search || "").toLowerCase().includes(q);
          let descendantMatches = false;
          node.children.forEach(child => {
            descendantMatches = walk(child) || descendantMatches;
          });
          const visible = selfMatches || descendantMatches || !q;
          if (node.treeLi) {
            node.treeLi.hidden = !visible;
          }
          const forcedOpen = Boolean(q && hasChildren && (selfMatches || descendantMatches));
          syncTreeNode(node, forcedOpen);
          return visible;
        }

        roots.forEach(root => walk(root));
        scheduleMainColumnHeightSync();
      }

      const treeElement = document.getElementById("tree");
      const detailElement = document.getElementById("detail");

      function renderApplication(activeId = "") {
        nodes.length = 0;
        roots.length = 0;
        nodeById.clear();
        treeElement.replaceChildren();
        detailElement.replaceChildren();
        buildTree();
        renderGlobal();
        const treeRoot = document.createElement("ol");
        roots.forEach(root => treeRoot.appendChild(renderTreeNode(root)));
        treeElement.appendChild(treeRoot);
        const targetId = activeId && nodeById.has(activeId) ? activeId : nodes[0]?.id;
        if (targetId) activate(targetId);
      }

      function toggleLanguage() {
        const activeId = document.querySelector(".tree-node.is-active")?.dataset.target || "";
        language = language === "zh" ? "en" : "zh";
        localStorage.setItem("moleculeReviewLanguage", language);
        applyStaticLanguage();
        renderApplication(activeId);
      }

      document.getElementById("tree-search").addEventListener("input", applyTreeFilter);
      document.getElementById("language-toggle").addEventListener("click", toggleLanguage);
      detailElement.addEventListener("click", event => {
        const box = event.target.closest(".image-box.is-zoomable");
        if (box) openImageLightbox(box);
      });
      detailElement.addEventListener("keydown", event => {
        if (event.key !== "Enter" && event.key !== " ") return;
        const box = event.target.closest(".image-box.is-zoomable");
        if (!box) return;
        event.preventDefault();
        openImageLightbox(box);
      });
      window.addEventListener("resize", scheduleMainColumnHeightSync);
      window.visualViewport?.addEventListener("resize", scheduleMainColumnHeightSync);
      document.getElementById("image-lightbox-close").addEventListener("click", closeImageLightbox);
      document.getElementById("image-lightbox").addEventListener("click", event => {
        if (event.target === event.currentTarget) closeImageLightbox();
      });
      applyStaticLanguage();
      renderApplication();
      scheduleMainColumnHeightSync();
    })();
  </script>
</body>
</html>
"""
    )


def _score_details(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key in _SCORE_DETAIL_KEYS or key.startswith(_SCORE_DETAIL_PREFIXES)
    }


def _candidate_identity(
    candidate: MetalCandidateState,
    *,
    fallback_candidate_index: int = 0,
) -> tuple[int, int]:
    return (
        int(candidate.metadata.get("search_layer_index", -1)),
        int(candidate.metadata.get("combination_index", fallback_candidate_index)),
    )


def _copy_metadata_by_identity(
    candidates: Sequence[MetalCandidateState],
) -> dict[tuple[int, int], dict[str, Any]]:
    copied: dict[tuple[int, int], dict[str, Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        copied[_candidate_identity(candidate, fallback_candidate_index=candidate_index)] = deepcopy(
            candidate.metadata
        )
    return copied


def _no_metal_candidate_trace(
    candidate: ReconstructionState,
    *,
    selected: bool,
    render_context: DofRenderContext | None = None,
    case_id: str = "",
    image_label: str = "no_metal_candidate",
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    selection_key = no_metal_selection._no_metal_candidate_selection_key(candidate, config=config)
    state = _omol_state_snapshot(
        candidate.omol,
        given_charge=candidate.given_charge,
        target_charge=candidate.total_charge,
        target_radical_electrons=candidate.total_radical_electrons,
    )
    image = _render_dof_molecule(
        candidate.omol,
        render_context=render_context,
        case_id=case_id,
        label=image_label,
        kind="no_metal_candidate",
    )
    if image is not None:
        state["dof_image"] = image
    return {
        "selected": selected,
        "phase_history": list(candidate.phase_history),
        "state": state,
        "score": candidate.metadata.get("score"),
        "force_field_energy": candidate.metadata.get("force_field_energy"),
        "organic_topology_selection_key": selection_key,
        "metadata": _jsonable(candidate.metadata),
    }


def _production_phase_steps(
    state: ReconstructionState,
    step_images: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expose the phases recorded by the production no-metal state machine."""

    steps: list[dict[str, Any]] = []
    for step_index, phase in enumerate(state.phase_history):
        if phase.startswith("recover_"):
            kind = "recovery"
        elif phase.startswith("resolve_neighbor_radicals"):
            kind = "neighbor_radical_seed"
        elif "resonance" in phase:
            kind = "resonance"
        else:
            kind = "stage"
        step = {"step_index": step_index, "phase": phase, "kind": kind}
        if step_images is not None and step_index in step_images:
            step["dof_image"] = step_images[step_index]
        steps.append(step)
    return steps


def _trace_node_kind(phase: str, event: dict[str, Any]) -> str:
    event_kind = str(event.get("kind", "stage"))
    if event_kind in {"branch", "checkpoint"}:
        return event_kind
    if phase.startswith("resolve_neighbor_radicals") or phase.startswith("neighbor_radicals"):
        return "neighbor_radical_seed"
    if "resonance" in phase or "resonance" in str(event.get("stage", "")):
        return "resonance"
    if phase.startswith("recover_"):
        return "recovery"
    return event_kind


def _trace_node_reports(
    recorder: OmolTraceRecorder,
    *,
    selected_ids: set[int],
    target_charge: int,
    target_radical_electrons: int,
    render_context: DofRenderContext | None = None,
    case_id: str = "",
) -> list[dict[str, Any]]:
    """Materialize every recorder node, including discarded resonance branches."""

    reports: list[dict[str, Any]] = []
    tree_scope_by_node: dict[int, int] = {}
    tree_depth_by_node: dict[int, int] = {}
    for node_id, record in sorted(recorder.records.items()):
        phase = str(record.get("phase", ""))
        event = cast(Dict[str, Any], record.get("event", {}))
        execution_parent_id = int(record.get("parent_id", -1))
        parent_scope_id = tree_scope_by_node.get(execution_parent_id, -1)
        is_expansion = event.get("kind") == "branch" or (
            bool(event.get("hit"))
            and phase.startswith(
                (
                    "resolve_neighbor_radicals",
                    "relocate_carbene_radical_for_resonance",
                    "recover_deformed_pi_bonds",
                    "recover_by_breaking_bonds",
                )
            )
        )
        tree_parent_id = parent_scope_id
        tree_depth = 0 if tree_parent_id < 0 else tree_depth_by_node.get(tree_parent_id, 0) + 1
        tree_scope_by_node[node_id] = node_id if is_expansion else parent_scope_id
        tree_depth_by_node[node_id] = tree_depth
        dof_image = _render_dof_molecule(
            cast(pybel.Molecule, record["omol"]),
            render_context=render_context,
            case_id=case_id,
            label=f"trace_node_{node_id}_{phase}",
            kind="no_metal_trace_node",
        )
        reports.append(
            {
                "node_id": node_id,
                "parent_id": execution_parent_id,
                "tree_parent_id": tree_parent_id,
                "tree_depth": tree_depth,
                "expansion": is_expansion,
                "selected_path": node_id in selected_ids,
                "phase": phase,
                "kind": _trace_node_kind(phase, event),
                "event": _jsonable(event),
                "metadata": _jsonable(record.get("metadata", {})),
                "dof_image": dof_image,
                "state": _omol_state_snapshot(
                    cast(pybel.Molecule, record["omol"]),
                    given_charge=int(record.get("given_charge", 0)),
                    target_charge=target_charge,
                    target_radical_electrons=target_radical_electrons,
                ),
            }
        )
    return reports


def _first_different_key_index(left: Sequence[Any], right: Sequence[Any]) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _resonance_selection_report(
    recorder: OmolTraceRecorder,
    *,
    selected_ids: set[int],
    trace_nodes: Sequence[dict[str, Any]],
    target_charge: int,
    target_radical_electrons: int,
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    """Explain the exact comparison among validated resonance candidates."""

    image_by_node_id = {
        int(node["node_id"]): node.get("dof_image")
        for node in trace_nodes
        if isinstance(node, dict) and int(node.get("node_id", -1)) >= 0
    }
    candidates: list[dict[str, Any]] = []
    candidate_graph_keys: list[tuple[int, ...]] = []
    for node_id, record in sorted(recorder.records.items()):
        if record.get("phase") != "accept_no_metal_candidate":
            continue
        event = cast(Dict[str, Any], record.get("event", {}))
        metadata = dict(cast(Dict[str, Any], record.get("metadata", {})))
        score = event.get("score")
        if score is not None:
            metadata["score"] = float(score)
            metadata["force_field_energy"] = float(score)
        candidate_state = ReconstructionState(
            omol=cast(pybel.Molecule, record["omol"]),
            given_charge=int(record.get("given_charge", 0)),
            total_charge=target_charge,
            total_radical_electrons=target_radical_electrons,
            metadata=metadata,
        )
        selection_key = no_metal_selection._no_metal_candidate_selection_key(
            candidate_state,
            config=config,
        )
        graph_tie_break_key = no_metal_selection._no_metal_candidate_graph_tie_break_key(
            candidate_state
        )
        selection_values = {
            key: candidate_state.metadata.get(key)
            for key, _label, _direction, _description in _RESONANCE_SELECTION_KEY_FIELDS
        }
        candidate_index = len(candidates)
        selected = node_id in selected_ids
        candidates.append(
            {
                "candidate_index": candidate_index,
                "trace_node_id": node_id,
                "selected": selected,
                "resonance_seed_index": metadata.get("resonance_seed_index"),
                "resonance_index": metadata.get("resonance_index"),
                "resonance_raw_index": metadata.get("resonance_raw_index"),
                "resonance_normalization": metadata.get("resonance_normalization"),
                "score": score,
                "selection_key": selection_key,
                "selection_values": selection_values,
                "graph_tie_break_hash": _short_hash(graph_tie_break_key),
                "metadata": _jsonable(candidate_state.metadata),
                "state": _omol_state_snapshot(
                    candidate_state.omol,
                    given_charge=candidate_state.given_charge,
                    target_charge=target_charge,
                    target_radical_electrons=target_radical_electrons,
                ),
                "dof_image": image_by_node_id.get(node_id),
            }
        )
        candidate_graph_keys.append(graph_tie_break_key)

    selected_candidate = next(
        (candidate for candidate in candidates if candidate["selected"]), None
    )
    if selected_candidate is not None:
        selected_key = tuple(selected_candidate["selection_key"])
        selected_graph_key = candidate_graph_keys[int(selected_candidate["candidate_index"])]
        for candidate, candidate_graph_key in zip(candidates, candidate_graph_keys):
            candidate_key = tuple(candidate["selection_key"])
            if candidate["selected"]:
                candidate["decision"] = "selected_lexicographic_minimum"
                candidate["decisive_field"] = ""
                continue
            key_index = _first_different_key_index(candidate_key, selected_key)
            if key_index is not None and key_index < len(_RESONANCE_SELECTION_KEY_FIELDS):
                candidate["decision"] = "rejected_by_selection_key"
                candidate["decisive_field"] = _RESONANCE_SELECTION_KEY_FIELDS[key_index][0]
                candidate["candidate_value"] = candidate["selection_values"].get(
                    candidate["decisive_field"]
                )
                candidate["selected_value"] = selected_candidate["selection_values"].get(
                    candidate["decisive_field"]
                )
            elif candidate_graph_key != selected_graph_key:
                candidate["decision"] = "rejected_by_graph_tie_break"
                candidate["decisive_field"] = "graph_tie_break_key"
                candidate["candidate_value"] = candidate["graph_tie_break_hash"]
                candidate["selected_value"] = selected_candidate["graph_tie_break_hash"]
            else:
                candidate["decision"] = "selection_key_tie_not_selected"
                candidate["decisive_field"] = ""

    return {
        "algorithm": "lexicographic_minimum_then_backend_independent_graph_tie_break",
        "candidate_count": len(candidates),
        "selected_candidate_index": (
            selected_candidate.get("candidate_index") if selected_candidate is not None else None
        ),
        "selected_trace_node_id": (
            selected_candidate.get("trace_node_id") if selected_candidate is not None else None
        ),
        "selected_selection_key": (
            selected_candidate.get("selection_key") if selected_candidate is not None else None
        ),
        "selection_key_fields": [
            {
                "priority": priority,
                "key": key,
                "label": label,
                "direction": direction,
                "description": description,
            }
            for priority, (key, label, direction, description) in enumerate(
                _RESONANCE_SELECTION_KEY_FIELDS,
                start=1,
            )
        ],
        "graph_tie_break": {
            "priority": len(_RESONANCE_SELECTION_KEY_FIELDS) + 1,
            "key": "graph_tie_break_key",
            "label": "显式电子态图稳定排序键",
            "direction": "min",
            "description": "仅在选择键完全相同时比较原子、显式电子标签和键表",
        },
        "candidates": candidates,
    }


def _trace_no_metal_reconstruction(
    xyz_block: str,
    total_charge: int,
    total_radical_electrons: int,
    *,
    render_context: DofRenderContext | None = None,
    case_id: str = "",
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "target": {
            "total_charge": total_charge,
            "total_radical_electrons": total_radical_electrons,
        },
        "status": "pending",
    }
    if total_radical_electrons < 0:
        trace["status"] = "invalid_negative_radicals"
        return trace

    seed_state = no_metal_preparation._seed_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
    )
    recorder = OmolTraceRecorder()
    read_snapshot_id = recorder.record_initial(seed_state.omol, "read_xyz")
    normalized_snapshot_id = recorder.record_initial(
        seed_state.omol,
        "normalize_seed_electronic_labels",
        parent_id=read_snapshot_id,
    )
    seed_state.metadata[TRACE_NODE_METADATA_KEY] = normalized_snapshot_id

    with trace_omol_state_machine(recorder):
        selected = reconstruct_without_metals._run_no_metal_pipeline_from_state(
            seed_state,
            config=config,
        )
    if selected is None:
        trace["status"] = "no_valid_no_metal_candidate"
        trace["trace_nodes"] = _trace_node_reports(
            recorder,
            selected_ids=set(),
            target_charge=total_charge,
            target_radical_electrons=total_radical_electrons,
            render_context=render_context,
            case_id=case_id,
        )
        trace["trace_node_count"] = len(trace["trace_nodes"])
        return trace

    selected_snapshot_id = int(selected.metadata.get(TRACE_NODE_METADATA_KEY, -1))
    selected_path_records: list[dict[str, Any]] = []
    selected_path_node_ids: list[int] = []
    seen_snapshot_ids: set[int] = set()
    while selected_snapshot_id >= 0 and selected_snapshot_id not in seen_snapshot_ids:
        seen_snapshot_ids.add(selected_snapshot_id)
        record = recorder.records.get(selected_snapshot_id)
        if record is None:
            break
        selected_path_records.append(record)
        selected_path_node_ids.append(selected_snapshot_id)
        selected_snapshot_id = int(record.get("parent_id", -1))
    selected_path_records.reverse()
    selected_path_node_ids.reverse()
    selected.metadata.pop(TRACE_NODE_METADATA_KEY, None)
    selected_ids = {
        int(record_id) for record_id in seen_snapshot_ids if record_id in recorder.records
    }
    trace_nodes = _trace_node_reports(
        recorder,
        selected_ids=selected_ids,
        target_charge=total_charge,
        target_radical_electrons=total_radical_electrons,
        render_context=render_context,
        case_id=case_id,
    )
    image_by_node_id = {
        int(node["node_id"]): cast(Dict[str, Any], node["dof_image"])
        for node in trace_nodes
        if isinstance(node.get("dof_image"), dict)
    }
    step_images = {
        step_index: image_by_node_id[node_id]
        for step_index, node_id in enumerate(selected_path_node_ids)
        if node_id in image_by_node_id
    }

    trace["status"] = "selected"
    trace["pipeline_steps"] = _production_phase_steps(selected, step_images)
    trace["trace_nodes"] = trace_nodes
    trace["trace_node_count"] = len(trace["trace_nodes"])
    trace["neighbor_radical_actions"] = list(selected.metadata.get("neighbor_radical_actions", ()))
    trace["recovery_tier"] = int(selected.metadata.get("recovery_tier", 0))
    trace["resonance"] = {
        "seed_index": selected.metadata.get("resonance_seed_index"),
        "resonance_index": selected.metadata.get("resonance_index"),
        "raw_index": selected.metadata.get("resonance_raw_index"),
        "normalization": selected.metadata.get("resonance_normalization"),
    }
    resonance_selection = _resonance_selection_report(
        recorder,
        selected_ids=selected_ids,
        trace_nodes=trace_nodes,
        target_charge=total_charge,
        target_radical_electrons=total_radical_electrons,
        config=config,
    )
    if render_context is not None:
        resonance_selection["image_comparison"] = {
            "max_molecules_per_grid": render_context.grid_max_mols,
            "molecules_per_row": render_context.grid_mols_per_row,
            "sub_image_size": list(render_context.grid_sub_img_size),
        }
    trace["resonance_selection_basis"] = resonance_selection
    trace["selected_candidate"] = _no_metal_candidate_trace(
        selected,
        selected=True,
        render_context=render_context,
        case_id=case_id,
        image_label="selected_no_metal_candidate",
        config=config,
    )

    animation = _render_dof_animation(
        [(record["omol"], str(record["phase"])) for record in selected_path_records],
        render_context=render_context,
        case_id=case_id,
        label="selected_no_metal_path",
        kind="selected_path_animation",
    )
    if animation is not None:
        trace["selected_path_animation"] = animation
    return trace


def _render_selected_metal_path_animation(
    selected_candidate: MetalCandidateState | None,
    *,
    no_metal_xyz_block: str,
    render_context: DofRenderContext | None,
    case_id: str,
) -> dict[str, Any] | None:
    if selected_candidate is None or selected_candidate.no_metal_state is None:
        return None
    seed_state = no_metal_preparation._seed_state(
        no_metal_xyz_block,
        selected_candidate.no_metal_charge_target,
        selected_candidate.no_metal_radical_target,
    )
    try:
        combined_omol = selected_candidate.materialize_combined_omol(
            preparation.combine_metal_with_omol
        )
    except Exception as exc:
        error = {
            "kind": "selected_metal_path_animation",
            "label": "selected_metal_path",
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "animation": True,
        }
        if render_context is not None:
            render_context.errors.append(error)
        return error
    return _render_dof_animation(
        [
            (_copy_omol(seed_state.omol), "normalized organic seed"),
            (_copy_omol(selected_candidate.no_metal_state.omol), "selected organic part"),
            (combined_omol, "selected metal candidate"),
        ],
        render_context=render_context,
        case_id=case_id,
        label="selected_metal_path",
        kind="selected_metal_path_animation",
    )


def _annotate_analysis_scores_for_all_candidates(
    candidates: Sequence[MetalCandidateState],
    *,
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    if not candidates:
        return {}

    scoring._annotate_candidate_set_discordance_features(candidates, config=config)
    for candidate in candidates:
        scoring._annotate_selected_candidate_metrics(candidate, config=config)

    for candidate in candidates:
        score_value = float(cast(float, candidate.score))
        candidate.metadata["analysis_selection_key_all_candidates"] = (
            float(candidate.metadata.get("metal_discordance_count", 0)),
            score_value,
            int(candidate.metadata.get("combination_index", 0)),
        )

    return {}


def _candidate_report(
    candidate: MetalCandidateState,
    *,
    candidate_index: int,
    selected_candidate_identity: Optional[tuple[int, int]],
    production_metadata_by_identity: dict[tuple[int, int], dict[str, Any]],
    render_context: DofRenderContext | None = None,
    case_id: str = "",
    field_analysis_by_state: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    combination_index = int(candidate.metadata.get("combination_index", candidate_index))
    candidate_identity = _candidate_identity(
        candidate,
        fallback_candidate_index=candidate_index,
    )
    no_metal_state = candidate.no_metal_state
    no_metal: dict[str, Any] = {}
    if no_metal_state is not None:
        no_metal = {
            "target_charge": int(no_metal_state.total_charge),
            "target_radical_electrons": int(no_metal_state.total_radical_electrons),
            "given_charge": int(no_metal_state.given_charge),
            "phase_history": list(no_metal_state.phase_history),
            "smiles": _omol_smiles_detail(no_metal_state.omol),
            "score": no_metal_state.metadata.get("score"),
            "metadata": _jsonable(no_metal_state.metadata),
        }

    metadata = candidate.metadata
    production_metadata = production_metadata_by_identity.get(candidate_identity, {})
    metal_valence_sum = sum(int(metal_state.valence) for metal_state in candidate.metal_states)
    no_metal_total_charge = int(no_metal_state.total_charge) if no_metal_state is not None else None
    candidate_total_charge = (
        no_metal_total_charge + metal_valence_sum if no_metal_total_charge is not None else None
    )
    report = {
        "candidate_index": candidate_index,
        "candidate_identity": {
            "search_layer_index": candidate_identity[0],
            "combination_index": candidate_identity[1],
        },
        "search_layer_index": candidate_identity[0],
        "combination_index": combination_index,
        "selected": selected_candidate_identity == candidate_identity,
        "in_production_selection_layer": bool(production_metadata),
        "candidate_total_charge": candidate_total_charge,
        "metal_valence_sum": metal_valence_sum,
        "target": {
            "no_metal_charge": int(candidate.no_metal_charge_target),
            "no_metal_radical_electrons": int(candidate.no_metal_radical_target),
        },
        "metal_states": [
            _metal_state_to_dict(
                metal_state,
                field_analysis=(field_analysis_by_state or {}).get(
                    (int(metal_state.idx), int(metal_state.valence))
                ),
            )
            for metal_state in candidate.metal_states
        ],
        "organic_part": {
            "smiles": no_metal.get("smiles", {}).get("smiles", ""),
            "canonical_smiles": no_metal.get("smiles", {}).get("canonical_smiles", ""),
        },
        "score": _jsonable(candidate.score),
        "score_details": _jsonable(_score_details(metadata)),
        "metadata": _jsonable(metadata),
        "production_metadata": _jsonable(production_metadata),
        "no_metal_state": no_metal,
        "phase_history": list(candidate.phase_history),
    }
    if candidate.no_metal_state is not None:
        image_label = f"L{candidate_identity[0]}_C{combination_index}"
        try:
            combined_omol = candidate.materialize_combined_omol(preparation.combine_metal_with_omol)
            image = _render_dof_molecule(
                combined_omol,
                render_context=render_context,
                case_id=case_id,
                label=image_label,
                kind="metal_candidate",
            )
        except Exception as exc:
            image = {
                "kind": "metal_candidate",
                "label": image_label,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if render_context is not None:
                render_context.errors.append(image)
        if image is not None:
            report["dof_image"] = image
    return report


def _metal_selection_basis(
    candidate_reports: Sequence[dict[str, Any]],
    *,
    selected_layer_index: int | None,
) -> dict[str, Any]:
    production_candidates = [
        candidate
        for candidate in candidate_reports
        if candidate.get("in_production_selection_layer")
    ]
    selected_candidate = next(
        (candidate for candidate in production_candidates if candidate.get("selected")),
        None,
    )
    selected_details = (
        cast(Dict[str, Any], selected_candidate.get("score_details", {}))
        if selected_candidate is not None
        else {}
    )
    selected_key = selected_details.get("selection_key")
    selected_key_values = list(selected_key) if isinstance(selected_key, (list, tuple)) else []
    discordance_values = [
        float(_score_detail_value(candidate, "metal_discordance_count", 0.0))
        for candidate in production_candidates
    ]
    minimum_discordance = min(discordance_values) if discordance_values else None

    candidate_decisions: list[dict[str, Any]] = []
    for candidate in candidate_reports:
        details = cast(Dict[str, Any], candidate.get("score_details", {}))
        raw_key = details.get("selection_key")
        selection_key = list(raw_key) if isinstance(raw_key, (list, tuple)) else []
        selection_values = {
            key: _score_detail_value(candidate, key, None)
            for key, _label, _description in _METAL_SELECTION_KEY_FIELDS
        }
        discordance_details = {
            key: _score_detail_value(candidate, key, None)
            for key in (
                "metal_discordance_count",
                "metal_discordance_structural_count",
                "metal_discordance_inner_visible_diradical_count",
                "metal_discordance_excess_visible_singlet_two_electron_center_count",
                "metal_discordance_bent_cumulated_ring_allene_count",
                "metal_discordance_outer_or_invisible_adjacent_double_charge_count",
                "metal_discordance_inner_visible_adjacent_carbanion_pair_count",
                "metal_discordance_inner_visible_conjugated_carbanion_pair_count",
                "metal_discordance_inner_visible_same_sign_charge_count",
                "metal_discordance_negative_metal_count",
                "metal_discordance_negative_metal_penalty",
                "metal_discordance_zero_valent_metals_with_organic_cation_count",
                "metal_discordance_unsaturated_organic_cation_count",
                "metal_discordance_repeated_component_charge_asymmetry_count",
                "metal_discordance_haptic_arene_reduction_count",
                "metal_discordance_coordination_geometry_count",
                "metal_discordance_outer_or_invisible_adjacent_same_sign_double_charge_count",
                "metal_discordance_outer_or_invisible_adjacent_opposite_sign_double_charge_count",
                "metal_discordance_outer_or_invisible_adjacent_unknown_metal_sign_double_charge_count",
                "metal_discordance_negative_metal_outer_sphere_cation_exception",
                "metal_discordance_negative_metal_positive_metal_counterion_exception",
            )
        }
        decision = ""
        decisive_field = ""
        candidate_value: Any = None
        selected_value: Any = None
        if not candidate.get("in_production_selection_layer"):
            decision = "analysis_only_later_layer"
        elif candidate.get("selected"):
            decision = "selected_lexicographic_minimum"
        elif not _score_detail_value(candidate, "passes_metal_discordance_filter", False):
            decision = "rejected_by_discordance_filter"
            decisive_field = "metal_discordance_count"
            candidate_value = selection_values[decisive_field]
            selected_value = minimum_discordance
        elif not selection_key or not selected_key_values:
            decision = "missing_selection_key"
        else:
            difference_index = next(
                (
                    index
                    for index, (value, best_value) in enumerate(
                        zip(selection_key, selected_key_values)
                    )
                    if value != best_value
                ),
                None,
            )
            if difference_index is None:
                decision = "selection_key_tie_not_selected"
            else:
                decisive_field = _METAL_SELECTION_KEY_FIELDS[difference_index][0]
                candidate_value = selection_key[difference_index]
                selected_value = selected_key_values[difference_index]
                try:
                    candidate_is_better = candidate_value < selected_value
                except TypeError:
                    candidate_is_better = False
                decision = (
                    "selection_inconsistency_candidate_beats_selected"
                    if candidate_is_better
                    else "rejected_by_lexicographic_key"
                )

        candidate_decisions.append(
            {
                "candidate_index": candidate.get("candidate_index"),
                "combination_index": candidate.get("combination_index"),
                "search_layer_index": candidate.get("search_layer_index"),
                "selected": bool(candidate.get("selected")),
                "in_production_selection_layer": bool(
                    candidate.get("in_production_selection_layer")
                ),
                "metal_states": candidate.get("metal_states", []),
                "metal_assignment_rank": _score_detail_value(
                    candidate, "metal_assignment_rank", None
                ),
                "passes_metal_discordance_filter": _score_detail_value(
                    candidate, "passes_metal_discordance_filter", None
                ),
                "selection_key": selection_key,
                "selection_values": selection_values,
                "discordance_details": discordance_details,
                "decision": decision,
                "decisive_field": decisive_field,
                "candidate_value": candidate_value,
                "selected_value": selected_value,
            }
        )

    return {
        "algorithm": "first_scored_layer_then_min_discordance_then_lexicographic_minimum",
        "selected_layer_index": selected_layer_index,
        "production_candidate_count": len(production_candidates),
        "minimum_metal_discordance_count": minimum_discordance,
        "selected_candidate_index": (
            selected_candidate.get("candidate_index") if selected_candidate is not None else None
        ),
        "selected_combination_index": (
            selected_candidate.get("combination_index") if selected_candidate is not None else None
        ),
        "selected_metal_states": (
            selected_candidate.get("metal_states", []) if selected_candidate is not None else []
        ),
        "selected_selection_key": selected_key_values,
        "selection_key_fields": [
            {
                "priority": priority,
                "key": key,
                "label": label,
                "direction": "min",
                "description": description,
            }
            for priority, (key, label, description) in enumerate(
                _METAL_SELECTION_KEY_FIELDS,
                start=1,
            )
        ],
        "candidate_decisions": candidate_decisions,
        "assignment_rank_policy": {
            "participates_in_final_selection_key": False,
            "use": "search_order_layering_and_per_target_pruning",
            "prior_valence_penalty": 0.0,
            "minor_valence_penalty": 10.0,
            "other_valence_penalty": 20.0,
            "nonpositive_valence_penalty": "10 * max(abs(valence), 1)",
        },
    }


def _target_summary(
    target: tuple[int, int],
    candidates: Sequence[MetalCandidateState],
) -> dict[str, Any]:
    return {
        "target": {
            "no_metal_charge": int(target[0]),
            "no_metal_radical_electrons": int(target[1]),
        },
        "candidate_count": len(candidates),
        "prepared_candidate_count": 0,
        "status": "pending",
    }


def _oxidative_addition_valence_deltas(
    selected: MetalCandidateState,
    candidate: MetalCandidateState,
) -> dict[int, int] | None:
    """Return same-site metal valence deltas for an oxidative-addition pair."""

    selected_by_idx = {int(state.idx): int(state.valence) for state in selected.metal_states}
    candidate_by_idx = {int(state.idx): int(state.valence) for state in candidate.metal_states}
    if not selected_by_idx or selected_by_idx.keys() != candidate_by_idx.keys():
        return None
    deltas = {
        atom_idx: candidate_by_idx[atom_idx] - selected_valence
        for atom_idx, selected_valence in selected_by_idx.items()
    }
    if not any(abs(delta) == 2 for delta in deltas.values()):
        return None
    if any(delta != 0 and abs(delta) != 2 for delta in deltas.values()):
        return None
    return deltas


def _additional_negative_charge_visibility(
    selected: MetalCandidateState,
    candidate: MetalCandidateState,
    *,
    config: MolGRConfig | None,
) -> dict[str, Any]:
    """Require charge gained by the selected structure to be visible to a metal."""

    result: dict[str, Any] = {
        "required": False,
        "passed": True,
        "selected_negative_charge_total": 0,
        "candidate_negative_charge_total": 0,
        "additional_negative_charge_atom_indices": [],
        "visible_additional_negative_charge_atom_indices": [],
        "invisible_additional_negative_charge_atom_indices": [],
    }
    if selected.no_metal_state is None or candidate.no_metal_state is None:
        result["passed"] = False
        result["reason"] = "no_metal_state_missing"
        return result

    selected_obmol = cast(ob.OBMol, selected.no_metal_state.omol.OBMol)
    candidate_obmol = cast(ob.OBMol, candidate.no_metal_state.omol.OBMol)
    selected_atoms = {
        int(atom.GetIdx()): atom for atom in scoring._non_metal_atom_entries(selected_obmol)
    }
    candidate_atoms = {
        int(atom.GetIdx()): atom for atom in scoring._non_metal_atom_entries(candidate_obmol)
    }
    if selected_atoms.keys() != candidate_atoms.keys() or any(
        int(selected_atoms[idx].GetAtomicNum()) != int(candidate_atoms[idx].GetAtomicNum())
        for idx in selected_atoms
    ):
        result["passed"] = False
        result["reason"] = "candidate_atom_mapping_mismatch"
        return result

    selected_negative_total = sum(
        max(0, -int(atom.GetFormalCharge())) for atom in selected_atoms.values()
    )
    candidate_negative_total = sum(
        max(0, -int(atom.GetFormalCharge())) for atom in candidate_atoms.values()
    )
    result["selected_negative_charge_total"] = selected_negative_total
    result["candidate_negative_charge_total"] = candidate_negative_total
    if selected_negative_total <= candidate_negative_total:
        result["reason"] = "selected_structure_has_no_additional_negative_charge"
        return result

    additional_indices = sorted(
        idx
        for idx, selected_atom in selected_atoms.items()
        if int(selected_atom.GetFormalCharge()) < int(candidate_atoms[idx].GetFormalCharge())
        and int(selected_atom.GetFormalCharge()) < 0
    )
    result["required"] = True
    result["additional_negative_charge_atom_indices"] = additional_indices

    resolved_config = CONFIG if config is None else config
    blocker_arrays = scoring._build_coordination_blocker_arrays(
        selected_obmol,
        metal_scoring_config=resolved_config.metal_scoring,
    )
    non_metal_atoms = tuple(selected_atoms.values())
    visible_indices: set[int] = set()
    for metal_state in selected.metal_states:
        visible_indices.update(
            int(atom.GetIdx())
            for atom in scoring._inner_visible_atoms_to_metal(
                non_metal_atoms,
                metal_state,
                metal_scoring_config=resolved_config.metal_scoring,
                blocker_arrays=blocker_arrays,
            )
        )

    visible_additional = sorted(set(additional_indices) & visible_indices)
    invisible_additional = sorted(set(additional_indices) - visible_indices)
    result["visible_additional_negative_charge_atom_indices"] = visible_additional
    result["invisible_additional_negative_charge_atom_indices"] = invisible_additional
    result["passed"] = not invisible_additional
    result["reason"] = (
        "additional_negative_charge_is_metal_visible"
        if result["passed"]
        else "additional_negative_charge_is_not_metal_visible"
    )
    return result


def _oxidative_addition_reference_match(
    candidates: Sequence[MetalCandidateState],
    selected_candidate: MetalCandidateState | None,
    reference_smiles: str,
    *,
    config: MolGRConfig | None,
) -> dict[str, Any]:
    """Find a reference-equivalent candidate separated by metal valence +/-2."""

    result: dict[str, Any] = {
        "matched": False,
        "recommended_review_status": "",
        "reason": "",
        "matching_candidates": [],
    }
    if selected_candidate is None:
        result["reason"] = "no_selected_candidate"
        return result
    if not reference_smiles.strip():
        result["reason"] = "reference_smiles_missing"
        return result
    reference = _mol_from_unsanitized_smiles(reference_smiles)
    if reference is None:
        result["reason"] = "reference_smiles_invalid"
        return result

    selected_identity = _candidate_identity(selected_candidate, fallback_candidate_index=-1)
    comparison_reasons: list[str] = []
    matching_candidates: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    for fallback_index, candidate in enumerate(candidates):
        candidate_identity = _candidate_identity(
            candidate,
            fallback_candidate_index=fallback_index,
        )
        if candidate_identity == selected_identity:
            continue
        deltas = _oxidative_addition_valence_deltas(selected_candidate, candidate)
        if deltas is None:
            continue
        try:
            combined = candidate.materialize_combined_omol(preparation.combine_metal_with_omol)
            candidate_mol = pybel_to_rdmol(combined, sanitize=False, kekulize=False)
            candidate_mol = make_dative_bond(candidate_mol, config=config)
            equivalent, info = check_equivalence(
                reference,
                candidate_mol,
                use_chirality=False,
            )
        except Exception as exc:  # noqa: BLE001
            comparison_reasons.append(f"C{candidate_identity[1]}:{type(exc).__name__}:{exc}")
            continue
        if not equivalent:
            comparison_reasons.append(f"C{candidate_identity[1]}:{info.reason or 'not_equivalent'}")
            continue
        negative_charge_visibility = _additional_negative_charge_visibility(
            selected_candidate,
            candidate,
            config=config,
        )
        if not bool(negative_charge_visibility["passed"]):
            comparison_reasons.append(
                f"C{candidate_identity[1]}:{negative_charge_visibility['reason']}"
            )
            rejected_candidates.append(
                {
                    "search_layer_index": candidate_identity[0],
                    "combination_index": candidate_identity[1],
                    "valence_deltas_by_atom": {
                        str(atom_idx): delta for atom_idx, delta in sorted(deltas.items())
                    },
                    "negative_charge_visibility": negative_charge_visibility,
                }
            )
            continue
        matching_candidates.append(
            {
                "search_layer_index": candidate_identity[0],
                "combination_index": candidate_identity[1],
                "metal_states": [
                    _metal_state_to_dict(metal_state) for metal_state in candidate.metal_states
                ],
                "valence_deltas_by_atom": {
                    str(atom_idx): delta for atom_idx, delta in sorted(deltas.items())
                },
                "equivalence_method": info.method.value if info.method else "",
                "equivalence_reason": info.reason,
                "negative_charge_visibility": negative_charge_visibility,
                "candidate_smiles": _safe_candidate_smiles(candidate_mol),
            }
        )

    result["matching_candidates"] = matching_candidates
    result["rejected_candidates"] = rejected_candidates
    if matching_candidates:
        result["matched"] = True
        result["recommended_review_status"] = "accept_both"
        result["reason"] = "reference_matches_candidate_with_metal_valence_delta_plus_or_minus_2"
    else:
        result["reason"] = "no_reference_equivalent_plus_or_minus_2_valence_candidate"
        if comparison_reasons:
            result["comparison_reasons"] = comparison_reasons
    return result


def classify_oxidative_addition_reference_match(
    xyz_block: str,
    *,
    total_charge: int,
    total_radical_electrons: int,
    reference_smiles: str,
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    """Run the +/-2 reference check without building browser trace details."""

    base_state = preparation.prepare_metal_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    state_search_groups = search._build_metal_state_search_groups(
        base_state.available_valence_radical_states,
        config=config,
    )
    layered_state_search_groups = search._build_layered_metal_state_search_groups(
        state_search_groups,
        total_radical_electrons,
        config=config,
    )

    all_scored_candidates: list[MetalCandidateState] = []
    selected_candidate: MetalCandidateState | None = None
    for layer_index, available_valence_radical_states in enumerate(layered_state_search_groups):
        grouped_candidates = search._group_candidates_by_target_dp(
            base_state.phase_history,
            available_valence_radical_states,
            total_charge,
            total_radical_electrons,
            config=config,
        )
        current_layer_scored_candidates: list[MetalCandidateState] = []
        for candidates in grouped_candidates.values():
            if not candidates:
                continue
            prototype = candidates[0]
            try:
                no_metal_state = reconstruct_without_metals.xyz_to_omol_no_metal_state(
                    base_state.no_metal_xyz_block,
                    prototype.no_metal_charge_target,
                    prototype.no_metal_radical_target,
                    config=config,
                )
            except (OSError, ValueError):
                continue
            if no_metal_state is None:
                continue
            for candidate in candidates:
                try:
                    scored_candidate = scoring._prepare_candidate_with_no_metal_state(
                        candidate,
                        no_metal_state,
                        config=config,
                    )
                except ValueError:
                    continue
                if cast(Optional[float], scored_candidate.score) is None:
                    continue
                scored_candidate.metadata["search_layer_index"] = layer_index
                current_layer_scored_candidates.append(scored_candidate)

        all_scored_candidates.extend(current_layer_scored_candidates)
        if selected_candidate is None and current_layer_scored_candidates:
            selected_candidate = scoring.select_best_candidate(
                current_layer_scored_candidates,
                config=config,
            )

    return _oxidative_addition_reference_match(
        all_scored_candidates,
        selected_candidate,
        reference_smiles,
        config=config,
    )


def _trace_candidates(
    xyz_block: str,
    *,
    total_charge: int,
    total_radical_electrons: int,
    score_all_candidates: bool,
    render_context: DofRenderContext | None = None,
    case_id: str = "",
    reference_smiles: str = "",
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    base_state = preparation.prepare_metal_state(
        xyz_block,
        total_charge,
        total_radical_electrons,
        config=config,
    )
    field_analysis_by_state = _metal_field_analysis_by_state(
        xyz_block,
        base_state.available_valence_radical_states,
        config=config,
    )
    state_search_groups = search._build_metal_state_search_groups(
        base_state.available_valence_radical_states,
        config=config,
    )
    layered_state_search_groups = search._build_layered_metal_state_search_groups(
        state_search_groups,
        total_radical_electrons,
        config=config,
    )

    layer_summaries: list[dict[str, Any]] = []
    all_scored_candidates: list[MetalCandidateState] = []
    production_scored_candidates: list[MetalCandidateState] = []
    scored_candidates_by_layer: dict[int, list[MetalCandidateState]] = {}
    selected_layer_index: Optional[int] = None

    for layer_index, available_valence_radical_states in enumerate(layered_state_search_groups):
        grouped_candidates = search._group_candidates_by_target_dp(
            base_state.phase_history,
            available_valence_radical_states,
            total_charge,
            total_radical_electrons,
            config=config,
        )
        layer_summary: dict[str, Any] = {
            "layer_index": layer_index,
            "state_group_count": len(available_valence_radical_states),
            "state_options_per_group": [
                len(state_options) for state_options in available_valence_radical_states
            ],
            "target_bucket_count": len(grouped_candidates),
            "candidate_count": sum(len(candidates) for candidates in grouped_candidates.values()),
            "prepared_candidate_count": 0,
            "target_buckets": [],
            "status": "no_candidate_targets" if not grouped_candidates else "pending",
        }
        if not grouped_candidates:
            layer_summaries.append(layer_summary)
            continue

        current_layer_scored_candidates: list[MetalCandidateState] = []
        for target, candidates in grouped_candidates.items():
            target_summary = _target_summary(target, candidates)
            if not candidates:
                target_summary["status"] = "empty_candidate_bucket"
                layer_summary["target_buckets"].append(target_summary)
                continue

            prototype = candidates[0]
            try:
                target_summary["no_metal_trace"] = _trace_no_metal_reconstruction(
                    base_state.no_metal_xyz_block,
                    prototype.no_metal_charge_target,
                    prototype.no_metal_radical_target,
                    render_context=render_context,
                    case_id=(
                        f"{case_id}_L{layer_index}_Q{prototype.no_metal_charge_target}"
                        f"_R{prototype.no_metal_radical_target}"
                    ),
                    config=config,
                )
            except Exception as exc:
                target_summary["no_metal_trace"] = {
                    "status": "trace_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            try:
                no_metal_state = reconstruct_without_metals.xyz_to_omol_no_metal_state(
                    base_state.no_metal_xyz_block,
                    prototype.no_metal_charge_target,
                    prototype.no_metal_radical_target,
                    config=config,
                )
            except (OSError, ValueError) as exc:
                target_summary["status"] = "no_metal_reconstruction_error"
                target_summary["error"] = str(exc)
                layer_summary["target_buckets"].append(target_summary)
                continue

            if no_metal_state is None:
                target_summary["status"] = "no_metal_reconstruction_none"
                # Failed targets still carry the complete state-machine trace and
                # must remain in the same collection consumed by every renderer.
                layer_summary["target_buckets"].append(target_summary)
                continue

            target_summary["status"] = "prepared"
            target_summary["organic_part"] = _omol_smiles_detail(no_metal_state.omol)
            target_summary["no_metal_phase_history"] = list(no_metal_state.phase_history)
            organic_image = _render_dof_molecule(
                no_metal_state.omol,
                render_context=render_context,
                case_id=case_id,
                label=(
                    f"L{layer_index}_organic_Q{prototype.no_metal_charge_target}"
                    f"_R{prototype.no_metal_radical_target}"
                ),
                kind="metal_target_organic_part",
            )
            if organic_image is not None:
                target_summary["dof_image"] = organic_image

            for candidate in candidates:
                try:
                    scored_candidate = scoring._prepare_candidate_with_no_metal_state(
                        candidate,
                        no_metal_state,
                        config=config,
                    )
                except ValueError as exc:
                    target_summary.setdefault("candidate_errors", []).append(
                        {
                            "combination_index": candidate.metadata.get("combination_index"),
                            "error": str(exc),
                        }
                    )
                    continue
                if cast(Optional[float], scored_candidate.score) is None:
                    continue
                scored_candidate.metadata["search_layer_index"] = layer_index
                current_layer_scored_candidates.append(scored_candidate)

            target_summary["no_metal_score"] = no_metal_state.metadata.get("score")
            target_summary["prepared_candidate_count"] = len(
                [
                    candidate
                    for candidate in current_layer_scored_candidates
                    if candidate.no_metal_charge_target == target[0]
                    and candidate.no_metal_radical_target == target[1]
                ]
            )
            layer_summary["target_buckets"].append(target_summary)

        layer_summary["prepared_candidate_count"] = len(current_layer_scored_candidates)
        if current_layer_scored_candidates:
            scored_candidates_by_layer[layer_index] = current_layer_scored_candidates
            all_scored_candidates.extend(current_layer_scored_candidates)
            if selected_layer_index is None:
                selected_layer_index = layer_index
                production_scored_candidates = current_layer_scored_candidates
                layer_summary["status"] = "production_scoring_layer"
                layer_summary["production_selected_layer"] = True
            else:
                layer_summary["status"] = "additional_scored_layer"
                layer_summary["production_selected_layer"] = False
            layer_summaries.append(layer_summary)
            continue

        layer_summary["status"] = "no_scored_candidates"
        layer_summary["production_selected_layer"] = False
        layer_summaries.append(layer_summary)

    selected_candidate = None
    production_metadata_by_identity: dict[tuple[int, int], dict[str, Any]] = {}
    analysis_score_context: dict[str, Any] = {}
    selected_candidate_identity: Optional[tuple[int, int]] = None
    if production_scored_candidates:
        selected_candidate = scoring.select_best_candidate(
            production_scored_candidates, config=config
        )
        production_metadata_by_identity = _copy_metadata_by_identity(production_scored_candidates)
        if selected_candidate is not None:
            selected_candidate_identity = _candidate_identity(
                selected_candidate,
                fallback_candidate_index=-1,
            )

        if score_all_candidates:
            analysis_score_context = {
                str(layer_index): _annotate_analysis_scores_for_all_candidates(
                    layer_candidates,
                    config=config,
                )
                for layer_index, layer_candidates in sorted(scored_candidates_by_layer.items())
            }

    if score_all_candidates and analysis_score_context:
        for layer_summary in layer_summaries:
            layer_context = analysis_score_context.get(str(layer_summary["layer_index"]))
            if layer_context is not None:
                layer_summary["analysis_score_context"] = _jsonable(layer_context)

    candidate_reports = [
        _candidate_report(
            candidate,
            candidate_index=candidate_index,
            selected_candidate_identity=selected_candidate_identity,
            production_metadata_by_identity=production_metadata_by_identity,
            render_context=render_context,
            case_id=case_id,
            field_analysis_by_state=field_analysis_by_state,
        )
        for candidate_index, candidate in enumerate(all_scored_candidates)
    ]
    candidate_grid_items: list[tuple[pybel.Molecule, str]] = []
    for candidate_index, candidate in enumerate(all_scored_candidates):
        if candidate.no_metal_state is None:
            continue
        try:
            candidate_identity = _candidate_identity(
                candidate,
                fallback_candidate_index=candidate_index,
            )
            candidate_grid_items.append(
                (
                    candidate.materialize_combined_omol(preparation.combine_metal_with_omol),
                    (
                        f"{'selected ' if selected_candidate_identity == candidate_identity else ''}"
                        f"L{candidate_identity[0]}/C{candidate_identity[1]} "
                        f"score={_format_value(candidate.score)}"
                    ),
                )
            )
        except Exception:
            continue
    candidate_grid = _render_dof_grid(
        candidate_grid_items,
        render_context=render_context,
        case_id=case_id,
        label="metal_candidates",
        kind="metal_candidate_grid",
    )

    selected_summary = None
    if selected_candidate_identity is not None:
        selected_summary = next(
            (
                {
                    "candidate_index": candidate["candidate_index"],
                    "combination_index": candidate["combination_index"],
                    "target": candidate["target"],
                    "metal_states": candidate["metal_states"],
                    "organic_part": candidate["organic_part"],
                    "score": candidate["score"],
                    "score_details": candidate["score_details"],
                }
                for candidate in candidate_reports
                if (
                    candidate["candidate_identity"]["search_layer_index"],
                    candidate["candidate_identity"]["combination_index"],
                )
                == selected_candidate_identity
            ),
            None,
        )
        if selected_summary is not None and selected_candidate is not None:
            try:
                selected_omol = selected_candidate.materialize_combined_omol(
                    preparation.combine_metal_with_omol
                )
                selected_summary["graph"] = _omol_smiles_detail(selected_omol)
            except Exception as exc:
                selected_summary["graph_error"] = f"{type(exc).__name__}: {exc}"
    selected_path_animation = _render_selected_metal_path_animation(
        selected_candidate,
        no_metal_xyz_block=base_state.no_metal_xyz_block,
        render_context=render_context,
        case_id=case_id,
    )
    metal_selection_basis = _metal_selection_basis(
        candidate_reports,
        selected_layer_index=selected_layer_index,
    )
    oxidative_addition_match = _oxidative_addition_reference_match(
        all_scored_candidates,
        selected_candidate,
        reference_smiles,
        config=config,
    )

    return {
        "status": "ok" if production_scored_candidates else "no_scored_metal_candidates",
        "base_state": {
            "metal_atom_count": int(base_state.metadata.get("metal_atom_count", 0)),
            "available_metal_states_by_site": [
                [
                    _metal_state_to_dict(
                        metal_state,
                        field_analysis=field_analysis_by_state.get(
                            (int(metal_state.idx), int(metal_state.valence))
                        ),
                    )
                    for metal_state in state_options
                ]
                for state_options in base_state.available_valence_radical_states
            ],
            "no_metal_xyz_block": base_state.no_metal_xyz_block,
            "phase_history": list(base_state.phase_history),
            "metadata": _jsonable(base_state.metadata),
        },
        "search": {
            "state_search_group_count": len(state_search_groups),
            "layer_count": len(layered_state_search_groups),
            "selected_layer_index": selected_layer_index,
            "layer_summaries": _jsonable(layer_summaries),
        },
        "analysis": {
            "score_all_candidates": score_all_candidates,
            "score_context": _jsonable(analysis_score_context),
            "note": (
                "production_metadata records the exact metadata after select_best_candidate; "
                "score_details also includes analysis_* metrics when score-all-candidates is enabled."
            ),
        },
        "candidate_count": len(candidate_reports),
        "production_candidate_count": len(production_scored_candidates),
        "metal_selection_basis": metal_selection_basis,
        "oxidative_addition_reference_match": oxidative_addition_match,
        "selected_candidate": selected_summary,
        "selected_path_animation": selected_path_animation,
        "candidates": candidate_reports,
        "dof_candidate_grid": candidate_grid,
    }


def _assign_global_trace_node_indices(cases: list[dict[str, Any]]) -> None:
    """Assign report-wide stable locators to every inspectable reconstruction node.

    State-machine ``node_id`` values are intentionally local to one no-metal
    recorder, so they cannot identify a node when a report contains multiple
    targets or cases.  The report-wide index is assigned after all cases have
    been built, in deterministic case/trace/candidate order.  Parent links are
    translated at the same time without changing the execution-local ids.
    """

    next_index = 0
    seen_traces: set[int] = set()

    def assign_node(node: dict[str, Any], *, case_id: str, kind: str) -> int:
        nonlocal next_index
        global_index = next_index
        next_index += 1
        node["global_node_index"] = global_index
        node["global_node_kind"] = kind
        prefix = "N" if kind == "no_metal_trace" else "C"
        node["global_node_locator"] = f"{case_id}:{prefix}{global_index:06d}"
        return global_index

    for case in cases:
        case_start_index = next_index
        case_id = str(case.get("id", "unknown"))
        no_metal_traces: list[dict[str, Any]] = []
        direct_trace = case.get("no_metal_trace")
        if isinstance(direct_trace, dict):
            no_metal_traces.append(direct_trace)
        search = case.get("search")
        if isinstance(search, dict):
            for layer in search.get("layer_summaries", []):
                if not isinstance(layer, dict):
                    continue
                for bucket in layer.get("target_buckets", []):
                    if not isinstance(bucket, dict):
                        continue
                    trace = bucket.get("no_metal_trace")
                    if isinstance(trace, dict):
                        no_metal_traces.append(trace)

        for trace in no_metal_traces:
            trace_identity = id(trace)
            if trace_identity in seen_traces:
                continue
            seen_traces.add(trace_identity)
            trace_nodes = trace.get("trace_nodes", [])
            if not isinstance(trace_nodes, list):
                continue
            local_to_global: dict[int, int] = {}
            for node in trace_nodes:
                if not isinstance(node, dict):
                    continue
                local_id = int(node.get("node_id", -1))
                local_to_global[local_id] = assign_node(
                    node,
                    case_id=case_id,
                    kind="no_metal_trace",
                )
            for node in trace_nodes:
                if not isinstance(node, dict):
                    continue
                parent_id = int(node.get("tree_parent_id", -1))
                node["global_tree_parent_index"] = local_to_global.get(parent_id)
            resonance_basis = trace.get("resonance_selection_basis")
            if isinstance(resonance_basis, dict):
                trace_node_by_id = {
                    int(node.get("node_id", -1)): node
                    for node in trace_nodes
                    if isinstance(node, dict)
                }
                for candidate in resonance_basis.get("candidates", []):
                    if not isinstance(candidate, dict):
                        continue
                    trace_node = trace_node_by_id.get(int(candidate.get("trace_node_id", -1)))
                    if trace_node is None:
                        continue
                    candidate["global_node_index"] = trace_node.get("global_node_index")
                    candidate["global_node_locator"] = trace_node.get("global_node_locator")
                    if candidate.get("selected"):
                        resonance_basis["selected_global_node_index"] = trace_node.get(
                            "global_node_index"
                        )
                        resonance_basis["selected_global_node_locator"] = trace_node.get(
                            "global_node_locator"
                        )
            trace["global_trace_node_count"] = sum(
                1 for node in trace_nodes if isinstance(node, dict)
            )

        candidates = case.get("candidates", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                assign_node(candidate, case_id=case_id, kind="metal_candidate")

        case["global_node_index_start"] = (
            case_start_index if next_index > case_start_index else None
        )
        case["global_node_index_end"] = next_index - 1 if next_index > case_start_index else None

    for case in cases:
        case["report_global_node_count"] = next_index


def _metal_atom_count_from_xyz(xyz_block: str) -> int:
    omol = pybel.readstring("xyz", xyz_block)
    return sum(1 for atom in omol.atoms if atom.OBAtom.IsMetal())


def _selected_trace_smiles(trace: dict[str, Any]) -> str:
    selected = trace.get("selected_candidate")
    if not isinstance(selected, dict):
        return ""
    if trace.get("trace_kind") == "no_metal":
        state = selected.get("state")
        if isinstance(state, dict):
            return str(state.get("canonical_smiles") or state.get("smiles") or "")
        return ""
    graph = selected.get("graph")
    if isinstance(graph, dict):
        return str(graph.get("canonical_smiles") or graph.get("smiles") or "")
    return ""


def _mol_from_unsanitized_smiles(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles, sanitize=False) if smiles.strip() else None
    if mol is not None:
        mol.UpdatePropertyCache(strict=False)
    return mol


def _review_fixture_trace_check(
    input_case: TraceInputCase,
    trace: dict[str, Any],
) -> dict[str, Any]:
    trace_smiles = _selected_trace_smiles(trace)
    expected_smiles_options = input_case.expected_smiles_options or (
        (input_case.expected_smiles,) if input_case.expected_smiles else ()
    )
    check: dict[str, Any] = {
        "kind": input_case.fixture_kind,
        "structure_file": input_case.fixture_structure_file,
        "expected_smiles": input_case.expected_smiles,
        "accepted_smiles": list(expected_smiles_options),
        "trace_smiles": trace_smiles,
        "equivalent": None,
        "equivalence_method": "",
        "equivalence_reason": "",
    }
    if not expected_smiles_options:
        check["equivalence_reason"] = "fixture_has_no_approved_answer"
        return check
    if not trace_smiles:
        check["equivalent"] = False
        check["equivalence_reason"] = "trace_has_no_selected_graph"
        return check

    traced = _mol_from_unsanitized_smiles(trace_smiles)
    if traced is None:
        check["equivalent"] = False
        check["equivalence_reason"] = "trace_selected_smiles_invalid"
        return check

    comparison_reasons: list[str] = []
    for option_index, expected_smiles in enumerate(expected_smiles_options):
        expected = _mol_from_unsanitized_smiles(expected_smiles)
        if expected is None:
            comparison_reasons.append("fixture_approved_smiles_invalid")
            continue
        equivalent, info = check_equivalence(expected, traced, use_chirality=False)
        comparison_reasons.append(info.reason)
        if equivalent:
            check["equivalent"] = True
            check["matched_answer_index"] = option_index
            check["equivalence_method"] = info.method.value if info.method else ""
            check["equivalence_reason"] = info.reason
            return check
    check["equivalent"] = False
    check["equivalence_reason"] = "; ".join(comparison_reasons)
    return check


def trace_reconstruction_case(
    input_case: TraceInputCase,
    *,
    score_all_candidates: bool,
    render_context: DofRenderContext | None = None,
    config: MolGRConfig | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    total_charge = int(input_case.total_charge)
    total_radicals = int(input_case.total_radical_electrons)
    metal_atom_count = _metal_atom_count_from_xyz(input_case.xyz_block)

    if metal_atom_count == 0:
        no_metal_trace = _trace_no_metal_reconstruction(
            input_case.xyz_block,
            total_charge,
            total_radicals,
            render_context=render_context,
            case_id=input_case.id,
            config=config,
        )
        trace = {
            "status": no_metal_trace.get("status", "unknown"),
            "trace_kind": "no_metal",
            "base_state": {
                "metal_atom_count": 0,
                "phase_history": ["read_xyz"],
            },
            "no_metal_trace": no_metal_trace,
            "selected_candidate": no_metal_trace.get("selected_candidate"),
            "candidate_count": 1 if no_metal_trace.get("selected_candidate") else 0,
            "production_candidate_count": 1 if no_metal_trace.get("selected_candidate") else 0,
        }
    else:
        trace = _trace_candidates(
            input_case.xyz_block,
            total_charge=total_charge,
            total_radical_electrons=total_radicals,
            score_all_candidates=score_all_candidates,
            render_context=render_context,
            case_id=input_case.id,
            reference_smiles=input_case.reference_smiles,
            config=config,
        )
        trace["trace_kind"] = "metal"

    if input_case.fixture_kind:
        trace["review_fixture"] = _review_fixture_trace_check(input_case, trace)

    trace.update(
        {
            "id": input_case.id,
            "charge": total_charge,
            "xyz_path": str(input_case.xyz_path) if input_case.xyz_path is not None else "",
            "xyz_source": input_case.xyz_source,
            "total_radical_electrons": total_radicals,
            "spin_multiplicity": total_radicals + 1,
            "reference_smiles": input_case.reference_smiles,
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    _assign_global_trace_node_indices([trace])
    return trace


def trace_reconstruction_cases(
    input_cases: Sequence[TraceInputCase],
    *,
    score_all_candidates: bool = True,
    render_context: DofRenderContext | None = None,
    config: MolGRConfig | None = None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for input_case in input_cases:
        started = time.perf_counter()
        try:
            cases.append(
                trace_reconstruction_case(
                    input_case,
                    score_all_candidates=score_all_candidates,
                    render_context=render_context,
                    config=config,
                )
            )
        except Exception as exc:
            cases.append(
                {
                    "id": input_case.id,
                    "status": "error",
                    "trace_kind": "unknown",
                    "charge": input_case.total_charge,
                    "total_radical_electrons": input_case.total_radical_electrons,
                    "spin_multiplicity": input_case.total_radical_electrons + 1,
                    "xyz_path": str(input_case.xyz_path) if input_case.xyz_path is not None else "",
                    "xyz_source": input_case.xyz_source,
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
    _assign_global_trace_node_indices(cases)
    return cases


def render_trace_report(
    input_cases: Sequence[TraceInputCase],
    *,
    score_all_candidates: bool = True,
    dof_max_images: int | None = 1000,
    defer_dof_images: bool = False,
    config: MolGRConfig | None = None,
    case_observer: Callable[[dict[str, Any]], None] | None = None,
) -> str:
    """Build the browser trace report for already-resolved input cases.

    This is the shared programmatic entry point for browser-facing callers.
    Keeping report assembly here prevents the development UI from maintaining
    a second trace renderer or case representation.
    """

    render_context = DofRenderContext(
        image_dir=Path("molgr_trace_dof_images"),
        display_base_dir=None,
        image_format="svg",
        defer_images=defer_dof_images,
        max_images=dof_max_images,
    )
    cases = trace_reconstruction_cases(
        input_cases,
        score_all_candidates=score_all_candidates,
        render_context=render_context,
        config=config,
    )
    if case_observer is not None:
        for case in cases:
            case_observer(case)
    output: dict[str, Any] = {
        "input": {
            "source": "review_page",
            "ids": [case.id for case in input_cases],
        },
        "case_count": len(cases),
        "global_node_count": max(
            (int(case.get("report_global_node_count", 0)) for case in cases),
            default=0,
        ),
        "cases": cases,
        "dof_rendering": dof_rendering_summary(render_context),
    }
    if defer_dof_images:
        output["cases"], output["dof_payloads"] = _extract_deferred_dof_payloads(output["cases"])
    return _render_html_browser_report(_jsonable(output))


def main() -> int:
    args = _parse_args()
    config: MolGRConfig | None = None
    input_cases = _collect_input_cases(args)
    render_context = _make_dof_render_context(args)
    cases = trace_reconstruction_cases(
        input_cases,
        score_all_candidates=not args.no_score_all_candidates,
        render_context=render_context,
        config=config,
    )

    output = {
        "input": {
            "source": (
                "review_fixtures" if args.review_fixtures_manifest is not None else "generic"
            ),
            "ids": [case.id for case in input_cases],
            "total_charge": args.total_charge,
            "total_radical_electrons": _direct_total_radicals(args),
        },
        "case_count": len(cases),
        "global_node_count": max(
            (int(case.get("report_global_node_count", 0)) for case in cases),
            default=0,
        ),
        "cases": cases,
    }
    if render_context is not None:
        output["dof_rendering"] = dof_rendering_summary(render_context)
    output_format = _resolve_output_format(args)
    if output_format == "json":
        output_text = json.dumps(
            _jsonable(output),
            ensure_ascii=False,
            indent=None if args.indent == 0 else args.indent,
            allow_nan=False,
        )
    else:
        output_text = _render_html_browser_report(_jsonable(output))

    if args.out is None:
        print(output_text, end="" if output_text.endswith("\n") else "\n")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            output_text if output_text.endswith("\n") else output_text + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
