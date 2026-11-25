/*
 * @Author: TMJ
 * @Date: 2026-01-01 23:15:31
 * @LastEditors: TMJ
 * @LastEditTime: 2026-01-01 23:28:31
 * @Description: 请填写简介
 */
#include "bindings.h"
#include "molgr/python_config.h"
#include "molgr/utils/consts.h"
#include "molgr/utils/force_field.h"
#include "molgr/utils/scoring.h"
#include "molgr/utils/conversions.h"
#include "molgr/utils/stereo.h"
#include "molgr/utils/organic_topology.h"
#include "molgr/utils/utils.h"
#include "molgr/utils/xyz.h"
#include "molgr/vendor/forcefielduff.h"
#include "molgr/vendor/openbabel_conversion.h"
#include "molgr/vendor/openbabel_threading.h"
#include "molgr/utils/logger.h"

#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <pybind11/stl.h> // 必须包含
#include <pybind11/numpy.h>

#include <array>
#include <limits>
#include <memory>
#include <stdexcept>

/// Parse one SMILES string into an independently owned OBMol; return null on reader failure.
/// @param smiles SMILES representation of the input molecular graph.
/// @return Newly owned molecule, or null on a parse failure.
static std::unique_ptr<OpenBabel::OBMol> mol_from_smiles(const std::string &smiles)
{
    auto mol = std::make_unique<OpenBabel::OBMol>();
    return molgr::vendor::openbabel_conversion::ReadSmiles(smiles, mol.get())
               ? std::move(mol)
               : nullptr;
}

/// Parse an XYZ block into an owned molecular graph, returning null if the coordinates cannot be read.
/// @param xyz XYZ text containing the declared atom count and Cartesian coordinates in angstroms.
/// @return Newly owned molecule, or null on a parse failure.
static std::unique_ptr<OpenBabel::OBMol> mol_from_xyz(const std::string &xyz)
{
    auto mol = std::make_unique<OpenBabel::OBMol>();
    return molgr::utils::ReadXyzBlockToMol(xyz, mol.get()) ? std::move(mol) : nullptr;
}

/// @param xyz_block XYZ text containing the declared atom count and Cartesian coordinates in angstroms.
/// @return The native XYZ seed topology as transferable molecule records; reject malformed XYZ.
static molgr::utils::MoleculeData xyz_seed_molecule_data(const std::string &xyz_block)
{
    OpenBabel::OBMol mol;
    py::gil_scoped_release release;
    if (!molgr::utils::ReadXyzBlockToMol(xyz_block, &mol))
    {
        throw std::invalid_argument("failed to parse XYZ block");
    }
    return molgr::utils::MoleculeDataFromOBMol(mol);
}

void molgr::bind::bind_utils(py::module_ &m)
{
    m.def("xyz_seed_molecule_data", &xyz_seed_molecule_data,
          "Read XYZ using the shared seed bond-perception implementation. Returns owned data, not an OBMol pointer.",
          py::arg("xyz_block"));
    m.def("get_possible_metal_radicals", &molgr::GetPossibleMetalRadicals,
          R"pbdoc(
            Get possible radical electron counts for a metal given its valence.

            Args:
                metal (str): The chemical symbol (e.g., "Fe").
                valence (int): The oxidation state.

            Returns:
                set[int]: A set of possible unpaired electron counts.
        )pbdoc",
          py::arg("metal"), py::arg("valence"));

    m.def("calculate_tetrahedron_volume", [](const std::vector<double> &p1, const std::vector<double> &p2, const std::vector<double> &p3, const std::vector<double> &p4)
          { return molgr::utils::CalculateTetrahedronVolume(
                molgr::utils::ToVector3(p1),
                molgr::utils::ToVector3(p2),
                molgr::utils::ToVector3(p3),
                molgr::utils::ToVector3(p4)); },
          R"pbdoc(
            Calculate the volume of a tetrahedron defined by 4 points.
            
            Parameters:
                p1 (list[float]): Coordinates of the first atom.
                p2 (list[float]): Coordinates of the second atom.
                p3 (list[float]): Coordinates of the third atom.
                p4 (list[float]): Coordinates of the fourth atom.
            
            Returns:
                float: The volume.
        )pbdoc",
          py::arg("p1"), py::arg("p2"), py::arg("p3"), py::arg("p4"));

    m.def("calculate_shape_quality", [](const std::vector<double> &p1, const std::vector<double> &p2, const std::vector<double> &p3, const std::vector<double> &p4)
          { return molgr::utils::CalculateShapeQuality(
                molgr::utils::ToVector3(p1),
                molgr::utils::ToVector3(p2),
                molgr::utils::ToVector3(p3),
                molgr::utils::ToVector3(p4)); },
          R"pbdoc(
            Calculate the shape quality score of a tetrahedron.
            
            Parameters:
                p1 (list[float]): Coordinates of the first atom.
                p2 (list[float]): Coordinates of the second atom.
                p3 (list[float]): Coordinates of the third atom.
                p4 (list[float]): Coordinates of the fourth atom.
            
            Returns:
                float: Quality score between 0.0 (coplanar/bad) and 1.0 (ideal).
        )pbdoc",
          py::arg("p1"), py::arg("p2"), py::arg("p3"), py::arg("p4"));

    py::class_<molgr::utils::AtomData>(m, "AtomData",
        "Element, isotope, Cartesian coordinates and explicit electron labels for one graph atom.")
        .def(py::init<>())
        .def_readwrite("atomic_num", &molgr::utils::AtomData::atomic_num,
                       "Atomic number of the element.")
        .def_readwrite("isotope", &molgr::utils::AtomData::isotope,
                       "Isotope mass number, or zero for the default isotope.")
        .def_readwrite("formal_charge", &molgr::utils::AtomData::formal_charge,
                       "Signed formal charge in elementary-charge units.")
        .def_readwrite("radical_num", &molgr::utils::AtomData::radical_num,
                       "Explicit count of unpaired electrons on this atom.")
        .def_readwrite("lone_pair_count", &molgr::utils::AtomData::lone_pair_count,
                       "Number of reconstruction-active lone pairs tracked on this atom.")
        .def_readwrite("unresolved_two_electron_center", &molgr::utils::AtomData::unresolved_two_electron_center,
                       "Whether two electrons still require a singlet or triplet assignment.")
        .def_readwrite("electron_property_presence", &molgr::utils::AtomData::electron_property_presence,
                       "Presence mask for unpaired electrons (1), active lone pairs (2) and unresolved occupancy (4).")
        .def_readwrite("hybridization", &molgr::utils::AtomData::hybridization,
                       "Hybridization code carried by the Open Babel atom.")
        .def_readwrite("x", &molgr::utils::AtomData::x,
                       "Cartesian x coordinate in angstroms.")
        .def_readwrite("y", &molgr::utils::AtomData::y,
                       "Cartesian y coordinate in angstroms.")
        .def_readwrite("z", &molgr::utils::AtomData::z,
                       "Cartesian z coordinate in angstroms.")
        .def("__repr__", [](const molgr::utils::AtomData &a)
             { return "<AtomData Z=" + std::to_string(a.atomic_num) +
                      " charge=" + std::to_string(a.formal_charge) +
                      " radical_num=" + std::to_string(a.radical_num) +
                      " lone_pair_count=" + std::to_string(a.lone_pair_count) +
                      " unresolved_two_electron_center=" +
                      (a.unresolved_two_electron_center ? "true" : "false") +
                      " hyb=" + std::to_string(a.hybridization) +
                      " pos=(" + std::to_string(a.x) + "," +
                      std::to_string(a.y) + "," + std::to_string(a.z) + ")>"; },
             "Return a concise debug representation of AtomData.");

    py::enum_<molgr::utils::BondKind>(m, "BondKind",
        "Distinguishes covalent order, directed dative donation and unspecified coordination.")
        .value("Covalent", molgr::utils::BondKind::Covalent)
        .value("Dative", molgr::utils::BondKind::Dative)
        .value("CoordinationUnspecified", molgr::utils::BondKind::CoordinationUnspecified);

    py::class_<molgr::utils::BondData>(m, "BondData",
        "An edge with one-based atom indices. A dative edge runs from donor to acceptor.")
        .def(py::init<>())
        .def_readwrite("begin_atom_idx", &molgr::utils::BondData::begin_atom_idx,
                       "One-based start atom index, the donor for a dative edge.")
        .def_readwrite("end_atom_idx", &molgr::utils::BondData::end_atom_idx,
                       "One-based end atom index, the acceptor for a dative edge.")
        .def_readwrite("order", &molgr::utils::BondData::order,
                       "Integer covalent bond order. Coordination uses the kind field.")
        .def_readwrite("aromatic", &molgr::utils::BondData::aromatic,
                       "Whether the source graph marked this bond aromatic.")
        .def_readwrite("kind", &molgr::utils::BondData::kind,
                       "Covalent, dative or unspecified coordination edge.")
        .def("__repr__", [](const molgr::utils::BondData &b)
             { return "<BondData " + std::to_string(b.begin_atom_idx) + "-" +
                      std::to_string(b.end_atom_idx) + " order=" + std::to_string(b.order) +
                      " aromatic=" + (b.aromatic ? "true" : "false") + ">"; },
             "Return a concise debug representation of BondData.");

    py::enum_<molgr::utils::StereoKind>(m, "StereoKind",
        "Geometry that determines how a stereo frame interprets its ordered neighbor references.")
        .value("Tetrahedral", molgr::utils::StereoKind::Tetrahedral)
        .value("DoubleBond", molgr::utils::StereoKind::DoubleBond)
        .value("SquarePlanar", molgr::utils::StereoKind::SquarePlanar)
        .value("TrigonalBipyramidal", molgr::utils::StereoKind::TrigonalBipyramidal)
        .value("Octahedral", molgr::utils::StereoKind::Octahedral);
    py::class_<molgr::utils::StereoData>(m, "StereoData",
        "One atom or double-bond stereo frame. References are one-based and zero denotes a virtual non-hydrogen ligand.")
        .def(py::init<>())
        .def_readwrite("kind", &molgr::utils::StereoData::kind,
                       "Stereo geometry of this atom or double bond.")
        .def_readwrite("center", &molgr::utils::StereoData::center,
                       "One-based atom center or first endpoint of a double bond.")
        .def_readwrite("end", &molgr::utils::StereoData::end,
                       "One-based second endpoint for double-bond stereo.")
        .def_readwrite("refs", &molgr::utils::StereoData::refs,
                       "Ordered one-based ligand references. Zero is a virtual non-hydrogen ligand.")
        .def_readwrite("configuration", &molgr::utils::StereoData::configuration,
                       "Zero denotes unspecified stereo. Tetrahedral uses 1=CW or 2=CCW, double bonds use 1=cis or 2=trans, and SP/TB/OH use SMILES permutation numbers.");
    py::enum_<molgr::utils::StereoGroupKind>(m, "StereoGroupKind",
        "Absolute, relative (AND) or alternative (OR) stereochemistry of a group of centers.")
        .value("Absolute", molgr::utils::StereoGroupKind::Absolute)
        .value("And", molgr::utils::StereoGroupKind::And)
        .value("Or", molgr::utils::StereoGroupKind::Or);
    py::class_<molgr::utils::StereoGroupData>(m, "StereoGroupData",
        "Enhanced stereo group with one-based atom members and the group identifier read from the input.")
        .def(py::init<>())
        .def_readwrite("kind", &molgr::utils::StereoGroupData::kind,
                       "Absolute, AND or OR relation among member centers.")
        .def_readwrite("atoms", &molgr::utils::StereoGroupData::atoms,
                       "One-based indices of the atom stereocenters in this group.")
        .def_readwrite("read_id", &molgr::utils::StereoGroupData::read_id,
                       "Group identifier from the input stereo annotation.");

    py::class_<molgr::utils::MoleculeData>(m, "MoleculeData",
        "Molecular graph exchanged by the native and Python backends, with explicit atoms, bonds, electron totals and stereo frames.")
        .def(py::init<>())
        .def_readwrite("atoms", &molgr::utils::MoleculeData::atoms,
                       "Atom records in input order. Bond and stereo references count from one in this array.")
        .def_readwrite("bonds", &molgr::utils::MoleculeData::bonds,
                       "Edge records including covalent and coordination bonds.")
        .def_readwrite("total_charge", &molgr::utils::MoleculeData::total_charge,
                       "Net molecular charge in elementary-charge units.")
        .def_readwrite("total_radical_num", &molgr::utils::MoleculeData::total_radical_num,
                       "Total unpaired-electron count recorded for the molecule.")
        .def_readwrite("stereo", &molgr::utils::MoleculeData::stereo,
                       "Sparse, ordered stereo frames for specified or explicitly unknown centers and bonds.")
        .def_readwrite("stereo_groups", &molgr::utils::MoleculeData::stereo_groups,
                       "Enhanced stereo relations between atom centers.");

    // Transfer primitive rows once instead of allocating Python AtomData /
    // BondData wrappers and crossing the binding for every field assignment.
    m.def("_molecule_data_from_rows", [](const std::vector<std::array<int, 7>> &atoms,
                                         const std::vector<std::array<int, 5>> &bonds,
                                         py::object coordinates)
    {
        molgr::utils::MoleculeData data;
        data.atoms.reserve(atoms.size());
        data.bonds.reserve(bonds.size());
        std::int64_t charge = 0, radicals = 0;
        for (const auto &row : atoms)
        {
            if (row[6] < 0 || row[6] > 7 || (row[5] != 0 && row[5] != 1))
                throw py::value_error("Invalid electron presence/boolean field");
            molgr::utils::AtomData atom{};
            atom.atomic_num = row[0]; atom.isotope = row[1];
            atom.formal_charge = row[2]; atom.radical_num = row[3];
            atom.lone_pair_count = row[4]; atom.unresolved_two_electron_center = row[5];
            atom.electron_property_presence = static_cast<std::uint8_t>(row[6]);
            charge += atom.formal_charge; radicals += atom.radical_num;
            data.atoms.push_back(atom);
        }
        if (charge < std::numeric_limits<int>::min() || charge > std::numeric_limits<int>::max() ||
            radicals < std::numeric_limits<int>::min() || radicals > std::numeric_limits<int>::max())
            throw py::value_error("Electron totals exceed the native integer range");
        data.total_charge = static_cast<int>(charge);
        data.total_radical_num = static_cast<int>(radicals);
        for (const auto &row : bonds)
        {
            if (row[4] < 0 || row[4] > 2 || (row[3] != 0 && row[3] != 1))
                throw py::value_error("Invalid bond kind/aromatic field");
            molgr::utils::BondData bond{};
            bond.begin_atom_idx = row[0]; bond.end_atom_idx = row[1];
            bond.order = row[2]; bond.aromatic = row[3];
            bond.kind = static_cast<molgr::utils::BondKind>(row[4]);
            data.bonds.push_back(bond);
        }
        if (!coordinates.is_none())
        {
            auto positions = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(coordinates);
            if (!positions || positions.ndim() != 2 || positions.shape(0) != static_cast<py::ssize_t>(atoms.size()) || positions.shape(1) != 3)
                throw py::value_error("Coordinates must have shape (atom_count, 3)");
            auto values = positions.unchecked<2>();
            for (std::size_t i = 0; i < atoms.size(); ++i)
            {
                data.atoms[i].x = values(i, 0);
                data.atoms[i].y = values(i, 1);
                data.atoms[i].z = values(i, 2);
            }
        }
        return data;
    }, "Build a molecular graph from electron-labeled atom rows and one-based bond rows. Optional XYZ coordinates have shape (N, 3).",
       py::arg("atoms"), py::arg("bonds"), py::arg("coordinates") = py::none());

    m.def("apply_molecule_data_stereo_to_obmol",
          [](const molgr::utils::MoleculeData &data, intptr_t mol_ptr)
          {
              if (!mol_ptr)
                  throw std::invalid_argument("OBMol pointer must not be null");
              molgr::utils::ApplyStereoDataToOBMol(data, *reinterpret_cast<OpenBabel::OBMol *>(mol_ptr));
          },
          "Replace stereo annotations on an existing covalent OBMol projection. The caller owns the pointer and must keep it confined to one thread.",
          py::arg("molecule_data"), py::arg("mol_ptr"));

    // 绑定提取函数
    m.def("extract_molecule_data",
          [](intptr_t mol_ptr)
          {
              molgr::WarnUnsafeOpenBabelUse(
                  "molgr.utils.extract_molecule_data",
                  "The input OBMol pointer is not thread-safe; keep it confined to one thread and do not use it after free_obmol_ptr.");
              return molgr::utils::ExtractMoleculeData(mol_ptr);
          },
          "Extracts OBMol content into a structured object.",
          py::arg("mol_ptr"));

    m.def("molecule_data_to_obmol_ptr",
          [](const molgr::utils::MoleculeData &molecule_data)
          {
              molgr::WarnUnsafeOpenBabelUse(
                  "molgr.utils.molecule_data_to_obmol_ptr",
                  "The returned OBMol pointer has manual ownership and is not safe to share across threads; prefer MoleculeData or pipeline APIs.");
              OpenBabel::OBMol *mol = new OpenBabel::OBMol(molgr::utils::MolFromMoleculeData(molecule_data));
              return reinterpret_cast<intptr_t>(mol);
          },
          "Converts MoleculeData to a covalent OBMol projection, omitting coordination edges. Hydrogens must be explicit. This raw pointer is not thread-safe; free it exactly once with _core.free_obmol_ptr.",
          py::arg("molecule_data"));
}

void molgr::bind::bind_dev_utils(py::module_ &m)
{
    m.def("debug_roundtrip_xyz_block",
          [](const std::string &xyz_block)
          {
              OpenBabel::OBMol mol;
              py::gil_scoped_release release;
              if (!molgr::utils::ReadXyzBlockToMol(xyz_block, &mol))
              {
                  throw std::runtime_error("failed to parse XYZ block");
              }
              return molgr::utils::WriteXyzBlock(mol);
          },
          "Round-trip an XYZ block through the plugin-free C++ reader and writer.",
          py::arg("xyz_block"));

    m.def("debug_xyz_seed_molecule_data",
          &xyz_seed_molecule_data,
          "Return the C++ vendor-perceived seed molecule data for an XYZ block.",
          py::arg("xyz_block"));

    m.def("compute_organic_topology_metrics_ptr",
          [](intptr_t mol_ptr, py::object config)
          {
              molgr::WarnUnsafeOpenBabelUse(
                  "molgr.dev.utils.compute_organic_topology_metrics_ptr",
                  "OBMol pointers are not thread-safe; keep the pointer confined to one thread for its entire lifetime.");
              if (mol_ptr == 0)
              {
                  throw std::runtime_error("null OBMol pointer");
              }
              const auto runtime_config = molgr::config::FromPython(config);
              auto *mol = reinterpret_cast<OpenBabel::OBMol *>(mol_ptr);
              const auto metrics = molgr::organic_topology::ComputeOrganicTopologyMetrics(
                  *mol,
                  runtime_config.organic_topology);
              py::dict out;
              out["aromatic_atom_count"] = metrics.aromatic_atom_count;
              out["aromatic_ring_count"] = metrics.aromatic_ring_count;
              out["aromatic_stability_score"] = metrics.aromatic_stability_score;
              out["conjugated_atom_count"] = metrics.conjugated_atom_count;
              out["conjugated_bond_count"] = metrics.conjugated_bond_count;
              out["max_conjugated_component_size"] = metrics.max_conjugated_component_size;
              out["conjugated_atom_indices"] = metrics.conjugated_atom_indices;
              out["conjugated_atom_components"] = metrics.conjugated_atom_components;
              out["hyperconjugative_donor_count"] = metrics.hyperconjugative_donor_count;
              out["hyperconjugation_score"] = metrics.hyperconjugation_score;
              return out;
          },
          "Compute C++ organic topology metrics for an OBMol pointer.",
          py::arg("mol_ptr"),
          py::kw_only(),
          py::arg("config") = py::none());

    m.def("organic_force_field_energy_ptr",
          [](intptr_t mol_ptr, py::object config)
          {
              molgr::WarnUnsafeOpenBabelUse(
                  "molgr.dev.utils.organic_force_field_energy_ptr",
                  "OBMol pointers are not thread-safe; keep the pointer confined to one thread for its entire lifetime.");
              if (mol_ptr == 0)
              {
                  throw std::runtime_error("null OBMol pointer");
              }
              const auto runtime_config = molgr::config::FromPython(config);
              auto *mol = reinterpret_cast<OpenBabel::OBMol *>(mol_ptr);
              return molgr::scoring::OrganicForceFieldEvaluation(*mol, runtime_config).energy_kj_mol;
          },
          "Compute C++ organic force-field energy for an OBMol pointer.",
          py::arg("mol_ptr"),
          py::kw_only(),
          py::arg("config") = py::none());

    m.def("debug_vendor_uff_ptr",
          [](intptr_t mol_ptr)
          {
              molgr::WarnUnsafeOpenBabelUse(
                  "molgr.dev.utils.debug_vendor_uff_ptr",
                  "OBMol pointers are not thread-safe; keep the pointer confined to one thread for its entire lifetime.");
              if (mol_ptr == 0)
              {
                  throw std::runtime_error("null OBMol pointer");
              }
              auto *mol = reinterpret_cast<OpenBabel::OBMol *>(mol_ptr);
              OpenBabel::OBMol working = molgr::utils::CloneMolTopologyOnly(*mol);
              molgr::vendor::openbabel_threading::SetAromaticPerceived(working, false);
              OpenBabel::MolgrForceFieldUFF force_field;
              force_field.SetLogLevel(OBFF_LOGLVL_NONE);
              py::dict out;
              const bool setup_ok = force_field.Setup(working);
              out["setup_ok"] = setup_ok;
              py::list atom_types;
              if (setup_ok)
              {
                  for (const std::string &atom_type : force_field.DebugAtomTypes())
                  {
                      atom_types.append(atom_type);
                  }
                  out["energy"] = force_field.Energy(false);
                  out["bond"] = force_field.E_Bond(false);
                  out["angle"] = force_field.E_Angle(false);
                  out["torsion"] = force_field.E_Torsion(false);
                  out["oop"] = force_field.E_OOP(false);
                  out["vdw"] = force_field.E_VDW(false);
              }
              out["atom_types"] = atom_types;
              return out;
          },
          "Return MolGR vendor UFF atom types and energy terms for an OBMol pointer.",
          py::arg("mol_ptr"));

    m.def("test_symmetry_penalty", [](const std::string &smiles)
          {
              auto mol = mol_from_smiles(smiles);
              return molgr::scoring::CalcSymmetryPenalty(*mol);
          },
          "Calculate symmetry penalty from SMILES (For Testing)",
          py::arg("smiles"));

    m.def("test_physchem_penalty", [](const std::string &smiles)
          {
              auto mol = mol_from_smiles(smiles);
              return molgr::scoring::CalculatePhysChemPenalty(*mol);
          },
          "Calculate PhysChem penalty from SMILES (For Testing)",
          py::arg("smiles"));

    m.def("test_deviation_score", [](const std::string &xyz_block, int atom_idx)
          {
              auto mol = mol_from_xyz(xyz_block);
              if (!mol)
              {
                  return -1.0;
              }
              OpenBabel::OBAtom *atom = mol->GetAtom(atom_idx);
              if (!atom)
              {
                  return -1.0;
              }
              return molgr::scoring::GetDeviationScore(*mol, atom);
          },
          "Calculate geometry deviation for atom (1-based index) from XYZ (For Testing)",
          py::arg("xyz_block"),
          py::arg("atom_idx"));

    m.def("test_total_score", [](const std::string &xyz_block)
          {
              auto mol = mol_from_xyz(xyz_block);
              if (!mol)
              {
                  throw std::runtime_error("failed to parse XYZ");
              }
              double score = molgr::scoring::OmolScore(*mol);
              const int n = mol->NumAtoms();
              for (int i = 1; i <= n; ++i)
              {
                  OpenBabel::OBAtom *a = mol->GetAtom(i);
                  if (!a)
                  {
                      continue;
                  }
                  for (int j = i + 1; j <= n; ++j)
                  {
                      OpenBabel::OBAtom *b = mol->GetAtom(j);
                      if (!b)
                      {
                          continue;
                      }
                      const double dist = a->GetDistance(b);
                      if (dist < 0.5)
                      {
                          score += (0.5 - dist) * 1000.0;
                      }
                  }
              }
              return score;
          },
          "Calculate total OMolScore from XYZ block (For Testing)",
          py::arg("xyz_block"));
}
