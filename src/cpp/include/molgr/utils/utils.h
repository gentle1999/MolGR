#pragma once

#include <cstdint>

#include <openbabel/math/vector3.h>
#include <openbabel/mol.h>

#include <vector>

#define FOR_NB_OF_ATOM(a, p) for (OpenBabel::OBAtomAtomIter a(p); a; ++a)

namespace molgr
{
    namespace utils
    {
        double CalculateTetrahedronVolume(const OpenBabel::vector3 &p1, const OpenBabel::vector3 &p2,
                                          const OpenBabel::vector3 &p3, const OpenBabel::vector3 &p4);

        double CalculateShapeQuality(const OpenBabel::vector3 &p1, const OpenBabel::vector3 &p2,
                                     const OpenBabel::vector3 &p3, const OpenBabel::vector3 &p4);

        OpenBabel::vector3 ToVector3(const std::vector<double> &coords);

        struct AtomData
        {
            int atomic_num;
            int formal_charge;
            int radical_num;
            int lone_pair_count = 0;
            bool unresolved_two_electron_center = false;
            // Presence bits: unpaired=1, active lone pairs=2, unresolved=4.
            // Internal pipeline output is fully specified; conversion from a
            // generic OBMol replaces this mask with actual property presence.
            std::uint8_t electron_property_presence = 7;
            int hybridization = 0;
            double x, y, z;
            int isotope = 0;
        };

        enum class BondKind : std::uint8_t
        {
            Covalent = 0,
            Dative,
            CoordinationUnspecified,
        };

        struct BondData
        {
            int begin_atom_idx;
            int end_atom_idx;
            int order;
            bool aromatic = false;
            // Endpoints retain Open Babel's one-based indexing. For Dative,
            // begin is the donor and end is the acceptor; order is unused.
            BondKind kind = BondKind::Covalent;
        };

        enum class StereoKind : std::uint8_t
        {
            Tetrahedral,
            DoubleBond,
            SquarePlanar,
            TrigonalBipyramidal,
            Octahedral,
        };

        // Sparse stereo records; all references use one-based atom indices.
        // Zero is a virtual non-H reference (e.g. a lone pair), never implicit H.
        struct StereoData
        {
            StereoKind kind = StereoKind::Tetrahedral;
            int center = 0;
            int end = 0; // Double-bond end; unused for atom stereo.
            std::vector<int> refs;
            // 0 = explicitly unspecified. Tetrahedral: 1=CW, 2=CCW viewed
            // from refs[0] towards center. DoubleBond: 1=cis, 2=trans between
            // refs[0] and refs[2]; refs[0:2] attach to center, refs[2:4] to end.
            // SP/TB/OH: standard SMILES permutation relative to ordered refs.
            int configuration = 0;
        };

        enum class StereoGroupKind : std::uint8_t { Absolute, And, Or };

        struct StereoGroupData
        {
            StereoGroupKind kind = StereoGroupKind::Absolute;
            std::vector<int> atoms;
            int read_id = 0;
        };

        struct MoleculeData
        {
            std::vector<AtomData> atoms;
            std::vector<BondData> bonds;
            int total_charge;
            int total_radical_num;
            std::vector<StereoData> stereo;
            std::vector<StereoGroupData> stereo_groups;
        };

        MoleculeData ExtractMoleculeData(intptr_t mol_ptr);
        // Covalent projection: coordination edges stay in the source data.
        // Hydrogens must already be explicit atom nodes.
        OpenBabel::OBMol MolFromMoleculeData(const MoleculeData &data);
        MoleculeData MoleculeDataFromOBMol(
            const OpenBabel::OBMol &mol,
            bool assume_electron_properties_known = false);

    }
}
