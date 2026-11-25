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
        /// @param p1 Cartesian vertex of the tetrahedron, in argument order.
        /// @param p2 Cartesian vertex of the tetrahedron, in argument order.
        /// @param p3 Cartesian vertex of the tetrahedron, in argument order.
        /// @param p4 Cartesian vertex of the tetrahedron, in argument order.
        /// @return The unsigned volume of the tetrahedron spanned by the four points.
        double CalculateTetrahedronVolume(const OpenBabel::vector3 &p1, const OpenBabel::vector3 &p2,
                                          const OpenBabel::vector3 &p3, const OpenBabel::vector3 &p4);

        /// @param p1 Cartesian vertex of the tetrahedron, in argument order.
        /// @param p2 Cartesian vertex of the tetrahedron, in argument order.
        /// @param p3 Cartesian vertex of the tetrahedron, in argument order.
        /// @param p4 Cartesian vertex of the tetrahedron, in argument order.
        /// @return The normalized tetrahedron shape score for the supplied four points.
        double CalculateShapeQuality(const OpenBabel::vector3 &p1, const OpenBabel::vector3 &p2,
                                     const OpenBabel::vector3 &p3, const OpenBabel::vector3 &p4);

        /// Convert a three-component coordinate array to an Open Babel vector.
        /// @param coords Cartesian coordinates in angstroms.
        /// @return Vector of the first three components, or the zero vector if fewer were supplied.
        OpenBabel::vector3 ToVector3(const std::vector<double> &coords);

        /// Element, isotope, Cartesian coordinates and explicit electron labels for one graph atom.
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

        /// Distinguishes covalent order, directed dative donation and unspecified coordination.
        enum class BondKind : std::uint8_t
        {
            Covalent = 0,
            Dative,
            CoordinationUnspecified,
        };

        /// An edge with one-based atom indices. A dative edge runs from donor to acceptor.
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

        /// Geometry that determines how a stereo frame interprets its ordered neighbor references.
        enum class StereoKind : std::uint8_t
        {
            Tetrahedral,
            DoubleBond,
            SquarePlanar,
            TrigonalBipyramidal,
            Octahedral,
        };

        /// A sparse stereo frame for an atom or double bond. References are one-based.
        /// Zero represents a virtual non-hydrogen ligand, such as a lone pair.
        struct StereoData
        {
            StereoKind kind = StereoKind::Tetrahedral;
            int center = 0; ///< Atom center or first endpoint of a double bond.
            int end = 0; ///< Second endpoint for double-bond stereo.
            std::vector<int> refs; ///< Ordered ligand references defining the frame.
            /// Zero denotes explicitly unspecified stereo. Tetrahedral uses 1=CW or 2=CCW
            /// viewed from refs[0] towards center. Double bonds use 1=cis or 2=trans
            /// between refs[0] and refs[2]. SP/TB/OH use the SMILES permutation number.
            int configuration = 0;
        };

        /// Absolute, relative (AND) or alternative (OR) stereochemistry of a group of centers.
        enum class StereoGroupKind : std::uint8_t { Absolute, And, Or };

        /// Enhanced stereo group with one-based atom members and the group identifier read from the input.
        struct StereoGroupData
        {
            StereoGroupKind kind = StereoGroupKind::Absolute;
            std::vector<int> atoms;
            int read_id = 0;
        };

        /// Molecular graph exchanged by the native and Python backends, with explicit atoms, bonds, electron totals and stereo frames.
        struct MoleculeData
        {
            std::vector<AtomData> atoms;
            std::vector<BondData> bonds;
            int total_charge;
            int total_radical_num;
            std::vector<StereoData> stereo;
            std::vector<StereoGroupData> stereo_groups;
        };

        /// Copy the graph and electron metadata from a caller-owned OBMol pointer.
        /// @param mol_ptr Address of the caller-owned Open Babel molecule.
        /// @return Copied graph records. A null pointer yields an empty graph with zero charge and
        ///     radicals.
        MoleculeData ExtractMoleculeData(intptr_t mol_ptr);
        // Covalent projection: coordination edges stay in the source data.
        // Hydrogens must already be explicit atom nodes.
        /// Build the covalent Open Babel projection from explicit atom records.
        /// @param data Molecular records with explicit atoms, including hydrogens.
        /// @return Covalent projection with coordination edges retained only in the source records.
        OpenBabel::OBMol MolFromMoleculeData(const MoleculeData &data);
        /// Copy atoms, bonds, explicit electron labels and supported stereo from an OBMol.
        /// @param mol Source graph in Open Babel atom order.
        /// @param assume_electron_properties_known Whether to mark all three electron labels as known.
        /// @return Transferable molecular records with one-based atom indices.
        MoleculeData MoleculeDataFromOBMol(
            const OpenBabel::OBMol &mol,
            bool assume_electron_properties_known = false);

    }
}
