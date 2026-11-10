#include "molgr/utils/conversions.h"
#include "molgr/utils/electrons.h"
#include "molgr/utils/stereo.h"
#include "molgr/vendor/openbabel_threading.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include "molgr/compat/openbabel_iter.h"

namespace molgr
{
    namespace utils
    {
        // Electron bookkeeping: despite the historical name, clone topology plus
        // all three MolGR atom electron fields verbatim; do not infer one from another.
        static void PopulateMolTopologyOnly(
            const OpenBabel::OBMol &mol,
            OpenBabel::OBMol &clone,
            bool preserve_implicit_hydrogens)
        {
            OpenBabel::OBMol &mutable_source = const_cast<OpenBabel::OBMol &>(mol);
            const bool source_has_hybridization =
                mutable_source.HasHybridizationPerceived();
            const bool source_has_aromaticity = mutable_source.HasAromaticPerceived();
            const bool source_has_ring_flags =
                mutable_source.HasRingAtomsAndBondsPerceived();
            clone.BeginModify();
            clone.ReserveAtoms(mol.NumAtoms());
            clone.SetDimension(mol.GetDimension());
            clone.SetTitle(mol.GetTitle());

            FOR_ATOMS_OF_MOL(atom, const_cast<OpenBabel::OBMol &>(mol))
            {
                OpenBabel::OBAtom *new_atom = clone.NewAtom();
                new_atom->SetAtomicNum(atom->GetAtomicNum());
                new_atom->SetIsotope(atom->GetIsotope());
                new_atom->SetFormalCharge(atom->GetFormalCharge());
                if (preserve_implicit_hydrogens)
                {
                    // Aromatic topology scoring depends on the implicit H count.
                    new_atom->SetImplicitHCount(atom->GetImplicitHCount());
                }
                CopyElectronData(*atom, *new_atom);
                if (source_has_hybridization)
                {
                    new_atom->SetHyb(atom->GetHyb());
                }
                // Query only already-perceived source flags. Otherwise the
                // clone stays unperceived and computes them when needed.
                new_atom->SetAromatic(source_has_aromaticity && atom->IsAromatic());
                new_atom->SetInRing(source_has_ring_flags && atom->IsInRing());
                new_atom->SetVector(atom->GetX(), atom->GetY(), atom->GetZ());
            }

            FOR_BONDS_OF_MOL(bond, const_cast<OpenBabel::OBMol &>(mol))
            {
                const int begin_idx = bond->GetBeginAtom()->GetIdx();
                const int end_idx = bond->GetEndAtom()->GetIdx();
                clone.AddBond(
                    begin_idx,
                    end_idx,
                    bond->GetBondOrder());
                OpenBabel::OBBond *new_bond = clone.GetBond(
                    begin_idx,
                    end_idx);
                if (new_bond != nullptr)
                {
                    const bool source_bond_aromatic =
                        (bond->GetFlags() & OB_AROMATIC_BOND) != 0;
                    const bool source_bond_in_ring =
                        (bond->GetFlags() & OB_RING_BOND) != 0;
                    const bool source_bond_closure =
                        (bond->GetFlags() & OB_CLOSURE_BOND) != 0;
                    new_bond->SetAromatic(source_bond_aromatic);
                    new_bond->SetInRing(source_bond_in_ring);
                    new_bond->SetClosure(source_bond_closure);
                }
            }

            clone.EndModify();
            clone.SetRingAtomsAndBondsPerceived(source_has_ring_flags);
            clone.SetClosureBondsPerceived(mutable_source.HasClosureBondsPerceived());
            clone.SetSSSRPerceived(false);
            clone.SetLSSRPerceived(mutable_source.HasLSSRPerceived());
            clone.SetAromaticPerceived(source_has_aromaticity);
            clone.SetHybridizationPerceived(source_has_hybridization);
            clone.SetAtomTypesPerceived(mutable_source.HasAtomTypesPerceived());
            // Internal topology/scoring clones also accept implicit hydrogens.
            // Keep stereo transfer at the explicit-H MoleculeData boundary;
            // topology scoring does not consume stereo annotations.
        }

        OpenBabel::OBMol CloneMolTopologyOnly(const OpenBabel::OBMol &mol)
        {
            OpenBabel::OBMol clone;
            PopulateMolTopologyOnly(mol, clone, false);
            return clone;
        }

        OpenBabel::OBMol CloneMolTopologyForScoring(const OpenBabel::OBMol &mol)
        {
            OpenBabel::OBMol clone;
            PopulateMolTopologyOnly(mol, clone, true);
            return clone;
        }

        std::shared_ptr<OpenBabel::OBMol> CloneMolTopologyShared(const OpenBabel::OBMol &mol)
        {
            auto clone = std::make_shared<OpenBabel::OBMol>();
            PopulateMolTopologyOnly(mol, *clone, false);
            return clone;
        }

        // Restore each known electron field independently, omitting unknown data.
        OpenBabel::OBMol MolFromMoleculeData(const MoleculeData &data)
        {
            OpenBabel::OBMol mol;
            mol.BeginModify();

            for (const AtomData &atom_data : data.atoms)
            {
                OpenBabel::OBAtom *atom = mol.NewAtom();
                atom->SetAtomicNum(atom_data.atomic_num);
                atom->SetIsotope(atom_data.isotope);
                atom->SetImplicitHCount(0);
                atom->SetFormalCharge(atom_data.formal_charge);
                // Reconstruction's sparse setters omit zero; serialization must
                // retain a known zero as distinct from an absent annotation.
                const auto restore_known_count = [atom](const char *name, int value)
                {
                    if (value != 0)
                    {
                        SetAtomIntegerData(*atom, name, value);
                        return;
                    }
                    auto *field = new OpenBabel::OBPairData();
                    field->SetAttribute(name);
                    field->SetValue("0");
                    field->SetOrigin(OpenBabel::local);
                    atom->SetData(field);
                };
                if (atom_data.electron_property_presence & 1)
                    restore_known_count(kUnpairedElectronCountProp, atom_data.radical_num);
                if (atom_data.electron_property_presence & 2)
                    restore_known_count(kLonePairCountProp, atom_data.lone_pair_count);
                if (atom_data.electron_property_presence & 4)
                    restore_known_count(
                        kUnresolvedTwoElectronCenterProp,
                        atom_data.unresolved_two_electron_center ? 1 : 0);
                atom->SetHyb(atom_data.hybridization);
                atom->SetVector(atom_data.x, atom_data.y, atom_data.z);
            }

            for (const BondData &bond_data : data.bonds)
            {
                // OBMol is a covalent projection, not a lossless container for
                // coordination. Keep the original MoleculeData for those edges.
                if (bond_data.kind != BondKind::Covalent)
                    continue;
                mol.AddBond(bond_data.begin_atom_idx, bond_data.end_atom_idx, bond_data.order);
                if (bond_data.aromatic || bond_data.order == 5)
                {
                    OpenBabel::OBBond *bond = mol.GetBond(
                        bond_data.begin_atom_idx,
                        bond_data.end_atom_idx);
                    if (bond != nullptr)
                    {
                        bond->SetAromatic();
                    }
                }
            }

            mol.EndModify();
            ApplyStereoDataToOBMol(data, mol);
            return mol;
        }

        // Electron bookkeeping: serialize all three classifications independently;
        // only real unpaired electrons contribute to total_radical_num.
        MoleculeData MoleculeDataFromOBMol(
            const OpenBabel::OBMol &mol,
            bool assume_electron_properties_known)
        {
            MoleculeData data;
            data.total_charge = 0;
            data.total_radical_num = 0;
            data.stereo = StereoDataFromOBMol(mol);

            OpenBabel::OBMol &mutable_mol = const_cast<OpenBabel::OBMol &>(mol);
            molgr::vendor::openbabel_threading::EnsureHybridizationPerceived(mutable_mol);
            data.atoms.reserve(mol.NumAtoms());
            FOR_ATOMS_OF_MOL(atom, mutable_mol)
            {
                AtomData atom_data;
                atom_data.atomic_num = atom->GetAtomicNum();
                atom_data.isotope = atom->GetIsotope();
                atom_data.formal_charge = atom->GetFormalCharge();
                atom_data.radical_num = GetUnpairedElectronCount(*atom);
                atom_data.lone_pair_count = GetLonePairCount(*atom);
                atom_data.unresolved_two_electron_center =
                    HasUnresolvedTwoElectronCenter(*atom);
                if (!assume_electron_properties_known)
                {
                    atom_data.electron_property_presence =
                        (atom->HasData(kUnpairedElectronCountProp) ? 1 : 0) |
                        (atom->HasData(kLonePairCountProp) ? 2 : 0) |
                        (atom->HasData(kUnresolvedTwoElectronCenterProp) ? 4 : 0);
                }
                atom_data.hybridization = atom->GetHyb();
                atom_data.x = atom->GetX();
                atom_data.y = atom->GetY();
                atom_data.z = atom->GetZ();
                data.atoms.push_back(atom_data);

                data.total_charge += atom_data.formal_charge;
                data.total_radical_num += atom_data.radical_num;
            }

            data.bonds.reserve(mol.NumBonds());
            FOR_BONDS_OF_MOL(bond, mutable_mol)
            {
                BondData bond_data;
                bond_data.begin_atom_idx = bond->GetBeginAtom()->GetIdx();
                bond_data.end_atom_idx = bond->GetEndAtom()->GetIdx();
                bond_data.order = bond->GetBondOrder();
                bond_data.aromatic = (bond->GetFlags() & OB_AROMATIC_BOND) != 0;
                data.bonds.push_back(bond_data);
            }

            return data;
        }
    }
}
