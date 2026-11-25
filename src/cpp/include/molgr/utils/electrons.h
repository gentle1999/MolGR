#pragma once

#include <openbabel/atom.h>
#include <openbabel/generic.h>

#include <stdexcept>
#include <string>

namespace molgr
{
    namespace utils
    {
        inline constexpr const char *kUnpairedElectronCountProp =
            "MOLGR_UNPAIRED_ELECTRON_COUNT";
        inline constexpr const char *kLonePairCountProp = "MOLGR_LONE_PAIR_COUNT";
        inline constexpr const char *kUnresolvedTwoElectronCenterProp =
            "MOLGR_UNRESOLVED_TWO_ELECTRON_CENTER";

        /// Read a nonnegative integer-valued atom annotation, treating a missing value as zero.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param attribute Name of the Open Babel atom property.
        /// @return Integer annotation value, or zero for an absent or unreadable property.
        inline int GetAtomIntegerData(
            const OpenBabel::OBAtom &atom,
            const char *attribute)
        {
            if (atom.DataSize() == 0) return 0;
            auto &mutable_atom = const_cast<OpenBabel::OBAtom &>(atom);
            OpenBabel::OBGenericData *data = mutable_atom.GetData(attribute);
            if (data == nullptr)
            {
                return 0;
            }
            if (const auto *integer = dynamic_cast<const OpenBabel::OBPairInteger *>(data))
            {
                return integer->GetGenericValue();
            }
            if (const auto *pair = dynamic_cast<const OpenBabel::OBPairData *>(data))
            {
                try
                {
                    return std::stoi(pair->GetValue());
                }
                catch (const std::exception &)
                {
                    return 0;
                }
            }
            return 0;
        }

        /// Store a nonnegative atom annotation. Zero removes the stored property.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param attribute Name of the Open Babel atom property.
        /// @param value Nonnegative annotation value. Zero clears the property.
        inline void SetAtomIntegerData(
            OpenBabel::OBAtom &atom,
            const char *attribute,
            int value)
        {
            if (value < 0)
            {
                throw std::invalid_argument("atom electron count must be nonnegative");
            }
            if (atom.DataSize() != 0) atom.DeleteData(attribute);
            if (value == 0)
            {
                return;
            }
            auto *data = new OpenBabel::OBPairData();
            data->SetAttribute(attribute);
            data->SetValue(std::to_string(value));
            data->SetOrigin(OpenBabel::local);
            atom.SetData(data);
        }

        /// Return the number of explicitly assigned unpaired electrons.
        /// @param atom Atom whose local chemical environment is examined.
        /// @return Number of explicitly assigned unpaired electrons.
        inline int GetUnpairedElectronCount(const OpenBabel::OBAtom &atom)
        {
            return GetAtomIntegerData(atom, kUnpairedElectronCountProp);
        }

        // Store only real unpaired electrons; active lone pairs must use their
        // independent field and never be folded into this count.
        /// Store the number of real unpaired electrons; never encode lone pairs here.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param value Number of real unpaired electrons.
        inline void SetUnpairedElectronCount(OpenBabel::OBAtom &atom, int value)
        {
            SetAtomIntegerData(atom, kUnpairedElectronCountProp, value);
        }

        /// Return the number of active lone pairs stored by reconstruction.
        /// @param atom Atom whose local chemical environment is examined.
        /// @return Number of active lone pairs stored by reconstruction.
        inline int GetLonePairCount(const OpenBabel::OBAtom &atom)
        {
            return GetAtomIntegerData(atom, kLonePairCountProp);
        }

        // Store reconstruction-active lone pairs, not the full Lewis lone-pair
        // inventory implied by ordinary valence and formal charge.
        /// Store reconstruction-active lone pairs, not the atom's full Lewis count.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param value Number of active lone pairs.
        inline void SetLonePairCount(OpenBabel::OBAtom &atom, int value)
        {
            SetAtomIntegerData(atom, kLonePairCountProp, value);
        }

        /// Check whether the radical budget still has to resolve a two-electron center.
        /// @param atom Atom whose local chemical environment is examined.
        /// @return True when a two-electron center still needs a singlet or triplet assignment.
        inline bool HasUnresolvedTwoElectronCenter(const OpenBabel::OBAtom &atom)
        {
            return GetAtomIntegerData(atom, kUnresolvedTwoElectronCenterProp) != 0;
        }

        // Mark a deferred two-electron occupancy without selecting singlet (0,1)
        // or triplet (2,0) classification.
        /// Mark a C/N/Si/P-like two-electron deficit whose spin occupancy is deferred.
        /// @param atom Atom whose local chemical environment is examined.
        /// @param value Whether to defer classification of the two electrons.
        inline void SetUnresolvedTwoElectronCenter(OpenBabel::OBAtom &atom, bool value)
        {
            SetAtomIntegerData(atom, kUnresolvedTwoElectronCenterProp, value ? 1 : 0);
        }

        /// Copy the three independent classifications verbatim; never reconstruct
        /// lone-pair or unresolved state from unpaired-electron parity.
        /// @param source Atom supplying the electron annotations.
        /// @param target Atom receiving the copied electron annotations.
        inline void CopyElectronData(
            const OpenBabel::OBAtom &source,
            OpenBabel::OBAtom &target)
        {
            if (source.DataSize() == 0 && target.DataSize() == 0) return;
            SetUnpairedElectronCount(target, GetUnpairedElectronCount(source));
            SetLonePairCount(target, GetLonePairCount(source));
            SetUnresolvedTwoElectronCenter(
                target,
                HasUnresolvedTwoElectronCenter(source));
        }
    }
}
