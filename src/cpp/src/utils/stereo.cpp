#include "molgr/utils/stereo.h"

#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/stereo/tetrahedral.h>
#include <openbabel/stereo/cistrans.h>
#include <openbabel/stereo/squareplanar.h>

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

namespace molgr::utils
{
    namespace
    {
        int IndexForRef(const OpenBabel::OBMol &mol, OpenBabel::OBStereo::Ref ref)
        {
            if (ref == OpenBabel::OBStereo::ImplicitRef)
                return 0;
            auto *atom = mol.GetAtomById(ref);
            if (!atom)
                throw std::invalid_argument("Stereo references a missing Open Babel atom");
            return atom->GetIdx();
        }

        OpenBabel::OBStereo::Ref RefForIndex(const OpenBabel::OBMol &mol, int index)
        {
            if (!index)
                return OpenBabel::OBStereo::ImplicitRef;
            auto *atom = index > 0 ? mol.GetAtom(index) : nullptr;
            if (!atom)
                throw std::invalid_argument("Stereo references a missing MoleculeData atom");
            return atom->GetId();
        }
    }

    std::vector<StereoData> StereoDataFromOBMol(const OpenBabel::OBMol &mol)
    {
        std::vector<StereoData> result;
        auto &source = const_cast<OpenBabel::OBMol &>(mol);
        if (!source.HasData(OpenBabel::OBGenericDataType::StereoData))
            return result;
        OpenBabel::OBStereoFacade facade(&source, false);
        for (auto *stereo : facade.GetAllTetrahedralStereo())
        {
            const auto cfg = stereo->GetConfig(); // CW, ViewFrom, normalized refs.
            StereoData record;
            record.kind = StereoKind::Tetrahedral;
            record.center = IndexForRef(mol, cfg.center);
            record.configuration = cfg.specified ? 1 : 0;
            record.refs.push_back(IndexForRef(mol, cfg.from));
            for (auto ref : cfg.refs)
                record.refs.push_back(IndexForRef(mol, ref));
            if (mol.GetAtom(record.center)->GetImplicitHCount())
                throw std::invalid_argument("Stereo extraction requires explicit hydrogen nodes");
            result.push_back(std::move(record));
        }
        for (auto *stereo : facade.GetAllCisTransStereo())
        {
            const auto cfg = stereo->GetConfig(); // ShapeU: refs[0] trans to refs[2].
            StereoData record;
            record.kind = StereoKind::DoubleBond;
            record.center = IndexForRef(mol, cfg.begin);
            record.end = IndexForRef(mol, cfg.end);
            record.configuration = cfg.specified ? 2 : 0;
            for (auto ref : cfg.refs)
                record.refs.push_back(IndexForRef(mol, ref));
            if (mol.GetAtom(record.center)->GetImplicitHCount() ||
                mol.GetAtom(record.end)->GetImplicitHCount())
                throw std::invalid_argument("Stereo extraction requires explicit hydrogen nodes");
            result.push_back(std::move(record));
        }
        for (auto *stereo : facade.GetAllSquarePlanarStereo())
        {
            const auto cfg = stereo->GetConfig(); // ShapeU == SMILES SP1.
            StereoData record;
            record.kind = StereoKind::SquarePlanar;
            record.center = IndexForRef(mol, cfg.center);
            record.configuration = cfg.specified ? 1 : 0;
            for (auto ref : cfg.refs)
                record.refs.push_back(IndexForRef(mol, ref));
            if (mol.GetAtom(record.center)->GetImplicitHCount())
                throw std::invalid_argument("Stereo extraction requires explicit hydrogen nodes");
            result.push_back(std::move(record));
        }
        return result;
    }

    void ApplyStereoDataToOBMol(const MoleculeData &data, OpenBabel::OBMol &mol)
    {
        std::unordered_set<int> relative_centers;
        for (const auto &group : data.stereo_groups)
            if (group.kind != StereoGroupKind::Absolute)
                relative_centers.insert(group.atoms.begin(), group.atoms.end());
        mol.DeleteData(OpenBabel::OBGenericDataType::StereoData);
        for (const auto &record : data.stereo)
        {
            // Open Babel does not encode enhanced relative groups or TB/OH.
            if (relative_centers.count(record.center) ||
                record.kind == StereoKind::TrigonalBipyramidal ||
                record.kind == StereoKind::Octahedral)
                continue;
            if (record.refs.size() != 4 || record.configuration < 0 ||
                record.configuration > (record.kind == StereoKind::SquarePlanar ? 3 : 2))
                throw std::invalid_argument("Invalid stereo configuration");
            const auto center = RefForIndex(mol, record.center);
            bool complete = true;
            OpenBabel::OBStereo::Refs refs;
            for (std::size_t i = 0; i < record.refs.size(); ++i)
            {
                int owner = record.kind == StereoKind::DoubleBond && i >= 2
                                ? record.end : record.center;
                refs.push_back(RefForIndex(mol, record.refs[i]));
                if (record.refs[i] && !mol.GetBond(owner, record.refs[i]))
                    complete = false;
            }
            // A removed coordination edge invalidates its local stereo frame.
            // Preserve that frame only in the original data, not the projection.
            if (!complete)
                continue;
            if (record.kind == StereoKind::Tetrahedral)
            {
                OpenBabel::OBTetrahedralStereo::Config cfg;
                cfg.center = center;
                cfg.from = refs[0];
                cfg.refs.assign(refs.begin() + 1, refs.end());
                cfg.specified = record.configuration != 0;
                cfg.winding = record.configuration == 2
                                  ? OpenBabel::OBStereo::AntiClockwise
                                  : OpenBabel::OBStereo::Clockwise;
                auto *stereo = new OpenBabel::OBTetrahedralStereo(&mol);
                stereo->SetConfig(cfg);
                mol.SetData(stereo);
            }
            else if (record.kind == StereoKind::DoubleBond)
            {
                auto *bond = mol.GetBond(record.center, record.end);
                if (!bond || bond->GetBondOrder() != 2)
                    throw std::invalid_argument("Double-bond stereo no longer matches the graph");
                if (record.configuration == 1)
                    std::swap(refs[2], refs[3]);
                OpenBabel::OBCisTransStereo::Config cfg(center, RefForIndex(mol, record.end), refs);
                cfg.specified = record.configuration != 0;
                auto *stereo = new OpenBabel::OBCisTransStereo(&mol);
                stereo->SetConfig(cfg);
                mol.SetData(stereo);
            }
            else if (record.kind == StereoKind::SquarePlanar)
            {
                if (record.configuration == 2)
                    std::swap(refs[1], refs[2]);
                else if (record.configuration == 3)
                    std::swap(refs[2], refs[3]);
                OpenBabel::OBSquarePlanarStereo::Config cfg(center, refs);
                cfg.specified = record.configuration != 0;
                auto *stereo = new OpenBabel::OBSquarePlanarStereo(&mol);
                stereo->SetConfig(cfg);
                mol.SetData(stereo);
            }
        }
        // Coordinates must not overwrite the supplied (possibly empty) projection.
        mol.SetChiralityPerceived(true);
    }
}
