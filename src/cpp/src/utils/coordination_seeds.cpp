#include "molgr/utils/coordination_seeds.h"
#include "molgr/utils/coordination_geometry.h"
#include "molgr/utils/search_constraints.h"
#include "molgr/utils/electrons.h"
#include "molgr/compat/openbabel_iter.h"
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/elements.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <tuple>

namespace molgr::coordination
{
// Removing the metal before distance perception can attach its hydride
// to a neighboring carbonyl. An ordinary formyl C-H, a bent M-C-O,
// or a remote metal provides no evidence for deleting the bond.
void DetachCarbonylHydrideContacts(
    OpenBabel::OBMol &ligands,
    const std::vector<molgr::MetalSite> &metals)
{
    using OpenBabel::OBElements::GetCovalentRad;
    for (unsigned int i = 1; i <= ligands.NumAtoms(); ++i)
    {
        auto *h = ligands.GetAtom(i);
        if (h->GetAtomicNum() != 1 || h->GetExplicitDegree() != 1) continue;
        OpenBabel::OBBondIterator bonds;
        auto *carbon = h->BeginNbrAtom(bonds);
        if (carbon->GetAtomicNum() != 6 || carbon->GetExplicitDegree() != 2 ||
            h->GetDistance(carbon) <= GetCovalentRad(1) + GetCovalentRad(6) + 0.20) continue;
        OpenBabel::OBAtom *oxygen = nullptr;
        FOR_NBORS_OF_ATOM(n, carbon)
            if (n->GetAtomicNum() == 8 && n->GetExplicitDegree() == 1) oxygen = &*n;
        if (!oxygen || carbon->GetDistance(oxygen) > 1.25) continue;
        for (const auto &metal : metals)
        {
            const double radius = GetCovalentRad(metal.element_idx);
            if (DistanceToMetal(*h, metal) <= radius + GetCovalentRad(1) &&
                DistanceToMetal(*carbon, metal) <= radius + GetCovalentRad(6) &&
                MetalDonorAngle(metal, *carbon, *oxygen) > 160.0)
            {
                ligands.DeleteBond(h->GetBond(carbon));
                break;
            }
        }
    }
}

namespace hydrazido
{
    Sites LocalSites(OpenBabel::OBMol &mol)
    {
        Sites sites;
        FOR_ATOMS_OF_MOL(first, mol)
        {
            if (first->GetAtomicNum() != 7 || first->GetExplicitDegree() != 1 || first->GetImplicitHCount()) continue;
            OpenBabel::OBAtom *second = nullptr;
            FOR_NBORS_OF_ATOM(n, &*first) second = &*n;
            if (!second || second->GetAtomicNum() != 7 || second->GetExplicitDegree() != 3 ||
                second->GetImplicitHCount()) continue;
            bool valid = true;
            FOR_NBORS_OF_ATOM(n, second)
                if (n->GetIdx() != first->GetIdx() && n->GetAtomicNum() != 6) valid = false;
            for (auto *a : {&*first, second})
                if (molgr::utils::GetUnpairedElectronCount(*a) || molgr::utils::GetLonePairCount(*a) ||
                    molgr::utils::HasUnresolvedTwoElectronCenter(*a)) valid = false;
            const Electronic state{first->GetFormalCharge(), second->GetFormalCharge(), first->GetBond(second)->GetBondOrder()};
            if (valid && (state == neutral || state == reduced)) sites[{first->GetIdx(), second->GetIdx()}] = state;
        }
        return sites;
    }

    bool SupportedGeometry(OpenBabel::OBAtom &a, OpenBabel::OBAtom &b,
                           const molgr::MetalSite &m)
    {
        const double ux = m.position_x - a.GetX(), uy = m.position_y - a.GetY(), uz = m.position_z - a.GetZ();
        const double vx = b.GetX() - a.GetX(), vy = b.GetY() - a.GetY(), vz = b.GetZ() - a.GetZ();
        const double mn = std::sqrt(ux * ux + uy * uy + uz * uz), nn = std::sqrt(vx * vx + vy * vy + vz * vz);
        const double radius = OpenBabel::OBElements::GetCovalentRad(m.element_idx) + OpenBabel::OBElements::GetCovalentRad(7);
        if (!(nn > 1.20 && nn < 1.45 && mn > 0.65 * radius && mn < 0.85 * radius)) return false;
        const double cosine = (ux * vx + uy * vy + uz * vz) / (mn * nn);
        return std::acos(std::clamp(cosine, -1.0, 1.0)) * 180.0 / std::acos(-1.0) > 160.0;
    }

}

namespace
{
/// Preserve the seed and spend only an available two-charge deficit.
/// @param state Ligand graph and electronic budgets at this stage of reconstruction.
/// @param metal Metal site being considered.
/// @return Original ligand state and charge-budgeted hydrazido alternatives.
std::vector<molgr::state::ReconstructionState> ExpandHydrazidoSeeds(
    const molgr::state::ReconstructionState &state, const molgr::MetalSite &metal)
{
    using namespace hydrazido;
    std::vector<molgr::state::ReconstructionState> pool{state};
    auto &mol = const_cast<OpenBabel::OBMol &>(state.Mol());
    // A terminal hydrazonido precursor can arrive as N*=N-C*. Closing
    // it to N#N+-C- loses the bent N-N-C geometry. Retain the reduced
    // N(2-)-N=C hypothesis beside the original, using the same short,
    // nearly linear metal-N-N support required for hydrazido seeds.
    FOR_ATOMS_OF_MOL(first, mol)
    {
        if (first->GetAtomicNum() != 7 || first->GetExplicitDegree() != 1) continue;
        OpenBabel::OBAtom *second = nullptr;
        FOR_NBORS_OF_ATOM(n, &*first) second = &*n;
        if (!second || second->GetAtomicNum() != 7 || second->GetExplicitDegree() != 2 ||
            first->GetBond(second)->GetBondOrder() != 2 ||
            !SupportedGeometry(*first, *second, metal)) continue;
        OpenBabel::OBAtom *carbon = nullptr;
        FOR_NBORS_OF_ATOM(n, second)
            if (n->GetIdx() != first->GetIdx()) carbon = &*n;
        if (!carbon || carbon->GetAtomicNum() != 6 || carbon->GetExplicitDegree() != 3 ||
            carbon->GetTotalValence() != 3 || second->GetBond(carbon)->GetBondOrder() != 1)
            continue;
        const double cn = second->GetDistance(carbon);
        const double angle = mol.GetAngle(&*first, second, carbon);
        if (!(cn > 1.20 && cn < 1.38 && angle > 90.0 && angle < 150.0)) continue;
        bool valid = true;
        for (auto *a : {&*first, second, carbon})
            if (a->GetFormalCharge() || a->GetImplicitHCount() ||
                molgr::utils::GetLonePairCount(*a) ||
                molgr::utils::HasUnresolvedTwoElectronCenter(*a) ||
                molgr::utils::GetUnpairedElectronCount(*a) != (a == second ? 0 : 1))
                valid = false;
        FOR_NBORS_OF_ATOM(n, carbon)
            if (n->GetIdx() != second->GetIdx() &&
                n->GetAtomicNum() != 6 && n->GetAtomicNum() != 1) valid = false;
        if (!valid) continue;
        const auto source_count = pool.size();
        for (std::size_t i = 0; i < source_count && pool.size() < 256; ++i)
        {
            const auto source = pool[i];
            if (source.given_charge > -2) continue;
            auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
            machine.RunOmolChargeStage("explore_terminal_hydrazonido",
                [](OpenBabel::OBMol &m, int &charge, int a, int b, int c)
                {
                    m.GetBond(a, b)->SetBondOrder(1);
                    m.GetBond(b, c)->SetBondOrder(2);
                    m.GetAtom(a)->SetFormalCharge(-2);
                    molgr::utils::SetUnpairedElectronCount(*m.GetAtom(a), 0);
                    molgr::utils::SetUnpairedElectronCount(*m.GetAtom(c), 0);
                    charge += 2;
                    return true;
                }, first->GetIdx(), second->GetIdx(), carbon->GetIdx());
            pool.push_back(machine.FreezeLike(source));
        }
    }
    for (const auto &[pair, electronic] : LocalSites(mol))
    {
        if (electronic != neutral || !SupportedGeometry(*mol.GetAtom(pair.first), *mol.GetAtom(pair.second), metal)) continue;
        const auto source_count = pool.size();
        for (std::size_t i = 0; i < source_count; ++i)
        {
            if (pool.size() >= 256) return pool;
            const auto source = pool[i];
            if (source.given_charge > -2) continue;
            auto machine = molgr::state::OmolStateMachine::FromReconstructionState(source);
            machine.Annotate("branch_terminal_hydrazido_seed");
            machine.RunOmolChargeStage("explore_terminal_hydrazido",
                [](OpenBabel::OBMol &m, int &charge, int a, int b)
                {
                    m.GetBond(a, b)->SetBondOrder(1);
                    m.GetAtom(a)->SetFormalCharge(-2);
                    m.GetAtom(b)->SetFormalCharge(0);
                    charge += 2;
                    return true;
                }, pair.first, pair.second);
            pool.push_back(machine.FreezeLike(source));
        }
    }
    return pool;
}

}

std::vector<molgr::state::ReconstructionState> ExpandCoordinationSeeds(
    const molgr::state::ReconstructionState &seed, const std::vector<molgr::MetalSite> &metals,
    const molgr::config::MetalScoringConfig &config)
{
    std::vector<molgr::state::ReconstructionState> seeds{seed};
    if (metals.empty()) return seeds;
    auto &mol = const_cast<OpenBabel::OBMol &>(seed.Mol());
    std::vector<OpenBabel::OBAtom *> terminals;
    bool has_terminal_no = false;
    FOR_ATOMS_OF_MOL(atom, mol)
    {
        if (atom->GetExplicitDegree() != 1) continue;
        terminals.push_back(&*atom);
        if (atom->GetAtomicNum() == 7)
            FOR_NBORS_OF_ATOM(n, &*atom)
                if (n->GetAtomicNum() == 8) has_terminal_no = true;
    }
    if (has_terminal_no)
    {
        const auto blockers = BuildCoordinationBlockers(mol, config);
        std::vector<std::vector<OpenBabel::OBAtom *>> visible(metals.size());
        for (std::size_t i = 0; i < metals.size(); ++i)
            for (auto *atom : terminals)
                if (IsInnerSphereAtom(*atom, metals[i], config) && IsVisibleToMetalAtom(*atom, metals[i], blockers))
                    visible[i].push_back(atom);
        const auto contacts = TerminalNitrosylContacts(mol, metals, visible);
        if (!contacts.empty())
        {
            auto alternative = std::make_shared<OpenBabel::OBMol>(mol);
            int charge = seed.given_charge;
            for (const auto &contact : contacts)
            {
                auto *n = alternative->GetAtom(contact.nitrogen->GetIdx());
                auto *o = alternative->GetAtom(contact.oxygen->GetIdx());
                charge += n->GetFormalCharge() + o->GetFormalCharge() - 1;
                n->SetFormalCharge(0);
                o->SetFormalCharge(1);
                n->GetBond(o)->SetBondOrder(3);
                for (auto *atom : {n, o})
                {
                    molgr::utils::SetUnpairedElectronCount(*atom, 0);
                    molgr::utils::SetLonePairCount(*atom, 0);
                    molgr::utils::SetUnresolvedTwoElectronCenter(*atom, false);
                }
                molgr::search_constraints::RequireNitrosonium(*n, *o);
            }
            auto branch = molgr::state::OmolStateMachine::FromReconstructionState(seed).Branch(
                "geometry_nitrosonium_seed", std::move(alternative), charge);
            seeds.push_back(branch.FreezeLike(seed));
        }
    }
    if (metals.size() != 1) return seeds;
    std::vector<molgr::state::ReconstructionState> expanded;
    for (const auto &local_seed : seeds)
        for (auto &alternative : ExpandHydrazidoSeeds(local_seed, metals.front()))
            expanded.push_back(std::move(alternative));
    return expanded;
}
}
