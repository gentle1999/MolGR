#pragma once

#include "molgr/state.h"
#include <vector>
#include <cstddef>

namespace molgr::metal::geometry
{
    /// Remove the previous amino-C5 geometry contribution before scoring the new candidates.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearAminoC5Conflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Score an exocyclic iminium against a charge-balanced aminocyclopentadienyl or indenyl partner.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateAminoC5Alternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Subtract the previous amido–carbene chelate conflict from both metal scores.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearArylAmidoNHCConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Score the quinonoid form against the matched aromatic amido–carbene chelate.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateArylAmidoNHCAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Subtract the previous phenoxide–carbene chelate conflict from both metal scores.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearArylPhenoxideNHCConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Score the quinonoid form against the matched aromatic phenoxide–carbene chelate.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateArylPhenoxideNHCAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Subtract the previous thiolate–carbene chelate conflict from both metal scores.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearArylThiolateNHCConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Score the quinonoid form against the matched aromatic thiolate–carbene chelate.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateArylThiolateNHCAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Subtract the previous benzyl–carbene chelate conflict from both metal scores.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearArylBenzylNHCConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Compare an exocyclic C=C assignment with a metal-facing pyramidal benzyl anion.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateArylBenzylNHCAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Remove the previous score for the adjacent oxygen donors on a shared aryl ring.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearOrthoDioxygenConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Score a quinonoid assignment against a matched reduced partner on the same ortho O/O ring.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateOrthoDioxygenAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Remove prior P–S conflicts before pairing pyramidal phosphorus assignments again.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearPyramidalPSConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Count P(+)=S assignments with a charge-balanced P–S(−) partner at a pyramidal donor.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotatePyramidalPSAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Remove prior cyclic P–N conflicts before pairing phosphorus assignments again.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearPyramidalPNConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Count cyclic P(+)=N assignments with a charge-balanced P–N(−) partner at pyramidal P.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotatePyramidalPNAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Remove the previous pyridyl penalty before comparing alternative charge localizations.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearPyramidalPyridylConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Compare matched pyridyl electronic assignments within the alternating-path normalization budgets.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    /// @param maxPathStates Maximum number of alternating-path states to explore.
    /// @param maxVariants Maximum number of distinct normalizations to retain.
    void AnnotatePyramidalPyridylAlternatives(
        std::vector<state::MetalCandidateState> &candidates,
        std::size_t maxPathStates = 128, std::size_t maxVariants = 16);
    /// Remove the previous contribution of the two cis-diimine units to metal discordance.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearPairedDiimineConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Prefer a total donor charge of −2 when matched cis-diimine units have a compensating metal state.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotatePairedDiimineAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Remove cached quinodiimine and polyenamine penalties before evaluating a new candidate pool.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void ClearConjugatedGeometryConflicts(std::vector<state::MetalCandidateState> &candidates);
    /// Flag a diimine assignment when the metal charge balances reduction at every changed site.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotateQuinodiimineAlternatives(std::vector<state::MetalCandidateState> &candidates);
    /// Count iminium assignments whose reduced partner preserves the graph outside the chain.
    /// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
    void AnnotatePolyenamineAlternatives(std::vector<state::MetalCandidateState> &candidates);
}

namespace molgr::metal::geometry {
/// Subtract earlier C–C geometry conflicts before comparing a different pool of metal states.
/// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
void ClearLocalCCRedoxConflicts(std::vector<molgr::state::MetalCandidateState> &candidates);
/// Penalize a localized C–C assignment only if its reduced partner preserves outside labels.
/// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
void AnnotateLocalCCRedoxAlternatives(std::vector<molgr::state::MetalCandidateState> &candidates);
}

namespace molgr::metal::geometry {
/// Remove the aryl carbenoid contribution from both structural and total discordance.
/// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
void ClearArylCarbenoidConflicts(std::vector<state::MetalCandidateState> &candidates);
/// Prefer a supported aryl carbene over a long exocyclic carbanion C=C.
/// @param candidates Validated metal-ligand candidates whose conflict metadata is updated in place.
void AnnotateArylCarbenoidAlternatives(std::vector<state::MetalCandidateState> &candidates);
}
