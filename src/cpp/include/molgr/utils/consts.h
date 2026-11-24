#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

namespace molgr
{
    /// Outer-shell electron count and reference valence used to classify nonmetal centers.
    struct ElementInfo
    {
        std::string symbol;
        std::string name;
        int atomic_number;
        double atomic_mass;
        int num_outer_electrons;
        int default_valence;
    };

    /// Store the equilibrium separation and energy scale used by the Lennard-Jones model.
    struct LennardJonesParams
    {
        double epsilon;
        double sigma;
        double cutoff;
    };

    /// Neutral-atom subshell occupations used when enumerating metal charge and spin.
    struct FDSP
    {
        int f, d, s, p;
    };

    extern const std::map<int, ElementInfo> kNonMetalDict;
    extern const std::map<int, double> kNonMetalPaulingElectronegativity;
    extern const std::map<int, double> kNonMetalFirstIonizationEnergyEv;
    extern const std::set<int> kHeteroatoms;
    extern const std::map<std::string, LennardJonesParams> kLJParams;
    extern const std::map<std::string, FDSP> kMetalFDSP;
    extern const std::vector<std::vector<int>> kDElectronsSpin;
    extern const std::map<std::string, std::vector<int>> kMetalValencePrior;
    extern const std::map<std::string, std::vector<int>> kMetalValenceMinor;

    /// @param metal_symbol Element symbol of the metal.
    /// @param valence Proposed formal oxidation state of the metal.
    /// @return Heuristic radical counts for a metal under a candidate valence.
    std::set<int> GetPossibleMetalRadicals(const std::string &metal_symbol, int valence);
}
