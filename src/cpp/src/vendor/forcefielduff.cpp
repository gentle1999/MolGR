/**********************************************************************
forcefielduff.cpp - UFF force field.

Copyright (C) 2007-2008 by Geoffrey Hutchison
Some portions Copyright (C) 2006-2008 by Tim Vandermeersch

This file is part of the Open Babel project.
For more information, see <http://openbabel.org/>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
************************************************************************/

#include <openbabel/babelconfig.h>
#include <openbabel/mol.h>
#include <openbabel/elements.h>
#include <openbabel/atom.h>
#include "molgr/compat/openbabel_iter.h"
#include <openbabel/generic.h>
#include <openbabel/bond.h>

#include "molgr/vendor/openbabel_copy.h"
#include "molgr/vendor/openbabel_threading.h"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <locale>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "molgr/utils/lru_cache.h"
#include "molgr/vendor/forcefielduff.h"


using namespace std;

// This implementation was created based on open code and reference websites:
// http://towhee.sourceforge.net/forcefields/uff.html
// http://rdkit.org/
// http://franklin.chm.colostate.edu/mmac/uff.html
// (for the last, use the Wayback Machine: http://www.archive.org/

// As well, the main UFF paper:
// Rappe, A. K., et. al.; J. Am. Chem. Soc. (1992) 114(25) p. 10024-10035.

namespace
{
  constexpr std::size_t kMolgrUffAtomTypeAssignmentCacheMaxSize = 4096;

#if defined(_WIN32)
  void MolgrUffModuleAnchor() {}

  std::filesystem::path MolgrInstalledUffParamPath()
  {
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&MolgrUffModuleAnchor), &module)) {
      return {};
    }

    std::wstring module_path(32768, L'\0');
    const DWORD path_length = GetModuleFileNameW(
        module, module_path.data(), static_cast<DWORD>(module_path.size()));
    if (path_length == 0 || path_length >= module_path.size()) {
      return {};
    }
    module_path.resize(path_length);
    return std::filesystem::path(module_path).parent_path() / L"data" / L"UFF.prm";
  }
#endif

  bool ParseClassicDouble(const std::string &text, double *value)
  {
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    stream >> *value;
    return stream && stream.eof();
  }

  struct MolgrUffAtomTypeRule
  {
    std::string smarts;
    std::string atom_type;
  };

  struct MolgrUffAtomPredicate
  {
    enum class Aromaticity
    {
      Any,
      Aromatic,
      Aliphatic,
    };

    int atomic_num = -1;
    Aromaticity aromaticity = Aromaticity::Any;
    std::optional<int> explicit_degree;
    std::optional<int> total_h_count;
    std::optional<int> ring_size;
    std::optional<int> formal_charge;
    std::optional<int> hybridization;
  };

  struct MolgrCompiledUffAtomTypeRule
  {
    std::string smarts;
    std::string atom_type;
    MolgrUffAtomPredicate atom;
    std::optional<MolgrUffAtomPredicate> default_bond_neighbor;
  };

  struct MolgrUffSharedData
  {
    bool loaded = false;
    std::vector<OpenBabel::OBFFParameter> ffparams;
    std::unordered_map<std::string, std::size_t> ffparam_index;
    std::vector<MolgrUffAtomTypeRule> atom_type_rules;
  };

  bool ConsumeChar(const std::string &text, std::size_t &pos, char expected)
  {
    if (pos < text.size() && text[pos] == expected) {
      ++pos;
      return true;
    }
    return false;
  }

  std::optional<int> ConsumeUnsignedInt(const std::string &text, std::size_t &pos)
  {
    if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
      return std::nullopt;
    }
    int value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
      value = value * 10 + (text[pos] - '0');
      ++pos;
    }
    return value;
  }

  bool ConsumeAtomicNumberPrimitive(
      const std::string &text,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    if (!ConsumeChar(text, pos, '#')) {
      return false;
    }
    std::optional<int> atomic_num = ConsumeUnsignedInt(text, pos);
    if (!atomic_num.has_value()) {
      return false;
    }
    predicate.atomic_num = *atomic_num;
    return true;
  }

  bool ConsumeElementPrimitive(
      const std::string &text,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    if (pos >= text.size()) {
      return false;
    }

    switch (text[pos]) {
    case 'B':
      ++pos;
      predicate.atomic_num = 5;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aliphatic;
      return true;
    case 'C':
      ++pos;
      predicate.atomic_num = 6;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aliphatic;
      return true;
    case 'N':
      ++pos;
      predicate.atomic_num = 7;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aliphatic;
      return true;
    case 'O':
      ++pos;
      predicate.atomic_num = 8;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aliphatic;
      return true;
    case 'S':
      ++pos;
      predicate.atomic_num = 16;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aliphatic;
      return true;
    case 'c':
      ++pos;
      predicate.atomic_num = 6;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aromatic;
      return true;
    case 'n':
      ++pos;
      predicate.atomic_num = 7;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aromatic;
      return true;
    case 'o':
      ++pos;
      predicate.atomic_num = 8;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aromatic;
      return true;
    case 's':
      ++pos;
      predicate.atomic_num = 16;
      predicate.aromaticity = MolgrUffAtomPredicate::Aromaticity::Aromatic;
      return true;
    default:
      return false;
    }
  }

  bool ConsumeFormalChargePrimitive(
      const std::string &text,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    if (pos >= text.size() || (text[pos] != '+' && text[pos] != '-')) {
      return false;
    }
    const int sign = text[pos] == '-' ? -1 : 1;
    ++pos;
    std::optional<int> magnitude = ConsumeUnsignedInt(text, pos);
    predicate.formal_charge = sign * magnitude.value_or(1);
    return true;
  }

  bool ConsumeAtomPrimitive(
      const std::string &text,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    if (pos >= text.size()) {
      return false;
    }

    if (text[pos] == '#') {
      return ConsumeAtomicNumberPrimitive(text, pos, predicate);
    }
    if (text[pos] == '+' || text[pos] == '-') {
      return ConsumeFormalChargePrimitive(text, pos, predicate);
    }
    if (ConsumeElementPrimitive(text, pos, predicate)) {
      return true;
    }

    switch (text[pos]) {
    case 'D': {
      ++pos;
      std::optional<int> degree = ConsumeUnsignedInt(text, pos);
      predicate.explicit_degree = degree.value_or(1);
      return true;
    }
    case 'H': {
      ++pos;
      std::optional<int> total_h_count = ConsumeUnsignedInt(text, pos);
      predicate.total_h_count = total_h_count.value_or(1);
      return true;
    }
    case 'r': {
      ++pos;
      std::optional<int> ring_size = ConsumeUnsignedInt(text, pos);
      if (!ring_size.has_value()) {
        return false;
      }
      predicate.ring_size = *ring_size;
      return true;
    }
    case '^': {
      ++pos;
      std::optional<int> hybridization = ConsumeUnsignedInt(text, pos);
      predicate.hybridization = hybridization.value_or(1);
      return true;
    }
    default:
      return false;
    }
  }

  bool ConsumePredicateBody(
      const std::string &body,
      MolgrUffAtomPredicate &predicate)
  {
    std::size_t pos = 0;
    while (pos < body.size()) {
      if (!ConsumeAtomPrimitive(body, pos, predicate)) {
        return false;
      }
    }
    return true;
  }

  bool ParseBracketAtomPredicate(
      const std::string &smarts,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    if (!ConsumeChar(smarts, pos, '[')) {
      return false;
    }
    const std::size_t body_start = pos;
    while (pos < smarts.size() && smarts[pos] != ']') {
      ++pos;
    }
    if (pos >= smarts.size()) {
      return false;
    }
    const std::string body = smarts.substr(body_start, pos - body_start);
    ++pos;
    return ConsumePredicateBody(body, predicate);
  }

  bool ParseBareAtomPredicate(
      const std::string &smarts,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    return ConsumeElementPrimitive(smarts, pos, predicate);
  }

  bool ParseAtomPredicate(
      const std::string &smarts,
      std::size_t &pos,
      MolgrUffAtomPredicate &predicate)
  {
    if (pos < smarts.size() && smarts[pos] == '[') {
      return ParseBracketAtomPredicate(smarts, pos, predicate);
    }
    return ParseBareAtomPredicate(smarts, pos, predicate);
  }

  MolgrCompiledUffAtomTypeRule CompileMolgrUffAtomTypeRule(
      const MolgrUffAtomTypeRule &rule)
  {
    MolgrCompiledUffAtomTypeRule compiled;
    compiled.smarts = rule.smarts;
    compiled.atom_type = rule.atom_type;

    std::size_t pos = 0;
    if (!ParseAtomPredicate(rule.smarts, pos, compiled.atom) ||
        compiled.atom.atomic_num < 0) {
      throw std::runtime_error("Unsupported UFF atom-typing SMARTS: " + rule.smarts);
    }

    if (pos < rule.smarts.size()) {
      MolgrUffAtomPredicate neighbor;
      if (!ParseAtomPredicate(rule.smarts, pos, neighbor) ||
          neighbor.atomic_num < 0 ||
          pos != rule.smarts.size()) {
        throw std::runtime_error("Unsupported UFF atom-typing SMARTS: " + rule.smarts);
      }
      compiled.default_bond_neighbor = neighbor;
    }

    return compiled;
  }

  bool HasSimplePathWithExactLength(
      OpenBabel::OBAtom &current,
      int target_idx,
      int remaining_edges,
      int blocked_atom_idx,
      std::vector<unsigned char> &visited)
  {
    const int current_idx = static_cast<int>(current.GetIdx());
    if (remaining_edges == 0) {
      return current_idx == target_idx;
    }
    if (current_idx == target_idx) {
      return false;
    }

    visited[static_cast<std::size_t>(current_idx)] = 1;
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBAtom *neighbor = current.BeginNbrAtom(bond_iter);
         neighbor != nullptr;
         neighbor = current.NextNbrAtom(bond_iter)) {
      const int neighbor_idx = static_cast<int>(neighbor->GetIdx());
      if (neighbor_idx == blocked_atom_idx) {
        continue;
      }
      if (neighbor_idx == target_idx && remaining_edges == 1) {
        visited[static_cast<std::size_t>(current_idx)] = 0;
        return true;
      }
      if (neighbor_idx < 0 ||
          static_cast<std::size_t>(neighbor_idx) >= visited.size() ||
          visited[static_cast<std::size_t>(neighbor_idx)] != 0) {
        continue;
      }
      if (HasSimplePathWithExactLength(
              *neighbor,
              target_idx,
              remaining_edges - 1,
              blocked_atom_idx,
              visited)) {
        visited[static_cast<std::size_t>(current_idx)] = 0;
        return true;
      }
    }
    visited[static_cast<std::size_t>(current_idx)] = 0;
    return false;
  }

  bool RingSizeMatchesWithoutLazySssr(OpenBabel::OBAtom &atom, int ring_size)
  {
    OpenBabel::OBMol *mol = atom.GetParent();
    if (mol == nullptr || ring_size < 3) {
      return false;
    }

    std::vector<OpenBabel::OBAtom*> neighbors;
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBAtom *neighbor = atom.BeginNbrAtom(bond_iter);
         neighbor != nullptr;
         neighbor = atom.NextNbrAtom(bond_iter)) {
      neighbors.push_back(neighbor);
    }

    const int atom_idx = static_cast<int>(atom.GetIdx());
    const int path_edges = ring_size - 2;
    for (std::size_t first = 0; first < neighbors.size(); ++first) {
      for (std::size_t second = first + 1; second < neighbors.size(); ++second) {
        std::vector<unsigned char> visited(
            static_cast<std::size_t>(mol->NumAtoms()) + 1,
            0);
        visited[static_cast<std::size_t>(atom_idx)] = 1;
        if (HasSimplePathWithExactLength(
                *neighbors[first],
                static_cast<int>(neighbors[second]->GetIdx()),
                path_edges,
                atom_idx,
                visited)) {
          return true;
        }
      }
    }
    return false;
  }

  int ExplicitValenceWithoutLazyPerception(const OpenBabel::OBAtom &atom)
  {
    int valence = 0;
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBBond *bond =
             const_cast<OpenBabel::OBAtom &>(atom).BeginBond(bond_iter);
         bond != nullptr;
         bond = const_cast<OpenBabel::OBAtom &>(atom).NextBond(bond_iter)) {
      valence += bond->GetBondOrder();
    }
    return valence;
  }

  int TotalHydrogenCountWithoutLazyPerception(const OpenBabel::OBAtom &atom)
  {
    int hydrogen_count = atom.GetImplicitHCount();
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBAtom *neighbor =
             const_cast<OpenBabel::OBAtom &>(atom).BeginNbrAtom(bond_iter);
         neighbor != nullptr;
         neighbor = const_cast<OpenBabel::OBAtom &>(atom).NextNbrAtom(bond_iter)) {
      if (neighbor->GetAtomicNum() == OpenBabel::OBElements::Hydrogen) {
        ++hydrogen_count;
      }
    }
    return hydrogen_count;
  }

  bool HasNeighborThroughDefaultSmartsBond(
      OpenBabel::OBAtom &atom,
      const MolgrUffAtomPredicate &neighbor_predicate);

  bool HasDoubleBondToAny(OpenBabel::OBAtom &atom)
  {
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBBond *bond = atom.BeginBond(bond_iter);
         bond != nullptr;
         bond = atom.NextBond(bond_iter)) {
      if (bond->GetBondOrder() == 2 && !bond->IsAromatic()) {
        return true;
      }
    }
    return false;
  }

  bool HasTripleBondToAny(OpenBabel::OBAtom &atom)
  {
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBBond *bond = atom.BeginBond(bond_iter);
         bond != nullptr;
         bond = atom.NextBond(bond_iter)) {
      if (bond->GetBondOrder() == 3) {
        return true;
      }
    }
    return false;
  }

  int MolgrUffSubsetHybridization(OpenBabel::OBAtom &atom)
  {
    const unsigned int perceived_hybridization = atom.GetHyb();
    if (perceived_hybridization != 0) {
      return static_cast<int>(perceived_hybridization);
    }

    const int atomic_num = static_cast<int>(atom.GetAtomicNum());
    const int degree = static_cast<int>(atom.GetExplicitDegree());
    const int total_valence =
        ExplicitValenceWithoutLazyPerception(atom) + atom.GetImplicitHCount();

    if (degree == 4) {
      return 3;
    }
    if (degree == 5) {
      return 5;
    }
    if (degree == 6) {
      return 6;
    }

    const bool is_aromatic = molgr::vendor::openbabel_threading::AtomIsAromatic(atom);
    if (atomic_num == OpenBabel::OBElements::Carbon) {
      if (HasTripleBondToAny(atom)) {
        return 1;
      }
      if (HasDoubleBondToAny(atom) || is_aromatic) {
        return 2;
      }
      return 3;
    }

    if (atomic_num == OpenBabel::OBElements::Nitrogen) {
      if (HasTripleBondToAny(atom)) {
        return 1;
      }
      if (HasDoubleBondToAny(atom) || is_aromatic) {
        return 2;
      }
      return 3;
    }

    if (atomic_num == OpenBabel::OBElements::Oxygen) {
      if (HasTripleBondToAny(atom)) {
        return 1;
      }
      if (HasDoubleBondToAny(atom) || is_aromatic) {
        return 2;
      }
      return 3;
    }

    if (atomic_num == OpenBabel::OBElements::Sulfur) {
      if (degree == 6) {
        return 6;
      }
      if (HasDoubleBondToAny(atom) || is_aromatic) {
        return 2;
      }
      return 3;
    }

    if (atomic_num == OpenBabel::OBElements::Boron) {
      return degree == 4 ? 3 : 2;
    }
    if (atomic_num == OpenBabel::OBElements::Phosphorus) {
      if (degree == 5 || total_valence == 5) {
        return 5;
      }
      if (degree == 1 && HasDoubleBondToAny(atom)) {
        return 2;
      }
      return 3;
    }

    switch (atomic_num) {
    case OpenBabel::OBElements::Beryllium:
    case OpenBabel::OBElements::Magnesium:
    case OpenBabel::OBElements::Calcium:
    case OpenBabel::OBElements::Strontium:
    case OpenBabel::OBElements::Barium:
    case OpenBabel::OBElements::Radium:
      return 1;
    case OpenBabel::OBElements::Aluminium:
    case OpenBabel::OBElements::Gallium:
    case OpenBabel::OBElements::Indium:
    case OpenBabel::OBElements::Thallium:
      return degree == 4 ? 3 : 2;
    case OpenBabel::OBElements::Silicon:
    case OpenBabel::OBElements::Lead:
    case OpenBabel::OBElements::Germanium:
    case OpenBabel::OBElements::Tin:
    case OpenBabel::OBElements::Selenium:
    case OpenBabel::OBElements::Tellurium:
    case OpenBabel::OBElements::Polonium:
      return 3;
    case OpenBabel::OBElements::Arsenic:
    case OpenBabel::OBElements::Antimony:
      return degree == 3 ? 3 : (degree == 5 ? 5 : degree);
    case OpenBabel::OBElements::Bismuth:
      if (degree == 3) {
        return 3;
      }
      if (degree == 5) {
        return 6;
      }
      return degree;
    default:
      if (degree <= 2) {
        return 1;
      }
      if (degree == 3) {
        return 2;
      }
      if (degree == 4) {
        return 3;
      }
      return degree;
    }
  }

  bool AtomMatchesUffPredicate(
      OpenBabel::OBAtom &atom,
      const MolgrUffAtomPredicate &predicate)
  {
    if (predicate.atomic_num >= 0 &&
        static_cast<int>(atom.GetAtomicNum()) != predicate.atomic_num) {
      return false;
    }
    if (predicate.aromaticity == MolgrUffAtomPredicate::Aromaticity::Aromatic &&
        !molgr::vendor::openbabel_threading::AtomIsAromatic(atom)) {
      return false;
    }
    if (predicate.aromaticity == MolgrUffAtomPredicate::Aromaticity::Aliphatic &&
        molgr::vendor::openbabel_threading::AtomIsAromatic(atom)) {
      return false;
    }
    if (predicate.explicit_degree.has_value() &&
        static_cast<int>(atom.GetExplicitDegree()) != *predicate.explicit_degree) {
      return false;
    }
    if (predicate.total_h_count.has_value() &&
        TotalHydrogenCountWithoutLazyPerception(atom) != *predicate.total_h_count) {
      return false;
    }
    if (predicate.ring_size.has_value() &&
        !RingSizeMatchesWithoutLazySssr(atom, *predicate.ring_size)) {
      return false;
    }
    if (predicate.formal_charge.has_value() &&
        atom.GetFormalCharge() != *predicate.formal_charge) {
      return false;
    }
    if (predicate.hybridization.has_value() &&
        MolgrUffSubsetHybridization(atom) != *predicate.hybridization) {
      return false;
    }
    return true;
  }

  bool HasNeighborThroughDefaultSmartsBond(
      OpenBabel::OBAtom &atom,
      const MolgrUffAtomPredicate &neighbor_predicate)
  {
    OpenBabel::OBBondIterator bond_iter;
    for (OpenBabel::OBBond *bond = atom.BeginBond(bond_iter);
         bond != nullptr;
         bond = atom.NextBond(bond_iter)) {
      const bool default_bond_matches =
          bond->GetBondOrder() == 1 ||
          molgr::vendor::openbabel_threading::BondIsAromatic(*bond);
      if (!default_bond_matches) {
        continue;
      }
      OpenBabel::OBAtom *neighbor = bond->GetNbrAtom(&atom);
      if (neighbor != nullptr && AtomMatchesUffPredicate(*neighbor, neighbor_predicate)) {
        return true;
      }
    }
    return false;
  }

  bool AtomMatchesUffRule(
      OpenBabel::OBAtom &atom,
      const MolgrCompiledUffAtomTypeRule &rule)
  {
    if (!AtomMatchesUffPredicate(atom, rule.atom)) {
      return false;
    }
    if (!rule.default_bond_neighbor.has_value()) {
      return true;
    }
    return HasNeighborThroughDefaultSmartsBond(atom, *rule.default_bond_neighbor);
  }

  MolgrUffSharedData LoadMolgrUffSharedData()
  {
    MolgrUffSharedData data;
    std::vector<std::string> vs;
    char buffer[BUFF_SIZE];

    OpenBabel::OBFFParameter parameter;
    std::ifstream ifs;
    OpenBabel::OpenDatafile(ifs, "UFF.prm");
#if defined(_WIN32)
    if (!ifs.is_open()) {
      ifs.clear();
      const auto installed_path = MolgrInstalledUffParamPath();
      if (!installed_path.empty()) {
        ifs.open(installed_path, std::ios::in);
      }
    }
#endif
    if (!ifs.is_open()) {
      return data;
    }

    while (ifs.getline(buffer, BUFF_SIZE)) {
      OpenBabel::tokenize(vs, buffer);
      if (EQn(buffer, "atom", 4) && vs.size() >= 3) {
        data.atom_type_rules.push_back(MolgrUffAtomTypeRule{vs[1], vs[2]});
        continue;
      }

      if (!EQn(buffer, "param", 5) || vs.size() < 13) {
        continue;
      }

      parameter.clear();
      parameter._a = vs[1];
      bool parameter_parse_ok = true;
      for (std::size_t index = 2; index < 13; ++index) {
        double value = 0.0;
        if (!ParseClassicDouble(vs[index], &value)) {
          parameter_parse_ok = false;
          break;
        }
        parameter._dpar.push_back(value);
      }
      if (!parameter_parse_ok) {
        continue;
      }

      parameter.b = 0;
      parameter.c = 0;

      // UFF type labels are usually three characters wide (e.g. "C_3", "C_R"),
      // but the parameter file also contains short labels such as "H_", "Li",
      // "Cl", and "Du". The legacy OpenBabel parser indexed type[2]
      // unconditionally, which turns these entries into undefined behavior.
      // All short labels in UFF.prm are effectively monocoordinate types, so
      // handle them explicitly instead of reading past the end of the string.
      const char coord = vs[1].size() > 2 ? vs[1][2] : '1';
      switch (coord) {
      case '1':
        parameter._ipar.push_back(1);
        break;
      case '2':
      case 'R':
        parameter._ipar.push_back(2);
        break;
      case '3':
        parameter._ipar.push_back(3);
        break;
      case '4':
        parameter._ipar.push_back(4);
        break;
      case '5':
        parameter._ipar.push_back(5);
        break;
      case '6':
        parameter._ipar.push_back(6);
        break;
      case '7':
        parameter._ipar.push_back(7);
        break;
      default:
        parameter._ipar.push_back(1);
        break;
      }

      data.ffparam_index[parameter._a] = data.ffparams.size();
      data.ffparams.push_back(parameter);
    }
    data.loaded = !data.ffparams.empty() && !data.atom_type_rules.empty();
    return data;
  }

  const MolgrUffSharedData& GetMolgrUffSharedData()
  {
    static const MolgrUffSharedData data = LoadMolgrUffSharedData();
    return data;
  }

  bool GetMolgrCompiledUffAtomTypeRules(
      const std::vector<MolgrCompiledUffAtomTypeRule> *&compiled_rules)
  {
    static const std::vector<MolgrCompiledUffAtomTypeRule> rules = []()
    {
      const auto &shared = GetMolgrUffSharedData();
      std::vector<MolgrCompiledUffAtomTypeRule> compiled;
      compiled.reserve(shared.atom_type_rules.size());
      for (const auto &rule : shared.atom_type_rules) {
        compiled.push_back(CompileMolgrUffAtomTypeRule(rule));
      }
      return compiled;
    }();
    compiled_rules = &rules;
    return compiled_rules != nullptr && !compiled_rules->empty();
  }

  molgr::utils::StringLruCache<std::vector<std::string>>& MolgrUffAtomTypeAssignmentCache()
  {
    static molgr::utils::StringLruCache<std::vector<std::string>> cache(
        kMolgrUffAtomTypeAssignmentCacheMaxSize);
    return cache;
  }

  std::vector<std::string> CaptureMolgrUffAtomTypes(OpenBabel::OBMol &mol)
  {
    std::vector<std::string> atom_types;
    atom_types.reserve(static_cast<std::size_t>(mol.NumAtoms()));
    FOR_ATOMS_OF_MOL(atom_iter, mol) {
      atom_types.emplace_back(atom_iter->GetType());
    }
    return atom_types;
  }

  bool ApplyMolgrUffAtomTypes(
      OpenBabel::OBMol &mol,
      const std::vector<std::string> &atom_types)
  {
    if (atom_types.size() != static_cast<std::size_t>(mol.NumAtoms())) {
      return false;
    }

    for (std::size_t atom_index = 0; atom_index < atom_types.size(); ++atom_index) {
      OpenBabel::OBAtom *atom = mol.GetAtom(static_cast<int>(atom_index + 1));
      if (atom == nullptr) {
        return false;
      }
      atom->SetType(atom_types[atom_index].c_str());
    }
    return true;
  }
}

namespace OpenBabel {

  thread_local MolgrForceFieldUFF* MolgrForceFieldUFF::active_instance_ = nullptr;

  template<bool gradients>
  void MolgrFFBondCalculationUFF::Compute()
  {
    if (MolgrForceFieldUFF::IgnoreCalculation(idx_a, idx_b)) {
      energy = 0.0;
      return;
    }

    vector3 vab, da, db;
    double delta2, dE;

    if (gradients) {
      rab = OBForceField::VectorBondDerivative(pos_a, pos_b, force_a, force_b);
    } else {
      rab = OBForceField::VectorDistance(pos_a, pos_b);
    }

    // Harmonic bond stretching
    delta = rab - r0; // we pre-compute the r0 below
    delta2 = delta * delta;
    energy = kb * delta2; // we fold the 1/2 into kb below

    if (gradients) {
      dE = 2.0 * kb * delta;
      OBForceField::VectorSelfMultiply(force_a, dE);
      OBForceField::VectorSelfMultiply(force_b, dE);
    }
  }

  template<bool gradients>
  double MolgrForceFieldUFF::E_Bond()
  {
    vector<MolgrFFBondCalculationUFF>::iterator i;
    double energy = 0.0;

    IF_OBFF_LOGLVL_HIGH {
      OBFFLog("\nB O N D   S T R E T C H I N G\n\n");
      OBFFLog("ATOM TYPES  BOND    BOND       IDEAL       FORCE\n");
      OBFFLog(" I      J   TYPE   LENGTH     LENGTH     CONSTANT      DELTA      ENERGY\n");
      OBFFLog("------------------------------------------------------------------------\n");
    }

    for (i = _bondcalculations.begin(); i != _bondcalculations.end(); ++i) {

      i->template Compute<gradients>();
      energy += i->energy;

      if (gradients) {
        AddGradient((*i).force_a, (*i).idx_a);
        AddGradient((*i).force_b, (*i).idx_b);
      }

      IF_OBFF_LOGLVL_HIGH {
        snprintf(_logbuf, BUFF_SIZE, "%-5s %-5s  %4.2f%8.3f   %8.3f     %8.3f   %8.3f   %8.3f\n",
                 (*i).a->GetType(), (*i).b->GetType(),
                 (*i).bt, (*i).rab, (*i).r0, (*i).kb, (*i).delta, (*i).energy);
        OBFFLog(_logbuf);
      }
    }

    IF_OBFF_LOGLVL_MEDIUM {
      snprintf(_logbuf, BUFF_SIZE, "     TOTAL BOND STRETCHING ENERGY = %8.3f %s\n",  energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }
    return energy;
  }

  template<bool gradients>
  void MolgrFFAngleCalculationUFF::Compute()
  {
    if (MolgrForceFieldUFF::IgnoreCalculation(idx_a, idx_b, idx_c)) {
      energy = 0.0;
      return;
    }

    vector3 da, db, dc;
		double dE;

    if (gradients) {
      theta = OBForceField::VectorAngleDerivative(pos_a, pos_b, pos_c, force_a, force_b, force_c);

      // Supply a small nudge if the angle is degenerate
      if (theta < 2.5 || theta > 357.5) {
        vector3 v1;
        v1.randomUnitVector();
        for (int i = 0; i < 3; ++i)
          force_a[i] += v1[i]*0.1;
      }
      theta *= DEG_TO_RAD;
    } else {
      theta = a->GetAngle(b, c) * DEG_TO_RAD;
		}

    if (!isfinite(theta))
      theta = 0.0; // doesn't explain why GetAngle is returning NaN but solves it for us

    double cosT;

    switch (coord) {
    case 1: // sp -- linear case, minima at 180 degrees, max (amplitude 2*ka) at 0, 360
      // Fixed typo from Rappe paper (i.e., it's NOT 1 - cosT)
      energy = ka*(1.0 + cos(theta));
      break;
    case 2: // sp2 -- trigonal planar or equatorial plane of trigonal bipyramidal
    case 4: // square planar
    case 6: // octahedral
      // ka already is pre-computed as ka/n^2 to save CPU cycles
      // UNLIKE Rappe paper, we add a penalty for angles close to zero, based on ESFF
      // i.e., if the angle is less than approx theta0, energy goes up exponentially
      energy = ka * (1 - cos(n*theta)) + exp(-20.0*(theta - theta0 + 0.25));
      break;
    case 7: // IF7 pentagonal -- pentagonal bipyramidal
      /* theta = 1/5 * 2 pi.  cosT = .30901699
       * theta = 2/5 * 2 pi.  cosT = -.80901699
       * theta = 3/5 * 2 pi.  cosT = -.80901699
       * theta = 4/5 * 2 pi.  cosT = .30901699
       */
      cosT = cos(theta);
      energy = ka * c1 * (cosT - .30901699) * (cosT - .30906199) * (cosT + .80901699) * (cosT + .8091699);
      break;
    default: // general (sp3) coordination
      cosT = cos(theta);
      energy = ka*(c0 + c1*cosT + c2*(2.0*cosT*cosT - 1.0)); // use cos 2t = (2cos^2 - 1)
    }

    if (gradients) {
      double sinT;

      switch (coord) {
      case 1: // sp -- linear case
        dE = -ka * sin(theta);
        break;
      case 2: // sp2 -- trigonal planar
      case 6: // octahedral
      case 4: // square planar
        dE = ka * n * sin(n * theta)  -20.0 * exp(-20.0*(theta - theta0 + 0.25));
        break;
      case 7: // pentagonal bipyramidal
        sinT = sin(theta);
        cosT = cos(theta);
        dE =
          c1 * -ka * (2 * sinT * (cosT - .30906199) * (cosT + .80901699) * (cosT + .8091699) +
                      2 * sinT * (cosT - .30901699) * (cosT - .30906199) * (cosT + .8091699));
        //dE = -ka * c1 * sin(5*theta) * 5;
        break;
      default: // general (sp3) coordination
        dE = -ka * (c1*sin(theta) + 2.0 * c2*sin(2 * theta));
      }

      OBForceField::VectorSelfMultiply(force_a, dE);
      OBForceField::VectorSelfMultiply(force_b, dE);
      OBForceField::VectorSelfMultiply(force_c, dE);
    }
  }

  template<bool gradients>
  double MolgrForceFieldUFF::E_Angle()
  {
    vector<MolgrFFAngleCalculationUFF>::iterator i;
    double energy = 0.0;

    IF_OBFF_LOGLVL_HIGH {
      OBFFLog("\nA N G L E   B E N D I N G\n\n");
      OBFFLog("ATOM TYPES       VALENCE     IDEAL      FORCE\n");
      OBFFLog(" I    J    K      ANGLE      ANGLE     CONSTANT      DELTA      ENERGY\n");
      OBFFLog("-----------------------------------------------------------------------------\n");
    }

    for (i = _anglecalculations.begin(); i != _anglecalculations.end(); ++i) {

      i->template Compute<gradients>();
      energy += i->energy;

      if (gradients) {
        AddGradient((*i).force_a, (*i).idx_a);
        AddGradient((*i).force_b, (*i).idx_b);
        AddGradient((*i).force_c, (*i).idx_c);
      }

      IF_OBFF_LOGLVL_HIGH {
        snprintf(_logbuf, BUFF_SIZE, "%-5s %-5s %-5s%8.3f  %8.3f     %8.3f   %8.3f   %8.3f\n", (*i).a->GetType(), (*i).b->GetType(),
                 (*i).c->GetType(), (*i).theta * RAD_TO_DEG, (*i).theta0, (*i).ka, (*i).delta, (*i).energy);
        OBFFLog(_logbuf);
      }
    }

    IF_OBFF_LOGLVL_MEDIUM {
      snprintf(_logbuf, BUFF_SIZE, "     TOTAL ANGLE BENDING ENERGY = %8.3f %s\n", energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }
    return energy;
  }

  template<bool gradients>
  void MolgrFFTorsionCalculationUFF::Compute()
  {
    if (MolgrForceFieldUFF::IgnoreCalculation(idx_a, idx_b, idx_c, idx_d)) {
      energy = 0.0;
      return;
    }

    vector3 da, db, dc, dd;
    double cosine;
    double dE;

    if (gradients) {
      tor = OBForceField::VectorTorsionDerivative(pos_a, pos_b, pos_c, pos_d,
                                                  force_a, force_b, force_c, force_d);
      if (!isfinite(tor))
        tor = 1.0e-3;
      tor *= DEG_TO_RAD;
    } else {
      vector3 vab, vbc, vcd, abbc, bccd;
      vab = a->GetVector() - b->GetVector();
      vbc = b->GetVector() - c->GetVector();
      vcd = c->GetVector() - d->GetVector();
      abbc = cross(vab, vbc);
      bccd = cross(vbc, vcd);

      double dotAbbcBccd = dot(abbc,bccd);
      tor = acos(dotAbbcBccd / (abbc.length() * bccd.length()));
      if (IsNearZero(dotAbbcBccd) || !isfinite(tor)) { // stop any NaN or infinity
        tor = 1.0e-3; // rather than NaN
      }
      else if (dotAbbcBccd > 0.0) {
        tor = -tor;
      }
    }

    cosine = cos(tor * n);
    energy = V * (1.0 - cosNPhi0*cosine);

    if (gradients) {
      dE = -(V * n * cosNPhi0 * sin(n * tor));
      OBForceField::VectorSelfMultiply(force_a, dE);
      OBForceField::VectorSelfMultiply(force_b, dE);
      OBForceField::VectorSelfMultiply(force_c, dE);
      OBForceField::VectorSelfMultiply(force_d, dE);
    }
  }

  template<bool gradients>
  double MolgrForceFieldUFF::E_Torsion()
  {
    vector<MolgrFFTorsionCalculationUFF>::iterator i;
    double energy = 0.0;

    IF_OBFF_LOGLVL_HIGH {
      OBFFLog("\nT O R S I O N A L\n\n");
      OBFFLog("----ATOM TYPES-----    FORCE         TORSION\n");
      OBFFLog(" I    J    K    L     CONSTANT        ANGLE         ENERGY\n");
      OBFFLog("----------------------------------------------------------------\n");
    }

    for (i = _torsioncalculations.begin(); i != _torsioncalculations.end(); ++i) {

      i->template Compute<gradients>();
      energy += i->energy;

      if (gradients) {
        AddGradient((*i).force_a, (*i).idx_a);
        AddGradient((*i).force_b, (*i).idx_b);
        AddGradient((*i).force_c, (*i).idx_c);
        AddGradient((*i).force_d, (*i).idx_d);
      }

      IF_OBFF_LOGLVL_HIGH {
        snprintf(_logbuf, BUFF_SIZE, "%-5s %-5s %-5s %-5s%6.3f       %8.3f     %8.3f\n",
                 (*i).a->GetType(), (*i).b->GetType(),
                 (*i).c->GetType(), (*i).d->GetType(), (*i).V,
                 (*i).tor * RAD_TO_DEG, (*i).energy);
        OBFFLog(_logbuf);
      }
    }

    IF_OBFF_LOGLVL_MEDIUM {
      snprintf(_logbuf, BUFF_SIZE, "     TOTAL TORSIONAL ENERGY = %8.3f %s\n", energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }

    return energy;
  }

  /*
  //  a
  //   \
  //    b---d      plane = a-b-c
  //   /
  //  c
  */
  template<bool gradients>
  void MolgrFFOOPCalculationUFF::Compute()
  {
    if (MolgrForceFieldUFF::IgnoreCalculation(idx_a, idx_b, idx_c, idx_d)) {
      energy = 0.0;
      return;
    }

    vector3 da, db, dc, dd;
    double dE;

    if (gradients) {
      angle = OBForceField::VectorOOPDerivative(pos_a, pos_b, pos_c, pos_d,
                                                force_a, force_b, force_c, force_d);
      angle *= DEG_TO_RAD;

	    if (!isfinite(angle))
	      angle = 0.0; // doesn't explain why GetAngle is returning NaN but solves it for us;

      // somehow we already get the -1 from the OOPDeriv -- so we'll omit it here
      dE = koop * (c1*sin(angle) + 2.0 * c2 * sin(2.0*angle));
      OBForceField::VectorSelfMultiply(force_a, dE);
      OBForceField::VectorSelfMultiply(force_b, dE);
      OBForceField::VectorSelfMultiply(force_c, dE);
      OBForceField::VectorSelfMultiply(force_d, dE);
    } else {
      angle = DEG_TO_RAD*Point2PlaneAngle(d->GetVector(), a->GetVector(), b->GetVector(), c->GetVector());
      if (!isfinite(angle))
        angle = 0.0; // doesn't explain why GetAngle is returning NaN but solves it for us;
    }

    energy = koop * (c0 + c1 * cos(angle) + c2 * cos(2.0*angle));
  }

  template<bool gradients>
  double MolgrForceFieldUFF::E_OOP()
  {
    vector<MolgrFFOOPCalculationUFF>::iterator i;
    double energy = 0.0;

    IF_OBFF_LOGLVL_HIGH {
      OBFFLog("\nO U T - O F - P L A N E   B E N D I N G\n\n");
      OBFFLog("ATOM TYPES                 OOP     FORCE \n");
      OBFFLog(" I    J     K     L       ANGLE   CONSTANT     ENERGY\n");
      OBFFLog("----------------------------------------------------------\n");
    }

    for (i = _oopcalculations.begin(); i != _oopcalculations.end(); ++i) {
      i->template Compute<gradients>();
      energy += i->energy;

      if (gradients) {
        AddGradient((*i).force_a, (*i).idx_a);
        AddGradient((*i).force_b, (*i).idx_b);
        AddGradient((*i).force_c, (*i).idx_c);
        AddGradient((*i).force_d, (*i).idx_d);
      }

      IF_OBFF_LOGLVL_HIGH {
        snprintf(_logbuf, BUFF_SIZE, "%-5s %-5s %-5s %-5s%8.3f   %8.3f     %8.3f\n", (*i).a->GetType(), (*i).b->GetType(), (*i).c->GetType(), (*i).d->GetType(),
                 (*i).angle * RAD_TO_DEG, (*i).koop, (*i).energy);
        OBFFLog(_logbuf);
      }
    }

    IF_OBFF_LOGLVL_HIGH {
      snprintf(_logbuf, BUFF_SIZE, "     TOTAL OUT-OF-PLANE BENDING ENERGY = %8.3f %s\n", energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }
    return energy;
  }

  template<bool gradients>
  void MolgrFFVDWCalculationUFF::Compute()
  {
    if (MolgrForceFieldUFF::IgnoreCalculation(idx_a, idx_b)) {
      energy = 0.0;
      return;
    }

    vector3 da, db;
    double term6, term12, dE, term7, term13, rabSquared = 0.0;

    if (gradients) {
      rab = OBForceField::VectorDistanceDerivative(pos_a, pos_b, force_a, force_b);

      if (rab < 1.0e-3)
        rab = 1.0e-3;

      rabSquared = SQUARE(rab);
    } else {
      // Get distance squared (saves a sqrt and multiply)
      // for every energy evaluation
      double ab[3];
      for (unsigned int c = 0; c < 3; ++c)
        rabSquared += SQUARE(a->GetCoordinate()[c] - b->GetCoordinate()[c]);

      // make sure the energy doesn't blow up
      if (rabSquared < 1.0e-5)
        rabSquared = 1.0e-5;
    }

    // TODO: This actually should include zetas (not always exactly 6-12 for VDW paper)

    term6 = kaSquared / rabSquared; // ^2
    term6 = term6 * term6 * term6; // ^6
    term12 = term6 * term6; // ^12

    energy = kab * ((term12) - (2.0 * term6));

    if (gradients) {
      term13 = term12 / rab; // ^13
      term7 = term6 / rab; // ^7
      dE = kab * 12.0 * (term7 - term13);
      OBForceField::VectorSelfMultiply(force_a, dE);
      OBForceField::VectorSelfMultiply(force_b, dE);
    }
  }

  template<bool gradients>
  double MolgrForceFieldUFF::E_VDW()
  {
    vector<MolgrFFVDWCalculationUFF>::iterator i;
    double energy = 0.0;

    IF_OBFF_LOGLVL_HIGH {
      OBFFLog("\nV A N   D E R   W A A L S\n\n");
      OBFFLog("ATOM TYPES\n");
      OBFFLog(" I    J        Rij       kij       ENERGY\n");
      OBFFLog("-----------------------------------------\n");
      //          XX   XX     -000.000  -000.000  -000.000  -000.000
    }

    unsigned int j = 0;
    for (i = _vdwcalculations.begin(); i != _vdwcalculations.end(); ++i, ++j) {
      // Cut-off check
      if (_cutoff)
        if (!_vdwpairs.BitIsSet(j))
          continue;

      i->template Compute<gradients>();
      energy += i->energy;

      if (gradients) {
        AddGradient((*i).force_a, (*i).idx_a);
        AddGradient((*i).force_b, (*i).idx_b);
      }

      IF_OBFF_LOGLVL_HIGH {
        snprintf(_logbuf, BUFF_SIZE, "%-5s %-5s %8.3f  %8.3f  %8.3f\n", (*i).a->GetType(), (*i).b->GetType(),
                 (*i).rab, (*i).kab, (*i).energy);
        OBFFLog(_logbuf);
      }
    }

    IF_OBFF_LOGLVL_MEDIUM {
      snprintf(_logbuf, BUFF_SIZE, "     TOTAL VAN DER WAALS ENERGY = %8.3f %s\n", energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }

    return energy;
  }

  template<bool gradients>
  void MolgrFFElectrostaticCalculationUFF::Compute()
  {
    if (MolgrForceFieldUFF::IgnoreCalculation(idx_a, idx_b)) {
      energy = 0.0;
      return;
    }

    vector3 da, db;
    double dE, rab2;

    if (gradients) {
      da = a->GetVector();
      db = b->GetVector();
      rab = OBForceField::VectorLengthDerivative(da, db);
    } else
      rab = a->GetDistance(b);

    if (IsNearZero(rab, 1.0e-3))
      rab = 1.0e-3;

    energy = qq / rab;

    if (gradients) {
      rab2 = rab * rab;
      dE = -qq / rab2;
      da *= dE;
      db *= dE;
      da.Get(force_a);
      db.Get(force_b);
    }
  }

  template<bool gradients>
  double MolgrForceFieldUFF::E_Electrostatic()
  {
    vector<MolgrFFElectrostaticCalculationUFF>::iterator i;
    double energy = 0.0;

    IF_OBFF_LOGLVL_HIGH {
      OBFFLog("\nE L E C T R O S T A T I C   I N T E R A C T I O N S\n\n");
      OBFFLog("ATOM TYPES\n");
      OBFFLog(" I    J           Rij   332.17*QiQj  ENERGY\n");
      OBFFLog("-------------------------------------------\n");
      //            XX   XX     -000.000  -000.000  -000.000
    }

    unsigned int j = 0;
    for (i = _electrostaticcalculations.begin(); i != _electrostaticcalculations.end(); ++i, ++j) {
      // Cut-off check
      if (_cutoff)
        if (!_elepairs.BitIsSet(j))
          continue;

      i->template Compute<gradients>();
      energy += i->energy;

      if (gradients) {
        AddGradient((*i).force_a, (*i).idx_a);
        AddGradient((*i).force_b, (*i).idx_b);
      }

      IF_OBFF_LOGLVL_HIGH {
        snprintf(_logbuf, BUFF_SIZE, "%-5s %-5s   %8.3f  %8.3f  %8.3f\n", (*i).a->GetType(), (*i).b->GetType(),
                 (*i).rab, (*i).qq, (*i).energy);
        OBFFLog(_logbuf);
      }
    }

    IF_OBFF_LOGLVL_MEDIUM {
      snprintf(_logbuf, BUFF_SIZE, "     TOTAL ELECTROSTATIC ENERGY = %8.3f %s\n", energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }

    return energy;
  }

  MolgrForceFieldUFF::~MolgrForceFieldUFF()
  {
    if (active_instance_ == this) {
      active_instance_ = nullptr;
    }
  }

  void MolgrForceFieldUFF::ActivateThreadLocalInstance()
  {
    active_instance_ = this;
  }

  MolgrForceFieldUFF* MolgrForceFieldUFF::ActiveInstance()
  {
    return active_instance_;
  }

  MolgrForceFieldUFF &MolgrForceFieldUFF::operator=(MolgrForceFieldUFF &src)
  {
    molgr::vendor::openbabel_threading::AssignMol(_mol, src._mol);

    _ffparams    = src._ffparams;
    _ffparam_index = src._ffparam_index;

    _bondcalculations          = src._bondcalculations;
    _anglecalculations         = src._anglecalculations;
    _torsioncalculations       = src._torsioncalculations;
    _oopcalculations           = src._oopcalculations;
    _vdwcalculations           = src._vdwcalculations;
    _electrostaticcalculations = src._electrostaticcalculations;
    _init                      = src._init;
    constraints_               = src.constraints_;
    fix_atom_                  = src.fix_atom_;
    ignore_atom_               = src.ignore_atom_;

    return *this;
  }

  OBFFConstraints& MolgrForceFieldUFF::GetConstraints()
  {
    return constraints_;
  }

  void MolgrForceFieldUFF::SetConstraints(OBFFConstraints& constraints)
  {
    ActivateThreadLocalInstance();
    if (!(constraints_.GetIgnoredBitVec() == constraints.GetIgnoredBitVec())) {
      constraints_ = constraints;
      if (!SetupCalculations()) {
        _validSetup = false;
        return;
      }
    } else {
      constraints_ = constraints;
    }

    constraints_.Setup(_mol);
  }

  void MolgrForceFieldUFF::SetFixAtom(int index)
  {
    fix_atom_ = static_cast<unsigned int>(index);
  }

  void MolgrForceFieldUFF::UnsetFixAtom()
  {
    fix_atom_ = 0;
  }

  void MolgrForceFieldUFF::SetIgnoreAtom(int index)
  {
    ignore_atom_ = static_cast<unsigned int>(index);
  }

  void MolgrForceFieldUFF::UnsetIgnoreAtom()
  {
    ignore_atom_ = 0;
  }

  bool MolgrForceFieldUFF::IgnoreCalculation(int a, int b)
  {
    MolgrForceFieldUFF* instance = ActiveInstance();
    if (instance == nullptr || instance->ignore_atom_ == 0)
      return false;

    return instance->ignore_atom_ == static_cast<unsigned int>(a) ||
           instance->ignore_atom_ == static_cast<unsigned int>(b);
  }

  bool MolgrForceFieldUFF::IgnoreCalculation(int a, int b, int c)
  {
    return IgnoreCalculation(a, c) ||
           (ActiveInstance() != nullptr &&
            ActiveInstance()->ignore_atom_ == static_cast<unsigned int>(b));
  }

  bool MolgrForceFieldUFF::IgnoreCalculation(int a, int b, int c, int d)
  {
    return IgnoreCalculation(a, b, c) ||
           (ActiveInstance() != nullptr &&
            ActiveInstance()->ignore_atom_ == static_cast<unsigned int>(d));
  }

  void MolgrForceFieldUFF::ConfigureAtomTypingCache(bool enabled, std::string cache_key)
  {
    use_atom_typing_cache_ = enabled;
    atom_typing_cache_key_ = std::move(cache_key);
  }

  void MolgrForceFieldUFF::CopyPerceivedHybridizationTo(OBMol &mol) const
  {
    if (_mol.NumAtoms() != mol.NumAtoms()) {
      return;
    }
    OpenBabel::OBMol &source_mol = const_cast<OpenBabel::OBMol &>(_mol);
    FOR_ATOMS_OF_MOL(source_atom, source_mol) {
      OpenBabel::OBAtom *target_atom = mol.GetAtom(source_atom->GetIdx());
      if (target_atom != nullptr) {
        target_atom->SetHyb(source_atom->GetHyb());
      }
    }
    mol.SetHybridizationPerceived(source_mol.HasHybridizationPerceived());
  }

  std::vector<std::string> MolgrForceFieldUFF::DebugAtomTypes() const
  {
    std::vector<std::string> atom_types;
    OpenBabel::OBMol &mutable_mol = const_cast<OpenBabel::OBMol &>(_mol);
    atom_types.reserve(static_cast<std::size_t>(mutable_mol.NumAtoms()));
    FOR_ATOMS_OF_MOL(atom_iter, mutable_mol) {
      atom_types.emplace_back(atom_iter->GetType());
    }
    return atom_types;
  }

  bool MolgrForceFieldUFF::Setup(OBMol &mol)
  {
    ActivateThreadLocalInstance();
    if (!_init) {
      ParseParamFile();
      _init = true;
      _velocityPtr = nullptr;
      _gradientPtr = nullptr;
      _grad1 = nullptr;
    }

    if (IsSetupNeeded(mol)) {
      molgr::vendor::openbabel_threading::AssignMol(_mol, mol);
      _ncoords = _mol.NumAtoms() * 3;

      delete [] _velocityPtr;
      _velocityPtr = nullptr;

      delete [] _gradientPtr;
      _gradientPtr = new double[_ncoords];

      if (_mol.NumAtoms() && constraints_.Size())
        constraints_.Setup(_mol);

      _mol.SetSSSRPerceived(false);
      _mol.DeleteData(OBGenericDataType::TorsionData);

      if (!SetTypes()) {
        _validSetup = false;
        return false;
      }

      SetFormalCharges();
      SetPartialCharges();

      if (!SetupCalculations()) {
        _validSetup = false;
        return false;
      }
    } else {
      if (_validSetup) {
        PrintTypes();
        PrintFormalCharges();
        PrintPartialCharges();
        SetCoordinates(mol);
        return true;
      }
      return false;
    }

    _validSetup = true;
    return true;
  }

  bool MolgrForceFieldUFF::Setup(OBMol &mol, OBFFConstraints &constraints)
  {
    ActivateThreadLocalInstance();
    if (!_init) {
      ParseParamFile();
      _init = true;
      _velocityPtr = nullptr;
      _gradientPtr = nullptr;
      _grad1 = nullptr;
    }

    if (IsSetupNeeded(mol)) {
      molgr::vendor::openbabel_threading::AssignMol(_mol, mol);
      _ncoords = _mol.NumAtoms() * 3;

      delete [] _velocityPtr;
      _velocityPtr = nullptr;

      delete [] _gradientPtr;
      _gradientPtr = new double[_ncoords];

      constraints_ = constraints;
      if (_mol.NumAtoms() && constraints_.Size())
        constraints_.Setup(_mol);

      _mol.SetSSSRPerceived(false);
      _mol.DeleteData(OBGenericDataType::TorsionData);

      if (!SetTypes()) {
        _validSetup = false;
        return false;
      }

      SetFormalCharges();
      SetPartialCharges();

      if (!SetupCalculations()) {
        _validSetup = false;
        return false;
      }
    } else {
      if (_validSetup) {
        if (!(constraints_.GetIgnoredBitVec() == constraints.GetIgnoredBitVec())) {
          constraints_ = constraints;
          if (!SetupCalculations()) {
            _validSetup = false;
            return false;
          }
        } else {
          constraints_ = constraints;
        }

        constraints_.Setup(_mol);
        SetCoordinates(mol);
        return true;
      }
      return false;
    }

    _validSetup = true;
    return true;
  }

  double CalculateBondDistance(OBFFParameter *i, OBFFParameter *j, double bondorder)
  {
    double ri, rj;
    double chiI, chiJ;
    double rbo, ren;
    ri = i->_dpar[0];
    rj = j->_dpar[0];
    chiI = i->_dpar[8];
    chiJ = j->_dpar[8];

    // Precompute the equilibrium bond distance
    // From equation 3
    rbo = -0.1332*(ri+rj)*log(bondorder);
    // From equation 4
    ren = ri*rj*(pow((sqrt(chiI) - sqrt(chiJ)),2.0)) / (chiI*ri + chiJ*rj);
    // From equation 2
    // NOTE: See http://towhee.sourceforge.net/forcefields/uff.html
    // There is a typo in the published paper
    return(ri + rj + rbo - ren);
  }

  bool MolgrForceFieldUFF::SetupVDWCalculation(OBAtom *a, OBAtom *b, MolgrFFVDWCalculationUFF &vdwcalc)
  {
    OBFFParameter *parameterA, *parameterB;
    parameterA = GetParameterUFF(a->GetType(), _ffparams);
    parameterB = GetParameterUFF(b->GetType(), _ffparams);

    if (parameterA == NULL || parameterB == NULL) {
      IF_OBFF_LOGLVL_LOW {
        snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR VDW INTERACTION %d-%d (IDX)...\n",
                 a->GetIdx(), b->GetIdx());
        OBFFLog(_logbuf);
      }
      return false;
    }

    vdwcalc.Ra = parameterA->_dpar[2];
    vdwcalc.ka = parameterA->_dpar[3];
    vdwcalc.Rb = parameterB->_dpar[2];
    vdwcalc.kb = parameterB->_dpar[3];

    vdwcalc.a = &*a;
    vdwcalc.b = &*b;

    //this calculations only need to be done once for each pair,
    //we do them now and save them for later use
    vdwcalc.kab = KCAL_TO_KJ * sqrt(vdwcalc.ka * vdwcalc.kb);

    // 1-4 scaling
    // This isn't mentioned in the UFF paper, but is common for other methods
    //       if (a->IsOneFour(b))
    //         vdwcalc.kab *= 0.5;

    // ka now represents the xij in equation 20 -- the expected vdw distance
    vdwcalc.kaSquared = (vdwcalc.Ra * vdwcalc.Rb);
    vdwcalc.ka = sqrt(vdwcalc.kaSquared);

    vdwcalc.SetupPointers();
    return true;
  }

  int GetCoordination(OBAtom *b, int ipar)
  {
    int coordination;

    // Work out coordination
    // including possible hypervalent compounds
    int valenceElectrons = 0;
    switch(b->GetAtomicNum())
      {
      case 15:
      case 33:
      case 51:
      case 83:
        // old "group 5": P, As, Sb, Bi
        valenceElectrons = 5;
        break;
      case 16:
      case 34:
      case 52:
      case 84:
        // old "group 6": S, Se, Te, Po
        valenceElectrons = 6;
        break;
      case 35:
      case 53:
      case 85:
        // old "group 7": Br, I, At
        valenceElectrons = 7;
        break;
      case 36:
      case 54:
      case 86:
        // hypervalent noble gases (Kr, Xe, Rn)
        valenceElectrons = 8;
        break;
      }
    if (valenceElectrons) {
      // calculate the number of lone pairs
      // e.g. for IF3 => "T-shaped"
      valenceElectrons -= b->GetFormalCharge(); // make sure to look for I+F4 -> see-saw
      double lonePairs = (valenceElectrons - b->GetExplicitValence()) / 2.0;
      // we actually need to round up here -- single e- take room too.
      int sites = (int)ceil(lonePairs);
      coordination = b->GetExplicitDegree() + sites;
      if (coordination <= 4) { // normal valency
        coordination = ipar;
      } else if (b->GetAtomicNum() == OBElements::Sulfur && b->CountFreeOxygens() == 3) {
        // SO3, should be planar
        // PR#2971473, thanks to Philipp Rumpf
        coordination = 2; // i.e., sp2
      }
      /* planar coordination of hexavalent molecules.*/
      if (lonePairs == 0 && b->GetExplicitDegree() == 3 && b->GetExplicitValence() == 6) {
        coordination = 2;
      }
      if (lonePairs == 0 && b->GetExplicitDegree() == 7) {
        coordination = 7;
      }
      // Check to see if coordination is really correct
      // if not (e.g., 5- or 7- or 8-coord...)
      // then create approximate angle bending terms
    } else {
      coordination = ipar; // coordination of central atom
    }
    if (b->GetExplicitDegree() > 4) {
      coordination = b->GetExplicitDegree();
    } else {
      int coordDifference = ipar - b->GetExplicitDegree();
      if (abs(coordDifference) > 2)
        // low valent, but very different than expected by ipar
        coordination = b->GetExplicitDegree() - 1; // 4 coordinate == sp3
    }
    return coordination;
  }

  bool MolgrForceFieldUFF::SetupCalculations()
  {
    OBFFParameter *parameterA, *parameterB, *parameterC;
    OBAtom *a, *b, *c, *d;
    double bondorder;
    MolgrFFBondCalculationUFF bondcalc;
    MolgrFFAngleCalculationUFF anglecalc;
    MolgrFFTorsionCalculationUFF torsioncalc;
    MolgrFFOOPCalculationUFF oopcalc;
    MolgrFFVDWCalculationUFF vdwcalc;

    IF_OBFF_LOGLVL_LOW
      OBFFLog("\nS E T T I N G   U P   C A L C U L A T I O N S\n\n");

    const unsigned int atom_count = _mol.NumAtoms();
    const unsigned int bond_count = _mol.NumBonds();

    std::vector<OBFFParameter *> atom_parameters(atom_count + 1, NULL);
    std::vector<int> atom_coordination(atom_count + 1, 0);
    std::vector<double> atom_vdw_radii(atom_count + 1, 0.0);
    std::vector<double> atom_vdw_well_depths(atom_count + 1, 0.0);
    std::vector<unsigned char> vdw_excluded((atom_count + 1) * (atom_count + 1), 0);

    const auto pair_offset = [atom_count](unsigned int lhs, unsigned int rhs)
    {
      return static_cast<std::size_t>(lhs) * (atom_count + 1) + rhs;
    };

    const auto mark_vdw_excluded = [&](unsigned int lhs, unsigned int rhs)
    {
      vdw_excluded[pair_offset(lhs, rhs)] = 1;
      vdw_excluded[pair_offset(rhs, lhs)] = 1;
    };

    const auto normalized_bond_order = [](OBBond *bond_ptr)
    {
      double normalized = bond_ptr->GetBondOrder();
      if (molgr::vendor::openbabel_threading::BondIsAromatic(*bond_ptr))
        normalized = 1.5;
      if (bond_ptr->IsAmide())
        normalized = 1.41;
      return normalized;
    };

    const auto setup_cached_vdw_calculation =
      [&](OBAtom *atom_a, OBAtom *atom_b, MolgrFFVDWCalculationUFF &cached_vdwcalc)
      {
        const unsigned int atom_a_idx = atom_a->GetIdx();
        const unsigned int atom_b_idx = atom_b->GetIdx();
        if (atom_parameters[atom_a_idx] == NULL || atom_parameters[atom_b_idx] == NULL)
          return false;

        cached_vdwcalc.Ra = atom_vdw_radii[atom_a_idx];
        cached_vdwcalc.ka = atom_vdw_well_depths[atom_a_idx];
        cached_vdwcalc.Rb = atom_vdw_radii[atom_b_idx];
        cached_vdwcalc.kb = atom_vdw_well_depths[atom_b_idx];
        cached_vdwcalc.a = atom_a;
        cached_vdwcalc.b = atom_b;
        cached_vdwcalc.kab = KCAL_TO_KJ * sqrt(cached_vdwcalc.ka * cached_vdwcalc.kb);
        cached_vdwcalc.kaSquared = cached_vdwcalc.Ra * cached_vdwcalc.Rb;
        cached_vdwcalc.ka = sqrt(cached_vdwcalc.kaSquared);
        cached_vdwcalc.SetupPointers();
        return true;
      };

    // Clear previous calculations
    _bondcalculations.clear();
    _anglecalculations.clear();
    _torsioncalculations.clear();
    _oopcalculations.clear();
    _vdwcalculations.clear();

    _bondcalculations.reserve(bond_count);
    _anglecalculations.reserve(bond_count * 4);
    _torsioncalculations.reserve(bond_count * 4);
    _oopcalculations.reserve(atom_count * 3);
    _vdwcalculations.reserve(
      atom_count > 1 ? (static_cast<std::size_t>(atom_count) * (atom_count - 1)) / 2 : 0);

    // Clear and reset any 5-coordinate axial/equatorial marks (i.e., strange coordination)
    // Now should fit standard VSEPR rules, although we can't easily handle lone pairs
    int coordination;
    FOR_ATOMS_OF_MOL(atom, _mol) {
      // remove any previous designation
      atom->DeleteData("UFF_AXIAL_ATOM");
      atom->DeleteData("UFF_CENTRAL_ATOM");
    }

    FOR_ATOMS_OF_MOL(atom, _mol) {
      parameterB = GetParameterUFF(atom->GetType(), _ffparams);

      // GitHub issue #1794
      if (parameterB == NULL) {
        snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR ATOM %d (IDX)...\n",
                 atom->GetIdx());
        IF_OBFF_LOGLVL_LOW
          OBFFLog(_logbuf);
        return false;
      }

      const unsigned int atom_idx = atom->GetIdx();
      const int cached_coordination = GetCoordination(&*atom, parameterB->_ipar[0]);
      atom_parameters[atom_idx] = parameterB;
      atom_coordination[atom_idx] = cached_coordination;
      atom_vdw_radii[atom_idx] = parameterB->_dpar[2];
      atom_vdw_well_depths[atom_idx] = parameterB->_dpar[3];

      if (cached_coordination == 5) { // we need to do work for trigonal-bipy!
        // First, find the two largest neighbors
        OBAtom *largestNbr, *current, *secondLargestNbr = 0;
        double largestRadius;
        OBBondIterator i;
        largestNbr = atom->BeginNbrAtom(i);
        // work out the radius
        parameterA = GetParameterUFF(largestNbr->GetType(), _ffparams);

        if (parameterA == NULL) {
          IF_OBFF_LOGLVL_LOW {
            snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR ATOM %d (IDX)...\n",
                largestNbr->GetIdx());
            OBFFLog(_logbuf);
          }
          return false;
        }

        largestRadius = parameterA->_dpar[0];

        for (current = atom->NextNbrAtom(i); current; current = atom->NextNbrAtom(i)) {
          parameterA = GetParameterUFF(current->GetType(), _ffparams);

          if (parameterA == NULL) {
            IF_OBFF_LOGLVL_LOW {
              snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR ATOM %d (IDX)...\n",
                  current->GetIdx());
              OBFFLog(_logbuf);
            }
            return false;
          }

          if (parameterA->_dpar[0] > largestRadius) {
            // New largest neighbor
            secondLargestNbr = largestNbr;
            largestRadius = parameterA->_dpar[0];
            largestNbr = current;
          }
          if (secondLargestNbr == NULL) {
            // save this atom
            secondLargestNbr = current;
          }
        }

        // OK, now we tag the central atom
        OBPairData *label = new OBPairData;
        label->SetAttribute("UFF_CENTRAL_ATOM");
        label->SetValue("True"); // doesn't really matter
        atom->SetData(label);

        label = new OBPairData;
        label->SetAttribute("UFF_AXIAL_ATOM");
        label->SetValue("True");
        largestNbr->SetData(label);

        if (secondLargestNbr != NULL) { // check for NULL, no guarantee
          label = new OBPairData;
          label->SetAttribute("UFF_AXIAL_ATOM");
          label->SetValue("True");
          secondLargestNbr->SetData(label);
        }

      } // end work for 5-coordinate angles
      if (cached_coordination == 7) { // pentagonal bipyramidal
        // First, find the two largest neighbors
        OBAtom *largestNbr, *current, *secondLargestNbr = 0;
        double largestRadius;
        OBBondIterator i;
        largestNbr = atom->BeginNbrAtom(i);
        // work out the radius
        parameterA = GetParameterUFF(largestNbr->GetType(), _ffparams);

        if (parameterA == NULL) {
          IF_OBFF_LOGLVL_LOW {
            snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR ATOM %d (IDX)...\n",
                largestNbr->GetIdx());
            OBFFLog(_logbuf);
          }
          return false;
        }

        largestRadius = parameterA->_dpar[0];

        for (current = atom->NextNbrAtom(i); current; current = atom->NextNbrAtom(i)) {
          parameterA = GetParameterUFF(current->GetType(), _ffparams);

          if (parameterA == NULL) {
            IF_OBFF_LOGLVL_LOW {
              snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR ATOM %d (IDX)...\n",
                  current->GetIdx());
              OBFFLog(_logbuf);
            }
            return false;
          }

          if (parameterA->_dpar[0] > largestRadius) {
            // New largest neighbor
            secondLargestNbr = largestNbr;
            largestRadius = parameterA->_dpar[0];
            largestNbr = current;
          }
          if (secondLargestNbr == NULL) {
            // save this atom
            secondLargestNbr = current;
          }
        }

        // OK, now we tag the central atom
        OBPairData *label = new OBPairData;
        label->SetAttribute("UFF_CENTRAL_ATOM");
        label->SetValue("True"); // doesn't really matter
        atom->SetData(label);
        // And tag the axial substituents
        label = new OBPairData;
        label->SetAttribute("UFF_AXIAL_ATOM");
        label->SetValue("True");
        largestNbr->SetData(label);
        if (secondLargestNbr != NULL) { // check for NULL, no guarantee
          label = new OBPairData;
          label->SetAttribute("UFF_AXIAL_ATOM");
          label->SetValue("True");
          secondLargestNbr->SetData(label);
        }
      }
    } // end loop through atoms

    FOR_BONDS_OF_MOL(bond_iter, _mol) {
      mark_vdw_excluded(
        bond_iter->GetBeginAtomIdx(),
        bond_iter->GetEndAtomIdx());
    }
    FOR_ATOMS_OF_MOL(atom, _mol) {
      std::vector<unsigned int> neighbor_indices;
      neighbor_indices.reserve(atom->GetExplicitDegree());
      OBBondIterator neighbor_bond_iter;
      for (OBAtom *neighbor = atom->BeginNbrAtom(neighbor_bond_iter);
           neighbor != NULL;
           neighbor = atom->NextNbrAtom(neighbor_bond_iter)) {
        neighbor_indices.push_back(neighbor->GetIdx());
      }
      for (std::size_t first = 0; first < neighbor_indices.size(); ++first) {
        for (std::size_t second = first + 1; second < neighbor_indices.size(); ++second) {
          mark_vdw_excluded(neighbor_indices[first], neighbor_indices[second]);
        }
      }
    }

    //
    // Bond Calculations
    IF_OBFF_LOGLVL_LOW
      OBFFLog("SETTING UP BOND CALCULATIONS...\n");

    FOR_BONDS_OF_MOL(bond, _mol) {
      a = bond->GetBeginAtom();
      b = bond->GetEndAtom();

      // skip this bond if the atoms are ignored
      if ( constraints_.IsIgnored(a->GetIdx()) || constraints_.IsIgnored(b->GetIdx()) )
        continue;

      // if there are any groups specified, check if the two bond atoms are in a single intraGroup
      if (HasGroups()) {
        bool validBond = false;
        for (unsigned int i=0; i < _intraGroup.size(); ++i) {
          if (_intraGroup[i].BitIsSet(a->GetIdx()) && _intraGroup[i].BitIsSet(b->GetIdx()))
            validBond = true;
        }
        if (!validBond)
          continue;
      }

      bondorder = normalized_bond_order(&*bond);
      // e.g., in Cp rings, may not be "aromatic" by OB
      // but check for explicit hydrogen counts (e.g., biphenyl inter-ring is not aromatic)
      if ((a->GetType()[2] == 'R' && b->GetType()[2] == 'R')
          && (a->ExplicitHydrogenCount() == 1 && b->ExplicitHydrogenCount() == 1))
        bondorder = 1.5;

      bondcalc.a = a;
      bondcalc.b = b;
      bondcalc.bt = bondorder;

      parameterA = atom_parameters[a->GetIdx()];
      parameterB = atom_parameters[b->GetIdx()];

      if (parameterA == NULL || parameterB == NULL) {
        IF_OBFF_LOGLVL_LOW {
          snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR BOND %d-%d (IDX)...\n",
                   a->GetIdx(), b->GetIdx());
          OBFFLog(_logbuf);
        }
        continue;
      }

      bondcalc.r0 = CalculateBondDistance(parameterA, parameterB, bondorder);

      // here we fold the 1/2 into the kij from equation 1a
      // Otherwise, this is equation 6 from the UFF paper.
      bondcalc.kb = (0.5 * KCAL_TO_KJ * 664.12
                     * parameterA->_dpar[5] * parameterB->_dpar[5])
        / (bondcalc.r0 * bondcalc.r0 * bondcalc.r0);

      bondcalc.SetupPointers();
      _bondcalculations.push_back(bondcalc);
    }

    //
    // Angle Calculations
    //
    IF_OBFF_LOGLVL_LOW
      OBFFLog("SETTING UP ANGLE CALCULATIONS...\n");

    double sinT0;
		double rab, rbc, rac;
		OBBond *bondPtr;
    FOR_ANGLES_OF_MOL(angle, _mol) {
      b = _mol.GetAtom((*angle)[0] + 1);
      a = _mol.GetAtom((*angle)[1] + 1);
      c = _mol.GetAtom((*angle)[2] + 1);

      // skip this angle if the atoms are ignored
      if ( constraints_.IsIgnored(a->GetIdx())
           || constraints_.IsIgnored(b->GetIdx())
           || constraints_.IsIgnored(c->GetIdx()) )
        continue;

      // if there are any groups specified,
      // check if the three angle atoms are in a single intraGroup
      if (HasGroups()) {
        bool validAngle = false;
        for (unsigned int i=0; i < _intraGroup.size(); ++i) {
          if (_intraGroup[i].BitIsSet(a->GetIdx()) && _intraGroup[i].BitIsSet(b->GetIdx()) &&
              _intraGroup[i].BitIsSet(c->GetIdx()))
            validAngle = true;
        }
        if (!validAngle)
          continue;
      }

      anglecalc.a = a;
      anglecalc.b = b;
      anglecalc.c = c;

      parameterA = atom_parameters[a->GetIdx()];
      parameterB = atom_parameters[b->GetIdx()];
      parameterC = atom_parameters[c->GetIdx()];

      if (parameterA == NULL || parameterB == NULL || parameterC == NULL) {
        IF_OBFF_LOGLVL_LOW {
          snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR ANGLE %d-%d-%d (IDX)...\n",
                   a->GetIdx(), b->GetIdx(), c->GetIdx());
          OBFFLog(_logbuf);
        }
        return false;
      }

      coordination = atom_coordination[b->GetIdx()];

      if (coordination != parameterB->_ipar[0]) {
        IF_OBFF_LOGLVL_LOW {
          snprintf(_logbuf, BUFF_SIZE, "    CORRECTED COORDINATION FOR ANGLE %d-%d-%d (IDX)... WAS %d NOW %d\n",
                   a->GetIdx(), b->GetIdx(), c->GetIdx(), parameterB->_ipar[0], coordination);
          OBFFLog(_logbuf);
        }
      }

      //double currentTheta;
      if (coordination > 7) {
        // large coordination sphere (e.g., [ReH9]-2 or [Ce(NO3)6]-2)
        // just resort to using VDW 1-3 interactions to push atoms into place
        // there's not much else we can do without real parameters
        if (setup_cached_vdw_calculation(a, c, vdwcalc)) {
          _vdwcalculations.push_back(vdwcalc);
        }
        // We're not installing an angle term for this set
        // We can't even approximate one.
        // The downside is that we can't easily handle lone pairs.
        continue;

      } else if (coordination == 7) { // pentagonal bipyramidal
        // This doesn't work so well because it's hard to classify between
        // axial-equatorial (90 degrees) and proximal equatorial (~72 degrees).
        double currentTheta;
        currentTheta =  a->GetAngle(&*b, &*c);

        anglecalc.c0 = 1.0;
        if (b->HasData("UFF_CENTRAL_ATOM")
              && a->HasData("UFF_AXIAL_ATOM")
              && c->HasData("UFF_AXIAL_ATOM")) { // axial ligands = linear
          anglecalc.coord = 1; // like sp
          anglecalc.theta0 = 180.0 * DEG_TO_RAD;
          anglecalc.c1 = 1.0;
        } else if ( (a->HasData("UFF_AXIAL_ATOM") && !c->HasData("UFF_AXIAL_ATOM"))
                    || (c->HasData("UFF_AXIAL_ATOM") && !a->HasData("UFF_AXIAL_ATOM")) ) { // axial-equatorial ligands
          anglecalc.coord = 4; // like sq. planar or octahedral
          anglecalc.theta0 = 90.0 * DEG_TO_RAD;
          anglecalc.c1 = 1.0;
        } else { // equatorial - equatorial
          anglecalc.coord = 7; // unlike anything else, as theta0 is ignored.
          anglecalc.theta0 = (currentTheta > 108.0 ? 144.0 : 72.0) * DEG_TO_RAD;
          anglecalc.c1 = 1.0;
        }
        anglecalc.c2 = 0.0;

        /*
        if (0) {
          if (currentTheta >= 155.0) { // axial ligands = linear
            anglecalc.coord = 1; // like sp
            anglecalc.theta0 = 180.0 * DEG_TO_RAD;
            anglecalc.c1 = 1.0;
          } else if (currentTheta < 155.0 && currentTheta >= 110.0) { // distal equatorial
            anglecalc.coord = 7; // like sp3
            anglecalc.theta0 = 144.0 * DEG_TO_RAD;
            anglecalc.c1 = 1.0;
          } else if (currentTheta < 110.0 && currentTheta >= 85.0) { // axial-equatorial
            anglecalc.coord = 4; // like sq. planar or octahedral
            anglecalc.theta0 = 90.0 * DEG_TO_RAD;
            anglecalc.c1 = 1.0;
          } else if (currentTheta < 85.0) { // proximal equatorial
            anglecalc.coord = 7; // general case (i.e., like sp3)
            anglecalc.theta0 = 72.0 * DEG_TO_RAD;
            anglecalc.c1 = 1.0;
          }
          anglecalc.c2 = 0.0;
        } else {
        */

      } else if (coordination == 5) { // trigonal bipyramidal
        anglecalc.c0 = 1.0;
        // We've already done some of our work above -- look for axial markings
        if (b->HasData("UFF_CENTRAL_ATOM")
            && a->HasData("UFF_AXIAL_ATOM")
            && c->HasData("UFF_AXIAL_ATOM")) { // axial ligands = linear
          anglecalc.coord = 1; // like sp
          anglecalc.theta0 = 180.0 * DEG_TO_RAD;
          anglecalc.c1 = 1.0;
        } else if ( (a->HasData("UFF_AXIAL_ATOM") && !c->HasData("UFF_AXIAL_ATOM"))
                    || (c->HasData("UFF_AXIAL_ATOM") && !a->HasData("UFF_AXIAL_ATOM")) ) { // axial-equatorial ligands
          anglecalc.coord = 4; // like sq. planar or octahedral
          anglecalc.theta0 = 90.0 * DEG_TO_RAD;
          anglecalc.c1 = 1.0;
        } else { // equatorial - equatorial
          anglecalc.coord = 2; // like sp2
          anglecalc.theta0 = 120.0 * DEG_TO_RAD;
          anglecalc.c1 = -1.0;
        }
        anglecalc.c2 = 0.0;
      }
      else { // normal coordination: sp, sp2, sp3, square planar, octahedral
        anglecalc.coord = coordination;
        anglecalc.theta0 = parameterB->_dpar[1] * DEG_TO_RAD;
        if (coordination != parameterB->_ipar[0]) {
          switch (coordination)
            {
            case 1:
              anglecalc.theta0 = 180.0 * DEG_TO_RAD;
              break;
            case 2:
              anglecalc.theta0 = 120.0 * DEG_TO_RAD;
              break;
            case 4: // sq. planar
            case 5: // axial / equatorial
            case 6: // octahedral
            case 7: // axial equatorial
              anglecalc.theta0 = 90.0 * DEG_TO_RAD;
              break;
            case 3: // tetrahedral
            default:
              anglecalc.theta0 = 109.5 * DEG_TO_RAD;
              break;
            }
        }
        anglecalc.cosT0 = cos(anglecalc.theta0);
        sinT0 = sin(anglecalc.theta0);
        anglecalc.c2 = 1.0 / (4.0 * sinT0 * sinT0);
        anglecalc.c1 = -4.0 * anglecalc.c2 * anglecalc.cosT0;
        anglecalc.c0 = anglecalc.c2*(2.0*anglecalc.cosT0*anglecalc.cosT0 + 1.0);
      }

      anglecalc.cosT0 = cos(anglecalc.theta0);
      anglecalc.zi = parameterA->_dpar[5];
      anglecalc.zk = parameterC->_dpar[5];
			// Precompute the force constant
			bondPtr = _mol.GetBond(a,b);
			bondorder = normalized_bond_order(bondPtr);
			rab = CalculateBondDistance(parameterA, parameterB, bondorder);

			bondPtr = _mol.GetBond(b,c);
			bondorder = normalized_bond_order(bondPtr);
			rbc = CalculateBondDistance(parameterB, parameterC, bondorder);
			rac = sqrt(rab*rab + rbc*rbc - 2.0 * rab*rbc*anglecalc.cosT0);

			// Equation 13 from paper -- corrected by Towhee
			// Note that 1/(rij * rjk) cancels with rij*rjk in eqn. 13
			anglecalc.ka = (664.12 * KCAL_TO_KJ) * (anglecalc.zi * anglecalc.zk / (pow(rac, 5.0)));
			anglecalc.ka *= (3.0*rab*rbc*(1.0 - anglecalc.cosT0*anglecalc.cosT0) - rac*rac*anglecalc.cosT0);
      // Make sure to divide by n^2 to save CPU cycles
      switch (anglecalc.coord) {
      case 2: // sp2, so divide by 3^2
        anglecalc.n = 3;
        anglecalc.ka = anglecalc.ka / 9.0;
        break;
      case 4: // divide by 4^2
      case 6:
        anglecalc.n = 4;
        anglecalc.ka = anglecalc.ka / 16.0;
        break;
      default:
        break;
      }

      anglecalc.SetupPointers();
      _anglecalculations.push_back(anglecalc);
    }

    //
    // Torsion Calculations
    //
    IF_OBFF_LOGLVL_LOW
      OBFFLog("SETTING UP TORSION CALCULATIONS...\n");

    double torsiontype;
    double phi0 = 0.0;

    double vi, vj;
    FOR_TORSIONS_OF_MOL(t, _mol) {
      a = _mol.GetAtom((*t)[0] + 1);
      b = _mol.GetAtom((*t)[1] + 1);
      c = _mol.GetAtom((*t)[2] + 1);
      d = _mol.GetAtom((*t)[3] + 1);

      // skip this torsion if the atoms are ignored
      if ( constraints_.IsIgnored(a->GetIdx()) || constraints_.IsIgnored(b->GetIdx()) ||
           constraints_.IsIgnored(c->GetIdx()) || constraints_.IsIgnored(d->GetIdx()) )
        continue;

      // if there are any groups specified, check if the four torsion atoms are in a single intraGroup
      if (HasGroups()) {
        bool validTorsion = false;
        for (unsigned int i=0; i < _intraGroup.size(); ++i) {
          if (_intraGroup[i].BitIsSet(a->GetIdx()) && _intraGroup[i].BitIsSet(b->GetIdx()) &&
              _intraGroup[i].BitIsSet(c->GetIdx()) && _intraGroup[i].BitIsSet(d->GetIdx()))
            validTorsion = true;
        }
        if (!validTorsion)
          continue;
      }

      OBBond *bc = _mol.GetBond(b, c);
      torsiontype = bc->GetBondOrder();
      if (molgr::vendor::openbabel_threading::BondIsAromatic(*bc))
        torsiontype = 1.5;
      if (bc->IsAmide())
        torsiontype = 1.41;

      torsioncalc.a = a;
      torsioncalc.b = b;
      torsioncalc.c = c;
      torsioncalc.d = d;
      torsioncalc.tt = torsiontype;

      parameterB = atom_parameters[b->GetIdx()];
      parameterC = atom_parameters[c->GetIdx()];

      if (parameterB == NULL || parameterC == NULL) {
        IF_OBFF_LOGLVL_LOW {
          snprintf(_logbuf, BUFF_SIZE, "    COULD NOT FIND PARAMETERS FOR TORSION X-%d-%d-X (IDX)...\n",
                   b->GetIdx(), c->GetIdx());
          OBFFLog(_logbuf);
        }
        return false;
      }

      if (parameterB->_ipar[0] == 3 && parameterC->_ipar[0] == 3) {
        // two sp3 centers
        phi0 = 60.0;
        torsioncalc.n = 3;
        vi = parameterB->_dpar[6];
        vj = parameterC->_dpar[6];

        // exception for a pair of group 6 sp3 atoms
        switch (b->GetAtomicNum()) {
        case 8:
          vi = 2.0;
          torsioncalc.n = 2;
          phi0 = 90.0;
          break;
        case 16:
        case 34:
        case 52:
        case 84:
          vi = 6.8;
          torsioncalc.n = 2;
          phi0 = 90.0;
        }
        switch (c->GetAtomicNum()) {
        case 8:
          vj = 2.0;
          torsioncalc.n = 2;
          phi0 = 90.0;
          break;
        case 16:
        case 34:
        case 52:
        case 84:
          vj = 6.8;
          torsioncalc.n = 2;
          phi0 = 90.0;
        }

        torsioncalc.V = 0.5 * KCAL_TO_KJ * sqrt(vi * vj);

      } else if (parameterB->_ipar[0] == 2 && parameterC->_ipar[0] == 2) {
        // two sp2 centers
        phi0 = 180.0;
        torsioncalc.n = 2;
        torsioncalc.V = 0.5 * KCAL_TO_KJ * 5.0 *
          sqrt(parameterB->_dpar[7]*parameterC->_dpar[7]) *
          (1.0 + 4.18 * log(torsiontype));
      } else if ((parameterB->_ipar[0] == 2 && parameterC->_ipar[0] == 3)
                 || (parameterB->_ipar[0] == 3 && parameterC->_ipar[0] == 2)) {
        // one sp3, one sp2
        phi0 = 0.0;
        torsioncalc.n = 6;
        torsioncalc.V = 0.5 * KCAL_TO_KJ * 1.0;

        // exception for group 6 sp3
        if (parameterC->_ipar[0] == 3) {
          switch (c->GetAtomicNum()) {
          case 8:
          case 16:
          case 34:
          case 52:
          case 84:
            torsioncalc.n = 2;
            phi0 = 90.0;
          }
        }
        if (parameterB->_ipar[0] == 3) {
          switch (b->GetAtomicNum()) {
          case 8:
          case 16:
          case 34:
          case 52:
          case 84:
            torsioncalc.n = 2;
            phi0 = 90.0;
          }
        }
      }

      if (IsNearZero(torsioncalc.V)) // don't bother calcuating this torsion
        continue;

      // still need to implement special case of sp2-sp3 with sp2-sp2

      torsioncalc.cosNPhi0 = cos(torsioncalc.n * DEG_TO_RAD * phi0);
      torsioncalc.SetupPointers();
      _torsioncalculations.push_back(torsioncalc);
    }

    //
    // OOP/Inversion Calculations
    //
    IF_OBFF_LOGLVL_LOW
      OBFFLog("SETTING UP OOP CALCULATIONS...\n");

    double phi;
    // The original Rappe paper in JACS isn't very clear about the parameters
    // The following was adapted from Towhee
    FOR_ATOMS_OF_MOL(atom, _mol) {
      b = (OBAtom*) &*atom;

      switch (b->GetAtomicNum()) {
      case 6: // carbon
      case 7: // nitrogen
      case 8: // oxygen
      case 15: // phos.
      case 33: // as
      case 51: // sb
      case 83: // bi
        break;
      default: // no inversion term for this element
        continue;
      }

      if (b->GetExplicitDegree() > 3) // no OOP for hypervalent atoms
        continue;

      a = NULL;
      c = NULL;
      d = NULL;

      if (EQn(b->GetType(), "N_3", 3) ||
          EQn(b->GetType(), "N_2", 3) ||
          EQn(b->GetType(), "N_R", 3) ||
          EQn(b->GetType(), "O_2", 3) ||
          EQn(b->GetType(), "O_R", 3)) {
        oopcalc.c0 = 1.0;
        oopcalc.c1 = -1.0;
        oopcalc.c2 = 0.0;
        oopcalc.koop = 6.0 * KCAL_TO_KJ;
      }
      else if (EQn(b->GetType(), "P_3+3", 5) ||
               EQn(b->GetType(), "As3+3", 5) ||
               EQn(b->GetType(), "Sb3+3", 5) ||
               EQn(b->GetType(), "Bi3+3", 5)) {

        if (EQn(b->GetType(), "P_3+3", 5))
          phi = 84.4339 * DEG_TO_RAD;
        else if (EQn(b->GetType(), "As3+3", 5))
          phi = 86.9735 * DEG_TO_RAD;
        else if (EQn(b->GetType(), "Sb3+3", 5))
          phi = 87.7047 * DEG_TO_RAD;
        else
          phi = 90.0 * DEG_TO_RAD;

        oopcalc.c1 = -4.0 * cos(phi);
        oopcalc.c2 = 1.0;
        oopcalc.c0 = -1.0*oopcalc.c1 * cos(phi) + oopcalc.c2*cos(2.0*phi);
        oopcalc.koop = 22.0 * KCAL_TO_KJ;
      }
      else if (!(EQn(b->GetType(), "C_2", 3) || EQn(b->GetType(), "C_R", 3)))
        continue; // inversion not defined for this atom type

      FOR_NBORS_OF_ATOM(nbr, b) {
        if (a == NULL)
          a = (OBAtom*) &*nbr;
        else if (c == NULL)
          c = (OBAtom*) &*nbr;
        else
          d = (OBAtom*) &*nbr;
      }

      if ((a == NULL) || (c == NULL) || (d == NULL))
        continue;

      // skip this oop if the atoms are ignored
      if ( constraints_.IsIgnored(a->GetIdx()) ||
           constraints_.IsIgnored(b->GetIdx()) ||
           constraints_.IsIgnored(c->GetIdx()) ||
           constraints_.IsIgnored(d->GetIdx()) )
        continue;

      // if there are any groups specified,
      // check if the four oop atoms are in a single intraGroup
      if (HasGroups()) {
        bool validOOP = false;
        for (unsigned int i=0; i < _intraGroup.size(); ++i) {
          if (_intraGroup[i].BitIsSet(a->GetIdx()) &&
              _intraGroup[i].BitIsSet(b->GetIdx()) &&
              _intraGroup[i].BitIsSet(c->GetIdx()) &&
              _intraGroup[i].BitIsSet(d->GetIdx()))
            validOOP = true;
        }
        if (!validOOP)
          continue;
      }

      // C atoms, we should check if we're bonded to O
      if (EQn(b->GetType(), "C_2", 3) || EQn(b->GetType(), "C_R", 3)) {
        oopcalc.c0 = 1.0;
        oopcalc.c1 = -1.0;
        oopcalc.c2 = 0.0;
        oopcalc.koop = 6.0 * KCAL_TO_KJ;
        if (EQn(a->GetType(), "O_2", 3) ||
            EQn(c->GetType(), "O_2", 3) ||
            EQn(d->GetType(), "O_2", 3)) {
          oopcalc.koop = 50.0 * KCAL_TO_KJ;
        }
      }

      // A-B-CD || C-B-AD  PLANE = ABC
      oopcalc.a = a;
      oopcalc.b = b;
      oopcalc.c = c;
      oopcalc.d = d;
      oopcalc.koop /= 3.0; // three OOPs to consider

      oopcalc.SetupPointers();
      _oopcalculations.push_back(oopcalc);

      // C-B-DA || D-B-CA  PLANE BCD
      oopcalc.a = d;
      oopcalc.d = a;

      oopcalc.SetupPointers();
      _oopcalculations.push_back(oopcalc);

      // A-B-DC || D-B-AC  PLANE ABD
      oopcalc.a = a;
      oopcalc.c = d;
      oopcalc.d = c;

      oopcalc.SetupPointers();
      _oopcalculations.push_back(oopcalc);
    } // for all atoms

    //
    // VDW Calculations
    //
    IF_OBFF_LOGLVL_LOW
      OBFFLog("SETTING UP VAN DER WAALS CALCULATIONS...\n");

    for (unsigned int atom_a_idx = 1; atom_a_idx <= atom_count; ++atom_a_idx) {
      a = _mol.GetAtom(atom_a_idx);
      if (a == NULL)
        continue;
      for (unsigned int atom_b_idx = atom_a_idx + 1; atom_b_idx <= atom_count; ++atom_b_idx) {
        b = _mol.GetAtom(atom_b_idx);
        if (b == NULL)
          continue;

        // skip this vdw if the atoms are ignored
        if ( constraints_.IsIgnored(a->GetIdx()) || constraints_.IsIgnored(b->GetIdx()) )
          continue;

        // if there are any groups specified, check if the two atoms are in a single _interGroup or if
        // two two atoms are in one of the _interGroups pairs.
        if (HasGroups()) {
          bool validVDW = false;
          for (unsigned int i=0; i < _interGroup.size(); ++i) {
            if (_interGroup[i].BitIsSet(a->GetIdx()) && _interGroup[i].BitIsSet(b->GetIdx()))
              validVDW = true;
          }
          for (unsigned int i=0; i < _interGroups.size(); ++i) {
            if (_interGroups[i].first.BitIsSet(a->GetIdx()) && _interGroups[i].second.BitIsSet(b->GetIdx()))
              validVDW = true;
            if (_interGroups[i].first.BitIsSet(b->GetIdx()) && _interGroups[i].second.BitIsSet(a->GetIdx()))
              validVDW = true;
          }

          if (!validVDW)
            continue;
        }

        if (vdw_excluded[pair_offset(atom_a_idx, atom_b_idx)]) {
          continue;
        }

        if (setup_cached_vdw_calculation(a, b, vdwcalc)) {
          _vdwcalculations.push_back(vdwcalc);
        }
      }
    }

    // NOTE: No electrostatics are set up
    // If you want electrostatics with UFF, you will need to call
    // SetupElectrostatics() manually

    return true;
  }

  bool MolgrForceFieldUFF::SetupElectrostatics()
  {
    //
    // Electrostatic Calculations
    //
    OBAtom *a, *b;

    IF_OBFF_LOGLVL_LOW
      OBFFLog("SETTING UP ELECTROSTATIC CALCULATIONS...\n");

    MolgrFFElectrostaticCalculationUFF elecalc;

    _electrostaticcalculations.clear();

    // Note that while the UFF paper mentions an electrostatic term,
    // it does not actually use it. Both Towhee and the UFF FAQ
    // discourage the use of electrostatics with UFF.

    FOR_PAIRS_OF_MOL(p, _mol) {
      a = _mol.GetAtom((*p)[0]);
      b = _mol.GetAtom((*p)[1]);

      // skip this ele if the atoms are ignored
      if ( constraints_.IsIgnored(a->GetIdx()) || constraints_.IsIgnored(b->GetIdx()) )
        continue;

      // if there are any groups specified, check if the two atoms are in a single _interGroup or if
      // two two atoms are in one of the _interGroups pairs.
      if (HasGroups()) {
        bool validEle = false;
        for (unsigned int i=0; i < _interGroup.size(); ++i) {
          if (_interGroup[i].BitIsSet(a->GetIdx()) && _interGroup[i].BitIsSet(b->GetIdx()))
            validEle = true;
        }
        for (unsigned int i=0; i < _interGroups.size(); ++i) {
          if (_interGroups[i].first.BitIsSet(a->GetIdx()) && _interGroups[i].second.BitIsSet(b->GetIdx()))
            validEle = true;
          if (_interGroups[i].first.BitIsSet(b->GetIdx()) && _interGroups[i].second.BitIsSet(a->GetIdx()))
            validEle = true;
        }

        if (!validEle)
          continue;
      }

      if (a->IsConnected(b)) {
        continue;
      }
      if (a->IsOneThree(b)) {
        continue;
      }

      // Remember that at the moment, this term is not currently used
      // These are also the Gasteiger charges, not the Qeq mentioned in the UFF paper
      elecalc.qq = KCAL_TO_KJ * 332.0637 * a->GetPartialCharge() * b->GetPartialCharge();

      if (elecalc.qq) {
        elecalc.a = &*a;
        elecalc.b = &*b;

        elecalc.SetupPointers();
        _electrostaticcalculations.push_back(elecalc);
      }
    }
    return true;
  }

  bool MolgrForceFieldUFF::SetupPointers()
  {
    for (unsigned int i = 0; i < _bondcalculations.size(); ++i)
      _bondcalculations[i].SetupPointers();
    for (unsigned int i = 0; i < _anglecalculations.size(); ++i)
      _anglecalculations[i].SetupPointers();
    for (unsigned int i = 0; i < _torsioncalculations.size(); ++i)
      _torsioncalculations[i].SetupPointers();
     for (unsigned int i = 0; i < _oopcalculations.size(); ++i)
      _oopcalculations[i].SetupPointers();
    for (unsigned int i = 0; i < _vdwcalculations.size(); ++i)
      _vdwcalculations[i].SetupPointers();
    for (unsigned int i = 0; i < _electrostaticcalculations.size(); ++i)
      _electrostaticcalculations[i].SetupPointers();

    return true;
  }

  bool MolgrForceFieldUFF::ParseParamFile()
  {
    const auto &shared = GetMolgrUffSharedData();
    if (!shared.loaded) {
      return false;
    }
    _ffparams = shared.ffparams;
    _ffparam_index = shared.ffparam_index;
    return true;
  }

  bool MolgrForceFieldUFF::SetTypes()
  {
    _mol.SetAtomTypesPerceived();

    if (use_atom_typing_cache_ && !atom_typing_cache_key_.empty()) {
      std::vector<std::string> cached_atom_types;
      if (MolgrUffAtomTypeAssignmentCache().Get(atom_typing_cache_key_, cached_atom_types) &&
          ApplyMolgrUffAtomTypes(_mol, cached_atom_types)) {
        return true;
      }
    }

    const std::vector<MolgrCompiledUffAtomTypeRule> *compiled_rules = nullptr;
    if (!GetMolgrCompiledUffAtomTypeRules(compiled_rules) || compiled_rules == nullptr) {
      return false;
    }

    for (const auto &rule : *compiled_rules) {
      FOR_ATOMS_OF_MOL(atom_iter, _mol) {
        if (AtomMatchesUffRule(*atom_iter, rule)) {
          atom_iter->SetType(rule.atom_type.c_str());
        }
      }
    }

    // Special atom types (i.e., P_3+q)
    // (We can't easily do this with a SMARTS)
    FOR_ATOMS_OF_MOL(a, _mol) {
      if (a->GetAtomicNum() == 15) {
        // loop through all the neighbors and see if we have a metal coordination
        bool organomet = false;
        int nbrElement;
        FOR_NBORS_OF_ATOM (nbr, &*a) {
          nbrElement = nbr->GetAtomicNum();
          if ( (nbrElement >= 21 && nbrElement <= 31) // Sc to Ga
               || (nbrElement >= 39 && nbrElement <= 50) // Y to Sn
               || (nbrElement >= 57 && nbrElement <= 83) // La to Bi
               || (nbrElement >= 89) ) {
            organomet = true;
            break; // done!
          }
        }
        if (organomet)
          a->SetType("P_3+q");
      }
      else if (a->GetAtomicNum() > 102) { // superheavy
        a->SetType("Lw6+3"); // prevent a crash with atoms beyond the parameterization Avogadro PR#741
      }
    }

    if (use_atom_typing_cache_ && !atom_typing_cache_key_.empty()) {
      MolgrUffAtomTypeAssignmentCache().Put(
          atom_typing_cache_key_,
          CaptureMolgrUffAtomTypes(_mol));
    }

    IF_OBFF_LOGLVL_LOW {
      OBFFLog("\nA T O M   T Y P E S\n\n");
      OBFFLog("IDX\tTYPE\tRING\n");

      FOR_ATOMS_OF_MOL (a, _mol) {
        snprintf(_logbuf, BUFF_SIZE, "%d\t%s\t%s\n", a->GetIdx(), a->GetType(),
	  (a->IsInRing()
       ? (molgr::vendor::openbabel_threading::AtomIsAromatic(*a) ? "AR" : "AL")
       : "NO"));
        OBFFLog(_logbuf);
      }

    }

    return true;
  }

  double MolgrForceFieldUFF::Energy(bool gradients)
  {
    ActivateThreadLocalInstance();
    double energy;

    IF_OBFF_LOGLVL_MEDIUM
      OBFFLog("\nE N E R G Y\n\n");

    if (gradients) {
      ClearGradients();
      energy  = E_Bond<true>();
      energy += E_Angle<true>();
      energy += E_Torsion<true>();
      energy += E_OOP<true>();
      energy += E_VDW<true>();
    } else {
      energy  = E_Bond<false>();
      energy += E_Angle<false>();
      energy += E_Torsion<false>();
      energy += E_OOP<false>();
      energy += E_VDW<false>();
    }

    // The electrostatic term, by default is 0.0
    // You will need to call SetupEletrostatics if you want it
    // energy += E_Electrostatic(gradients);

    IF_OBFF_LOGLVL_MEDIUM {
      snprintf(_logbuf, BUFF_SIZE, "\nTOTAL ENERGY = %8.5f %s\n", energy, GetUnit().c_str());
      OBFFLog(_logbuf);
    }

    return energy;
  }

  OBFFParameter* MolgrForceFieldUFF::GetParameterUFF(std::string a, vector<OBFFParameter> &parameter)
  {
    (void)parameter;
    const auto iter = _ffparam_index.find(a);
    if (iter == _ffparam_index.end())
      return NULL;
    return &_ffparams[iter->second];
  }

  bool MolgrForceFieldUFF::ValidateGradients ()
  {
    vector3 numgrad, anagrad, err;
    bool passed = true; // set to false if any component fails
    int coordIdx;

    OBFFLog("\nV A L I D A T E   G R A D I E N T S\n\n");
    OBFFLog("ATOM IDX      NUMERICAL GRADIENT           ANALYTICAL GRADIENT        REL. ERROR (%)   \n");
    OBFFLog("----------------------------------------------------------------------------------------\n");
    //     "XX       (000.000, 000.000, 000.000)  (000.000, 000.000, 000.000)  (00.00, 00.00, 00.00)"

    FOR_ATOMS_OF_MOL (a, _mol) {
      coordIdx = (a->GetIdx() - 1) * 3;

      // OBFF_ENERGY (i.e., overall)
      numgrad = NumericalDerivative(&*a, OBFF_ENERGY);
      Energy(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "%2d       (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", a->GetIdx(), numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);

      // OBFF_EBOND
      numgrad = NumericalDerivative(&*a, OBFF_EBOND);
      ClearGradients();
      E_Bond(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "    bond    (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);
      if (err.x() > 5.0 || err.y() > 5.0 || err.z() > 5.0)
        passed = false;

      // OBFF_EANGLE
      numgrad = NumericalDerivative(&*a, OBFF_EANGLE);
      ClearGradients();
      E_Angle(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "    angle   (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);
      if (err.x() > 8.0 || err.y() > 8.0 || err.z() > 8.0)
        passed = false;

      // OBFF_ETORSION
      numgrad = NumericalDerivative(&*a, OBFF_ETORSION);
      ClearGradients();
      E_Torsion(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "    torsion (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);
      // 8% tolerance here because some 180 torsions cause numerical instability
      if (err.x() > 8.0 || err.y() > 8.0 || err.z() > 8.0)
        passed = false;

      // OBFF_EOOP
      numgrad = NumericalDerivative(&*a, OBFF_EOOP);
      ClearGradients();
      E_OOP(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "    oop     (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);
      // We don't care if the OOP error is relatively large
      //      if (err.x() > 5.0 || err.y() > 5.0 || err.z() > 5.0)
      //        passed = false;

      // OBFF_EVDW
      numgrad = NumericalDerivative(&*a, OBFF_EVDW);
      ClearGradients();
      E_VDW(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "    vdw     (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);
      if (err.x() > 5.0 || err.y() > 5.0 || err.z() > 5.0)
        passed = false;

      // OBFF_EELECTROSTATIC
      numgrad = NumericalDerivative(&*a, OBFF_EELECTROSTATIC);
      ClearGradients();
      E_Electrostatic(); // compute
      anagrad.Set(_gradientPtr[coordIdx], _gradientPtr[coordIdx+1], _gradientPtr[coordIdx+2]);
      err = ValidateGradientError(numgrad, anagrad);

      snprintf(_logbuf, BUFF_SIZE, "    electro (%7.3f, %7.3f, %7.3f)  (%7.3f, %7.3f, %7.3f)  (%5.2f, %5.2f, %5.2f)\n", numgrad.x(), numgrad.y(), numgrad.z(),
               anagrad.x(), anagrad.y(), anagrad.z(), err.x(), err.y(), err.z());
      OBFFLog(_logbuf);
      if (err.x() > 5.0 || err.y() > 5.0 || err.z() > 5.0)
        passed = false;
    }

    return passed; // did we pass every single component?
  }

  void ClearMolgrUffAtomTypeAssignmentCache()
  {
    MolgrUffAtomTypeAssignmentCache().Clear();
  }

  std::tuple<std::size_t, std::size_t, std::size_t> MolgrUffAtomTypeAssignmentCacheInfo()
  {
    return MolgrUffAtomTypeAssignmentCache().Info();
  }

} // end namespace OpenBabel

//! \file forcefieldUFF.cpp
//! \brief UFF force field
