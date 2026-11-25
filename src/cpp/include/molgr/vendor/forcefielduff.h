#pragma once

/**********************************************************************
forcefielduff.h - UFF force field.

Copyright (C) 2007 by Geoffrey Hutchison
Some portions Copyright (C) 2006-2007 by Tim Vandermeersch

This file is part of the Open Babel project.
For more information, see <http://openbabel.org/>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 2 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
***********************************************************************/

#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <tuple>

#include <openbabel/forcefield.h>
#include <openbabel/base.h>
#include <openbabel/mol.h>

namespace OpenBabel
{
  /// Parameters and coordinates for one UFF bond stretching interaction.
  class MolgrFFBondCalculationUFF : public OBFFCalculation2
  {
    public:
      double bt = 0.0; // bond order (e.g., 1.41 for amide)
      double kb = 0.0;
      double r0 = 0.0;
      double rab = 0.0;
      double delta = 0.0;

      /// Evaluate this bond stretching interaction and accumulate gradients when requested by the template
      /// argument.
      template<bool> void Compute();
  };

  /// Parameters and coordinates for one UFF angle bending interaction.
  class MolgrFFAngleCalculationUFF : public OBFFCalculation3
  {
    public:
      int at = 0; //angletype (ATIJK)
      bool linear = false;
      double ka = 0.0;
      double theta0 = 0.0;
      double theta = 0.0;
      double delta = 0.0;
      double c0 = 0.0;
      double c1 = 0.0;
      double c2 = 0.0;
      double zi = 0.0;
      double zk = 0.0;
      double rij = 0.0;
      double rjk = 0.0;
      double rik = 0.0;
      double cosT0 = 0.0; // cos theta0
      int coord = 0;
      int n = 0;

      /// Evaluate this angle bending interaction and accumulate gradients when requested by the template
      /// argument.
      template<bool> void Compute();
  };

  /// Parameters and coordinates for one UFF torsion interaction.
  class MolgrFFTorsionCalculationUFF : public OBFFCalculation4
  {
    public:
      int n = 0;
      double tt = 0.0; //torsiontype (i.e. b-c bond order)
      double V = 0.0;
      double tor = 0.0;
      double cosNPhi0 = 0.0;

      /// Evaluate this torsion interaction and accumulate gradients when requested by the template
      /// argument.
      template<bool> void Compute();

  };

  /// Parameters and coordinates for one UFF out-of-plane bending interaction.
  class MolgrFFOOPCalculationUFF : public OBFFCalculation4
  {
    public:
      double koop = 0.0;
      double angle = 0.0;
      double c0 = 0.0;
      double c1 = 0.0;
      double c2 = 0.0;

      /// Evaluate this out-of-plane bending interaction and accumulate gradients when requested by the
      /// template argument.
      template<bool> void Compute();
  };

  /// Parameters and coordinates for one UFF van der Waals interaction.
  class MolgrFFVDWCalculationUFF : public OBFFCalculation2
  {
    public:
      bool is14 = false;
      bool samering = false;
      double ka = 0.0;
      double kaSquared = 0.0;
      double Ra = 0.0;
      double kb = 0.0;
      double Rb = 0.0;
      double kab = 0.0;
      double rab = 0.0;

      /// Evaluate this van der Waals interaction and accumulate gradients when requested by the template
      /// argument.
      template<bool> void Compute();
  };

  /// Parameters and coordinates for one UFF electrostatic interaction.
  class MolgrFFElectrostaticCalculationUFF : public OBFFCalculation2
  {
    public:
      double qq = 0.0;
      double rab = 0.0;

      /// Evaluate this electrostatic interaction and accumulate gradients when requested by the template
      /// argument.
      template<bool> void Compute();
  };

  // Class MolgrForceFieldUFF
  // class introduction in forcefieldUFF.cpp
  /// UFF evaluator with per-instance constraints, atom typing and interaction lists for one molecular
  /// topology.
  class MolgrForceFieldUFF: public OBForceField
  {
  protected:
    /// @return True when the UFF parameter table is available.
    bool ParseParamFile();
    /// @return True when every atom has a UFF type usable for parameter lookup.
    bool SetTypes();
    /// Build the bonded and nonbonded interaction lists for the typed molecule.
    /// @return True when the requested setup completed successfully.
    bool SetupCalculations();
    /// Bind the stored interaction terms to the active molecular coordinates.
    /// @return True when the requested setup completed successfully.
    bool SetupPointers();
    /// Initialize the mixed van der Waals parameters for a nonbonded atom pair.
    /// @param a First atom of the nonbonded pair.
    /// @param b Second atom of the nonbonded pair.
    /// @param vdwcalc Destination for the mixed van der Waals interaction parameters.
    /// @return True when the requested setup completed successfully.
    bool SetupVDWCalculation(OBAtom *a, OBAtom *b, MolgrFFVDWCalculationUFF &vdwcalc);
    /// Initialize electrostatic pairs when the electrostatic term is requested.
    /// @return True when the requested setup completed successfully.
    bool SetupElectrostatics();
    /// @param a UFF atom type name.
    /// @param parameter Compatibility argument. Lookup uses the evaluator's own indexed parameter table.
    /// @return The UFF parameter record for an atom type, or null if the type is unknown.
    OBFFParameter* GetParameterUFF(std::string a, std::vector<OBFFParameter> &parameter);

    // OBFFParameter vectors to contain the parameters
    std::vector<OBFFParameter> _ffparams;
    std::unordered_map<std::string, std::size_t> _ffparam_index;

    // OBFFXXXCalculationYYY vectors to contain the calculations
    std::vector<MolgrFFBondCalculationUFF>          _bondcalculations;
    std::vector<MolgrFFAngleCalculationUFF>         _anglecalculations;
    std::vector<MolgrFFTorsionCalculationUFF>       _torsioncalculations;
    std::vector<MolgrFFOOPCalculationUFF>           _oopcalculations;
    std::vector<MolgrFFVDWCalculationUFF>           _vdwcalculations;
    std::vector<MolgrFFElectrostaticCalculationUFF> _electrostaticcalculations;

    OBFFConstraints constraints_;
    unsigned int fix_atom_ = 0;
    unsigned int ignore_atom_ = 0;
    bool use_atom_typing_cache_ = false;
    std::string atom_typing_cache_key_;

    /// Make this evaluator's fixed and ignored atom settings available to this thread's calculations.
    void ActivateThreadLocalInstance();
    /// @return The UFF instance active on this thread.
    static MolgrForceFieldUFF* ActiveInstance();
    static thread_local MolgrForceFieldUFF* active_instance_;

  public:
    /// Initialize an independent evaluator with the default UFF cutoffs and minimization settings.
    MolgrForceFieldUFF() : OBForceField("", false)
    {
      _validSetup = false;
      _init = false;
      _rvdw = 7.0;
      _rele = 15.0;
      _epsilon = 1.0; // electrostatics not used
      _pairfreq = 10;
      _cutoff = false;
      _linesearch = LineSearchType::Newton2Num;
    }

    /// Release the evaluator and detach its thread-local instance reference when applicable.
    ~MolgrForceFieldUFF() override;

    /// @return A newly allocated UFF evaluator with independently initialized state.
    MolgrForceFieldUFF* MakeNewInstance() override
    {
       return new MolgrForceFieldUFF();
    }

    /// Prepare a typed copy of the molecule and the interaction terms used to evaluate its energy.
    /// @param mol Molecular graph on which the operation acts.
    /// @return True when the requested setup completed successfully.
    bool Setup(OBMol &mol);
    /// Prepare a typed copy of the molecule and the interaction terms used to evaluate its energy.
    /// @param mol Molecular graph on which the operation acts.
    /// @param constraints Coordinate constraints to apply to this UFF setup.
    /// @return True when the requested setup completed successfully.
    bool Setup(OBMol &mol, OBFFConstraints &constraints);

    /// Copy the source molecule, interaction lists and constraint state into this evaluator.
    /// @param src Evaluator supplying the molecule and UFF interaction state.
    /// @return Reference to this evaluator.
    MolgrForceFieldUFF &operator = (MolgrForceFieldUFF &src);

    /// @return The constraints used by this evaluator.
    OBFFConstraints& GetConstraints();
    /// Replace the evaluator's coordinate constraints.
    /// @param constraints Coordinate constraints to apply to this UFF setup.
    void SetConstraints(OBFFConstraints& constraints);
    /// Set the evaluator’s fixed-atom index.
    /// @param index One-based molecular atom index.
    void SetFixAtom(int index);
    /// Clear the fixed-atom index.
    void UnsetFixAtom();
    /// Select an atom to omit from interaction calculations.
    /// @param index One-based molecular atom index.
    void SetIgnoreAtom(int index);
    /// Clear the ignored-atom index so its interactions can be evaluated again.
    void UnsetIgnoreAtom();
    /// @param a First one-based atom index in the interaction.
    /// @param b Second one-based atom index in the interaction.
    /// @return True when the active evaluator excludes an atom participating in this interaction.
    static bool IgnoreCalculation(int a, int b);
    /// @param a First one-based atom index in the interaction.
    /// @param b Second one-based atom index in the interaction.
    /// @param c Third one-based atom index in the interaction.
    /// @return True when the active evaluator excludes an atom participating in this interaction.
    static bool IgnoreCalculation(int a, int b, int c);
    /// @param a First one-based atom index in the interaction.
    /// @param b Second one-based atom index in the interaction.
    /// @param c Third one-based atom index in the interaction.
    /// @param d Fourth one-based atom index in the interaction.
    /// @return True when the active evaluator excludes an atom participating in this interaction.
    static bool IgnoreCalculation(int a, int b, int c, int d);
    /// Configure reuse of UFF atom types for an equivalent typing environment.
    /// @param enabled Whether to reuse stored UFF atom-type assignments.
    /// @param cache_key Fingerprint identifying a reusable molecular environment.
    void ConfigureAtomTypingCache(bool enabled, std::string cache_key);
    /// Copy the evaluator's perceived hybridizations to corresponding atoms of the destination molecule.
    /// @param mol Molecular graph on which the operation acts.
    void CopyPerceivedHybridizationTo(OBMol &mol) const;
    /// @return The assigned UFF type strings in molecular atom order.
    std::vector<std::string> DebugAtomTypes() const;

    /// @return The force field's display name.
    const char* Description()
    {
      return "Universal Force Field.";
    }

    /// @return The energy unit used by this implementation (kJ/mol).
    std::string GetUnit()
      {
        return std::string("kJ/mol");  // Note that we convert from kcal/mol internally
      }

    /// @return True because UFF supplies analytical coordinate gradients.
    bool HasAnalyticalGradients() { return true; }

    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The sum of enabled UFF energy terms in kJ/mol.
    double Energy(bool gradients = true);
    /// @return The UFF bond stretching energy in kJ/mol.
    template<bool> double E_Bond();
    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The UFF bond stretching energy in kJ/mol.
    double E_Bond(bool gradients = true)
    {
      return gradients ? E_Bond<true>() : E_Bond<false>();
    }
    /// @return The UFF angle bending energy in kJ/mol.
    template<bool> double E_Angle();
    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The UFF angle bending energy in kJ/mol.
    double E_Angle(bool gradients = true)
    {
      return gradients ? E_Angle<true>() : E_Angle<false>();
    }
    /// @return The UFF torsion energy in kJ/mol.
    template<bool> double E_Torsion();
    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The UFF torsion energy in kJ/mol.
    double E_Torsion(bool gradients = true)
    {
      return gradients ? E_Torsion<true>() : E_Torsion<false>();
    }
    /// @return The UFF out-of-plane bending energy in kJ/mol.
    template<bool> double E_OOP();
    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The UFF out-of-plane bending energy in kJ/mol.
    double E_OOP(bool gradients = true)
    {
      return gradients ? E_OOP<true>() : E_OOP<false>();
    }
    /// @return The UFF van der Waals energy in kJ/mol.
    template<bool> double E_VDW();
    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The UFF van der Waals energy in kJ/mol.
    double E_VDW(bool gradients = true)
    {
      return gradients ? E_VDW<true>() : E_VDW<false>();
    }
    /// @return The UFF electrostatic energy in kJ/mol.
    template<bool> double E_Electrostatic();
    /// @param gradients Whether the evaluation should also accumulate coordinate gradients.
    /// @return The UFF electrostatic energy in kJ/mol.
    double E_Electrostatic(bool gradients = true)
    {
      return gradients ? E_Electrostatic<true>() : E_Electrostatic<false>();
    }

    /// Report whether analytical derivatives agree with numerical derivatives in the gradient check.
    /// @return True when every component passes the numerical gradient check.
    bool ValidateGradients();

  }; // class MolgrForceFieldUFF

  /// Clear stored UFF atom-type assignments and their cache counters.
  void ClearMolgrUffAtomTypeAssignmentCache();
  /// @return Atom-typing cache hits, misses and current entry count, in that order.
  std::tuple<std::size_t, std::size_t, std::size_t> MolgrUffAtomTypeAssignmentCacheInfo();

}// namespace OpenBabel

//! \file forcefieldUFF.h
//! \brief UFF force field
