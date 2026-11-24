/*
 * @Author: TMJ
 * @Date: 2026-01-01 23:12:22
 * @LastEditors: TMJ
 * @LastEditTime: 2026-01-01 23:35:25
 * @Description: 请填写简介
 */
#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace molgr {
namespace bind {

    /// Register graph records, stereo frames and conversions between MoleculeData and Open Babel.
    /// @param m Python module on which to register the native interface.
    void bind_utils(py::module_ &m);
    /// Register bounded graph equivalence and fixed-mapping stereo comparison.
    /// @param m Python module on which to register the native interface.
    void bind_equivalence(py::module_ &m);
    /// Register public reconstruction entry points and per-run diagnostic access.
    /// @param m Python module on which to register the native interface.
    void bind_pipeline(py::module_ &m);
    /// Register native scoring, geometry and force-field inspection helpers.
    /// @param m Python module on which to register the native interface.
    void bind_dev_utils(py::module_ &m);
    /// Register the development submodules for ligand search, metal assignments and resonance.
    /// @param m Python module on which to register the native interface.
    void bind_dev_pipeline(py::module_ &m);
    /// Expose the individual bond, charge and radical reconstruction stages to Python.
    /// @param m Python module on which to register the native interface.
    void bind_stages(py::module_ &m);

} // namespace bind
} // namespace molgr
