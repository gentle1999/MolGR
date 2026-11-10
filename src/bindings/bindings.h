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

    void bind_utils(py::module_ &m);
    void bind_equivalence(py::module_ &m);
    void bind_pipeline(py::module_ &m);
    void bind_dev_utils(py::module_ &m);
    void bind_dev_pipeline(py::module_ &m);
    void bind_stages(py::module_ &m);

} // namespace bind
} // namespace molgr
