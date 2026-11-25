#include "molgr/python_config.h"

#include <string>

#include <pybind11/stl.h>

namespace py = pybind11;

namespace molgr::config
{
    namespace
    {
        /// Read a required named attribute of a Python configuration object.
        /// @param object Python configuration object providing the named attribute.
        /// @param name Name of the requested field or counter.
        /// @return Named attribute, converted to the requested native type where applicable.
        py::object RequiredAttr(py::handle object, const char *name)
        {
            if (object.is_none())
            {
                throw py::type_error("MolGR config object is None while reading '" +
                                     std::string(name) + "'");
            }
            if (!py::hasattr(object, name))
            {
                throw py::attribute_error("MolGR config object is missing required attribute '" +
                                          std::string(name) + "'");
            }
            return py::reinterpret_borrow<py::object>(object.attr(name));
        }

        /// Read a required named attribute of a Python configuration object.
        /// @tparam T Native type required by the configuration schema.
        /// @param object Python configuration object providing the named attribute.
        /// @param name Name of the requested field or counter.
        /// @return Named attribute, converted to the requested native type where applicable.
        template <typename T>
        T CastRequiredAttr(py::handle object, const char *name)
        {
            return RequiredAttr(object, name).cast<T>();
        }

        /// @param object Python configuration object providing the named attribute.
        /// @param name Name of the requested field or counter.
        /// @return The configured integer, or an empty optional for Python None.
        std::optional<int> CastRequiredOptionalIntAttr(py::handle object, const char *name)
        {
            py::object value = RequiredAttr(object, name);
            if (value.is_none())
            {
                return std::nullopt;
            }
            return value.cast<int>();
        }

        /// @param config Settings for the current reconstruction or scoring operation.
        /// @return The supplied Python configuration, falling back to the package configuration when
        ///     omitted.
        py::object ResolvePythonConfig(py::handle config)
        {
            if (!config.is_none())
            {
                return py::reinterpret_borrow<py::object>(config);
            }
            py::module_ config_module = py::module_::import("molgr.config");
            return config_module.attr("CONFIG");
        }
    }

    const MolGRConfig &GetDefaultConfig()
    {
        static const MolGRConfig config;
        return config;
    }

    MolGRConfig FromPython(py::handle config)
    {
        py::object resolved = ResolvePythonConfig(config);
        MolGRConfig out;

        py::object resonance = RequiredAttr(resolved, "resonance");
        out.resonance.max_depth = CastRequiredAttr<int>(resonance, "max_depth");
        out.resonance.limited_discrepancy_max_discrepancy = CastRequiredAttr<int>(
            resonance,
            "limited_discrepancy_max_discrepancy");
        out.resonance.traversal_score = CastRequiredAttr<std::string>(
            resonance,
            "traversal_score");
        out.resonance.neighbor_radical_max_discrepancy = CastRequiredAttr<int>(
            resonance,
            "neighbor_radical_max_discrepancy");

        py::object cpp_backend = RequiredAttr(resolved, "cpp_backend");
        out.cpp_backend.max_threads = CastRequiredOptionalIntAttr(
            cpp_backend,
            "max_threads");
        out.cpp_backend.enable_target_bucket_parallelism = CastRequiredAttr<bool>(
            cpp_backend,
            "enable_target_bucket_parallelism");
        out.cpp_backend.enable_candidate_scoring_parallelism = CastRequiredAttr<bool>(
            cpp_backend,
            "enable_candidate_scoring_parallelism");
        out.cpp_backend.enable_uff_atom_typing_cache = CastRequiredAttr<bool>(
            cpp_backend,
            "enable_uff_atom_typing_cache");
        out.cpp_backend.target_bucket_parallel_threshold = CastRequiredAttr<int>(
            cpp_backend,
            "target_bucket_parallel_threshold");
        out.cpp_backend.target_bucket_parallel_max_threads = CastRequiredOptionalIntAttr(
            cpp_backend,
            "target_bucket_parallel_max_threads");
        out.cpp_backend.candidate_score_parallel_threshold = CastRequiredAttr<int>(
            cpp_backend,
            "candidate_score_parallel_threshold");

        py::object organic_topology = RequiredAttr(resolved, "organic_topology");
        out.organic_topology.conjugation_min_plane_overlap =
            CastRequiredAttr<double>(organic_topology, "conjugation_min_plane_overlap");
        out.organic_topology.conjugation_normalized_tetrahedron_volume_tolerance =
            CastRequiredAttr<double>(
                organic_topology,
                "conjugation_normalized_tetrahedron_volume_tolerance");
        out.organic_topology.aromatic_stability_benzene_score = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_benzene_score");
        out.organic_topology.aromatic_stability_other_ring_max_score = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_other_ring_max_score");
        out.organic_topology.aromatic_stability_ring_size_6_factor = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_ring_size_6_factor");
        out.organic_topology.aromatic_stability_ring_size_5_factor = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_ring_size_5_factor");
        out.organic_topology.aromatic_stability_other_ring_size_factor = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_other_ring_size_factor");
        out.organic_topology.aromatic_stability_hetero_atom_penalty = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_hetero_atom_penalty");
        out.organic_topology.aromatic_stability_min_hetero_factor = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_min_hetero_factor");
        out.organic_topology.aromatic_stability_formal_charge_penalty = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_formal_charge_penalty");
        out.organic_topology.aromatic_stability_min_charge_factor = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_min_charge_factor");
        out.organic_topology.aromatic_stability_radical_penalty = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_radical_penalty");
        out.organic_topology.aromatic_stability_min_radical_factor = CastRequiredAttr<double>(
            organic_topology,
            "aromatic_stability_min_radical_factor");

        py::object metal_scoring = RequiredAttr(resolved, "metal_scoring");
        out.metal_scoring.open_shell_multimetal_state_penalty_window = CastRequiredAttr<double>(
            metal_scoring,
            "open_shell_multimetal_state_penalty_window");
        out.metal_scoring.open_shell_multimetal_min_state_options = CastRequiredAttr<int>(
            metal_scoring,
            "open_shell_multimetal_min_state_options");
        out.metal_scoring.same_element_multimetal_unify_threshold = CastRequiredAttr<int>(
            metal_scoring,
            "same_element_multimetal_unify_threshold");
        out.metal_scoring.max_mixed_valence_spread = CastRequiredOptionalIntAttr(
            metal_scoring,
            "max_mixed_valence_spread");
        out.metal_scoring.max_assignments_per_target = CastRequiredAttr<int>(
            metal_scoring,
            "max_assignments_per_target");
        out.metal_scoring.metal_coordination_extra_tolerance_angstrom = CastRequiredAttr<double>(
            metal_scoring,
            "metal_coordination_extra_tolerance_angstrom");
        out.metal_scoring.pi_dative_distance_difference_tolerance_angstrom =
            CastRequiredAttr<double>(
                metal_scoring,
                "pi_dative_distance_difference_tolerance_angstrom");
        out.metal_scoring.metal_access_radius_scale = CastRequiredAttr<double>(
            metal_scoring,
            "metal_access_radius_scale");
        out.metal_scoring.metal_access_clearance_angstrom = CastRequiredAttr<double>(
            metal_scoring,
            "metal_access_clearance_angstrom");
        out.metal_scoring.charge_localization_selection_margin = CastRequiredAttr<double>(
            metal_scoring,
            "charge_localization_selection_margin");

        py::object metal_radical = RequiredAttr(resolved, "metal_radical_inference");
        out.metal_radical_inference.max_considered_donors = CastRequiredAttr<int>(
            metal_radical,
            "max_considered_donors");
        out.metal_radical_inference.square_planar_planarity_tolerance_angstrom =
            CastRequiredAttr<double>(
                metal_radical,
                "square_planar_planarity_tolerance_angstrom");
        out.metal_radical_inference.trigonal_planar_planarity_tolerance_angstrom =
            CastRequiredAttr<double>(
                metal_radical,
                "trigonal_planar_planarity_tolerance_angstrom");
        out.metal_radical_inference.linear_angle_min_degrees = CastRequiredAttr<double>(
            metal_radical,
            "linear_angle_min_degrees");
        out.metal_radical_inference.strong_field_threshold = CastRequiredAttr<double>(
            metal_radical,
            "strong_field_threshold");
        out.metal_radical_inference.weak_field_threshold = CastRequiredAttr<double>(
            metal_radical,
            "weak_field_threshold");
        out.metal_radical_inference.field_ambiguity_margin = CastRequiredAttr<double>(
            metal_radical,
            "field_ambiguity_margin");

        return out;
    }
}
