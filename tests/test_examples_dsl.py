"""
Regression tests for the bundled preset mechanisms (M1-M9).

These tests ensure that:
1. All bundled presets parse successfully with the DSL parser
2. All bundled presets can be loaded via the resource system
3. Parsing produces valid Mechanism objects with expected structure
4. A representative subset can run basic simulations without errors
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from kindred.core.simulator.dsl import parse_dsl_to_mechanism
from kindred.core.ode_builder import build_ode_rhs_from_mechanism
from kindred.core.simulator.solvers import SimulationRequest, solve_ode
from kindred.io.resources import get_all_example_specs, get_resource_text

pytestmark = pytest.mark.integration


_DOC_INTERVENTION_EXAMPLE_RE = re.compile(
    r"<!--\s*kindred-test:\s*intervention-example\s*-->\s*```text\n(?P<body>.*?)\n```",
    re.DOTALL,
)


def _run_scheduled_doc_example(text: str):
    mechanism = parse_dsl_to_mechanism(text)
    rhs = build_ode_rhs_from_mechanism(mechanism)
    species_names = mechanism.species_names()
    y0 = np.asarray([mechanism.species[name].initial_conc for name in species_names], dtype=float)
    request = SimulationRequest(
        rhs=rhs,
        t_span=(0.0, 4.0),
        y0=y0,
        solver="BDF",
        grid={"N": 9},
        species_names=tuple(species_names),
        intervention_schedule=mechanism.metadata.get("intervention_schedule"),
    )
    return solve_ode(request)


def test_dsl_reference_intervention_examples_are_executable() -> None:
    reference = Path("DSL_REFERENCE.md").read_text(encoding="utf-8")
    examples = [match.group("body").strip() for match in _DOC_INTERVENTION_EXAMPLE_RE.finditer(reference)]

    assert examples, "DSL_REFERENCE.md must include executable intervention examples."

    for example in examples:
        result = _run_scheduled_doc_example(example)
        assert result.provenance["has_intervention_schedule"] is True
        assert result.t.size > 0
        assert result.Y.shape[1] == result.t.size


def test_dsl_reference_describes_schedule_annotations_as_optional_display_aids() -> None:
    reference = Path("DSL_REFERENCE.md").read_text(encoding="utf-8")

    assert "off by default" in reference
    assert "Show Intervention Schedule Annotations" in reference
    assert "display truth" in reference



class TestExampleParsing:
    """Test that all bundled preset mechanisms parse successfully."""

    @pytest.fixture(scope="class")
    def all_examples(self):
        """Get metadata for all bundled preset examples."""
        examples = get_all_example_specs()
        assert examples, "No examples discovered"
        assert len(examples) == 9, f"Expected 9 presets, found {len(examples)}"
        return examples

    def test_all_examples_discovered(self, all_examples):
        """Verify we found the curated preset set."""
        example_ids = {ex["id"] for ex in all_examples}

        expected_presets = {f"M{i}" for i in range(1, 10)}
        assert example_ids == expected_presets, (
            f"Expected presets {expected_presets}, found {example_ids}"
        )

    @pytest.mark.parametrize("example_spec", get_all_example_specs())
    def test_example_loads_from_resources(self, example_spec):
        """Test that each bundled preset can be loaded via the resource system."""
        example_id = example_spec["id"]
        resource_path = example_spec["path"]

        # Load the example text
        text = get_resource_text(resource_path)

        # Basic validation
        assert text, f"Example {example_id} loaded empty text"
        assert len(text) > 0, f"Example {example_id} has no content"
        assert isinstance(text, str), f"Example {example_id} did not load as string"

    @pytest.mark.parametrize("example_spec", get_all_example_specs())
    def test_example_parses_successfully(self, example_spec):
        """Test that each bundled preset parses without errors."""
        example_id = example_spec["id"]
        resource_path = example_spec["path"]

        # Load and parse
        text = get_resource_text(resource_path)

        try:
            mechanism = parse_dsl_to_mechanism(text)
        except Exception as e:
            pytest.fail(
                f"Example {example_id} failed to parse: {e}\n"
                f"Resource path: {resource_path}"
            )

        # Validate the mechanism object
        assert mechanism is not None, f"Example {example_id} produced None mechanism"
        assert hasattr(mechanism, "reactions"), (
            f"Example {example_id} mechanism missing reactions attribute"
        )
        assert hasattr(mechanism, "species"), (
            f"Example {example_id} mechanism missing species attribute"
        )

    @pytest.mark.parametrize("example_spec", get_all_example_specs())
    def test_example_has_species(self, example_spec):
        """Test that each parsed preset contains at least one species."""
        example_id = example_spec["id"]
        resource_path = example_spec["path"]

        # Load and parse
        text = get_resource_text(resource_path)
        mechanism = parse_dsl_to_mechanism(text)

        # Check for species
        species_count = len(mechanism.species)
        assert species_count > 0, (
            f"Example {example_id} has no species. "
            f"Mechanism should define at least one species."
        )

    @pytest.mark.parametrize("example_spec", get_all_example_specs())
    def test_example_has_reactions_or_equilibria(self, example_spec):
        """Test that each bundled preset has reactions or equilibria."""
        example_id = example_spec["id"]
        resource_path = example_spec["path"]

        # Load and parse
        text = get_resource_text(resource_path)
        mechanism = parse_dsl_to_mechanism(text)

        # Check for reactions or equilibria
        has_reactions = len(mechanism.reactions) > 0
        has_equilibria = len(mechanism.equilibria) > 0

        assert has_reactions or has_equilibria, (
            f"Example {example_id} has no reactions or equilibria. "
            f"Reactions: {len(mechanism.reactions)}, "
            f"Equilibria: {len(mechanism.equilibria)}"
        )


class TestExampleSimulation:
    """Optional tests for running basic simulations on examples."""

    # Select a few representative presets for simulation tests
    # (not all, to keep test runtime reasonable)
    SIMULATION_TEST_EXAMPLES = [
        "M1",
        "M2",
        "M9",
    ]

    @pytest.mark.parametrize("example_id", SIMULATION_TEST_EXAMPLES)
    def test_example_runs_basic_simulation(self, example_id):
        """Test that selected bundled presets can run a short simulation."""
        # Find the example spec
        all_examples = get_all_example_specs()
        example_spec = next(
            (ex for ex in all_examples if ex["id"] == example_id),
            None,
        )
        assert example_spec is not None, f"Example {example_id} not found"

        # Load and parse
        text = get_resource_text(example_spec["path"])
        mechanism = parse_dsl_to_mechanism(text)

        # Build initial state vector
        y0 = np.array([mechanism.species[name].initial_conc for name in mechanism.species])

        # Build ODE RHS function
        rhs = build_ode_rhs_from_mechanism(mechanism)

        # Create simulation request
        request = SimulationRequest(
            rhs=rhs,
            t_span=(0.0, 1.0),  # Very short time span
            y0=y0,
            solver="BDF",
            rtol=1e-6,
            atol=1e-12,
            grid={"N": 10},  # Small grid for speed
        )

        # Run simulation
        try:
            result = solve_ode(request)
        except Exception as e:
            pytest.fail(
                f"Example {example_id} failed basic simulation: {e}\n"
                f"This suggests the mechanism may be invalid for integration."
            )

        # Basic validation of results
        # If solve_ode returns without raising, the simulation succeeded
        assert result.t.size > 0, (
            f"Example {example_id} simulation produced no time points"
        )
        assert result.Y.shape[0] == len(mechanism.species), (
            f"Example {example_id} simulation produced wrong number of species. "
            f"Expected {len(mechanism.species)}, got {result.Y.shape[0]}"
        )
        assert result.Y.shape[1] == result.t.size, (
            f"Example {example_id} simulation shape mismatch. "
            f"Y.shape[1]={result.Y.shape[1]}, t.size={result.t.size}"
        )

    def test_all_presets_have_initial_conditions(self):
        """Test that all bundled preset mechanisms (M1-M9) define initial conditions."""
        all_examples = get_all_example_specs()
        presets = [ex for ex in all_examples if ex["type"] == "preset"]

        for preset in presets:
            text = get_resource_text(preset["path"])
            mechanism = parse_dsl_to_mechanism(text)

            # Check that at least one species has a non-zero initial condition
            assert len(mechanism.species) > 0, (
                f"Preset {preset['id']} has no species defined"
            )

            # Check that at least one initial is > 0
            has_nonzero = any(
                species.initial_conc > 0
                for species in mechanism.species.values()
            )
            assert has_nonzero, (
                f"Preset {preset['id']} has no non-zero initial concentrations. "
                f"Initials: {[(name, sp.initial_conc) for name, sp in mechanism.species.items()]}"
            )


class TestExampleMetadata:
    """Test the example metadata helper itself."""

    def test_example_specs_structure(self):
        """Test that get_all_example_specs returns properly structured data."""
        examples = get_all_example_specs()

        for ex in examples:
            # Check required keys
            assert "id" in ex, f"Example missing 'id': {ex}"
            assert "type" in ex, f"Example missing 'type': {ex}"
            assert "path" in ex, f"Example missing 'path': {ex}"

            # Validate types
            assert isinstance(ex["id"], str), f"Example id should be str: {ex}"
            assert isinstance(ex["type"], str), f"Example type should be str: {ex}"
            assert isinstance(ex["path"], str), f"Example path should be str: {ex}"

            assert ex["type"] == "preset", (
                f"Example type should be 'preset', got: {ex['type']}"
            )

    def test_example_ids_are_unique(self):
        """Test that all example IDs are unique."""
        examples = get_all_example_specs()
        ids = [ex["id"] for ex in examples]

        # Check for duplicates
        id_set = set(ids)
        assert len(ids) == len(id_set), (
            f"Duplicate example IDs found. "
            f"Total: {len(ids)}, Unique: {len(id_set)}"
        )

    def test_preset_examples_count(self):
        """Test that we have exactly 9 preset examples (M1-M9)."""
        examples = get_all_example_specs()
        presets = [ex for ex in examples if ex["type"] == "preset"]
        assert len(presets) == 9, f"Expected 9 presets, found {len(presets)}"
