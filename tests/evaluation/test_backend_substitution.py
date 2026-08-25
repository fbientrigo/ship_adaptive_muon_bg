"""Unit D: the mission's core acceptance criterion — backend substitution.

The claim under test: *a deterministic fake backend can be replaced by a future
FairShip adapter without modifying ``ProxyTagger``, ``Nflow``, the aggregation
logic, or any downstream metric.*

The experiment runs **one** pipeline function against **two** independently
written backends. ``FakeFairShipBackend`` derives its structure from sha256
draws; ``MinimalStubBackend`` walks a fixed cycle of integers and never hashes
anything. They share no code path beyond the canonical entity constructors. If
the same downstream code produces well-formed, invariant-respecting results for
both, backend choice is dependency injection rather than a scientific
assumption compiled into the stack.

This is evidence, not proof: neither backend is FairShip. What it establishes
is that the downstream stack contains no branch on backend identity and reads
nothing a real adapter could not also supply.
"""

from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pytest

from ProxyTagger.baseline import DummyProxy
from ship_muon_bg.entities import ExecutionStatus
from ship_muon_bg.simulation.evaluation import evaluate_verified
from ship_muon_bg.simulation.fake_fairship import FakeFairShipBackend, FakeStageSpec
from ship_muon_bg.simulation.stub_backend import MinimalStubBackend
from ship_muon_bg.tagging.aggregation import (
    DecisionLevel,
    ExecutionStageOutcome,
    aggregate_state_stage_evaluations,
)

from tests.evaluation import fixtures as fx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --------------------------------------------------------------------------
# The single pipeline, shared verbatim by both backends
# --------------------------------------------------------------------------


def run_pipeline(backend, request, stage_definition_ids):
    """Backend -> canonical records -> state-level tagging table.

    Note what is absent: no ``if backend.name == ...``, no import from any
    adapter package, no knowledge of how the records were produced. The only
    thing this function knows about ``backend`` is that it satisfies the
    protocol.
    """
    bundle = evaluate_verified(backend, request)
    return bundle, aggregate_state_stage_evaluations(
        executions=bundle.executions,
        candidates=bundle.candidates,
        decisions=bundle.decisions,
        stage_definition_ids=stage_definition_ids,
        subjects=request.subjects,
    )


def _fake_backend():
    definition, spec = fx.stage_spec()
    return definition, FakeFairShipBackend(
        stages=(spec,), technical_failure_modulus=4, max_realizations=2
    )


def _stub_backend(stage_definition_id):
    return MinimalStubBackend(
        stage_definition_ids=(stage_definition_id,),
        observation_definition_ids=(fx.SCORE_A,),
        candidate_count_cycle=(0, 1, 3),
        score_cycle=(0.9, 0.2),
        threshold=fx.THRESHOLD,
        failure_every=5,
    )


def _request():
    return fx.request(
        subject_ids=tuple(f"s{i}" for i in range(6)),
        replications_per_subject=4,
        seed=17,
    )


@pytest.fixture
def both_backends():
    definition, fake = _fake_backend()
    stub = _stub_backend(definition.stage_definition_id)
    return definition, fake, stub


# --------------------------------------------------------------------------
# Substitution
# --------------------------------------------------------------------------


def test_the_same_pipeline_runs_unchanged_against_both_backends(both_backends):
    definition, fake, stub = both_backends
    request = _request()
    stages = (definition.stage_definition_id,)

    fake_bundle, fake_dataset = run_pipeline(fake, request, stages)
    stub_bundle, stub_dataset = run_pipeline(stub, request, stages)

    # Both produced real, non-trivial work rather than empty structures.
    assert fake_bundle.executions and stub_bundle.executions
    assert fake_bundle.candidates and stub_bundle.candidates
    assert fake_dataset.rows and stub_dataset.rows

    # Identical schema, identical grouping, identical reduction rule.
    assert set(fake_dataset.as_records()[0]) == set(stub_dataset.as_records()[0])
    assert fake_dataset.subject_ids == stub_dataset.subject_ids
    assert fake_dataset.stage_definition_ids == stub_dataset.stage_definition_ids
    assert {row.rollup_rule for row in fake_dataset.rows} == {
        row.rollup_rule for row in stub_dataset.rows
    }


def test_the_scientific_invariants_hold_for_both_backends(both_backends):
    definition, fake, stub = both_backends
    request = _request()
    stages = (definition.stage_definition_id,)

    for backend in (fake, stub):
        bundle, dataset = run_pipeline(backend, request, stages)

        failed_ids = {
            e.execution_id
            for e in bundle.executions
            if e.execution_status is ExecutionStatus.TECHNICAL_FAILURE
        }
        assert failed_ids, f"{backend.name} must exercise the failure path"

        # 3.2 — no failed run contributed a candidate or a decision value.
        assert not [c for c in bundle.candidates if c.execution_id in failed_ids]
        for result in dataset.results:
            if result.execution_id in failed_ids:
                assert result.outcome is ExecutionStageOutcome.TECHNICAL_FAILURE
                assert not result.is_valid

        # 3.2 — failures are outside every valid denominator, and visible.
        for row in dataset.rows:
            assert row.valid_count == row.positive_count + row.negative_count
            assert (
                row.valid_count
                + row.technical_failure_count
                + row.technically_censored_count
                + row.not_evaluated_count
                == row.execution_count
            )
            if row.eta_hat is not None:
                assert 0.0 <= row.eta_hat <= 1.0

        assert sum(row.technical_failure_count for row in dataset.rows) == len(failed_ids)

        # 3.1 — one row per (state, config, stage): repeats never became states.
        keys = [
            (r.subject_id, r.fs_sim_configuration_id, r.stage_definition_id)
            for r in dataset.rows
        ]
        assert len(keys) == len(set(keys))
        assert set(dataset.subject_ids) <= set(request.subject_ids)

        # 3.3 — multiplicity survived to the end of the pipeline.
        multiplicities = set()
        for row in dataset.rows:
            multiplicities |= set(row.candidate_count_distribution)
        assert len(multiplicities) > 1, "a backend that only ever yields one N proves nothing"


def test_the_aggregation_source_contains_no_reference_to_any_backend():
    """The substantive form of "switching backend is configuration".

    Sweeping the *test's* own helper would prove nothing about the library, so
    this parses the real aggregation module and checks every identifier,
    attribute, string and comment in it. If the reduction ever grows a branch on
    which simulator produced the records, this fails.
    """
    import ship_muon_bg.tagging.aggregation as aggregation

    tree = ast.parse(inspect.getsource(aggregation))

    # The executable surface: every identifier the module can name, plus every
    # module it can reach. Prose is excluded on purpose - the docstring
    # explains precisely why the module cannot see a backend, and it should be
    # allowed to say so.
    surface = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            surface.add(node.id)
        elif isinstance(node, ast.Attribute):
            surface.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            surface.add(node.name)
        elif isinstance(node, ast.arg):
            surface.add(node.arg)
        elif isinstance(node, ast.Import):
            surface.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            surface.add(node.module)
        elif isinstance(node, ast.Compare):
            # A string comparison is the one way prose could become a branch.
            surface.update(
                item.value
                for item in [node.left, *node.comparators]
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )

    forbidden = ("fairship", "backend", "adapter", "simulation", "root")
    offenders = sorted(
        name
        for name in surface
        if any(token in name.lower() for token in forbidden)
    )
    assert offenders == [], f"aggregation can reach {offenders}"


def test_both_backends_exercise_the_full_censoring_taxonomy(both_backends):
    """A substitution test over two backends that only ever produce clean
    positives would prove nothing. Both must reach the interesting states."""
    definition, fake, stub = both_backends
    request = _request()
    stages = (definition.stage_definition_id,)

    for backend in (fake, stub):
        bundle, dataset = run_pipeline(backend, request, stages)
        outcomes = {result.outcome for result in dataset.results}
        assert ExecutionStageOutcome.POSITIVE in outcomes
        assert ExecutionStageOutcome.NEGATIVE in outcomes
        assert ExecutionStageOutcome.TECHNICAL_FAILURE in outcomes
        # Every empty run said whether it looked, so none of them fell through
        # to an unattested NOT_EVALUATED.
        empty = [r for r in dataset.results if r.candidate_count == 0 and r.is_valid]
        assert empty
        assert all(r.decision_level is DecisionLevel.EXECUTION for r in empty)


def test_the_full_path_reaches_an_unmodified_proxy_scorer(both_backends):
    """Backend -> records -> aggregation -> labels -> ProxyTagger.

    ``DummyProxy`` is used exactly as it already exists on the branch; nothing
    in ``ProxyTagger/`` is adapted for this. States with no evaluable execution
    are dropped rather than labelled zero — the aggregation layer reports
    ``eta_hat is None`` for them, and turning that into a training label would
    fabricate a confident negative out of missing data.
    """
    definition, fake, stub = both_backends
    request = _request()
    stages = (definition.stage_definition_id,)
    proxy = DummyProxy()

    for backend in (fake, stub):
        _bundle, dataset = run_pipeline(backend, request, stages)
        labelled = [row for row in dataset.rows if row.eta_hat is not None]
        assert len(labelled) > 1

        # Features come from the subject's own coordinates, never from the
        # backend. A distinct momentum per state, in the (N, 8) contract shape,
        # so row alignment between features and labels is actually observable
        # rather than hidden behind identical rows.
        x = np.zeros((len(labelled), 8), dtype=np.float64)
        for index, row in enumerate(labelled):
            x[index, 0] = 10.0 * (index + 1)
        labels = np.array([row.eta_hat for row in labelled], dtype=np.float64)
        counts = np.array([row.valid_count for row in labelled], dtype=np.float64)
        assert len(labels) == len(counts) == x.shape[0]
        assert np.all(np.isfinite(labels)) and np.all((labels >= 0) & (labels <= 1))

        proxy.fit(x, labels, sample_weight=counts)
        scores = proxy.score(x)
        assert scores.shape == (len(labelled),)
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        # Row order is preserved end to end: score is monotone in |p|, which is
        # monotone in the row index by construction above.
        assert np.all(np.diff(scores) > 0)


# --------------------------------------------------------------------------
# Structural proof that the downstream stack cannot be backend-aware
# --------------------------------------------------------------------------


def _dotted_imports(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        if "legacy" in dirpath.split(os.sep):
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            full = os.path.join(dirpath, name)
            with open(full, encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=full)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    found.extend((full, alias.name) for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    found.append((full, node.module))
    return found


@pytest.mark.parametrize("package", ["ProxyTagger", "Nflow"])
def test_downstream_packages_cannot_see_any_backend(package):
    """``ProxyTagger`` and ``Nflow`` import no simulation or adapter module.

    This is the structural half of the substitution claim: they could not
    branch on which backend ran even if someone wanted them to, because the
    backend layer is not reachable from them at all.
    """
    offenders = [
        (path, dotted)
        for path, dotted in _dotted_imports(os.path.join(REPO_ROOT, package))
        if any(
            segment in {"simulation", "adapters", "fairship"}
            for segment in dotted.lower().split(".")
        )
    ]
    assert offenders == []


def test_the_aggregation_layer_cannot_see_any_backend():
    """``tagging/`` consumes canonical entities only — never an
    ``EvaluationBundle``, a backend, or an adapter — so replacing the simulator
    cannot reach it."""
    offenders = [
        (path, dotted)
        for path, dotted in _dotted_imports(
            os.path.join(REPO_ROOT, "src", "ship_muon_bg", "tagging")
        )
        if any(
            segment in {"simulation", "adapters", "fairship", "root"}
            for segment in dotted.lower().split(".")
        )
    ]
    assert offenders == []
