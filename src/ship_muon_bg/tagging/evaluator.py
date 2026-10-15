"""Backend-independent evaluation of declarative stage definitions."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

from ship_muon_bg.entities.decision import (
    DecisionEvaluationStatus,
    StageDecision,
    stage_decision_id,
)
from ship_muon_bg.entities.observation import (
    ObservationEnvelope,
    ObservationEvaluationStatus,
    ScalarObservationPayload,
)

from ship_muon_bg.tagging.definitions import StageDefinition


class StageEvaluator:
    """Apply a supported ``StageDefinition`` to one declared entity reference.

    The v0 evaluator intentionally supports only the controlled
    ``greater_than`` scalar rule.  It never aggregates observations across
    subjects, executions, realizations, or candidates; duplicate evidence for
    one required definition is rejected as ambiguous rather than reduced.
    """

    def evaluate(
        self,
        definition: StageDefinition,
        observations: Iterable[ObservationEnvelope],
        *,
        subject_ref: str,
    ) -> StageDecision:
        """Return one ``StageDecision`` for ``subject_ref`` and ``definition``.

        Availability precedence is deterministic when required evidence is
        mixed: ``TECHNICALLY_UNAVAILABLE`` wins over ``NOT_APPLICABLE``, which
        wins over a missing envelope.  The first produces a technically
        unavailable decision; either of the latter two produces
        ``NOT_EVALUATED``.  None of these paths produces ``False``.
        """
        if not isinstance(definition, StageDefinition):
            raise TypeError("definition must be a StageDefinition")
        if not isinstance(subject_ref, str) or not subject_ref:
            raise ValueError("subject_ref must be a non-empty string")

        required = definition.required_observation_definition_ids
        required_set = set(required)
        consumed_by_definition = {}
        for observation in tuple(observations):
            if not isinstance(observation, ObservationEnvelope):
                raise TypeError("observations must contain ObservationEnvelope objects")
            if observation.subject_ref != subject_ref:
                continue
            if observation.observation_definition_id not in required_set:
                continue
            if observation.observation_definition_id in consumed_by_definition:
                raise ValueError(
                    "multiple observations for one required definition are ambiguous; "
                    "the tagging core does not aggregate them"
                )
            consumed_by_definition[observation.observation_definition_id] = observation

        consumed = tuple(
            consumed_by_definition[observation_definition_id]
            for observation_definition_id in required
            if observation_definition_id in consumed_by_definition
        )
        evidence_references = tuple(observation.observation_id for observation in consumed)
        missing = tuple(
            observation_definition_id
            for observation_definition_id in required
            if observation_definition_id not in consumed_by_definition
        )
        technically_unavailable = tuple(
            observation.observation_definition_id
            for observation in consumed
            if observation.evaluation_status
            is ObservationEvaluationStatus.TECHNICALLY_UNAVAILABLE
        )
        not_applicable = tuple(
            observation.observation_definition_id
            for observation in consumed
            if observation.evaluation_status
            is ObservationEvaluationStatus.NOT_APPLICABLE
        )

        if technically_unavailable:
            return self._decision(
                definition,
                subject_ref,
                DecisionEvaluationStatus.TECHNICALLY_UNAVAILABLE,
                None,
                evidence_references,
                self._reason("required_observation_technically_unavailable", technically_unavailable),
            )
        if not_applicable:
            return self._decision(
                definition,
                subject_ref,
                DecisionEvaluationStatus.NOT_EVALUATED,
                None,
                evidence_references,
                self._reason("required_observation_not_applicable", not_applicable),
            )
        if missing:
            return self._decision(
                definition,
                subject_ref,
                DecisionEvaluationStatus.NOT_EVALUATED,
                None,
                evidence_references,
                self._reason("required_observation_missing", missing),
            )

        decision = self._evaluate_supported_rule(definition, consumed)
        return self._decision(
            definition,
            subject_ref,
            DecisionEvaluationStatus.EVALUATED,
            decision,
            evidence_references,
            "",
        )

    @staticmethod
    def _reason(prefix: str, observation_definition_ids: Tuple[str, ...]) -> str:
        return f"{prefix}:{','.join(observation_definition_ids)}"

    @staticmethod
    def _decision(
        definition: StageDefinition,
        subject_ref: str,
        evaluation_status: DecisionEvaluationStatus,
        decision,
        evidence_references: Tuple[str, ...],
        reason: str,
    ) -> StageDecision:
        # ObservationEnvelope.observation_id is the stable identity of the
        # evidence realization.  The evaluator canonicalizes evidence order
        # through StageDefinition's sorted dependency tuple, so this hash is
        # independent of the caller's iterable order.  ``decision`` is
        # intentionally absent: for the supported rule it is derived from the
        # stage definition, referenced evidence, and evaluation state.  The
        # human-readable reason is likewise derived censoring context, not an
        # independent semantic input.
        return StageDecision(
            decision_id=stage_decision_id(
                stage_definition_id=definition.stage_definition_id,
                subject_ref=subject_ref,
                evidence_references=evidence_references,
                evaluation_status=evaluation_status,
            ),
            subject_ref=subject_ref,
            stage_definition_id=definition.stage_definition_id,
            evaluation_status=evaluation_status,
            decision=decision,
            reason=reason,
            evidence_references=evidence_references,
        )

    @staticmethod
    def _evaluate_supported_rule(
        definition: StageDefinition, consumed: Tuple[ObservationEnvelope, ...]
    ) -> bool:
        content = definition.semantic_content
        if content.get("operation") != "greater_than":
            raise ValueError(
                "StageEvaluator v0 supports only the declarative greater_than operation"
            )
        if len(consumed) != 1 or len(definition.required_observation_definition_ids) != 1:
            raise ValueError("greater_than requires exactly one scalar observation")

        required_observation_definition_id = definition.required_observation_definition_ids[0]
        declared_observation_definition_id = content.get("observation_definition_id")
        if (
            declared_observation_definition_id is not None
            and declared_observation_definition_id != required_observation_definition_id
        ):
            raise ValueError(
                "semantic observation_definition_id does not match the required definition"
            )
        threshold = content.get("threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("greater_than threshold must be numeric")
        if not math.isfinite(float(threshold)):
            raise ValueError("greater_than threshold must be finite")

        observation = consumed[0]
        if not isinstance(observation.payload, ScalarObservationPayload):
            raise TypeError("greater_than requires a ScalarObservationPayload")
        return bool(observation.payload.value > float(threshold))
