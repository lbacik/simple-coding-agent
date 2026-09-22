from __future__ import annotations

from pathlib import Path

import pytest

from simple_coding_agent.command_runner import CommandExecution, CommandResult, CommandStatus
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionEvaluator,
    PublicationPath,
    ReviewEvidence,
    ReviewFinding,
    VerificationRunner,
    VerificationOrderError,
)
from simple_coding_agent.config import ProfileError, RepositoryProfile
from simple_coding_agent.model_execution import ModelExecutionStatus


def command_result(status: CommandStatus = CommandStatus.SUCCEEDED) -> CommandResult:
    exit_code = 0 if status is CommandStatus.SUCCEEDED else 1
    return CommandResult(status, (CommandExecution("check", exit_code, "", "", False),), 0.1)


def test_verification_runs_setup_then_an_uncached_baseline_then_final_check(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    class Runner:
        def run(self, commands, **kwargs):
            calls.append(commands)
            return command_result()

    profile = RepositoryProfile(("setup",), ("check",), "main", 30, 20, {})
    verifier = VerificationRunner(Runner())

    preparation = verifier.prepare(profile, tmp_path)
    with pytest.raises(VerificationOrderError, match="requires completed model review"):
        verifier.final_check(profile, tmp_path)
    verifier.mark_review_complete(
        model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=1,
        review=ReviewEvidence((), 0),
    )
    final_check = verifier.final_check(profile, tmp_path)

    assert preparation.setup.succeeded
    assert preparation.baseline.succeeded
    assert final_check.succeeded
    assert calls == [("setup",), ("check",), ("check",)]


def test_setup_failure_is_an_infrastructure_error_and_skips_baseline(tmp_path: Path) -> None:
    class Runner:
        def run(self, commands, **kwargs):
            return command_result(CommandStatus.FAILED)

    profile = RepositoryProfile(("setup",), ("check",), "main", 30, 20, {})
    preparation = VerificationRunner(Runner()).prepare(profile, tmp_path)

    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=preparation.setup,
        baseline=preparation.baseline,
        model_status=None,
        commit_count=0,
        acceptance_criteria_satisfied=False,
        review=ReviewEvidence((), 0),
        final_check=None,
    )
    assert preparation.baseline is None
    assert decision.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR


def test_new_preparation_discards_the_previous_attempts_review_gate(tmp_path: Path) -> None:
    class Runner:
        def run(self, commands, **kwargs):
            return command_result()

    profile = RepositoryProfile(("setup",), ("check",), "main", 30, 20, {})
    verifier = VerificationRunner(Runner())
    verifier.prepare(profile, tmp_path)
    verifier.mark_review_complete(
        model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=1,
        review=ReviewEvidence((), 0),
    )
    verifier.prepare(profile, tmp_path)

    with pytest.raises(VerificationOrderError, match="requires completed model review"):
        verifier.final_check(profile, tmp_path)


def test_profile_error_is_structured_as_an_infrastructure_error() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=None,
        baseline=None,
        model_status=None,
        commit_count=0,
        acceptance_criteria_satisfied=False,
        review=ReviewEvidence((), 0),
        final_check=None,
        profile_error=ProfileError("Repository profile could not be read"),
    )

    assert decision.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert decision.reasons == ("Repository profile could not be loaded.",)


def test_baseline_failure_is_incomplete_before_model_execution() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=command_result(), baseline=command_result(CommandStatus.FAILED), model_status=None,
        commit_count=0, acceptance_criteria_satisfied=False, review=ReviewEvidence((), 0), final_check=None,
    )
    assert decision.outcome is AttemptOutcome.INCOMPLETE
    assert decision.publication_eligible is False


def test_zero_commit_successful_skill_run_is_no_changes() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=command_result(), baseline=command_result(), model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=0, acceptance_criteria_satisfied=False, review=ReviewEvidence((), 0), final_check=None,
    )
    assert decision.outcome is AttemptOutcome.NO_CHANGES


def test_handoff_request_with_committed_progress_is_a_handoff_eligible_for_partial_publication() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=command_result(), baseline=command_result(),
        model_status=ModelExecutionStatus.HANDOFF_REQUESTED,
        commit_count=1, acceptance_criteria_satisfied=False, review=ReviewEvidence((), 0), final_check=None,
    )

    assert decision.outcome is AttemptOutcome.HANDOFF
    assert decision.publication_eligible is False
    assert decision.publication_path is PublicationPath.PARTIAL


def test_handoff_request_without_preserved_work_downgrades_to_no_changes() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=command_result(), baseline=command_result(),
        model_status=ModelExecutionStatus.HANDOFF_REQUESTED,
        commit_count=0, acceptance_criteria_satisfied=False, review=ReviewEvidence((), 0), final_check=None,
    )

    assert decision.outcome is AttemptOutcome.NO_CHANGES
    assert decision.publication_eligible is False
    assert decision.publication_path is PublicationPath.NONE


def test_local_success_is_only_eligible_until_push_and_pr_are_observed() -> None:
    evaluator = CompletionEvaluator({"must-fix"})
    arguments = dict(
        setup=command_result(), baseline=command_result(), model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=1, acceptance_criteria_satisfied=True, review=ReviewEvidence((), 2), final_check=command_result(),
    )

    eligible = evaluator.evaluate(**arguments)
    complete = evaluator.evaluate(**arguments, push_succeeded=True, pr_exists=True)

    assert eligible.outcome is None
    assert eligible.publication_eligible is True
    assert complete.outcome is AttemptOutcome.COMPLETE


def test_unsatisfied_acceptance_criteria_block_publication_eligibility() -> None:
    decision = CompletionEvaluator({"must-fix"}).evaluate(
        setup=command_result(), baseline=command_result(), model_status=ModelExecutionStatus.SUCCEEDED,
        commit_count=1, acceptance_criteria_satisfied=False, review=ReviewEvidence((), 0),
        final_check=command_result(),
    )

    assert decision.outcome is AttemptOutcome.INCOMPLETE
    assert decision.publication_eligible is False
    assert decision.publication_path is PublicationPath.PARTIAL


def test_review_threshold_and_model_limit_block_completion() -> None:
    evaluator = CompletionEvaluator({"must-fix"})
    common = dict(
        setup=command_result(), baseline=command_result(), commit_count=1,
        acceptance_criteria_satisfied=True, final_check=command_result(),
    )

    review_blocked = evaluator.evaluate(
        **common, model_status=ModelExecutionStatus.SUCCEEDED,
        review=ReviewEvidence((ReviewFinding("must-fix", "Fix this"),), 3),
    )
    limited = evaluator.evaluate(
        **common, model_status=ModelExecutionStatus.MODEL_LIMIT_REACHED,
        review=ReviewEvidence((), 0),
    )

    assert review_blocked.outcome is AttemptOutcome.INCOMPLETE
    assert limited.outcome is AttemptOutcome.INCOMPLETE


def test_model_and_publication_infrastructure_errors_are_not_reported_complete() -> None:
    evaluator = CompletionEvaluator({"must-fix"})
    common = dict(
        setup=command_result(), baseline=command_result(), commit_count=1,
        acceptance_criteria_satisfied=True, review=ReviewEvidence((), 0), final_check=command_result(),
    )

    model_error = evaluator.evaluate(**common, model_status=ModelExecutionStatus.INFRASTRUCTURE_ERROR)
    push_error = evaluator.evaluate(
        **common, model_status=ModelExecutionStatus.SUCCEEDED, push_succeeded=False
    )

    assert model_error.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
    assert push_error.outcome is AttemptOutcome.INFRASTRUCTURE_ERROR
