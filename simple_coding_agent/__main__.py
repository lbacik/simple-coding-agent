"""Runnable entry point for the unattended single-process agent."""

from __future__ import annotations

import sys
from pathlib import Path

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.command_runner import CommandRunner
from simple_coding_agent.completion import CompletionEvaluator, VerificationRunner
from simple_coding_agent.config import load_repository_profile, load_runtime_config
from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.github_tracker import GitHubGraphQLTransport, GitHubTracker
from simple_coding_agent.lifecycle import AgentLifecycle, ModelAttemptRunner
from simple_coding_agent.model_execution import ModelExecutor
from simple_coding_agent.observability import AttemptArchive, JsonEventLogger
from simple_coding_agent.operating import ConsecutiveErrorStore
from simple_coding_agent.publication import Publisher
from simple_coding_agent.provenance import ProvenanceError, ProvenanceVerifier
from dotenv import load_dotenv


def main() -> None:
    """Run sequential attempts forever for the one operator-configured repository."""

    load_dotenv(override=False)
    config = load_runtime_config()
    attempt_state = AttemptStateStore(config.data_dir)
    archive_factory = lambda issue_number, started_at: AttemptArchive(
        config.data_dir,
        issue_number=issue_number,
        started_at=started_at,
        redactions=(config.github_token, config.meta_api_key),
    )

    def _attempt_sink(issue_number: int) -> AttemptArchive | None:
        checkpoint = attempt_state.read()
        if checkpoint is None or checkpoint.issue_number != issue_number:
            return None
        return archive_factory(issue_number, checkpoint.started_at)

    logger = JsonEventLogger(
        sys.stdout,
        redactions=(config.github_token, config.meta_api_key),
        attempt_sink=_attempt_sink,
    )
    try:
        provenance = ProvenanceVerifier(
            home=Path.home(),
            expected_sdk_version=config.claude_agent_sdk_version,
            expected_cli_version=config.claude_code_version,
        ).verify()
    except ProvenanceError as error:
        logger.emit("provenance_verification_failed", phase="startup", detail=str(error), level="ERROR")
        raise SystemExit(1) from error
    logger.emit(
        "provenance_verified",
        phase="startup",
        detail=f"sdk={provenance.sdk_version}; cli={provenance.cli_version}",
    )
    github = GitHubGraphQLTransport(config.github_token, max_retries=config.max_retries)
    workspace = GitWorkspace(
        config.clone_dir,
        f"https://github.com/{config.target_repo}.git",
        token_provider=lambda: config.github_token,
    )
    tracker = GitHubTracker(github, config.target_repo)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=VerificationRunner(
            CommandRunner(
                redactions=(config.github_token, config.meta_api_key),
                extra_path=config.profile_extra_path,
            )
        ),
        evaluator=CompletionEvaluator(config.review_blocking_severities),
        issue_comments=tracker.trusted_comments,
        model_executor=ModelExecutor(
            config,
            event_log=lambda event, detail="", issue_number=None: logger.emit(
                event, phase="model_execution", detail=detail, issue_number=issue_number
            ),
        ),
        attempt_archive_factory=archive_factory,
        event_log=lambda event, detail="", level="ERROR", issue_number=None: logger.emit(
            event, phase="setup", detail=detail, level=level, issue_number=issue_number
        ),
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=attempt_state,
        workspace=workspace,
        profile_loader=lambda repository_dir: load_repository_profile(
            repository_dir, config.profile_path
        ),
        publisher=Publisher(
            workspace,
            github,
            config.target_repo,
            max_retries=config.max_retries,
            publish_timeout=config.publish_timeout,
            attempt_state=attempt_state,
        ),
        attempt_runner=runner,
        poll_interval=config.poll_interval,
        error_store=ConsecutiveErrorStore(config.data_dir),
        max_consecutive_errors=config.max_consecutive_errors,
        event_log=lambda event, detail="", level="INFO", issue_number=None: logger.emit(
            event, phase="polling", detail=detail, level=level, issue_number=issue_number
        ),
        attempt_archive_factory=archive_factory,
    )
    lifecycle.run_forever(stop=lambda: False)


if __name__ == "__main__":
    main()
