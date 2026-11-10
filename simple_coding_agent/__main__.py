"""Runnable entry point for the unattended single-process agent.

Startup order (operator contract, startup ticket):

1. configuration parsing;
2. volume lock: an exclusive advisory lock on a lock file in the state
   directory, held for the process lifetime. Failing to acquire it exits
   non-zero before serving or claiming;
3. control store open;
4. control server start, serving ``instance.state: starting``;
5. provenance verification, reported as ``instance.startup_check:
   provenance``;
6. remaining construction;
7. the lifecycle loop.

A configuration error before the volume lock cannot be recorded as a
halt: the engine exits non-zero with a stderr diagnostic (not an
uncaught traceback) and records nothing. This is a documented
limitation.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

from simple_coding_agent.agent_version import agent_version
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.command_runner import CommandRunner
from simple_coding_agent.completion import CompletionEvaluator, VerificationRunner
from simple_coding_agent.contract_events import (
    ContractEventsBuffer,
    halted_detail,
    process_started_detail,
    seed_recorded_halt,
    startup_check_detail,
)
from simple_coding_agent.config import (
    EXPECTED_CLAUDE_AGENT_SDK_VERSION,
    RuntimeConfig,
    load_repository_profile,
    load_runtime_config,
)
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer, socket_path_for
from simple_coding_agent.finalization import AttemptCompletionStore
from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.github_tracker import GitHubGraphQLTransport, GitHubTracker
from simple_coding_agent.handoff_record import HandoffPublicationStore
from simple_coding_agent.halt import (
    HALT_CONTROL_ENDPOINT_UNAVAILABLE,
    HALT_PROVENANCE_FAILED,
    HaltStore,
)
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptInterruptionHandler, ModelAttemptRunner
from simple_coding_agent.model_execution import (
    PROMPT_CACHE_MIN_MEASURED_RESPONSES,
    ModelExecutor,
)
from simple_coding_agent.observability import AttemptArchive, JsonEventLogger
from simple_coding_agent.operating import ConsecutiveErrorStore
from simple_coding_agent.publication import Publisher
from simple_coding_agent.provenance import ProvenanceError, ProvenanceVerifier
from simple_coding_agent.telemetry import RecentEventsBuffer, TelemetryReporter, format_timestamp
from simple_coding_agent.volume_lock import VolumeLock, VolumeLockError
from dotenv import load_dotenv


def _load_config() -> RuntimeConfig:
    """Parse operator configuration or exit non-zero with a stderr diagnostic.

    A configuration error happens before the volume lock, so it cannot be
    recorded as a halt: it exits non-zero with a stderr diagnostic (never an
    uncaught traceback) and records nothing.
    """

    try:
        return load_runtime_config()
    except Exception as error:
        print(f"configuration error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


def _acquire_volume_lock(data_dir: Path) -> VolumeLock:
    """Acquire the single-instance volume lock or exit before serving or claiming."""

    lock = VolumeLock(data_dir)
    try:
        lock.acquire()
    except VolumeLockError as error:
        print(f"volume lock: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    return lock


def emit_process_started(logger: JsonEventLogger) -> None:
    """Emit the ``process_started`` native event for the contract projection."""

    try:
        logger.emit(
            "process_started",
            phase="startup",
            detail=process_started_detail(agent_version()),
        )
    except Exception:
        pass


def emit_startup_check(logger: JsonEventLogger, check: str, result: str) -> None:
    """Emit one ``startup_check`` native event (result started/ok/skipped/failed)."""

    try:
        level = "ERROR" if result == "failed" else "INFO"
        logger.emit(
            "startup_check",
            phase="startup",
            detail=startup_check_detail(check, result),
            level=level,
        )
    except Exception:
        pass


def emit_halted(logger: JsonEventLogger, reason: str) -> None:
    """Emit the ``halted`` native event for an engine-decided exit."""

    try:
        logger.emit(
            "halted",
            phase="startup",
            detail=halted_detail(reason),
            level="ERROR",
        )
    except Exception:
        pass


def _record_startup_halt(halt_store: HaltStore | None, reason: str, detail: str) -> None:
    """Best-effort durable halt before an engine-decided exit (never raises)."""

    if halt_store is None:
        return
    try:
        halt_store.record(reason, detail)
    except Exception:
        pass


def main() -> None:
    """Run sequential attempts forever for the one operator-configured repository."""

    load_dotenv(override=False)
    process_started_at = format_timestamp(datetime.now(UTC))
    # 1. Configuration parsing (records nothing on failure: no halt possible
    # before the volume lock).
    config = _load_config()
    # 2. Volume lock (the only new single-instance guard; claim semantics
    # are unchanged). Held for the process lifetime.
    _volume_lock = _acquire_volume_lock(config.data_dir)
    halt_store = HaltStore(config.data_dir)
    # 3. Control store open.
    attempt_state = AttemptStateStore(config.data_dir)
    control_store = ControlStore(config.data_dir)
    archive_factory = lambda issue_number, started_at: AttemptArchive(
        config.data_dir,
        issue_number=issue_number,
        started_at=started_at,
        redactions=config.credential_redactions,
    )

    def _attempt_sink(issue_number: int) -> AttemptArchive | None:
        checkpoint = attempt_state.read()
        if checkpoint is None or checkpoint.issue_number != issue_number:
            return None
        return archive_factory(issue_number, checkpoint.started_at)

    recent_events = RecentEventsBuffer()
    contract_events = ContractEventsBuffer()
    # The previous process's ``halted`` event died with its buffer: replay
    # it from the durable halt record before this process emits anything.
    seed_recorded_halt(contract_events, halt_store)
    logger = JsonEventLogger(
        sys.stdout,
        redactions=config.credential_redactions,
        attempt_sink=_attempt_sink,
        recent_events=recent_events,
        contract_events=contract_events,
    )
    emit_process_started(logger)
    emit_startup_check(logger, "provenance", "started")
    # Early construction that needs no provenance, so the control endpoint
    # can serve ``starting`` while provenance verification runs.
    github = GitHubGraphQLTransport(config.github_token, max_retries=config.max_retries)
    workspace = GitWorkspace(
        config.clone_dir,
        f"https://github.com/{config.target_repo}.git",
        token_provider=lambda: config.github_token,
    )
    tracker = GitHubTracker(
        github,
        config.target_repo,
        max_auto_continuations=config.max_auto_continuations,
        max_conflict_attempts=config.max_conflict_attempts,
        worker_id=config.worker_id,
        event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: logger.emit(
            event, phase="polling", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
        ),
    )
    command_runner = CommandRunner(
        redactions=config.credential_redactions,
        extra_path=config.profile_extra_path,
    )
    publisher = Publisher(
        workspace,
        github,
        config.target_repo,
        max_retries=config.max_retries,
        publish_timeout=config.publish_timeout,
        auto_merge=config.pr_auto_merge,
        merge_method=config.pr_merge_method,
        attempt_state=attempt_state,
        handoff_store=HandoffPublicationStore(config.data_dir),
        handoff_comment_mode=config.handoff_comment_mode,
        event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: logger.emit(
            event, phase="publication", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
        ),
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=attempt_state,
        workspace=workspace,
        profile_loader=lambda repository_dir: load_repository_profile(
            repository_dir, config.profile_path
        ),
        publisher=publisher,
        attempt_runner=None,
        poll_interval=config.poll_interval,
        error_store=ConsecutiveErrorStore(config.data_dir),
        max_consecutive_errors=config.max_consecutive_errors,
        max_auto_continuations=config.max_auto_continuations,
        max_conflict_attempts=config.max_conflict_attempts,
        command_runner=command_runner,
        conflict_model_executor=None,
        conflict_issue_comments=None,
        conflict_prompt_redactions=config.credential_redactions,
        event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: logger.emit(
            event, phase="polling", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
        ),
        attempt_archive_factory=archive_factory,
        completion_store=AttemptCompletionStore(config.data_dir),
        control_store=control_store,
        contract_events=contract_events,
        repository=config.target_repo,
        telemetry=None,
        data_dir=config.data_dir,
        mode="online",
        process_started_at=process_started_at,
        starting=True,
        startup_check="provenance",
        halt_store=halt_store,
    )
    interruption_handler = AttemptInterruptionHandler(
        event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: logger.emit(
            event, phase="interruption", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
        ),
        issue_number_provider=lambda: lifecycle.active_issue_number,
        checkpoint=lifecycle.shutdown,
    )
    restore_interruption_handler = interruption_handler.install()
    control_server = ControlServer(
        socket_path_for(config.data_dir),
        lifecycle,
        event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: logger.emit(
            event, phase="control", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
        ),
    )
    # 4. Control server start, serving ``starting``.
    try:
        _start_control_endpoint(logger, control_server, halt_store)
        # 5. Provenance verification, reported as ``startup_check: provenance``.
        try:
            provenance = ProvenanceVerifier(
                home=Path.home(),
                expected_sdk_version=EXPECTED_CLAUDE_AGENT_SDK_VERSION,
            ).verify()
        except ProvenanceError as error:
            detail = str(error)
            _record_startup_halt(halt_store, HALT_PROVENANCE_FAILED, detail)
            emit_startup_check(logger, "provenance", "failed")
            emit_halted(logger, HALT_PROVENANCE_FAILED)
            logger.emit("provenance_verification_failed", phase="startup", detail=detail, level="ERROR")
            raise SystemExit(1) from None
        emit_startup_check(logger, "provenance", "ok")
        logger.emit(
            "provenance_verified",
            phase="startup",
            detail=provenance.summary(),
        )
        # 6. Remaining construction.
        model_executor = ModelExecutor(
            config,
            cli_path=provenance.cli_path,
            event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: logger.emit(
                event, phase="model_execution", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
            ),
        )
        runner = ModelAttemptRunner(
            attempt_state=attempt_state,
            workspace=workspace,
            verifier=VerificationRunner(command_runner),
            evaluator=CompletionEvaluator(config.review_blocking_severities),
            issue_comments=tracker.trusted_comments,
            prompt_comments=tracker.prompt_comments,
            prompt_redactions=config.credential_redactions,
            model_executor=model_executor,
            attempt_archive_factory=archive_factory,
            event_log=lambda event, detail="", level="ERROR", issue_number=None, attempt_id=None: logger.emit(
                event, phase="setup", detail=detail, level=level, issue_number=issue_number, attempt_id=attempt_id
            ),
        )
        telemetry = TelemetryReporter(
            repository=config.target_repo,
            agent_version=provenance.agent_version,
            sdk_version=provenance.sdk_version,
            cli_version=provenance.cli_version,
            process_started_at=process_started_at,
            hard_ceiling=config.max_budget_tokens,
            turns_max=config.max_turns,
            timeout_seconds=config.model_timeout,
            continuation_max=config.max_auto_continuations,
            usd_ceiling=config.max_budget_usd,
            # Single source of truth with the executor's prompt-cache warning:
            # the served hit rate stays unavailable below this many measured
            # main-thread responses.
            cache_min_measured_responses=PROMPT_CACHE_MIN_MEASURED_RESPONSES,
            recent_events=recent_events,
        )
        lifecycle.attach_post_provenance(
            attempt_runner=runner,
            telemetry=telemetry,
            conflict_model_executor=model_executor,
            conflict_issue_comments=tracker.prompt_comments,
        )
        lifecycle.mark_started()
        # 7. The lifecycle loop.
        lifecycle.run_forever(stop=lambda: False)
    finally:
        control_server.stop()
        # After the engine's last control write: merge the WAL into the
        # main database file so the stopped volume reads without it. The
        # SIGTERM/SIGINT handler above checkpoints the same way before it
        # resignals, since `finally` never runs on death by signal.
        lifecycle.shutdown()
        restore_interruption_handler()


def _start_control_endpoint(
    logger: JsonEventLogger,
    control_server: ControlServer,
    halt_store: HaltStore | None = None,
) -> None:
    """Start the operator control endpoint or log and exit.

    A failure to open the control socket is fatal: record a halt, emit the
    ``halted`` contract event plus one structured
    ``control_endpoint_unavailable`` event and exit 1 so the operator sees
    JSON instead of a raw traceback.
    """

    try:
        control_server.start()
    except Exception as error:
        detail = f"Operator control socket could not be opened: {error}"
        _record_startup_halt(
            halt_store, HALT_CONTROL_ENDPOINT_UNAVAILABLE, detail
        )
        emit_halted(logger, HALT_CONTROL_ENDPOINT_UNAVAILABLE)
        logger.emit(
            "control_endpoint_unavailable",
            phase="startup",
            detail=detail,
            level="ERROR",
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
