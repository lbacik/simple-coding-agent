"""Fakes-wired engine for the conformance harness (not shipped in production).

Runs the real :class:`AgentLifecycle
<simple_coding_agent.lifecycle.AgentLifecycle>` against the shared test
fakes (``FakeGitHub`` per issue plus ``FakeModelExecutor`` from
``tests/fakes.py`` — reused, never copied) over a real git remote, serving
the contract on a Unix control socket. The host-side harness CLI
(:mod:`conformance_harness.harness`) steers this process through flag
files: holds, one-shot crashes, scripted native outcomes, halt requests,
the deferred-publication one-shot, the claim gate and the engine lock.

The engine advances only when a harness verb asks it to, as the contract's
reference stub producer does: a Claim opens only through the gate that
``release`` arms (:class:`GatedTracker`), a claimed attempt waits for the
outcome ``finish-attempt`` scripts (:class:`BlockingExecutor`), and
``hold-lock`` makes this process hold the engine's own control lock
(:func:`_hold_control_lock_while_held`).

Only the ``harness`` Docker build stage contains this module and the
fakes; the production image has neither.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path


def _state_root() -> Path:
    raw = os.environ.get("SCA_HARNESS_STATE_DIR", "~/.cache/sca-conformance-harness")
    return Path(raw).expanduser()


def _read_json(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _take_json(path: Path) -> dict | None:
    """Read and remove one one-shot flag file, if present."""

    payload = _read_json(path)
    if payload is None and not path.exists():
        return None
    try:
        path.unlink()
    except OSError:
        pass
    return payload if payload is not None else {}


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


#: One-shot claim gate armed by ``release`` (``{"token": ...}``).
CLAIM_GATE = "claim_gate.json"
#: How the daemon answered one armed gate (``{"token": ..., "claimed": ...}``).
CLAIM_GATE_RESULT = "claim_gate_result.json"
#: Written while this process holds the engine control lock for ``hold-lock``.
ENGINE_LOCK_HELD = "engine_lock.held"
#: Armed by ``start-container``: the engine stays ``starting`` until ``release``.
STARTUP_HOLD = "startup_hold"
#: Armed by ``start-container`` over a retained attempt: reconciliation waits
#: (the engine reports ``recovering``) until ``release``.
RECOVERY_HOLD = "recovery_hold"
#: Written once the first engine iteration after startup has returned.
STARTUP_RECONCILED = "startup_reconciled"


def claim_gate_token(flags: Path) -> str | None:
    """The token of the armed claim gate, or ``None`` while it is closed."""

    payload = _read_json(flags / CLAIM_GATE)
    token = payload.get("token") if payload else None
    return token if isinstance(token, str) and token else None


def _answer_claim_gate(flags: Path, token: str, *, claimed: bool) -> None:
    """Close the gate ``token`` armed and tell ``release`` what happened."""

    try:
        _write_json(flags / CLAIM_GATE_RESULT, {"token": token, "claimed": claimed})
        if claim_gate_token(flags) == token:
            (flags / CLAIM_GATE).unlink(missing_ok=True)
    except OSError:
        pass


class HarnessTransport:
    """Route every transport call to the per-issue shared ``FakeGitHub``.

    The harness keeps one issue map (``issues.json``); each issue owns a
    ``FakeGitHub`` from the shared test fakes. This facade only routes —
    all behaviour stays in the shared doubles.
    """

    def __init__(self, issues: dict[int, object], spec_path: Path | None = None) -> None:
        self._issues = issues
        self._spec_path = spec_path
        self._sync_lock = threading.Lock()

    def sync(self) -> None:
        """Apply the harness issue map (``issues.json``) to the per-issue fakes.

        Reads sync first, so an issue ``set-issue`` just wrote is visible
        to the very next engine read (a ``next issue`` validation, a claim)
        without waiting for a daemon loop iteration.
        """

        if self._spec_path is None:
            return
        try:
            spec = json.loads(self._spec_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            spec = {}
        if not isinstance(spec, dict):
            return
        with self._sync_lock:
            _sync_issues(self._issues, spec)

    def _for_issue(self, number: int) -> object:
        try:
            return self._issues[number]
        except KeyError:
            raise AssertionError(f"harness issue #{number} is not seeded") from None

    def _for_head(self, head: str) -> object | None:
        from simple_coding_agent.github_tracker import conflict_branch_issue_number

        number = conflict_branch_issue_number(head)
        if number is None:
            return None
        return self._issues.get(number)

    def _for_issue_id(self, issue_id: str) -> object:
        for fake in self._issues.values():
            if fake._issue.id == issue_id:  # noqa: SLF001 - routing only
                return fake
        raise AssertionError(f"harness issue id {issue_id!r} is not seeded")

    # GitHubTransport
    def list_issues(self, repository: str, cursor: str | None) -> object:
        from simple_coding_agent.github_tracker import IssuePage

        self.sync()
        ordered = sorted(self._issues)
        return IssuePage(
            issues=tuple(self._issues[number]._current() for number in ordered),  # noqa: SLF001
            next_cursor=None,
        )

    def list_open_pull_requests(self, repository: str, cursor: str | None) -> object:
        from simple_coding_agent.github_tracker import PullRequestPage

        pulls: list = []
        for fake in self._issues.values():
            pulls.extend(fake.pull_requests.values())
        return PullRequestPage(pull_requests=tuple(pulls), next_cursor=None)

    def get_issue(self, repository: str, number: int) -> object | None:
        self.sync()
        fake = self._issues.get(number)
        if fake is None:
            return None
        return fake.get_issue(repository, number)

    def viewer(self) -> object:
        return next(iter(self._issues.values())).viewer()

    def assign_issue(self, issue_id: str, assignee_id: str) -> object:
        return self._for_issue_id(issue_id).assign_issue(issue_id, assignee_id)

    def remove_assignee(self, issue_id: str, assignee_id: str) -> None:
        self._for_issue_id(issue_id).remove_assignee(issue_id, assignee_id)

    def remove_label(self, issue_id: str, label: str) -> None:
        self._for_issue_id(issue_id).remove_label(issue_id, label)

    def add_label(self, issue_id: str, label: str) -> None:
        self._for_issue_id(issue_id).add_label(issue_id, label)

    def list_issue_comments(self, repository: str, issue_number: int) -> tuple:
        fake = self._issues.get(issue_number)
        if fake is None:
            return ()
        return fake.list_issue_comments(repository, issue_number)

    def add_comment(self, repository: str, issue_number: int, body: str) -> None:
        self._for_issue(issue_number).add_comment(repository, issue_number, body)

    # PublicationTransport
    def find_pull_request(self, repository: str, head: str) -> object | None:
        fake = self._for_head(head)
        if fake is None:
            return None
        return fake.find_pull_request(repository, head)

    def create_pull_request(
        self, repository: str, head: str, base: str, title: str, body: str
    ) -> object:
        fake = self._for_head(head)
        if fake is None:
            raise AssertionError(f"harness issue for head {head!r} is not seeded")
        return fake.create_pull_request(repository, head, base, title, body)

    def enable_auto_merge(
        self, repository: str, pull_request: object, merge_method: str
    ) -> None:
        return None

    def merge_pull_request(
        self, repository: str, pull_request: object, merge_method: str
    ) -> None:
        return None

    def find_attempt_comment(
        self, repository: str, issue_number: int, marker: str
    ) -> bool:
        fake = self._issues.get(issue_number)
        if fake is None:
            return False
        return fake.find_attempt_comment(repository, issue_number, marker)

    def find_handoff_record(self, repository: str, issue_number: int) -> object | None:
        fake = self._issues.get(issue_number)
        if fake is None:
            return None
        return fake.find_handoff_record(repository, issue_number)

    def create_handoff_record(
        self, repository: str, issue_number: int, body: str
    ) -> str:
        return self._for_issue(issue_number).create_handoff_record(
            repository, issue_number, body
        )

    def update_handoff_record(
        self, repository: str, comment_id: str, body: str
    ) -> None:
        for fake in self._issues.values():
            try:
                fake.update_handoff_record(repository, comment_id, body)
                return
            except AssertionError:
                continue
        raise AssertionError(f"unknown handoff comment {comment_id!r}")


class GatedTracker:
    """Let the engine cross the Claim boundary only through ``release``.

    Wraps the real tracker. The daemon loop allows one armed gate token per
    ``run_once``; a claim begins only while that same token is still armed,
    and a successful claim (FIFO ``claim_next`` or the prioritized
    ``claim_verified``) consumes it. With no gate allowed the FIFO queue
    reads as empty, so startup reconciliation, recovery and pending stop
    completion keep running while no claim opens.
    """

    def __init__(self, inner: object, flags: Path) -> None:
        self._inner = inner
        self._flags = flags
        #: The gate token the current ``run_once`` may consume, if any.
        self.allowed: str | None = None

    def _open(self) -> bool:
        return self.allowed is not None and claim_gate_token(self._flags) == self.allowed

    def _consume(self) -> None:
        if self.allowed is not None:
            _answer_claim_gate(self._flags, self.allowed, claimed=True)

    def claim_next(self) -> object | None:
        if not self._open():
            return None
        claim = self._inner.claim_next()
        if claim is not None:
            self._consume()
        return claim

    def claim_verified(self, issue: object) -> object:
        # Reached only with the gate allowed: the daemon loop does not run
        # the engine while a priority waits behind a closed gate.
        claim = self._inner.claim_verified(issue)
        self._consume()
        return claim

    def recover_claim(self, issue_number: int) -> object | None:
        # Startup reconciliation of an interrupted attempt passes here with
        # the engine reporting ``recovering``; after a restart the harness
        # keeps it there until ``release`` (spec: the recovery hold).
        while (self._flags / RECOVERY_HOLD).exists():
            time.sleep(0.05)
        return self._inner.recover_claim(issue_number)

    def __getattr__(self, name: str) -> object:
        return getattr(self.__dict__["_inner"], name)


def _wait_while_held(flags: Path, poll: float = 0.05) -> None:
    """Block while a hold flag file exists (release removes it)."""

    hold = flags / "hold.json"
    while hold.exists():
        time.sleep(poll)


def _checkpoint_phase(data_dir: Path) -> str | None:
    payload = _read_json(data_dir / "state" / "attempt.json")
    if not payload:
        return None
    phase = payload.get("phase")
    return phase if isinstance(phase, str) else None


def _checkpoint_attempt_id(data_dir: Path) -> str | None:
    payload = _read_json(data_dir / "state" / "attempt.json")
    if not payload:
        return None
    started_at = payload.get("started_at")
    return started_at if isinstance(started_at, str) and started_at else None


class BlockingExecutor:
    """Fake model execution that waits for the outcome ``finish-attempt`` scripts.

    Delegates the actual turn to the shared ``FakeModelExecutor`` (with the
    shared ``write_handoff_note`` standing in for the handoff skill's note
    commit). This wrapper only blocks while a hold is armed and until
    ``finish-attempt`` records a scripted native outcome for this attempt;
    a claimed attempt never finishes on its own.
    """

    def __init__(self, flags: Path, data_dir: Path) -> None:
        self._flags = flags
        self._data_dir = data_dir
        self.calls = 0

    def _hold_applies(self) -> bool:
        hold = _read_json(self._flags / "hold.json")
        if not hold:
            return False
        point = hold.get("point")
        if point == "after-claim":
            return True
        if point == "native-phase" and hold.get("phase") == _checkpoint_phase(
            self._data_dir
        ):
            return True
        return False

    def _take_scripted_outcome(self, attempt_id: str | None) -> dict | None:
        """Consume the outcome scripted for ``attempt_id``, if recorded yet.

        Taking it removes it, so it can never finish a later attempt; an
        outcome scripted for another attempt is stale and is dropped.
        """

        path = self._flags / "scripted_outcome.json"
        scripted = _read_json(path)
        if scripted is None:
            return None
        try:
            path.unlink()
        except OSError:
            pass
        if scripted.get("attempt_id") != attempt_id:
            return None
        return scripted

    async def execute(
        self,
        *,
        issue_body: str,
        working_directory: Path,
        issue_number: int | None = None,
        archive=None,
        **kwargs,
    ) -> object:
        from tests.fakes import FakeModelExecutor, write_handoff_note

        self.calls += 1
        # A one-shot crash armed for after-claim fires before the model runs.
        armed_path = self._flags / "armed_crash.json"
        armed = _read_json(armed_path)
        if armed is not None and armed.get("point") == "after-claim":
            try:
                armed_path.unlink()
            except OSError:
                pass
            os._exit(1)
        attempt_id = _checkpoint_attempt_id(self._data_dir)
        while True:
            if not self._hold_applies():
                scripted = self._take_scripted_outcome(attempt_id)
                if scripted is not None:
                    break
            time.sleep(0.05)
        outcome = scripted.get("native_outcome", "complete")
        if outcome == "handoff":
            inner = FakeModelExecutor(
                actions=[
                    lambda wd: write_handoff_note(
                        wd, issue_number or 0, "harness scripted handoff"
                    )
                ],
                handoff=True,
            )
            return await inner.execute(
                issue_body=issue_body,
                working_directory=working_directory,
                issue_number=issue_number,
                archive=archive,
            )
        if outcome == "no_changes":
            inner = FakeModelExecutor()
            return await inner.execute(
                issue_body=issue_body,
                working_directory=working_directory,
                issue_number=issue_number,
                archive=archive,
            )
        if outcome in ("incomplete", "infrastructure_error"):
            from simple_coding_agent.model_execution import ModelExecutionStatus

            status = (
                ModelExecutionStatus.MODEL_LIMIT_REACHED
                if outcome == "incomplete"
                else ModelExecutionStatus.INFRASTRUCTURE_ERROR
            )
            inner = FakeModelExecutor(status=status)
            return await inner.execute(
                issue_body=issue_body,
                working_directory=working_directory,
                issue_number=issue_number,
                archive=archive,
            )
        # The default scripted completion: one local file, sealed and checked.
        def _change(working: Path) -> None:
            (working / "harness-output.txt").write_text("harness scripted change\n")

        inner = FakeModelExecutor(actions=[_change])
        return await inner.execute(
            issue_body=issue_body,
            working_directory=working_directory,
            issue_number=issue_number,
            archive=archive,
        )


class GatedPublisher:
    """Real publication gated by harness holds, crashes and deferral.

    Delegates every call to the real publisher; the gates only pause or
    interrupt around it. ``after-outcome`` holds while the outcome is
    recorded but finalization has not run; ``defer-publication`` arms a
    one-shot that leaves the publication effect pending across the restart
    so recovery finishes it before new claims open.
    """

    def __init__(self, inner: object, flags: Path, data_dir: Path) -> None:
        self._inner = inner
        self._flags = flags
        self._data_dir = data_dir

    def _gate(self) -> None:
        armed_path = self._flags / "armed_crash.json"
        armed = _read_json(armed_path)
        if armed is not None and armed.get("point") == "during-finalization":
            try:
                armed_path.unlink()
            except OSError:
                pass
            os._exit(1)
        if (self._flags / "deferred_publication").exists():
            try:
                (self._flags / "deferred_publication").unlink()
            except OSError:
                pass
            pending = {
                "phase": _checkpoint_phase(self._data_dir),
                "at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            }
            try:
                (self._flags / "pending_publication.json").write_text(
                    json.dumps(pending, sort_keys=True), encoding="utf-8"
                )
            except OSError:
                pass
            os._exit(1)
        hold = _read_json(self._flags / "hold.json")
        if hold is not None and hold.get("point") == "after-outcome":
            _wait_while_held(self._flags)

    def publish(self, request: object, *, timeout: float | None = None) -> object:
        self._gate()
        return self._inner.publish(request, timeout=timeout)

    def publish_conflict(self, request: object, *, timeout: float | None = None) -> object:
        self._gate()
        return self._inner.publish_conflict(request, timeout=timeout)

    def find_pull_request(self, head: str) -> object | None:
        return self._inner.find_pull_request(head)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


class GateProxy:
    """Serialize control mutations with the harness engine lock and crash gate.

    Mutations wait while ``hold-lock`` holds the cross-process engine lock
    (reads keep serving prepared snapshots meanwhile), and a one-shot crash
    armed for ``between-commit-and-reply`` kills the process after the
    commit is durable but before the socket reply is written.
    """

    _MUTATIONS = (
        "submit_stop",
        "submit_stop_after",
        "submit_next_issue",
        "submit_resume",
        "submit_handoff",
        "submit_recovery_retry",
        "submit_recovery_release",
        "submit_errors_reset",
    )

    def __init__(self, inner: object, flags: Path) -> None:
        self._inner = inner
        self._flags = flags

    @property
    def repository(self) -> str:
        return self._inner.repository

    def _wait_for_engine_lock(self) -> None:
        lock_path = self._flags / "engine.lock"
        while True:
            try:
                handle = open(lock_path, "a+")
            except OSError:
                return
            try:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    time.sleep(0.05)
                    continue
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
                return
            finally:
                handle.close()

    def _maybe_crash(self) -> None:
        armed_path = self._flags / "armed_crash.json"
        armed = _read_json(armed_path)
        if armed is not None and armed.get("point") == "between-commit-and-reply":
            try:
                armed_path.unlink()
            except OSError:
                pass
            os._exit(1)

    def __getattr__(self, name: str) -> object:
        inner = self.__dict__["_inner"]
        member = getattr(inner, name)
        if name not in self._MUTATIONS:
            return member

        def _gated(*args: object, **kwargs: object) -> object:
            self._wait_for_engine_lock()
            record = member(*args, **kwargs)
            self._maybe_crash()
            return record

        return _gated


def _engine_lock_held(flags: Path) -> bool:
    """Whether ``hold-lock`` currently holds the cross-process engine lock."""

    lock_path = flags / "engine.lock"
    try:
        handle = open(lock_path, "a+")
    except OSError:
        return False
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        return False
    finally:
        handle.close()


def _hold_control_lock_while_held(flags: Path, control_lock: object) -> None:
    """Hold the engine's control lock whenever ``hold-lock`` holds its file lock.

    ``hold-lock`` runs in another process, so it cannot take the engine's
    in-process control lock itself; this watcher takes it on its behalf
    and marks that with :data:`ENGINE_LOCK_HELD`. Meanwhile the engine's
    own prepared-state path applies: a status read waits at most
    ``CONTROL_LOCK_TIMEOUT_SECONDS`` and serves the last built snapshot.
    """

    marker = flags / ENGINE_LOCK_HELD
    while True:
        if _engine_lock_held(flags):
            with control_lock:
                try:
                    marker.write_text(str(os.getpid()), encoding="utf-8")
                except OSError:
                    pass
                try:
                    while _engine_lock_held(flags):
                        time.sleep(0.02)
                finally:
                    marker.unlink(missing_ok=True)
        time.sleep(0.02)


def _sync_issues(issues: dict[int, object], spec: dict) -> None:
    """Create missing per-issue fakes and steer eligibility labels."""

    from datetime import UTC as _UTC
    from datetime import datetime as _datetime

    from simple_coding_agent.github_tracker import TrackerIssue
    from tests.fakes import FakeGitHub

    for raw_number, entry in spec.items():
        try:
            number = int(raw_number)
        except (TypeError, ValueError):
            continue
        eligible = bool(entry.get("eligible", False))
        title = entry.get("title") or f"Issue {number}"
        fake = issues.get(number)
        if fake is None:
            issue = TrackerIssue(
                id=f"issue-{number}",
                number=number,
                title=title,
                body=f"Harness issue {number}.",
                created_at=_datetime(2026, 9, 20, tzinfo=_UTC),
                state="OPEN",
                labels=frozenset({"ready-for-agent"} if eligible else set()),
                assignee_logins=(),
                blocked_by=0,
                author_login="reporter",
            )
            issues[number] = FakeGitHub(issue)
            continue
        current = fake._current()  # noqa: SLF001 - eligibility steering only
        if current.assignee_logins:
            continue  # never disturb an in-flight claim
        want = {"ready-for-agent"} if eligible else set()
        have = set(current.labels) & {"ready-for-agent"}
        if set(want) == have:
            continue
        if eligible:
            fake.labels.add("ready-for-agent")
        else:
            fake.labels.discard("ready-for-agent")


class _Engine:
    """The wired engine pieces the daemon loop steers."""

    def __init__(self, **parts: object) -> None:
        self.__dict__.update(parts)


def _build_engine(root: Path, repository: str, transport: HarnessTransport) -> _Engine:
    """Wire the real lifecycle to the shared fakes, mirroring production ``main``.

    Everything production wires is wired the same way (event logger with
    its per-attempt sink and contract buffers, attempt archives, halt
    store, telemetry); only the GitHub transport, the model executor and
    provenance are fakes.
    """

    from simple_coding_agent.__main__ import emit_process_started, emit_startup_check
    from simple_coding_agent.agent_version import agent_version
    from simple_coding_agent.attempt_state import AttemptStateStore
    from simple_coding_agent.command_runner import CommandRunner
    from simple_coding_agent.completion import CompletionEvaluator, VerificationRunner
    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.contract_events import ContractEventsBuffer
    from simple_coding_agent.control import ControlStore
    from simple_coding_agent.finalization import AttemptCompletionStore
    from simple_coding_agent.git_workspace import GitWorkspace
    from simple_coding_agent.github_tracker import GitHubTracker
    from simple_coding_agent.halt import HaltStore
    from simple_coding_agent.handoff_record import HandoffPublicationStore
    from simple_coding_agent.lifecycle import AgentLifecycle, ModelAttemptRunner
    from simple_coding_agent.observability import AttemptArchive, JsonEventLogger
    from simple_coding_agent.operating import ConsecutiveErrorStore
    from simple_coding_agent.publication import Publisher
    from simple_coding_agent.telemetry import (
        RecentEventsBuffer,
        TelemetryReporter,
        format_timestamp,
    )

    data_dir = root / "instance"
    clone_dir = root / "clone"
    remote = root / "remote.git"
    flags = root / "flags"
    process_started_at = format_timestamp(datetime.now(UTC))

    def default_profile() -> RepositoryProfile:
        return RepositoryProfile(
            setup=(),
            check=("true",),
            base_branch="main",
            timeout=30,
            setup_timeout=30,
            env={},
        )

    halt_store = HaltStore(data_dir)
    attempt_state = AttemptStateStore(data_dir)
    control_store = ControlStore(data_dir)

    def archive_factory(issue_number: int, started_at: str) -> AttemptArchive:
        return AttemptArchive(
            data_dir, issue_number=issue_number, started_at=started_at, redactions=()
        )

    def attempt_sink(issue_number: int) -> AttemptArchive | None:
        checkpoint = attempt_state.read()
        if checkpoint is None or checkpoint.issue_number != issue_number:
            return None
        return archive_factory(issue_number, checkpoint.started_at)

    recent_events = RecentEventsBuffer()
    contract_events = ContractEventsBuffer()
    logger = JsonEventLogger(
        sys.stdout,
        attempt_sink=attempt_sink,
        recent_events=recent_events,
        contract_events=contract_events,
    )

    def event_log(phase: str, default_level: str = "INFO"):
        def emit(event, detail="", level=default_level, issue_number=None, attempt_id=None):
            logger.emit(
                event,
                phase=phase,
                detail=detail,
                level=level,
                issue_number=issue_number,
                attempt_id=attempt_id,
            )

        return emit

    emit_process_started(logger)
    emit_startup_check(logger, "provenance", "started")
    workspace = GitWorkspace(
        clone_dir, str(remote), token_provider=lambda: "token"
    )
    github_tracker = GitHubTracker(transport, repository, event_log=event_log("polling"))
    tracker = GatedTracker(github_tracker, flags)
    publisher = GatedPublisher(
        Publisher(
            workspace,
            transport,
            repository,
            attempt_state=attempt_state,
            sleeper=lambda _: None,
            handoff_comment_mode="verbose",
            handoff_store=HandoffPublicationStore(data_dir),
            event_log=event_log("publication"),
        ),
        flags,
        data_dir,
    )
    lifecycle = AgentLifecycle(
        tracker=tracker,
        attempt_state=attempt_state,
        workspace=workspace,
        profile_loader=lambda _: default_profile(),
        publisher=publisher,
        attempt_runner=None,
        sleeper=lambda _: None,
        poll_interval=60,
        error_store=ConsecutiveErrorStore(data_dir),
        event_log=event_log("polling"),
        attempt_archive_factory=archive_factory,
        completion_store=AttemptCompletionStore(data_dir),
        control_store=control_store,
        contract_events=contract_events,
        repository=repository,
        data_dir=data_dir,
        mode="online",
        process_started_at=process_started_at,
        starting=True,
        startup_check="provenance",
        halt_store=halt_store,
    )
    # Post-provenance construction (the fake provenance always verifies).
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=VerificationRunner(CommandRunner()),
        evaluator=CompletionEvaluator(frozenset({"blocking"})),
        model_executor=BlockingExecutor(flags, data_dir),
        issue_comments=github_tracker.trusted_comments,
        prompt_comments=github_tracker.prompt_comments,
        attempt_archive_factory=archive_factory,
        event_log=event_log("setup", "ERROR"),
    )
    telemetry = TelemetryReporter(
        repository=repository,
        agent_version=agent_version(),
        sdk_version="harness",
        cli_version="harness",
        process_started_at=process_started_at,
        recent_events=recent_events,
    )
    return _Engine(
        lifecycle=lifecycle,
        tracker=tracker,
        control_store=control_store,
        halt_store=halt_store,
        logger=logger,
        event_log=event_log,
        attach=lambda: lifecycle.attach_post_provenance(
            attempt_runner=runner, telemetry=telemetry
        ),
        provenance_ok=lambda: emit_startup_check(logger, "provenance", "ok"),
    )


def _priority_waits_behind_closed_gate(control_store: object, flags: Path) -> bool:
    """Whether a ``next issue`` priority would be claimed with the gate closed.

    The prioritized claim does not go through ``claim_next``, so while the
    gate is closed the daemon keeps the engine away from a Claim boundary
    with intake open and a priority pending.
    """

    from simple_coding_agent.control import IntakeState

    if claim_gate_token(flags) is not None:
        return False
    try:
        return (
            control_store.next_issue_snapshot() is not None
            and control_store.intake_state() is IntakeState.RUNNING
        )
    except Exception:
        return False


def run_daemon(root: Path) -> int:
    """Run the fakes-wired engine until asked to stop; never returns normally."""

    from simple_coding_agent.__main__ import emit_halted
    from simple_coding_agent.agent_version import agent_version
    from simple_coding_agent.control_server import ControlServer
    from simple_coding_agent.instance import record_instance
    from simple_coding_agent.lifecycle import LifecycleStatus
    from simple_coding_agent.volume_lock import VolumeLock, VolumeLockError
    from tests.fakes import repository_with_main

    state_path = root / "state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError) as error:
        print(f"harness daemon: unreadable state: {error}", file=sys.stderr)
        return 1
    repository = state.get("repository", "")
    startup_hold_ms = int(state.get("startup_hold_ms", 500))
    try:
        data_dir = root / "instance"
        lock = VolumeLock(data_dir)
        lock.acquire()
    except VolumeLockError as error:
        print(f"harness daemon: volume lock: {error}", file=sys.stderr)
        return 1
    record_instance(data_dir, repository=repository, agent_version=agent_version())
    if not (root / "remote.git").exists():
        repository_with_main(root)
    flags = root / "flags"
    flags.mkdir(parents=True, exist_ok=True)
    issues: dict[int, object] = {}
    transport = HarnessTransport(issues, root / "issues.json")
    engine = _build_engine(root, repository, transport)
    lifecycle = engine.lifecycle
    tracker = engine.tracker
    control_store = engine.control_store
    halt_store = engine.halt_store
    threading.Thread(
        target=_hold_control_lock_while_held,
        args=(flags, control_store.lock),
        name="harness-engine-lock",
        daemon=True,
    ).start()
    proxy = GateProxy(lifecycle, flags)
    server = ControlServer(root / "daemon.sock", proxy, event_log=engine.event_log("control"))
    try:
        server.start()
    except Exception as error:
        print(f"harness daemon: control endpoint: {error}", file=sys.stderr)
        return 1
    (root / "daemon.serving").write_text(str(os.getpid()), encoding="utf-8")
    # Early serving: the control endpoint answers ``starting`` while the
    # (instant, fake) provenance check is reported in flight: for at least
    # ``native.startup_hold_ms``, and after a restart until ``release``.
    deadline = time.monotonic() + max(0, startup_hold_ms) / 1000.0
    while time.monotonic() < deadline or (flags / STARTUP_HOLD).exists():
        if (flags / "halt_request.json").exists():
            break
        time.sleep(0.05)
    engine.provenance_ok()
    engine.attach()
    lifecycle.mark_started()
    try:
        lifecycle.contract_status_snapshot()
    except Exception:
        pass
    (root / "daemon.ready").write_text(str(os.getpid()), encoding="utf-8")
    poll_interval = 0.2
    reconciled = False
    while True:
        halt_request = _take_json(flags / "halt_request.json") if (flags / "halt_request.json").exists() else None
        if halt_request is not None:
            reason = halt_request.get("reason", "halted")
            detail = halt_request.get("detail", "")
            try:
                halt_store.record(reason, detail if isinstance(detail, str) else "")
            except Exception as error:
                print(f"harness daemon: halt record: {error}", file=sys.stderr)
            emit_halted(engine.logger, reason)
            server.stop()
            try:
                lifecycle.shutdown()
            except Exception:
                pass
            try:
                (root / "daemon.ready").unlink(missing_ok=True)
            except OSError:
                pass
            return 0
        armed = _read_json(flags / "armed_crash.json")
        if armed is not None and armed.get("point") == "before-claim":
            try:
                (flags / "armed_crash.json").unlink()
            except OSError:
                pass
            os._exit(1)
        transport.sync()
        hold = _read_json(flags / "hold.json")
        if (
            hold is not None
            and hold.get("point") == "before-claim"
            and _checkpoint_phase(data_dir) is None
        ):
            time.sleep(poll_interval)
            continue
        if _engine_lock_held(flags):
            time.sleep(poll_interval)
            continue
        if _priority_waits_behind_closed_gate(control_store, flags):
            time.sleep(poll_interval)
            continue
        gate = claim_gate_token(flags)
        tracker.allowed = gate
        result = None
        try:
            result = lifecycle.run_once()
        except SystemExit:
            raise
        except Exception as error:
            print(f"harness daemon: run_once: {type(error).__name__}: {error}", file=sys.stderr)
        finally:
            tracker.allowed = None
        if not reconciled:
            reconciled = True
            try:
                (flags / STARTUP_RECONCILED).write_text("1", encoding="utf-8")
            except OSError:
                pass
        if (
            gate is not None
            and claim_gate_token(flags) == gate
            and (result is None or result.status is LifecycleStatus.IDLE)
        ):
            # The engine passed the Claim boundary (or failed before it)
            # without claiming: release stops waiting for a claim.
            _answer_claim_gate(flags, gate, claimed=False)
        time.sleep(poll_interval)


def main(argv: list[str] | None = None) -> int:
    """Run the fakes-wired engine for one harness state directory."""

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m conformance_harness.daemon <state-dir>", file=sys.stderr)
        return 2
    return run_daemon(Path(args[0]))


if __name__ == "__main__":  # pragma: no cover - exercised through run_daemon()
    raise SystemExit(main())
