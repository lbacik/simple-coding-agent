"""Fakes-wired engine for the conformance harness (not shipped in production).

Runs the real :class:`AgentLifecycle
<simple_coding_agent.lifecycle.AgentLifecycle>` against the shared test
fakes (``FakeGitHub`` per issue plus ``FakeModelExecutor`` from
``tests/fakes.py`` — reused, never copied) over a real git remote, serving
the contract on a Unix control socket. The host-side harness CLI
(:mod:`conformance_harness.harness`) steers this process through flag
files: holds, one-shot crashes, scripted native outcomes, halt requests
and the deferred-publication one-shot.

Only the ``harness`` Docker build stage contains this module and the
fakes; the production image has neither.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
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


class HarnessTransport:
    """Route every transport call to the per-issue shared ``FakeGitHub``.

    The harness keeps one issue map (``issues.json``); each issue owns a
    ``FakeGitHub`` from the shared test fakes. This facade only routes —
    all behaviour stays in the shared doubles.
    """

    def __init__(self, issues: dict[int, object]) -> None:
        self._issues = issues

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


class BlockingExecutor:
    """Fake model execution that honours harness holds and scripted outcomes.

    Delegates the actual turn to the shared ``FakeModelExecutor`` (with the
    shared ``write_handoff_note`` standing in for the handoff skill's note
    commit); this wrapper only blocks while a hold is armed and selects the
    scripted native outcome ``finish-attempt`` recorded.
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
        while self._hold_applies():
            time.sleep(0.05)
        scripted = _read_json(self._flags / "scripted_outcome.json") or {}
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


def _sync_issues(transport: HarnessTransport, issues: dict[int, object], spec: dict) -> None:
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


def _build_engine(root: Path, repository: str, transport: HarnessTransport) -> object:
    """Wire the real lifecycle to the shared fakes (mirrors ``build_lifecycle``)."""

    from simple_coding_agent.attempt_state import AttemptStateStore
    from simple_coding_agent.command_runner import CommandRunner
    from simple_coding_agent.completion import CompletionEvaluator, VerificationRunner
    from simple_coding_agent.config import RepositoryProfile
    from simple_coding_agent.control import ControlStore
    from simple_coding_agent.finalization import AttemptCompletionStore
    from simple_coding_agent.git_workspace import GitWorkspace
    from simple_coding_agent.github_tracker import GitHubTracker
    from simple_coding_agent.handoff_record import HandoffPublicationStore
    from simple_coding_agent.lifecycle import AgentLifecycle, ModelAttemptRunner
    from simple_coding_agent.observability import AttemptArchive
    from simple_coding_agent.operating import ConsecutiveErrorStore
    from simple_coding_agent.publication import Publisher

    data_dir = root / "instance"
    clone_dir = root / "clone"
    remote = root / "remote.git"
    flags = root / "flags"

    def default_profile() -> RepositoryProfile:
        return RepositoryProfile(
            setup=(),
            check=("true",),
            base_branch="main",
            timeout=30,
            setup_timeout=30,
            env={},
        )

    attempt_state = AttemptStateStore(data_dir)
    workspace = GitWorkspace(
        clone_dir, str(remote), token_provider=lambda: "token"
    )
    tracker = GitHubTracker(transport, repository)
    executor = BlockingExecutor(flags, data_dir)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=VerificationRunner(CommandRunner()),
        evaluator=CompletionEvaluator(frozenset({"blocking"})),
        model_executor=executor,
        issue_comments=tracker.trusted_comments,
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            data_dir, issue_number=issue_number, started_at=started_at, redactions=()
        ),
    )
    publisher = GatedPublisher(
        Publisher(
            workspace,
            transport,
            repository,
            attempt_state=attempt_state,
            sleeper=lambda _: None,
            handoff_comment_mode="verbose",
            handoff_store=HandoffPublicationStore(data_dir),
        ),
        flags,
        data_dir,
    )
    control_store = ControlStore(data_dir)
    return AgentLifecycle(
        tracker=tracker,
        attempt_state=attempt_state,
        workspace=workspace,
        profile_loader=lambda _: default_profile(),
        publisher=publisher,
        attempt_runner=runner,
        sleeper=lambda _: None,
        poll_interval=60,
        error_store=ConsecutiveErrorStore(data_dir),
        completion_store=AttemptCompletionStore(data_dir),
        control_store=control_store,
        repository=repository,
        data_dir=data_dir,
        mode="online",
        process_started_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        starting=True,
        startup_check="provenance",
    )


def run_daemon(root: Path) -> int:
    """Run the fakes-wired engine until asked to stop; never returns normally."""

    from simple_coding_agent.agent_version import agent_version
    from simple_coding_agent.control_server import ControlServer
    from simple_coding_agent.halt import HaltStore
    from simple_coding_agent.instance import record_instance
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
    transport = HarnessTransport(issues)
    lifecycle = _build_engine(root, repository, transport)
    proxy = GateProxy(lifecycle, flags)
    server = ControlServer(root / "daemon.sock", proxy)
    try:
        server.start()
    except Exception as error:
        print(f"harness daemon: control endpoint: {error}", file=sys.stderr)
        return 1
    # Early serving: the control endpoint answers ``starting`` while the
    # (instant, fake) provenance check is reported in flight.
    deadline = time.monotonic() + max(0, startup_hold_ms) / 1000.0
    while time.monotonic() < deadline:
        if (flags / "halt_request.json").exists():
            break
        time.sleep(0.05)
    lifecycle.mark_started()
    try:
        lifecycle.contract_status_snapshot()
    except Exception:
        pass
    (root / "daemon.ready").write_text(str(os.getpid()), encoding="utf-8")
    halt_store = HaltStore(data_dir)
    poll_interval = 0.2
    while True:
        halt_request = _take_json(flags / "halt_request.json") if (flags / "halt_request.json").exists() else None
        if halt_request is not None:
            reason = halt_request.get("reason", "halted")
            detail = halt_request.get("detail", "")
            try:
                halt_store.record(reason, detail if isinstance(detail, str) else "")
            except Exception as error:
                print(f"harness daemon: halt record: {error}", file=sys.stderr)
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
        try:
            spec = json.loads((root / "issues.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError, OSError):
            spec = {}
        if isinstance(spec, dict):
            _sync_issues(transport, issues, spec)
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
        try:
            lifecycle.run_once()
        except SystemExit:
            raise
        except Exception as error:
            print(f"harness daemon: run_once: {type(error).__name__}: {error}", file=sys.stderr)
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
