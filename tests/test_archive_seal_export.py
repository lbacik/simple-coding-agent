"""Archive sealing and ``attempts export`` (issue #344).

A dashboard can tell when an Attempt's evidence is final and fetch it as
a tar stream. Sealing freezes the archive after the finalization commit;
export streams an uncompressed pax tar whose first member is
``manifest.json``. Every test here is an acceptance criterion from #344.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from contextlib import redirect_stderr
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_coding_agent import agentctl, attempt_history
from simple_coding_agent.attempt_history import (
    AttemptHistoryError,
    export_projection,
    find_run_directory,
    read_sealing_record,
    seal_archive,
    show_attempt,
    write_export_tar,
)
from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.completion import (
    AttemptOutcome,
    CompletionDecision,
    PublicationPath,
)
from simple_coding_agent.control import ControlStore
from simple_coding_agent.control_server import ControlServer
from simple_coding_agent.lifecycle import AgentLifecycle
from simple_coding_agent.observability import (
    SEALING_FILENAME,
    AttemptArchive,
    AttemptArchiveSealedError,
)
from simple_coding_agent.publication import PullRequest
from simple_coding_agent.telemetry import TelemetryReporter
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace

REPOSITORY = "example-owner/example-repo"
ATTEMPT_ID = "2026-11-06T08:01:10.000001Z"


def _archive(data_dir: Path, issue_number: int = 41) -> AttemptArchive:
    return AttemptArchive(
        data_dir, issue_number=issue_number, started_at=ATTEMPT_ID
    )


def _write_attempt(
    data_dir: Path, issue_number: int = 41, **fields: object
) -> AttemptArchive:
    archive = _archive(data_dir, issue_number)
    record: dict[str, object] = {
        "started_at": ATTEMPT_ID,
        "issue_number": issue_number,
        "issue_title": "Fix pagination",
    }
    record.update(fields)
    archive.write_attempt(record)
    return archive


def _run_directory(data_dir: Path, issue_number: int = 41) -> Path:
    return _archive(data_dir, issue_number).directory


def _sealed_members(data: bytes) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as archive:
        return [member.name for member in archive]


def _read_tar(data: bytes) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as archive:
        for member in archive:
            extracted = archive.extractfile(member)
            members[member.name] = extracted.read() if extracted else b""
    return members


# ---------------------------------------------------------------------------
# sealing
# ---------------------------------------------------------------------------


def test_outcome_without_finalization_stays_unsealed(tmp_path: Path) -> None:
    """An outcome write alone never seals: finalization seals, not the outcome."""

    _write_attempt(
        tmp_path,
        outcome="complete",
        completed_at="2026-11-06T08:44:52Z",
    )

    assert read_sealing_record(_run_directory(tmp_path)) is None
    shown = show_attempt(tmp_path, ATTEMPT_ID)
    assert shown["attempt"]["outcome"] == {
        "label": "Complete",
        "native": "complete",
        "reason": None,
    }
    assert shown["attempt"]["sealed"] is False
    assert shown["attempt"]["sealed_at"] is None


def test_seal_sets_sealed_at_once_and_freezes_the_manifest(
    tmp_path: Path,
) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    archive.write_text("notes.txt", "evidence")

    first = seal_archive(archive.directory)
    assert first is not None
    assert isinstance(first["sealed_at"], str) and first["sealed_at"]
    frozen = [dict(entry) for entry in first["manifest"]]

    # Later files never join the frozen manifest, and the timestamp is stable.
    archive.directory.joinpath("late.txt").write_text("too late")
    second = seal_archive(archive.directory)
    assert second is not None
    assert second["sealed_at"] == first["sealed_at"]
    assert second["manifest"] == frozen

    shown = show_attempt(tmp_path, ATTEMPT_ID)
    assert shown["attempt"]["sealed"] is True
    assert shown["attempt"]["sealed_at"] == first["sealed_at"]
    assert shown["manifest"] == frozen
    assert [entry["path"] for entry in shown["manifest"]] == sorted(
        entry["path"] for entry in shown["manifest"]
    )


def test_sealed_archive_rejects_every_write(tmp_path: Path) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    assert seal_archive(archive.directory) is not None

    with pytest.raises(AttemptArchiveSealedError):
        archive.write_attempt({"outcome": "complete"})
    with pytest.raises(AttemptArchiveSealedError):
        archive.append_event({"event": "late"})
    with pytest.raises(AttemptArchiveSealedError):
        archive.append_contract_event({"event": "late"})
    with pytest.raises(AttemptArchiveSealedError):
        archive.write_text("late.txt", "nope")
    with pytest.raises(AttemptArchiveSealedError):
        archive.copy_cli_transcripts(
            "session", event_log=lambda *a, **k: None
        )


def test_reseal_writes_nothing_and_announces_nothing(tmp_path: Path) -> None:
    """Recovering an already Sealed attempt leaves the archive untouched."""

    archive = _write_attempt(tmp_path, outcome="complete")
    first = seal_archive(archive.directory)
    assert first is not None
    before = {
        path.relative_to(archive.directory).as_posix(): (
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in sorted(archive.directory.rglob("*"))
        if path.is_file()
    }

    lifecycle, _events = _sealing_lifecycle(tmp_path)
    lifecycle._seal_archive_and_emit(  # noqa: SLF001 - recovery seam
        archive, ATTEMPT_ID, issue_number=41
    )

    after = {
        path.relative_to(archive.directory).as_posix(): (
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in sorted(archive.directory.rglob("*"))
        if path.is_file()
    }
    assert after == before
    assert read_sealing_record(archive.directory) == first
    assert [event for event in _events if event[0] == "sealed"] == []


def test_private_state_never_appears_in_manifest_or_export(
    tmp_path: Path,
) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    archive.write_text("notes.txt", "public")
    run_dir = archive.directory
    run_dir.joinpath(".hidden").write_text("dotfile")
    run_dir.joinpath("scratch.tmp").write_text("temp")
    run_dir.joinpath("cache.sqlite3").write_text("db")
    run_dir.joinpath("api-credentials.json").write_text("secret")
    run_dir.joinpath("transcripts").mkdir(exist_ok=True)
    run_dir.joinpath("transcripts", "s.jsonl").write_text('{"redacted": true}\n')
    target = run_dir / "notes.txt"
    run_dir.joinpath("link.txt").symlink_to(target)
    sealed = seal_archive(run_dir)
    assert sealed is not None

    paths = [entry["path"] for entry in sealed["manifest"]]
    assert SEALING_FILENAME not in paths
    assert not any(part.startswith(".") for path in paths for part in path.split("/"))
    assert not any(path.endswith(".tmp") for path in paths)
    assert not any("sqlite" in path for path in paths)
    assert not any("credential" in path for path in paths)
    assert not any(entry.get("link", False) for entry in sealed["manifest"])

    stream = io.BytesIO()
    write_export_tar(
        stream,
        attempt=show_attempt(tmp_path, ATTEMPT_ID)["attempt"],
        entries=sealed["manifest"],
        read_bytes=lambda path: (run_dir / path).read_bytes(),
    )
    members = _sealed_members(stream.getvalue())
    assert SEALING_FILENAME not in members
    assert "link.txt" not in members


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_streams_pax_tar_with_manifest_first(tmp_path: Path) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    archive.write_text("b.txt", "bee")
    archive.write_text("a.txt", "aye")
    sealed = seal_archive(archive.directory)
    assert sealed is not None
    projection = export_projection(tmp_path, ATTEMPT_ID)

    stream = io.BytesIO()
    write_export_tar(
        stream,
        attempt=projection["attempt"],
        entries=projection["manifest"],
        read_bytes=lambda path: (archive.directory / path).read_bytes(),
    )
    payload = stream.getvalue()

    with tarfile.open(fileobj=io.BytesIO(payload), mode="r|") as tar:
        first = tar.next()
        assert first is not None and first.name == "manifest.json"
        assert first.pax_headers == {}
        assert first.mtime == 0
        extracted = tar.extractfile(first)
        assert extracted is not None
        manifest = json.loads(extracted.read().decode("utf-8"))
    assert manifest["contract"] == "1.0"
    assert manifest["attempt"]["attempt_id"] == ATTEMPT_ID
    assert manifest["attempt"]["sealed"] is True
    assert [entry["path"] for entry in manifest["files"]] == [
        entry["path"] for entry in projection["manifest"]
    ]
    members = _read_tar(payload)
    assert list(members)[0] == "manifest.json"
    assert set(members) == {"manifest.json"} | {
        entry["path"] for entry in projection["manifest"]
    }
    for entry in projection["manifest"]:
        body = members[entry["path"]]
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        assert len(body) == entry["size"]


def test_export_path_filter_limits_members(tmp_path: Path) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    archive.write_text("keep.txt", "keep")
    archive.write_text("drop.txt", "drop")
    assert seal_archive(archive.directory) is not None
    projection = export_projection(tmp_path, ATTEMPT_ID)
    wanted = [entry for entry in projection["manifest"] if entry["path"] == "keep.txt"]
    assert len(wanted) == 1

    stream = io.BytesIO()
    write_export_tar(
        stream,
        attempt=projection["attempt"],
        entries=wanted,
        read_bytes=lambda path: (archive.directory / path).read_bytes(),
    )

    assert set(_read_tar(stream.getvalue())) == {"manifest.json", "keep.txt"}


def test_unsealed_torn_read_surfaces_a_sha256_mismatch(tmp_path: Path) -> None:
    """A file grown past the snapshot keeps the snapshot entry in the
    manifest while the tar streams the live bytes, so the consumer sees a
    sha256 mismatch instead of silently absorbing the growth."""

    archive = _write_attempt(tmp_path)
    archive.write_text("growing.txt", "short")
    projection = export_projection(tmp_path, ATTEMPT_ID)
    (entry,) = [e for e in projection["manifest"] if e["path"] == "growing.txt"]
    snapshot_size, snapshot_sha = entry["size"], entry["sha256"]

    # A side channel (never an archive write) grows the file afterwards.
    (archive.directory / "growing.txt").write_text("short but much longer now")

    second = export_projection(tmp_path, ATTEMPT_ID)
    (kept,) = [e for e in second["manifest"] if e["path"] == "growing.txt"]
    assert (kept["size"], kept["sha256"]) == (snapshot_size, snapshot_sha)

    stream = io.BytesIO()
    write_export_tar(
        stream,
        attempt=second["attempt"],
        entries=second["manifest"],
        read_bytes=lambda path: (archive.directory / path).read_bytes(),
    )
    members = _read_tar(stream.getvalue())
    live = members["growing.txt"]
    assert len(live) != kept["size"]
    assert hashlib.sha256(live).hexdigest() != kept["sha256"]


def test_sealed_export_is_byte_identical_across_reads(tmp_path: Path) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    archive.write_text("notes.txt", "stable")
    assert seal_archive(archive.directory) is not None

    def _export() -> bytes:
        projection = export_projection(tmp_path, ATTEMPT_ID)
        stream = io.BytesIO()
        write_export_tar(
            stream,
            attempt=projection["attempt"],
            entries=projection["manifest"],
            read_bytes=lambda path: (archive.directory / path).read_bytes(),
        )
        return stream.getvalue()

    assert _export() == _export()


def test_unknown_attempt_answers_not_found(tmp_path: Path) -> None:
    with pytest.raises(AttemptHistoryError) as error:
        export_projection(tmp_path, "2026-01-01T00:00:00.000001Z")

    assert error.value.code == "not_found"


# ---------------------------------------------------------------------------
# lifecycle: finalization seals, release seals, sealed event
# ---------------------------------------------------------------------------


def _tracked_issue(number: int):
    from simple_coding_agent.github_tracker import TrackerIssue

    return TrackerIssue(
        id=f"issue-{number}",
        number=number,
        title=f"Issue {number}",
        body=f"Implement {number}.",
        created_at=datetime(2026, 11, 6, 8, 0, tzinfo=UTC),
        state="OPEN",
        labels=frozenset({"ready-for-agent"}),
        assignee_logins=(),
        blocked_by=0,
        author_login="reporter",
    )


def _reporter() -> TelemetryReporter:
    return TelemetryReporter(
        repository=REPOSITORY,
        agent_version="0.0-test",
        sdk_version="0.2.163-test",
        cli_version="2.1.286-test",
        process_started_at="2026-01-01T00:00:00Z",
        hard_ceiling=1000,
        turns_max=50,
        timeout_seconds=600,
        continuation_max=2,
    )


def _sealing_lifecycle(
    tmp_path: Path,
) -> tuple[AgentLifecycle, list[tuple[str, str | None]]]:
    from simple_coding_agent.config import RepositoryProfile

    events: list[tuple[str, str | None]] = []
    lifecycle = AgentLifecycle(
        tracker=FakeTracker({7: _tracked_issue(7)}, fifo=[7]),  # type: ignore[arg-type]
        attempt_state=AttemptStateStore(tmp_path),
        workspace=InMemoryWorkspace(tmp_path / "repo"),
        profile_loader=lambda _: RepositoryProfile(
            setup=("setup.sh",),
            check=("pytest",),
            base_branch="main",
            timeout=300,
            setup_timeout=120,
            env={},
        ),
        publisher=FakePublisher(
            pull_request=PullRequest(number=11, url="https://example.test/11")
        ),
        attempt_runner=lambda claim, profile, prepared: _complete_evidence(),
        control_store=ControlStore(tmp_path),
        telemetry=_reporter(),
        repository=REPOSITORY,
        sleeper=lambda seconds: None,
        event_log=lambda event, detail="", level="INFO", issue_number=None, attempt_id=None: events.append(
            (event, attempt_id)
        ),
        attempt_archive_factory=lambda issue_number, started_at: AttemptArchive(
            tmp_path, issue_number=issue_number, started_at=started_at
        ),
        data_dir=tmp_path,
    )
    return lifecycle, events


def _complete_evidence():
    from simple_coding_agent.lifecycle import AttemptEvidence

    return AttemptEvidence(
        decision=CompletionDecision(
            AttemptOutcome.COMPLETE, True, PublicationPath.COMPLETE, ("ready",)
        ),
        check_command="pytest",
        check_exit_code=0,
        review_cycles=1,
        review_findings="all clear",
        details="Implemented.",
    )


def test_finalization_seals_after_the_pull_request_backfill(
    tmp_path: Path,
) -> None:
    lifecycle, events = _sealing_lifecycle(tmp_path)

    assert lifecycle.run_once().status.value == "attempted"
    attempt_id: str = lifecycle.telemetry_snapshot()["last_finished"]["attempt_id"]
    run_dir = find_run_directory(tmp_path, attempt_id)
    assert run_dir is not None

    sealing = read_sealing_record(run_dir)
    assert sealing is not None
    assert isinstance(sealing["sealed_at"], str) and sealing["sealed_at"]
    record = json.loads((run_dir / "attempt.json").read_text(encoding="utf-8"))
    assert record["outcome"] == "complete"
    assert record["pull_request"] == {
        "number": 11,
        "url": "https://example.test/11",
    }
    manifest_paths = [entry["path"] for entry in sealing["manifest"]]
    assert "attempt.json" in manifest_paths

    sealed_events = [event for event in events if event[0] == "sealed"]
    assert len(sealed_events) == 1
    assert sealed_events[0][1] == attempt_id

    shown = show_attempt(tmp_path, attempt_id)
    assert shown["attempt"]["sealed"] is True
    assert shown["attempt"]["sealed_at"] == sealing["sealed_at"]


def test_release_without_a_terminal_outcome_seals_as_released(
    tmp_path: Path,
) -> None:
    """The recovery-release finalization seals after its last write."""

    lifecycle, events = _sealing_lifecycle(tmp_path)
    started_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat().replace(
        "+00:00", "Z"
    )
    archive = AttemptArchive(tmp_path, issue_number=7, started_at=started_at)
    archive.write_attempt(
        {
            "started_at": started_at,
            "issue_number": 7,
            "issue_title": "Issue 7",
        }
    )
    checkpoint = SimpleNamespace(
        started_at=started_at, issue_number=7, issue_title="Issue 7"
    )

    lifecycle._commit_finalization(  # noqa: SLF001 - release finalization seam
        checkpoint,
        7,
        "main",
        AttemptOutcome.INCOMPLETE,
        archive,
        released=True,
    )

    sealing = read_sealing_record(archive.directory)
    assert sealing is not None
    assert archive.read_attempt().get("lifecycle_state") == "released"
    shown = show_attempt(tmp_path, started_at)
    assert shown["attempt"]["sealed"] is True
    assert shown["attempt"]["lifecycle_state"] == "released"
    sealed_events = [event for event in events if event[0] == "sealed"]
    assert len(sealed_events) == 1
    assert sealed_events[0][1] == started_at


def test_sealed_event_carries_the_attempt_id_after_checkpoint_removal(
    tmp_path: Path,
) -> None:
    lifecycle, events = _sealing_lifecycle(tmp_path)

    assert lifecycle.run_once().status.value == "attempted"
    attempt_id: str = lifecycle.telemetry_snapshot()["last_finished"]["attempt_id"]

    sealed_events = [event for event in events if event[0] == "sealed"]
    assert sealed_events == [("sealed", attempt_id)]


# ---------------------------------------------------------------------------
# contract path: export over the control socket
# ---------------------------------------------------------------------------


def _export_server(tmp_path: Path):
    lifecycle, _ = _sealing_lifecycle(tmp_path)
    socket_path = tmp_path / "export.sock"
    server = ControlServer(socket_path, lifecycle)  # type: ignore[arg-type]
    server.start()
    return server, lifecycle, socket_path


def _run_export_raw(
    argv: list[str], monkeypatch: pytest.MonkeyPatch
) -> tuple[int, bytes, str]:
    """Run the CLI with a binary stdout; return (exit code, raw bytes, stderr)."""

    raw = io.BytesIO()
    wrapper = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")
    previous = sys.stdout
    sys.stdout = wrapper  # type: ignore[assignment]
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            code = agentctl.main(argv)
    finally:
        wrapper.flush()
        sys.stdout = previous
    assert isinstance(code, int)
    return code, raw.getvalue(), err.getvalue()


def test_contract_export_streams_tar_with_manifest_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, lifecycle, socket_path = _export_server(tmp_path)
    try:
        assert lifecycle.run_once().status.value == "attempted"
        attempt_id: str = lifecycle.telemetry_snapshot()["last_finished"][
            "attempt_id"
        ]
        monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
        monkeypatch.setenv("DATA_DIR", str(tmp_path))

        code, payload, err = _run_export_raw(
            ["--contract", "1", "attempts", "export", attempt_id], monkeypatch
        )
    finally:
        server.stop()

    assert err == ""
    assert code == 0
    members = _read_tar(payload)
    assert list(members)[:1] == ["manifest.json"]
    manifest = json.loads(members["manifest.json"].decode("utf-8"))
    assert manifest["attempt"]["attempt_id"] == attempt_id
    assert manifest["attempt"]["sealed"] is True


def test_contract_export_path_filter_and_unknown_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    archive.write_text("keep.txt", "keep")
    archive.write_text("drop.txt", "drop")
    assert seal_archive(archive.directory) is not None
    stub = SimpleNamespace(
        repository=REPOSITORY,
        control_identity=lambda: {
            "backend": "simple-coding-agent",
            "agent_version": "0.0-test",
            "contract": {"major": 1, "minor": 0},
            "repository": REPOSITORY,
            "mode": "online",
            "process_started_at": "2026-11-06T08:00:00Z",
            "capabilities": [],
        },
        control_attempts_export=lambda attempt_id: export_projection(
            tmp_path, attempt_id
        ),
    )
    socket_path = tmp_path / "export-paths.sock"
    server = ControlServer(socket_path, stub)  # type: ignore[arg-type]
    server.start()
    try:
        monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
        monkeypatch.setenv("DATA_DIR", str(tmp_path))

        code, payload, err = _run_export_raw(
            ["--contract", "1", "attempts", "export", ATTEMPT_ID, "--path", "keep.txt"],
            monkeypatch,
        )
        assert err == ""
        assert code == 0
        assert set(_read_tar(payload)) == {"manifest.json", "keep.txt"}

        code, payload, err = _run_export_raw(
            ["--contract", "1", "attempts", "export", ATTEMPT_ID, "--path", "nope.txt"],
            monkeypatch,
        )
    finally:
        server.stop()

    assert err == ""
    assert code == 5
    envelope = json.loads(payload.decode("utf-8"))
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "not_found"
    assert envelope["error"]["retryable"] is False


def test_contract_export_unknown_attempt_has_no_tar_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _, socket_path = _export_server(tmp_path)
    try:
        monkeypatch.setenv("AGENTCTL_SOCKET", str(socket_path))
        monkeypatch.setenv("DATA_DIR", str(tmp_path))

        code, payload, err = _run_export_raw(
            ["--contract", "1", "attempts", "export", "2026-01-01T00:00:00.000001Z"],
            monkeypatch,
        )
    finally:
        server.stop()

    assert err == ""
    assert code == 5
    envelope = json.loads(payload.decode("utf-8"))
    assert envelope["contract"] == "1.0"
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "not_found"
    assert envelope["error"]["retryable"] is False


def test_attempt_history_module_exports_only_public_paths(
    tmp_path: Path,
) -> None:
    archive = _write_attempt(tmp_path, outcome="complete")
    assert attempt_history.SEALING_FILENAME == "sealing.json"
    assert seal_archive(archive.directory) is not None
    projection = export_projection(tmp_path, ATTEMPT_ID)

    assert all(entry["path"] != "sealing.json" for entry in projection["manifest"])
