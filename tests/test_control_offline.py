"""Offline reads of the control store and WAL checkpointing on shutdown (issue #335).

The offline reader serves a stopped instance's committed Intake and
Operator Commands from a read-only volume: it copies the main database
file and its ``-wal`` file to a writable temp directory and opens only
the copy, so the source never gains a lock, a ``-wal`` or a ``-shm``
file. Every orderly shutdown path checkpoints the WAL after the engine's
last write as a second line of defence.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from simple_coding_agent.control import (
    CommandAcknowledgement,
    ControlStore,
    ControlStoreError,
    IntakeState,
)
from simple_coding_agent.control_offline import (
    OfflineReadError,
    read_offline_control,
)
from simple_coding_agent.lifecycle import AgentLifecycle, AttemptInterruptionHandler
from tests.fakes import FakePublisher, FakeTracker, InMemoryWorkspace


def _db_path(data_dir: Path) -> Path:
    return data_dir / "state" / "control.sqlite3"


def _wal_path(data_dir: Path) -> Path:
    return _db_path(data_dir).with_name("control.sqlite3-wal")


def _committed_store(data_dir: Path) -> ControlStore:
    """A store with two committed commands (intake stopping on ``req-1``)."""

    store = ControlStore(data_dir)
    store.submit_stop("req-1", has_active_attempt=True)
    store.submit_next_issue("req-2", 42)
    return store


def _insert_command_row(
    connection: sqlite3.Connection,
    request_id: str,
    sequence: int,
    kind: str = "resume",
    acknowledgement: str = "completed",
) -> None:
    connection.execute(
        "INSERT INTO commands (request_id, sequence, kind, payload,"
        " acknowledgement, detail, created_at, updated_at)"
        " VALUES (?, ?, ?, '{}', ?, ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
        (request_id, sequence, kind, acknowledgement, f"detail for {request_id}"),
    )


def _tree_snapshot(state_dir: Path) -> dict[str, tuple[int, int]]:
    """Map every source file to its size and mtime; the read must not move any."""

    snapshot: dict[str, tuple[int, int]] = {}
    for path in sorted(state_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            snapshot[str(path.relative_to(state_dir))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _make_read_only(state_dir: Path) -> None:
    for path in sorted(state_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o444)
    os.chmod(state_dir, 0o555)


def _make_writable(state_dir: Path) -> None:
    os.chmod(state_dir, 0o755)
    for path in sorted(state_dir.rglob("*")):
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o644)


def _raw_connection(data_dir: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(_db_path(data_dir)))
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _make_lifecycle(data_dir: Path, store: ControlStore) -> AgentLifecycle:
    from simple_coding_agent.attempt_state import AttemptStateStore

    return AgentLifecycle(
        tracker=FakeTracker(),
        attempt_state=AttemptStateStore(data_dir),
        workspace=InMemoryWorkspace(data_dir / "repo"),
        profile_loader=lambda _: type("Profile", (), {"base_branch": "main"})(),
        publisher=FakePublisher(),
        attempt_runner=lambda received_claim, profile, prepared: pytest.fail(
            "no attempt must run"
        ),
        control_store=store,
        sleeper=lambda seconds: None,
    )


# -- offline reads ----------------------------------------------------------


def test_offline_reader_sees_every_committed_row_with_uncheckpointed_wal(
    tmp_path: Path,
) -> None:
    """Rows committed only to the WAL read back from a read-only directory."""

    store = _committed_store(tmp_path)
    # Simulate an engine killed without a clean shutdown: a second
    # connection commits a row and never closes, so the WAL keeps frames
    # no orderly checkpoint ever merged.
    killer = _raw_connection(tmp_path)
    try:
        _insert_command_row(killer, "req-killed", 3)
        killer.commit()
        wal = _wal_path(tmp_path)
        assert wal.is_file() and wal.stat().st_size > 0
        _make_read_only(tmp_path / "state")
        try:
            snapshot = read_offline_control(tmp_path)
        finally:
            _make_writable(tmp_path / "state")
    finally:
        killer.close()

    assert snapshot.intake is IntakeState.STOPPING
    assert snapshot.pending_command_id == "req-1"
    assert [record.request_id for record in snapshot.commands] == [
        "req-1",
        "req-2",
        "req-killed",
    ]
    assert snapshot.commands[2].acknowledgement is CommandAcknowledgement.COMPLETED
    # The live store agrees: nothing was reordered or lost in the copy.
    assert [record.request_id for record in store.recent_commands(limit=10)] == [
        "req-1",
        "req-2",
        "req-killed",
    ]


def test_offline_read_leaves_the_source_directory_unchanged(tmp_path: Path) -> None:
    """No lock is taken and no -wal or -shm file is created or modified."""

    _committed_store(tmp_path)
    killer = _raw_connection(tmp_path)
    try:
        _insert_command_row(killer, "req-killed", 3)
        killer.commit()
        state_dir = tmp_path / "state"
        before = _tree_snapshot(state_dir)
        assert "control.sqlite3-wal" in before
        _make_read_only(state_dir)
        try:
            read_offline_control(tmp_path)
        finally:
            _make_writable(state_dir)
        after = _tree_snapshot(state_dir)
    finally:
        killer.close()

    # `after == before` already proves no file was created (no lock, no
    # fresh -shm) and none was modified; the -shm present here belongs to
    # the killed-engine fixture's still-open connection, not to the reader.
    assert after == before


def test_offline_reader_leaves_no_shm_behind_when_none_exists(
    tmp_path: Path,
) -> None:
    """A cleanly closed store reads back with no -shm created next to it."""

    import gc as _gc

    _committed_store(tmp_path)
    # ControlStore connections close via cyclic collection, so collect
    # first: a quiet store must leave no -wal and no -shm behind.
    _gc.collect()
    state_dir = tmp_path / "state"
    assert not (state_dir / "control.sqlite3-shm").exists()
    assert not (state_dir / "control.sqlite3-wal").exists()
    before = _tree_snapshot(state_dir)
    read_offline_control(tmp_path)
    assert _tree_snapshot(state_dir) == before
    assert not (state_dir / "control.sqlite3-shm").exists()
    assert not (state_dir / "control.sqlite3-wal").exists()


def test_offline_reader_works_when_no_wal_file_exists(tmp_path: Path) -> None:
    import gc as _gc

    _committed_store(tmp_path).checkpoint()
    # The checkpoint's own connection closes via cyclic collection, which
    # also drops the emptied -wal and the -shm: no WAL file at all remains.
    _gc.collect()
    assert not _wal_path(tmp_path).exists()

    snapshot = read_offline_control(tmp_path)

    assert snapshot.intake is IntakeState.STOPPING
    assert [record.request_id for record in snapshot.commands] == ["req-1", "req-2"]


def test_offline_reader_after_killed_engine_matches_live_state(tmp_path: Path) -> None:
    """A child that commits then dies without closing leaves a readable volume."""

    _committed_store(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    child = (
        "import os, sqlite3, sys\n"
        "db = os.path.join(sys.argv[1], 'state', 'control.sqlite3')\n"
        "con = sqlite3.connect(db)\n"
        "con.execute('PRAGMA journal_mode=WAL')\n"
        "row = con.execute('SELECT COALESCE(MAX(sequence), 0) FROM commands').fetchone()\n"
        "con.execute(\n"
        "    'INSERT INTO commands (request_id, sequence, kind, payload,'\n"
        "    ' acknowledgement, detail, created_at, updated_at)'\n"
        "    \" VALUES ('req-killed', ?, 'resume', '{}', 'completed',\"\n"
        "    \" 'killed engine row', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')\",\n"
        "    (int(row[0]) + 1,),\n"
        ")\n"
        "con.commit()\n"
        "print('committed', flush=True)\n"
        "os._exit(0)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "committed"
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    # The dead child never closed or checkpointed: WAL (and a stale -shm)
    # remain, exactly like an engine killed without a clean shutdown.
    assert _wal_path(tmp_path).stat().st_size > 0

    snapshot = read_offline_control(tmp_path)

    assert [record.request_id for record in snapshot.commands] == [
        "req-1",
        "req-2",
        "req-killed",
    ]


def test_offline_reader_missing_store_raises_instead_of_returning_empty(
    tmp_path: Path,
) -> None:
    with pytest.raises(OfflineReadError):
        read_offline_control(tmp_path / "does-not-exist")


def test_offline_reader_garbage_file_raises_instead_of_returning_rows(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "control.sqlite3").write_bytes(b"not a sqlite database at all")

    with pytest.raises(OfflineReadError):
        read_offline_control(tmp_path)


def test_offline_reader_non_file_wal_entry_raises_instead_of_stale_rows(
    tmp_path: Path,
) -> None:
    """A -wal entry that cannot be copied as a file must fail, never go stale."""

    import gc as _gc

    _committed_store(tmp_path)
    _gc.collect()
    wal = _wal_path(tmp_path)
    assert not wal.exists()
    os.symlink(tmp_path / "no-such-wal-target", wal)

    with pytest.raises(OfflineReadError):
        read_offline_control(tmp_path)


def test_offline_reader_unreadable_wal_magic_raises_instead_of_stale_rows(
    tmp_path: Path,
) -> None:
    """A WAL whose header is destroyed cannot be trusted: fail, never stale."""

    _committed_store(tmp_path)
    # Keep the writer connection open: closing the last connection would
    # checkpoint, so the committed row stays WAL-only while its header is
    # destroyed underneath it.
    killer = _raw_connection(tmp_path)
    try:
        _insert_command_row(killer, "req-killed", 3)
        killer.commit()
        wal = _wal_path(tmp_path)
        assert wal.stat().st_size > 0
        with open(wal, "r+b") as handle:
            handle.seek(0)
            handle.write(b"\x00\x00\x00\x00")

        with pytest.raises(OfflineReadError):
            read_offline_control(tmp_path)
    finally:
        killer.close()


# -- checkpoints on shutdown --------------------------------------------------


def test_checkpoint_merges_the_wal_into_the_main_database(tmp_path: Path) -> None:
    store = _committed_store(tmp_path)
    killer = _raw_connection(tmp_path)
    try:
        _insert_command_row(killer, "req-killed", 3)
        killer.commit()
        assert _wal_path(tmp_path).stat().st_size > 0

        store.checkpoint()
    finally:
        killer.close()

    wal = _wal_path(tmp_path)
    assert not wal.is_file() or wal.stat().st_size == 0
    assert [record.request_id for record in store.recent_commands(limit=10)] == [
        "req-1",
        "req-2",
        "req-killed",
    ]
    assert store.intake_state() is IntakeState.STOPPING


def test_online_journal_mode_and_durability_settings_are_unchanged(
    tmp_path: Path,
) -> None:
    """Checkpointing changes no online behaviour: WAL mode, FULL sync, same open order."""

    store = _committed_store(tmp_path)
    store.checkpoint()

    connection = sqlite3.connect(str(_db_path(tmp_path)))
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
        # 2 is FULL: every commit is durable before the call returns.
        assert connection.execute("PRAGMA synchronous").fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM commands").fetchone() == (2,)
    finally:
        connection.close()


def test_lifecycle_shutdown_checkpoints_the_wal(tmp_path: Path) -> None:
    store = _committed_store(tmp_path)
    lifecycle = _make_lifecycle(tmp_path, store)
    killer = _raw_connection(tmp_path)
    try:
        _insert_command_row(killer, "req-killed", 3)
        killer.commit()
        assert _wal_path(tmp_path).stat().st_size > 0

        lifecycle.shutdown()
    finally:
        killer.close()

    wal = _wal_path(tmp_path)
    assert not wal.is_file() or wal.stat().st_size == 0


def test_lifecycle_shutdown_failure_is_logged_and_never_raises(
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str, str]] = []

    class _FailingStore:
        def checkpoint(self) -> None:
            raise ControlStoreError("disk is gone")

    lifecycle = _make_lifecycle(tmp_path, _committed_store(tmp_path))
    lifecycle._control_store = _FailingStore()  # type: ignore[assignment]

    def event_log(event: str, detail: str = "", level: str = "INFO", **kwargs: object) -> None:
        events.append((event, detail, level))

    lifecycle._event_log = event_log  # type: ignore[assignment]

    lifecycle.shutdown()  # Must not raise.

    assert len(events) == 1
    event, detail, level = events[0]
    assert event == "control_checkpoint_failed"
    assert level == "WARNING"
    assert "disk is gone" in detail


def test_lifecycle_shutdown_without_a_store_is_a_noop(tmp_path: Path) -> None:
    lifecycle = _make_lifecycle(tmp_path, _committed_store(tmp_path))
    lifecycle._control_store = None

    lifecycle.shutdown()  # Must not raise.


def test_interruption_handler_checkpoints_before_resignalling(tmp_path: Path) -> None:
    """The SIGTERM path merges the WAL even though `finally` never runs."""

    _committed_store(tmp_path)
    killer = _raw_connection(tmp_path)
    calls: list[str] = []
    try:
        _insert_command_row(killer, "req-killed", 3)
        killer.commit()
        assert _wal_path(tmp_path).stat().st_size > 0

        events: list[str] = []
        handler = AttemptInterruptionHandler(
            event_log=lambda event, detail="", **kwargs: events.append(event),
            issue_number_provider=lambda: None,
            resignal=lambda signum: calls.append(f"resignal-{signum}"),
            checkpoint=lambda: calls.append("checkpoint"),
        )
        handler(signal.SIGTERM, None)
    finally:
        killer.close()

    assert calls == ["checkpoint", f"resignal-{signal.SIGTERM}"]
    assert events == ["attempt_interrupted"]


def test_sigterm_shutdown_checkpoints_the_wal(tmp_path: Path) -> None:
    """A real SIGTERM dies by signal after the handler merged the WAL."""

    _committed_store(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    child = (
        "import sqlite3, sys, time\n"
        "from pathlib import Path\n"
        "from simple_coding_agent.control import ControlStore\n"
        "from simple_coding_agent.lifecycle import AttemptInterruptionHandler\n"
        "data_dir = Path(sys.argv[1])\n"
        "store = ControlStore(data_dir)\n"
        "raw = sqlite3.connect(str(data_dir / 'state' / 'control.sqlite3'))\n"
        "raw.execute('PRAGMA journal_mode=WAL')\n"
        "row = raw.execute('SELECT COALESCE(MAX(sequence), 0) FROM commands').fetchone()\n"
        "raw.execute(\n"
        "    'INSERT INTO commands (request_id, sequence, kind, payload,'\n"
        "    ' acknowledgement, detail, created_at, updated_at)'\n"
        "    \" VALUES ('req-sigterm', ?, 'resume', '{}', 'completed',\"\n"
        "    \" 'sigterm row', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')\",\n"
        "    (int(row[0]) + 1,),\n"
        ")\n"
        "raw.commit()\n"
        "handler = AttemptInterruptionHandler(\n"
        "    event_log=lambda *args, **kwargs: None,\n"
        "    issue_number_provider=lambda: None,\n"
        "    checkpoint=store.checkpoint,\n"
        ")\n"
        "handler.install()\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    time.sleep(0.1)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", child, str(tmp_path)],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "ready"
        # The extra row is WAL-only while the child sleeps.
        assert _wal_path(tmp_path).stat().st_size > 0
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()

    wal = _wal_path(tmp_path)
    assert not wal.is_file() or wal.stat().st_size == 0
    snapshot = read_offline_control(tmp_path)
    assert [record.request_id for record in snapshot.commands] == [
        "req-1",
        "req-2",
        "req-sigterm",
    ]
