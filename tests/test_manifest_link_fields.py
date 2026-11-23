"""Contract 1.2 manifest link fields: ``at``/``call_id``/``tool``/``origin``/``conversation``.

Every archived ``tool_call``, ``tool_result`` and ``model_output`` file gets
a manifest entry pairing calls with results by the SDK ``tool_use_id``
(``call_id``), with the tool name, the write time and the thread it ran on.
The metadata is recorded when the evidence file is written, in a dotfile
index beside the evidence files, and read back by both the live and the
sealing manifest builders. Evidence file names, bytes and ``sha256`` stay
unchanged, and the ``native.tool_call`` / ``native.tool_result`` event text
is unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from agent_operator_contract import documents

from claude_agent_sdk import AssistantMessage, TextBlock

from simple_coding_agent.attempt_history import (
    export_projection,
    seal_archive,
    show_attempt,
    write_export_tar,
)
from simple_coding_agent.config import RuntimeConfig
from simple_coding_agent.model_execution import ModelExecutor
from simple_coding_agent.observability import (
    EVIDENCE_INDEX_FILENAME,
    AttemptArchive,
)

ATTEMPT_ID = "2026-11-23T12:00:00Z"
FIXED_NOW = datetime(2026, 11, 23, 12, 0, 0, tzinfo=UTC)


def runtime_config(tmp_path: Path) -> RuntimeConfig:
    return RuntimeConfig(
        github_token="github-secret",
        meta_api_key="meta-secret",
        target_repo="octo/example",
        data_dir=tmp_path,
        clone_dir=tmp_path / "repo",
        poll_interval=60,
        log_level="INFO",
        model_timeout=60,
        publish_timeout=120,
        max_retries=3,
        max_consecutive_errors=3,
        agent_trust_project_settings=False,
        review_blocking_severities=frozenset({"must-fix"}),
    )


class FakeClient:
    def __init__(self, options: object, messages: list[object]) -> None:
        self.options = options
        self._messages = messages

    async def disconnect(self) -> None: ...
    async def connect(self, prompt: str) -> None: ...
    async def interrupt(self) -> None: ...
    async def stop_task(self, task_id: str) -> None: ...
    async def query(self, prompt: str) -> None: ...

    async def receive_response(self):
        for message in self._messages:
            yield message


def _result() -> object:
    return SimpleNamespace(
        is_error=False,
        stop_reason="end_turn",
        model_usage={"muse-spark-1.3-contributor": {"input_tokens": 12}},
    )


def _archive(data_dir: Path) -> AttemptArchive:
    archive = AttemptArchive(data_dir, issue_number=41, started_at=ATTEMPT_ID)
    archive.write_attempt({"started_at": ATTEMPT_ID, "issue_number": 41})
    archive.append_contract_event(
        {
            "at": ATTEMPT_ID,
            "level": "info",
            "event": "claim_opened",
            "attempt_id": ATTEMPT_ID,
            "issue": 41,
            "text": "",
        }
    )
    return archive


def _run_attempt(archive: AttemptArchive, tmp_path: Path) -> list[tuple[str, str]]:
    """Run main-thread + subagent tool calls and model outputs; return native events."""

    events: list[tuple[str, str]] = []
    captured: list[FakeClient] = []

    def client_factory(options: object) -> FakeClient:
        client = FakeClient(
            options,
            [
                AssistantMessage(
                    content=[TextBlock(text="main answer")],
                    model="muse-spark-1.3-contributor",
                ),
                AssistantMessage(
                    content=[TextBlock(text="subagent answer")],
                    model="muse-spark-1.3-contributor",
                    parent_tool_use_id="toolu-launch-9",
                ),
                _result(),
            ],
        )
        captured.append(client)
        return client

    executor = ModelExecutor(
        runtime_config(tmp_path),
        client_factory=client_factory,
        clock=lambda: FIXED_NOW,
        event_log=lambda event, detail="", level="INFO", issue_number=None: events.append(
            (event, detail)
        ),
    )
    asyncio.run(
        executor.execute(
            issue_body="Fix it.", working_directory=tmp_path, archive=archive
        )
    )
    pre_hook = captured[0].options.hooks["PreToolUse"][0].hooks[0]
    post_hook = captured[0].options.hooks["PostToolUse"][0].hooks[0]

    asyncio.run(
        pre_hook(
            {"tool_name": "Bash", "tool_input": {"command": "pytest -q"}},
            "toolu-main-1",
            {},
        )
    )
    asyncio.run(
        post_hook(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "pytest -q"},
                "tool_response": {"stdout": "ok"},
            },
            "toolu-main-1",
            {},
        )
    )
    asyncio.run(
        pre_hook(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "src/a.py"},
                "agent_id": "sub-1",
            },
            "toolu-sub-1",
            {},
        )
    )
    asyncio.run(
        post_hook(
            {
                "tool_name": "Read",
                "tool_input": {"file_path": "src/a.py"},
                "tool_response": {"content": "hello"},
                "agent_id": "sub-1",
            },
            "toolu-sub-1",
            {},
        )
    )
    return events


def _by_path(manifest: list[dict]) -> dict[str, dict]:
    return {entry["path"]: entry for entry in manifest}


def test_live_manifest_pairs_calls_with_results_by_sdk_tool_use_id(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)

    shown = show_attempt(tmp_path, ATTEMPT_ID)
    manifest = _by_path(shown["manifest"])

    calls = [e for e in shown["manifest"] if e["kind"] == "tool_call"]
    results = [e for e in shown["manifest"] if e["kind"] == "tool_result"]
    assert {e["call_id"] for e in calls} == {"toolu-main-1", "toolu-sub-1"}
    assert {e["call_id"] for e in results} == {"toolu-main-1", "toolu-sub-1"}

    main_call = next(e for e in calls if e["call_id"] == "toolu-main-1")
    main_result = next(e for e in results if e["call_id"] == "toolu-main-1")
    assert main_call["tool"] == "Bash"
    assert main_result["tool"] == "Bash"
    assert main_call["origin"] == "main"
    assert main_call["conversation"] == "main"
    assert main_result["origin"] == "main"
    assert main_result["conversation"] == "main"

    sub_call = next(e for e in calls if e["call_id"] == "toolu-sub-1")
    sub_result = next(e for e in results if e["call_id"] == "toolu-sub-1")
    assert sub_call["tool"] == "Read"
    assert sub_result["tool"] == "Read"
    assert sub_call["origin"] == "subagent"
    assert sub_call["conversation"] == "subagent:sub-1"
    assert sub_result["origin"] == "subagent"
    assert sub_result["conversation"] == "subagent:sub-1"

    for entry in calls + results:
        assert entry["at"] == "2026-11-23T12:00:00Z"

    assert manifest  # every evidence file is still listed
    assert (archive.directory / EVIDENCE_INDEX_FILENAME).is_file()


def test_model_output_entries_carry_thread_fields_but_no_call_id_or_tool(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)

    shown = show_attempt(tmp_path, ATTEMPT_ID)
    outputs = [e for e in shown["manifest"] if e["kind"] == "model_output"]
    assert len(outputs) == 2
    assert {e["origin"] for e in outputs} == {"main", "subagent"}
    assert {e["conversation"] for e in outputs} == {"main", "subagent:toolu-launch-9"}
    for entry in outputs:
        assert entry["at"] == "2026-11-23T12:00:00Z"
        assert "call_id" not in entry
        assert "tool" not in entry


def test_sealed_manifest_carries_the_same_fields_as_the_live_manifest(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)
    live = show_attempt(tmp_path, ATTEMPT_ID)["manifest"]

    sealed = seal_archive(archive.directory)
    assert sealed is not None
    assert sealed["manifest"] == live
    assert show_attempt(tmp_path, ATTEMPT_ID)["manifest"] == live


def test_live_and_sealed_manifests_validate_against_contract_1_2(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)

    live = show_attempt(tmp_path, ATTEMPT_ID)
    assert documents.validate_attempts_show(live) == []

    sealed_record = seal_archive(archive.directory)
    assert sealed_record is not None
    sealed = show_attempt(tmp_path, ATTEMPT_ID)
    assert sealed["attempt"]["sealed"] is True
    assert documents.validate_attempts_show(sealed) == []

    projection = export_projection(tmp_path, ATTEMPT_ID)
    manifest_doc = {
        "contract": "1.2",
        "attempt": projection["attempt"],
        "files": projection["manifest"],
    }
    assert documents.validate_export_manifest(manifest_doc) == []

    for entry in live["manifest"]:
        assert documents.validate_manifest_entry(entry) == []


def test_sealed_result_without_an_archived_call_loses_its_call_id(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    archive.write_text("0001_tool_result.json", '{"stdout": "orphan"}')
    archive.record_evidence_metadata(
        "0001_tool_result.json",
        {
            "at": "2026-11-23T12:00:00Z",
            "call_id": "toolu-orphan",
            "tool": "Bash",
            "origin": "main",
            "conversation": "main",
        },
    )

    live = show_attempt(tmp_path, ATTEMPT_ID)["manifest"]
    (orphan_live,) = [e for e in live if e["kind"] == "tool_result"]
    assert orphan_live["call_id"] == "toolu-orphan"

    sealed = seal_archive(archive.directory)
    assert sealed is not None
    (orphan_sealed,) = [e for e in sealed["manifest"] if e["kind"] == "tool_result"]
    assert "call_id" not in orphan_sealed
    assert orphan_sealed["tool"] == "Bash"
    assert orphan_sealed["origin"] == "main"
    assert orphan_sealed["conversation"] == "main"
    assert documents.validate_attempts_show(show_attempt(tmp_path, ATTEMPT_ID)) == []


def test_missing_or_corrupt_index_leaves_entries_without_the_new_fields(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)
    (archive.directory / EVIDENCE_INDEX_FILENAME).unlink()

    shown = show_attempt(tmp_path, ATTEMPT_ID)
    assert shown["manifest"]
    assert all("call_id" not in e and "at" not in e for e in shown["manifest"])
    assert documents.validate_attempts_show(shown) == []

    (archive.directory / EVIDENCE_INDEX_FILENAME).write_text(
        "{not json", encoding="utf-8"
    )
    shown = show_attempt(tmp_path, ATTEMPT_ID)
    assert shown["manifest"]
    assert all("call_id" not in e and "at" not in e for e in shown["manifest"])
    assert documents.validate_attempts_show(shown) == []

    sealed = seal_archive(archive.directory)
    assert sealed is not None
    assert all("call_id" not in e and "at" not in e for e in sealed["manifest"])
    assert documents.validate_attempts_show(show_attempt(tmp_path, ATTEMPT_ID)) == []


def test_evidence_file_names_bytes_and_sha256_are_unchanged(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)

    shown = show_attempt(tmp_path, ATTEMPT_ID)
    kinds = {e["path"]: e["kind"] for e in shown["manifest"]}
    assert any(k == "tool_call" and p.endswith(".json") for p, k in kinds.items())
    for entry in shown["manifest"]:
        if entry["kind"] not in ("tool_call", "tool_result", "model_output"):
            continue
        raw = (archive.directory / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
        assert entry["size"] == len(raw)
    assert not (archive.directory / EVIDENCE_INDEX_FILENAME).name.endswith(".jsonl")
    assert all(not e["path"].startswith(".") for e in shown["manifest"])


def test_export_tar_reports_contract_1_2(tmp_path: Path) -> None:
    import io
    import tarfile

    archive = _archive(tmp_path)
    _run_attempt(archive, tmp_path)
    seal_archive(archive.directory)
    projection = export_projection(tmp_path, ATTEMPT_ID)

    stream = io.BytesIO()
    write_export_tar(
        stream,
        attempt=projection["attempt"],
        entries=projection["manifest"],
        read_bytes=lambda path: (archive.directory / path).read_bytes(),
    )
    with tarfile.open(fileobj=io.BytesIO(stream.getvalue()), mode="r|") as tar:
        first = tar.next()
        assert first is not None and first.name == "manifest.json"
        handle = tar.extractfile(first)
        assert handle is not None
        manifest = json.loads(handle.read().decode("utf-8"))
    assert manifest["contract"] == "1.2"
    assert documents.validate_export_manifest(manifest) == []


def test_native_tool_event_text_is_unchanged(tmp_path: Path) -> None:
    archive = _archive(tmp_path)
    events = _run_attempt(archive, tmp_path)

    call_event = next(detail for name, detail in events if name == "tool_call")
    result_event = next(detail for name, detail in events if name == "tool_result")
    assert call_event.startswith("Bash [main] -> ")
    assert call_event.endswith("_tool_call.json")
    assert result_event.startswith("Bash [main] -> ")
    assert "_tool_result.json" in result_event
    sub_call = next(
        detail for name, detail in events if name == "tool_call" and "subagent" in detail
    )
    assert sub_call.startswith("Read [subagent:sub-1] -> ")
    assert "call_id" not in call_event
