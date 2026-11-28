"""The Handoff note deep module (issue #112).

Every rule about the note file at ``.agent/handoff/<issue-number>.md`` lives
here: reading it verbatim, validating a committed note before it may publish,
synthesizing the runtime emergency note when the model left none, and
recognising a note recovered from the model's final response.

The workspace owns *where* the file lives (``git_workspace.handoff_note_relpath``);
this module owns *what counts* as a trustworthy note.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from simple_coding_agent.git_workspace import (
    AttemptCommit,
    PreparedAttempt,
    handoff_note_relpath,
)
from simple_coding_agent.attempt_workspace import AttemptWorkspace
from simple_coding_agent.model_execution import (
    ToolCallRecord,
    expected_handoff_last_work_commit,
)
from simple_coding_agent.completion import _select_failed_command

_OPERATOR_HANDOFF_REASON = "operator_request"


_ALLOWED_HANDOFF_REASONS = frozenset(
    {
        "token_soft_threshold",
        "token_hard_ceiling",
        "cost_soft_threshold",
        "cost_hard_limit",
        "turn_limit",
        "time_limit",
        "operator_request",
    }
)


_EMERGENCY_TOOL_TRAIL_LIMIT = 15
_EMERGENCY_FILE_LIST_LIMIT = 50
_EMERGENCY_ASSISTANT_TEXT_LIMIT = 1500
_EMERGENCY_PREVIOUS_NOTE_LIMIT = 3000


@dataclass(frozen=True)
class _PreviousCompleteNote:
    """A prior round's Handoff note commit and what it held.

    ``content`` is ``None`` when the revision could not be read: the
    emergency note then names the revision explicitly instead of pretending
    the older note describes the current branch.
    """

    revision: str
    content: str | None


_PREVIOUS_NOTE_ATTEMPT_ID = "previous-rounds"


_RUNTIME_WRITTEN_MARKER = "written by the runtime"

#: Provenance line appended to a note recovered from the model's final
#: response. It must not contain ``_RUNTIME_WRITTEN_MARKER``: a recovered
#: note carries the model's own assessment and still counts as a complete
#: note for later rounds' previous-note lookup.
_RECOVERED_NOTE_PROVENANCE = (
    "Recovered by the runtime from the model's final response: the model "
    "wrote this note but did not commit it."
)


_FENCED_BLOCK_PATTERN = re.compile(r"```[^\n]*\n?(.*?)```", re.DOTALL)


def _emit_handoff_note_events(
    event_log: Callable[..., None],
    issue_number: int,
    validation: _HandoffNoteValidation,
) -> None:
    """Emit the WARNING events for one handoff-note verdict.

    A bad ``started_at`` never rejects the note: it emits
    ``handoff_note_started_at_invalid`` and the note is otherwise judged as
    before. A mistyped ``last_work_commit`` (a near-copy of the expected
    revision that names no commit) likewise warns via
    ``handoff_note_last_work_commit_mistyped`` without rejecting. Any
    invalid note additionally emits ``handoff_note_rejected``
    with the validation reason, so a silent downgrade to ``incomplete`` is
    always visible in the event stream.
    """

    if validation.started_at_warning:
        event_log(
            "handoff_note_started_at_invalid",
            validation.started_at_warning,
            level="WARNING",
            issue_number=issue_number,
        )
    if validation.last_work_commit_warning:
        event_log(
            "handoff_note_last_work_commit_mistyped",
            validation.last_work_commit_warning,
            level="WARNING",
            issue_number=issue_number,
        )
    if not validation.valid:
        event_log(
            "handoff_note_rejected",
            validation.reason,
            level="WARNING",
            issue_number=issue_number,
        )


def _read_handoff_note(working_directory: Path, issue_number: int) -> str:
    """Read the model-authored handoff note verbatim, for the result comment.

    The comment copy is the durable, human-visible record; the file on the
    branch is what a continuation is pointed at (see _build_starting_prompt).
    Missing is reported rather than raised: the note is written by the model
    in the SDK session this function has no control over.
    """

    note_path = working_directory / handoff_note_relpath(issue_number)
    try:
        return note_path.read_text()
    except OSError:
        return "Handoff was requested, but the handoff note file could not be read."


def _classify_handoff_reason(execution: object) -> str:
    """Classify the emergency handoff note reason from structured signals.

    The latched token hard-ceiling flag wins over every terminal signal; the
    SDK budget signals map to ``cost_hard_limit``, turn signals to
    ``turn_limit``, and timeout signals to ``time_limit``. The explanation
    text is never read. Unknown signals default to ``cost_hard_limit``.
    """

    if bool(getattr(execution, "token_hard_ceiling_reached", False)):
        return "token_hard_ceiling"
    stop_reason = getattr(execution, "stop_reason", None) or ""
    terminal_reason = getattr(execution, "terminal_reason", None) or ""
    stop = stop_reason.lower() if isinstance(stop_reason, str) else ""
    terminal = terminal_reason.lower() if isinstance(terminal_reason, str) else ""
    if (
        terminal in ("budget_exhausted", "error_max_budget_usd")
        or "budget" in terminal
        or stop == "max_budget_usd_exceeded"
        or "budget" in stop
    ):
        return "cost_hard_limit"
    if (
        terminal in ("max_turns", "error_max_turns")
        or "turn" in terminal
        or stop == "max_turns_exceeded"
        or "turn" in stop
    ):
        return "turn_limit"
    if (
        stop == "timeout"
        or "timeout" in terminal
        or "timeout" in stop
        or stop.startswith("aborted_")
        or terminal.startswith("aborted_")
    ):
        return "time_limit"
    return "cost_hard_limit"


def _extract_recovered_handoff_note(
    last_assistant_text: object, issue_number: int, attempt_id: str
) -> str | None:
    """The first fenced block reading as this attempt's handoff note, if any.

    A block qualifies when it parses with the matching title and ``issue``
    field for this issue, ``started_at`` equal to the attempt id, and an
    allowed ``reason``. Anything else (another issue's note, a stale attempt
    id, prose) is ignored so the caller falls back to the emergency note.
    """

    if not isinstance(last_assistant_text, str) or not last_assistant_text:
        return None
    want = str(issue_number)
    for match in _FENCED_BLOCK_PATTERN.finditer(last_assistant_text):
        block = match.group(1).strip()
        if not block:
            continue
        fields = _parse_handoff_note_fields(block)
        if fields.get("title_issue") != want:
            continue
        if fields.get("issue") != want:
            continue
        if fields.get("started_at", "") != attempt_id:
            continue
        if fields.get("reason") not in _ALLOWED_HANDOFF_REASONS:
            continue
        return block
    return None


def _recovered_note_names_expected_work(
    block: str, expected_revision: str, working_directory: Path
) -> bool:
    """Whether ``block`` names the preserved work beneath the note commit.

    Mirrors the ``last_work_commit`` half of ``_validate_handoff_note`` (which
    stays unchanged): an exact prefix of at least 7 characters, or a
    transcription slip within two edits that names no real commit. A stale
    value naming a real (older) commit stays rejected, so the caller falls
    back to the emergency note instead of committing a note that could never
    validate.
    """

    last_work_commit = _parse_handoff_note_fields(block).get("last_work_commit", "")
    if expected_revision.startswith(last_work_commit) and len(last_work_commit) >= 7:
        return True
    return (
        len(last_work_commit) >= 7
        and _is_hex(last_work_commit)
        and not _resolves_to_commit(working_directory, last_work_commit)
        and _last_work_commit_typo_distance(last_work_commit, expected_revision, 2)
        <= 2
    )


def _is_complete_handoff_note(content: str, issue_number: int) -> bool:
    """Whether ``content`` is a complete note of an earlier round for this issue.

    Complete means it parses with a matching title and ``issue`` field and
    was not written by the runtime: an emergency note carries only runtime
    evidence, never a continuation-ready assessment.
    """

    fields = _parse_handoff_note_fields(content)
    return (
        fields.get("title_issue") == str(issue_number)
        and fields.get("issue") == str(issue_number)
        and _RUNTIME_WRITTEN_MARKER not in content
    )


def _previous_complete_note(
    workspace: AttemptWorkspace,
    prepared: PreparedAttempt,
    issue_number: int,
) -> _PreviousCompleteNote | None:
    """The newest complete Handoff note of an earlier round, if any.

    Never raises: an unreadable history degrades to an absent or unreadable
    note instead of failing the emergency path.
    """

    try:
        commits = workspace.handoff_note_commits(
            prepared, issue_number, attempt_id=_PREVIOUS_NOTE_ATTEMPT_ID
        )
    except Exception:
        return None
    unreadable: str | None = None
    for commit in commits:
        try:
            content = workspace.handoff_note_content_at_revision(
                commit.revision, issue_number
            )
        except Exception:
            content = None
        if content is None:
            if unreadable is None:
                unreadable = commit.revision
            continue
        if _is_complete_handoff_note(content, issue_number):
            return _PreviousCompleteNote(revision=commit.revision, content=content)
    if unreadable is not None:
        return _PreviousCompleteNote(revision=unreadable, content=None)
    return None


def _build_emergency_handoff_note(
    *,
    issue_number: int,
    started_at: str,
    reason: str,
    last_work_commit: str,
    explanation: str,
    diff_stat: str | None = None,
    attempt_commits: Sequence[str] = (),
    sealed_files: Sequence[str] | None = None,
    edited_files: Sequence[str] = (),
    last_assistant_text: str | None = None,
    handoff_skill_invoked: bool = False,
    tool_calls: Sequence[ToolCallRecord] = (),
    last_check_call: ToolCallRecord | None = None,
    baseline_result: object | None = None,
    token_budget: Mapping[str, Any] | None = None,
    archive_dir: str | None = None,
    session_id: str | None = None,
    previous_note: _PreviousCompleteNote | None = None,
) -> str:
    """Render the runtime-written handoff note for a limit stop without a model note.

    The title and four header fields are exactly as before so
    ``_validate_handoff_note`` still passes; every section below them
    degrades to ``"not available"`` / ``"none recorded"`` / ``"none"``
    instead of failing when a fact is missing.
    """

    header = (
        f"# Handoff note: issue #{issue_number}\n\n"
        f"- issue: {issue_number}\n"
        f"- started_at: {started_at}\n"
        f"- reason: {reason}\n"
        f"- last_work_commit: {last_work_commit}\n"
    )
    sections = [
        ("Provenance", _emergency_provenance(reason, explanation)),
        ("Previous complete note", _emergency_previous_note(previous_note)),
        (
            "Changes",
            _emergency_changes(
                diff_stat=diff_stat,
                attempt_commits=attempt_commits,
                sealed_files=sealed_files,
                edited_files=edited_files,
            ),
        ),
        (
            "Last model intent",
            _emergency_last_intent(
                last_assistant_text=last_assistant_text,
                handoff_skill_invoked=handoff_skill_invoked,
            ),
        ),
        (
            "Activity trail",
            _emergency_activity(
                edited_files=edited_files,
                tool_calls=tool_calls,
                last_check_call=last_check_call,
            ),
        ),
        ("Check state", _emergency_check_state(baseline_result=baseline_result)),
        ("Budget facts", _emergency_budget(token_budget=token_budget)),
        (
            "Pointers",
            _emergency_pointers(archive_dir=archive_dir, session_id=session_id),
        ),
        ("Remaining work", _emergency_remaining_work()),
    ]
    body = "".join(f"\n## {title}\n\n{content}\n" for title, content in sections)
    return f"{header}{body}"


def _emergency_provenance(reason: str, explanation: str) -> str:
    """Who wrote the note and which limit fired."""

    detail = (explanation or "").strip() or "not available"
    return (
        "This note was written by the runtime because the model did not write "
        "a handoff note before the attempt ended.\n"
        f"The limit that fired was `{reason}`.\n\n"
        "Model's terminal explanation:\n\n"
        f"{_blockquote(detail)}"
    )


def _emergency_previous_note(previous_note: _PreviousCompleteNote | None) -> str:
    """The last complete note of an earlier round, or why none is shown.

    A found note is quoted verbatim (capped) at its immutable revision and
    marked as older context, clearly distinguished from the runtime evidence
    below, which describes the current branch. A missing or unreadable note
    is said explicitly, so the continuation never mistakes the older note
    for the current branch state.
    """

    if previous_note is None:
        return (
            "No previous complete Handoff note was found on this branch: "
            "no earlier round committed a readable, complete note for this issue."
        )
    if previous_note.content is None:
        return (
            "The previous round's Handoff note commit "
            f"`{previous_note.revision}` could not be read from this branch, "
            "so its older context is unavailable; only the runtime evidence "
            "below describes the current branch."
        )
    content = previous_note.content.strip()
    if len(content) > _EMERGENCY_PREVIOUS_NOTE_LIMIT:
        content = content[:_EMERGENCY_PREVIOUS_NOTE_LIMIT] + "...[truncated]"
    return (
        "The last complete Handoff note "
        f"(revision `{previous_note.revision}`) is preserved below as older "
        "context: it describes the branch as an earlier round left it; the "
        "runtime evidence below describes the current branch.\n\n"
        f"{_blockquote(content)}"
    )


def _emergency_changes(
    *,
    diff_stat: str | None,
    attempt_commits: Sequence[str],
    sealed_files: Sequence[str] | None,
    edited_files: Sequence[str],
) -> str:
    """Preserved work: diff stat, attempt commits, sealed and edited files."""

    stat_block = (diff_stat or "").strip() or "not available"
    commits = [line for line in (str(line).strip() for line in attempt_commits) if line]
    commits_block = (
        "\n".join(f"- {line}" for line in commits) if commits else "none"
    )
    if sealed_files is None:
        sealed_block = "none"
    else:
        sealed_block = _format_capped_files(sealed_files, empty="none")
    return (
        "Changes since the verified base "
        "(`git diff --stat <base>..<seal or HEAD>`):\n\n"
        "```diff\n"
        f"{stat_block}\n"
        "```\n\n"
        "Attempt commits (`git log --oneline <base>..HEAD`):\n\n"
        f"{commits_block}\n\n"
        "Files that were uncommitted when sealing happened "
        "(committed by the seal commit):\n\n"
        f"{sealed_block}\n\n"
        "Files the model edited or created (Edit/Write calls):\n\n"
        f"{_format_capped_files(edited_files, empty='none recorded')}"
    )


def _emergency_last_intent(
    *,
    last_assistant_text: str | None,
    handoff_skill_invoked: bool,
) -> str:
    """The model's last words and whether it invoked the handoff skill."""

    detail = (last_assistant_text or "").strip()
    if not detail:
        text_block = "not available"
    elif len(detail) > _EMERGENCY_ASSISTANT_TEXT_LIMIT:
        text_block = _blockquote(
            detail[:_EMERGENCY_ASSISTANT_TEXT_LIMIT] + "...[truncated]"
        )
    else:
        text_block = _blockquote(detail)
    invoked = "invoked" if handoff_skill_invoked else "not invoked"
    return (
        "Last assistant text (truncated to about 1,500 characters):\n\n"
        f"{text_block}\n\n"
        f"Handoff skill on the main thread: {invoked}."
    )


def _emergency_activity(
    *,
    edited_files: Sequence[str],
    tool_calls: Sequence[ToolCallRecord],
    last_check_call: ToolCallRecord | None,
) -> str:
    """Edited files, the recent tool trail, and the last check-command call."""

    trail = list(tool_calls)[-_EMERGENCY_TOOL_TRAIL_LIMIT:]
    if trail:
        calls_block = "\n".join(_format_tool_call(record) for record in trail)
    else:
        calls_block = "none recorded"
    if last_check_call is None:
        check_block = "none recorded"
    else:
        command = " ".join((last_check_call.command or "").split())
        check_block = f"`{command}` ({last_check_call.outcome})"
    return (
        "Files the model edited or created (Edit/Write calls):\n\n"
        f"{_format_capped_files(edited_files, empty='none recorded')}\n\n"
        "Recent tool calls (last 15: tool, short target, ok/error flag):\n\n"
        f"{calls_block}\n\n"
        "Last Bash call running a profile `check` command:\n\n"
        f"{check_block}"
    )


def _emergency_check_state(baseline_result: object | None) -> str:
    """The final check never runs on this path; report the baseline if one ran."""

    if baseline_result is None:
        baseline_block = "not available (no baseline check ran)"
    else:
        succeeded = getattr(baseline_result, "succeeded", None)
        if succeeded is True:
            baseline_block = (
                f"passed (exit {getattr(baseline_result, 'exit_code', None)})"
            )
        elif succeeded is False:
            baseline_block = f"failed: {_command_failure_detail(baseline_result)}"
        else:
            baseline_block = "not available"
    return (
        "final check: not run (the runtime does not run the final check "
        "before writing an emergency note).\n\n"
        f"Baseline check (pre-model): {baseline_block}."
    )


def _emergency_budget(token_budget: Mapping[str, Any] | None) -> str:
    """Token-budget facts, each degrading to ``not available`` when missing."""

    budget = token_budget if isinstance(token_budget, Mapping) else {}
    soft_crossed = budget.get("soft_threshold_crossed")
    soft_at = budget.get("soft_threshold_crossed_at_tokens")
    if soft_crossed and soft_at is not None:
        soft_block = f"crossed at {soft_at} tokens"
    elif soft_crossed:
        soft_block = "crossed"
    else:
        soft_block = "not crossed"
    prompt_cache = budget.get("prompt_cache")
    hit_rate = (
        prompt_cache.get("main_hit_rate")
        if isinstance(prompt_cache, Mapping)
        else None
    )
    lines = [
        f"- budget_tokens: {_budget_value(budget.get('budget_tokens'))}",
        f"- max_budget_tokens: {_budget_value(budget.get('max_budget_tokens'))}",
        f"- soft threshold: {soft_block}",
        f"- peak_main_context_tokens: "
        f"{_budget_value(budget.get('peak_main_context_tokens'))}",
        f"- main prompt-cache hit rate: "
        f"{hit_rate if hit_rate is not None else 'not available'}",
        f"- turns: {_budget_value(budget.get('turns'))}",
        f"- elapsed: {_elapsed_value(budget.get('elapsed_seconds'))}",
    ]
    return "\n".join(lines)


def _emergency_pointers(
    *, archive_dir: str | None, session_id: str | None
) -> str:
    """Where to look next: the attempt log directory and the CLI session id."""

    directory = (archive_dir or "").strip() or "not available"
    session = (session_id or "").strip() or "not available"
    return (
        f"- attempt log directory: {directory}\n"
        f"- CLI transcript session id: {session}"
    )


def _emergency_remaining_work() -> str:
    """What is left: unknown to the runtime; the issue decides; nothing verified."""

    return (
        "The runtime could not determine what work remains: the attempt "
        "stopped at a limit without a model-written assessment of what is "
        "left.\n\n"
        "The issue's acceptance criteria are the reference for what \"done\" "
        "means. None were verified by the runtime on this path.\n\n"
        "Review the sections above and continue implementation from the "
        "preserved work on this branch."
    )


def _blockquote(text: str) -> str:
    """Quote free model text so it can never be misread as a note header field."""

    lines = text.strip().splitlines() or [text]
    return "\n".join(f"> {line}" if line.strip() else ">" for line in lines)


def _format_capped_files(files: Sequence[str], *, empty: str) -> str:
    """Render a file list capped at 50 entries with a ``+N more`` line."""

    paths = [str(path).strip() for path in files]
    paths = [path for path in paths if path]
    if not paths:
        return empty
    shown = paths[:_EMERGENCY_FILE_LIST_LIMIT]
    lines = [f"- `{path}`" for path in shown]
    remaining = len(paths) - len(shown)
    if remaining:
        lines.append(f"- +{remaining} more")
    return "\n".join(lines)


def _format_tool_call(record: ToolCallRecord) -> str:
    """Render one trail entry as tool name, short target, and ok/error flag."""

    target = f" {record.target}" if record.target else ""
    return f"- `{record.tool}`{target} ({record.outcome})"


def _budget_value(value: object) -> str:
    return "not available" if value is None else str(value)


def _elapsed_value(value: object) -> str:
    if value is None:
        return "not available"
    return f"{value}s"


def _main_thread_handoff_invoked(skill_events: object) -> bool:
    """Whether the ``handoff`` skill was invoked in the main thread."""

    events = skill_events if isinstance(skill_events, (list, tuple)) else ()
    return any(
        getattr(event, "name", None) == "handoff"
        and getattr(event, "agent_id", None) is None
        for event in events
    )


def _last_check_call(
    bash_calls: Sequence[ToolCallRecord], check_commands: Sequence[str]
) -> ToolCallRecord | None:
    """The last Bash call whose command contains a profile ``check`` command."""

    checks = tuple(
        command
        for command in check_commands
        if isinstance(command, str) and command
    )
    if not checks:
        return None
    for record in reversed(list(bash_calls)):
        command = record.command if isinstance(record, ToolCallRecord) else ""
        if any(check in command for check in checks):
            return record
    return None


def _command_failure_detail(result: object, *, limit: int = 4000) -> str:
    """Surface the failing command and its output for the live event stream."""

    failed = _select_failed_command(result)
    if failed is None:
        return "timed out before any command ran"
    stderr = (getattr(failed, "stderr", "") or "").strip()
    output = stderr or (getattr(failed, "stdout", "") or "").strip()
    detail = f"`{getattr(failed, 'command', '')}` (exit {getattr(failed, 'exit_code', None)}): {output}"
    return detail[:limit]


def _git_diff_stat(
    working_directory: Path, base_revision: str, endpoint: str
) -> str | None:
    """Best-effort ``git diff --stat`` for the emergency note; ``None`` when unavailable."""

    try:
        completed = subprocess.run(
            ("git", "diff", "--stat", f"{base_revision}..{endpoint}"),
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    output = (completed.stdout or "").strip()
    return output if output else "(no differences)"


_HANDOFF_NOTE_TITLE_PATTERN = re.compile(r"^#\s*Handoff note:\s*issue #(\d+)")
_HANDOFF_NOTE_FIELD_PATTERN = re.compile(r"^-\s*(issue|started_at|reason|last_work_commit):\s*(.+?)\s*$")


@dataclass(frozen=True)
class _HandoffNoteValidation:
    """Whether a handoff note commit is trustworthy enough to publish."""

    valid: bool
    reason: str = ""
    started_at_warning: str | None = None
    last_work_commit_warning: str | None = None


_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _is_hex(value: str) -> bool:
    """Whether ``value`` is a non-empty hexadecimal string."""

    return bool(value) and all(char in _HEX_DIGITS for char in value)


def _levenshtein_within(first: str, second: str, limit: int) -> int:
    """Levenshtein distance between two strings, capped at ``limit + 1``.

    The full matrix is tiny here (both sides are at most 40 characters),
    so this stays simple: a two-row dynamic program with an early exit
    when the current row's minimum already exceeds ``limit``.
    """

    if first == second:
        return 0
    if abs(len(first) - len(second)) > limit:
        return limit + 1
    previous = list(range(len(second) + 1))
    for i, first_char in enumerate(first, start=1):
        current = [i]
        row_min = i
        for j, second_char in enumerate(second, start=1):
            cost = 0 if first_char == second_char else 1
            best = min(
                previous[j] + 1,  # deletion
                current[j - 1] + 1,  # insertion
                previous[j - 1] + cost,  # substitution
            )
            current.append(best)
            if best < row_min:
                row_min = best
        if row_min > limit:
            return limit + 1
        previous = current
    distance = previous[-1]
    return distance if distance <= limit else limit + 1


def _last_work_commit_typo_distance(value: str, expected: str, limit: int) -> int:
    """Smallest edit distance from ``value`` to any long-enough prefix of ``expected``.

    Only prefixes of ``expected`` that are at least 7 characters long count:
    a short typed fragment matching an equally short prefix proves nothing.
    """

    best = limit + 1
    for end in range(7, len(expected) + 1):
        distance = _levenshtein_within(value, expected[:end], limit)
        if distance < best:
            best = distance
            if best == 0:
                break
    return best


def _resolves_to_commit(working_directory: Path, value: str) -> bool:
    """Whether ``value`` names a commit in the attempt's repository.

    A transcription slip names nothing; a stale ``last_work_commit`` names
    the real (older) commit the note was actually written against (#323).
    Anything git cannot report (not a repository, unreadable object store)
    counts as not resolving, so validation degrades to the typo check.
    """

    try:
        completed = subprocess.run(
            ("git", "rev-parse", "--verify", "--quiet", f"{value}^{{commit}}"),
            cwd=working_directory,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return False
    return completed.returncode == 0


def _this_attempt_note_commit_revisions(
    workspace: AttemptWorkspace,
    prepared: PreparedAttempt,
    issue_number: int,
    attempt_id: str,
) -> frozenset[str]:
    """Revisions of this attempt's note-run commits for note validation.

    The run is every note-touching commit of this attempt except work
    commits that only touched the note path by deleting a stale note
    alongside a real change: those end the run, so validation expects
    them as ``last_work_commit`` instead of skipping past them. A
    commit that cannot be inspected fails closed to work (it is not
    skipped), which is exactly the pre-fix strictness: validation then
    expects the commit directly beneath the note.
    """

    try:
        commits = workspace.handoff_note_commits(
            prepared, issue_number, attempt_id=attempt_id
        )
    except Exception:
        return frozenset()
    run: list[str] = []
    for commit in commits:
        try:
            carries_work = workspace.handoff_note_commit_carries_work(
                commit.revision, issue_number
            )
        except Exception:
            continue
        if not carries_work:
            run.append(commit.revision)
    return frozenset(run)


def _validate_handoff_note(
    note_commit: AttemptCommit | None,
    commits: Sequence[AttemptCommit],
    issue_number: int,
    working_directory: Path,
    base_revision: str,
    *,
    note_commit_revisions: Collection[str] | None = None,
) -> _HandoffNoteValidation:
    """Confirm a handoff has a trustworthy, committed note before it may publish.

    The note is located by path (the newest attempt commit touching the
    issue's handoff note file), not by position: a seal commit landing on
    top of a model-committed note does not hide it. Recognizing the note
    commit only proves a commit touched that path; it does not prove the
    note file was actually written, belongs to this issue and attempt, or
    reflects the work actually preserved. All of that is checked here,
    since a continuation trusts the note and a human trusts the
    ``round-finished`` label that only a valid handoff may add.
    """

    if note_commit is None:
        return _HandoffNoteValidation(
            False,
            "Handoff was requested, but no handoff note commit is on the branch.",
        )
    # The seal commit (when the model left dirty work after committing its
    # own note) only adds other files, so reading the note file from the
    # checked-out working tree reads the committed note content.
    note_path = working_directory / handoff_note_relpath(issue_number)
    try:
        text = note_path.read_text()
    except OSError:
        return _HandoffNoteValidation(
            False,
            "Handoff was requested, but the handoff note file could not be read "
            "from the committed branch state.",
        )
    fields = _parse_handoff_note_fields(text)
    # ``started_at`` is the attempt id owned by the runner: it is
    # informational, never a gate. A missing or malformed value produces a
    # warning carried alongside the verdict instead of rejecting the note.
    started_at = fields.get("started_at", "")
    started_at_warning: str | None = None
    if not started_at:
        started_at_warning = (
            f"issue #{issue_number}: the handoff note's `started_at` field is"
            " missing; it should carry the attempt id."
        )
    elif not _is_iso8601(started_at):
        started_at_warning = (
            f"issue #{issue_number}: the handoff note's `started_at` value"
            f" {started_at!r} is not valid ISO-8601; it should carry the attempt id."
        )
    if fields.get("title_issue") != str(issue_number):
        return _HandoffNoteValidation(
            False,
            "Handoff note does not carry the required title for this issue.",
            started_at_warning,
        )
    if fields.get("issue") != str(issue_number):
        return _HandoffNoteValidation(
            False,
            "Handoff note is missing a matching `issue` field.",
            started_at_warning,
        )
    if fields.get("reason") not in _ALLOWED_HANDOFF_REASONS:
        return _HandoffNoteValidation(
            False,
            "Handoff note is missing a valid `reason` field.",
            started_at_warning,
        )
    last_work_commit = fields.get("last_work_commit", "")
    # The note describes the work preserved beneath it: the attempt commit
    # directly under the note in newest-first order, or the base revision
    # when the note is the oldest attempt commit. When the model committed
    # the note more than once, the commits beneath the newest note commit
    # are earlier note commits of this same attempt, not work: skip the
    # unbroken run of this attempt's note-run commits below the note and
    # expect the first commit beneath that run. A note-touching commit
    # that deleted the note while carrying other files is work that
    # dropped a stale note, not a note commit, so it is already excluded
    # from ``note_commit_revisions`` and ends the run. Note commits from
    # earlier rounds of a continued branch are never in
    # ``note_commit_revisions``, so a new round that names the previous
    # round's note commit (HEAD at injection) still validates.
    if note_commit_revisions is None:
        note_revisions = {note_commit.revision}
    else:
        note_revisions = set(note_commit_revisions) | {note_commit.revision}
    note_index = next(
        (index for index, commit in enumerate(commits) if commit.revision == note_commit.revision),
        None,
    )
    if note_index is None:
        expected_last_work_commit = base_revision
    else:
        expected_last_work_commit = expected_handoff_last_work_commit(
            [commit.revision for commit in commits],
            note_revisions,
            base_revision,
            start=note_index + 1,
        )
    if expected_last_work_commit.startswith(last_work_commit) and len(last_work_commit) >= 7:
        return _HandoffNoteValidation(True, started_at_warning=started_at_warning)
    # Not an exact prefix: tell a transcription slip apart from a stale note.
    # A near-copy of the expected revision that names no commit is evidence
    # the model copied the right value and slipped; a value naming a real
    # (older) commit is the stale-HEAD case (#323) and stays rejected, as
    # does anything further than two edits away, non-hex, or missing.
    if (
        len(last_work_commit) >= 7
        and _is_hex(last_work_commit)
        and not _resolves_to_commit(working_directory, last_work_commit)
        and _last_work_commit_typo_distance(last_work_commit, expected_last_work_commit, 2)
        <= 2
    ):
        return _HandoffNoteValidation(
            True,
            started_at_warning=started_at_warning,
            last_work_commit_warning=(
                f"issue #{issue_number}: the handoff note's `last_work_commit` value"
                f" {last_work_commit!r} is not an exact prefix of the expected"
                f" revision {expected_last_work_commit!r}; accepting as a"
                " transcription slip."
            ),
        )
    return _HandoffNoteValidation(
        False,
        "Handoff note's `last_work_commit` does not match the last preserved "
        "work commit.",
        started_at_warning,
    )


def _parse_handoff_note_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        title_match = _HANDOFF_NOTE_TITLE_PATTERN.match(line)
        if title_match and "title_issue" not in fields:
            fields["title_issue"] = title_match.group(1)
            continue
        field_match = _HANDOFF_NOTE_FIELD_PATTERN.match(line)
        if field_match:
            fields[field_match.group(1)] = field_match.group(2)
    return fields


def _is_iso8601(value: str) -> bool:
    candidate = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        return False
    return True

# ---------------------------------------------------------------------------
# Public interface: every rule above is also available without the leading
# underscore so callers need not reach into private names. The private names
# stay as the canonical definitions to keep existing imports working.
# ---------------------------------------------------------------------------

ALLOWED_HANDOFF_REASONS = _ALLOWED_HANDOFF_REASONS
OPERATOR_HANDOFF_REASON = _OPERATOR_HANDOFF_REASON
EMERGENCY_TOOL_TRAIL_LIMIT = _EMERGENCY_TOOL_TRAIL_LIMIT
EMERGENCY_FILE_LIST_LIMIT = _EMERGENCY_FILE_LIST_LIMIT
EMERGENCY_ASSISTANT_TEXT_LIMIT = _EMERGENCY_ASSISTANT_TEXT_LIMIT
EMERGENCY_PREVIOUS_NOTE_LIMIT = _EMERGENCY_PREVIOUS_NOTE_LIMIT
PREVIOUS_NOTE_ATTEMPT_ID = _PREVIOUS_NOTE_ATTEMPT_ID
RUNTIME_WRITTEN_MARKER = _RUNTIME_WRITTEN_MARKER
RECOVERED_NOTE_PROVENANCE = _RECOVERED_NOTE_PROVENANCE
FENCED_BLOCK_PATTERN = _FENCED_BLOCK_PATTERN
HANDOFF_NOTE_TITLE_PATTERN = _HANDOFF_NOTE_TITLE_PATTERN
HANDOFF_NOTE_FIELD_PATTERN = _HANDOFF_NOTE_FIELD_PATTERN

PreviousCompleteNote = _PreviousCompleteNote
HandoffNoteValidation = _HandoffNoteValidation

emit_handoff_note_events = _emit_handoff_note_events
read_handoff_note = _read_handoff_note
classify_handoff_reason = _classify_handoff_reason
extract_recovered_handoff_note = _extract_recovered_handoff_note
recovered_note_names_expected_work = _recovered_note_names_expected_work
is_complete_handoff_note = _is_complete_handoff_note
previous_complete_note = _previous_complete_note
build_emergency_handoff_note = _build_emergency_handoff_note
parse_handoff_note_fields = _parse_handoff_note_fields
validate_handoff_note = _validate_handoff_note
this_attempt_note_commit_revisions = _this_attempt_note_commit_revisions
main_thread_handoff_invoked = _main_thread_handoff_invoked


def is_operator_request_note(text: str) -> bool:
    """Whether ``text`` cites the operator-request reason."""

    return _parse_handoff_note_fields(text).get("reason") == _OPERATOR_HANDOFF_REASON


__all__ = [
    "ALLOWED_HANDOFF_REASONS",
    "OPERATOR_HANDOFF_REASON",
    "PreviousCompleteNote",
    "HandoffNoteValidation",
    "FENCED_BLOCK_PATTERN",
    "HANDOFF_NOTE_TITLE_PATTERN",
    "HANDOFF_NOTE_FIELD_PATTERN",
    "RUNTIME_WRITTEN_MARKER",
    "RECOVERED_NOTE_PROVENANCE",
    "build_emergency_handoff_note",
    "classify_handoff_reason",
    "emit_handoff_note_events",
    "extract_recovered_handoff_note",
    "is_complete_handoff_note",
    "is_operator_request_note",
    "main_thread_handoff_invoked",
    "parse_handoff_note_fields",
    "previous_complete_note",
    "read_handoff_note",
    "recovered_note_names_expected_work",
    "this_attempt_note_commit_revisions",
    "validate_handoff_note",
]

