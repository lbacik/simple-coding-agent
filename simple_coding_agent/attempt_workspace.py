"""The complete attempt-workspace seam between orchestration and git.

The lifecycle and the model attempt runner treat the workspace as an
:class:`AttemptWorkspace`: every operation they use is a required member,
so there is no ``getattr``/``hasattr`` probing across this seam. Two
adapters satisfy it: the production :class:`GitWorkspace
<simple_coding_agent.git_workspace.GitWorkspace>` and the in-memory
double in ``tests/fakes.py`` (exercised together by the conformance
suite).

Sealing lives in exactly one operation, :meth:`AttemptWorkspace.seal`:
after every exit from model execution the runner seals the attempt's work
unconditionally, whatever the model execution status. Adding a new
``ModelExecutionStatus`` therefore cannot silently drop work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from simple_coding_agent.git_workspace import (
    AttemptCommit,
    ConflictState,
    PreparedAttempt,
    handoff_note_relpath,
)
from simple_coding_agent.model_execution import ModelExecutionStatus


__all__ = [
    "AttemptWorkspace",
    "BranchPusher",
    "ConflictState",
    "handoff_note_relpath",
]


class AttemptWorkspace(Protocol):
    """Everything orchestration needs from the workspace of one attempt."""

    working_directory: Path

    def prepare_for_profile_read(self, *, base_branch: str = "main") -> str: ...

    def prepare_attempt(self, *, base_branch: str, issue_number: int) -> PreparedAttempt: ...

    def is_clean(self) -> bool: ...

    def dirty_paths(self) -> tuple[str, ...]: ...

    def commits_added(self, prepared: PreparedAttempt) -> tuple[AttemptCommit, ...]: ...

    def conflict_state(self) -> ConflictState: ...

    def abort_unresolved_rebase(self, prepared: PreparedAttempt) -> None: ...

    def seal(
        self,
        *,
        issue_number: int,
        attempt_id: str,
        status: ModelExecutionStatus,
    ) -> AttemptCommit | None:
        """Commit every dirty and untracked change on the attempt branch.

        Called unconditionally after every exit from model execution.
        Returns ``None`` when the tree is already clean. Refuses (raises)
        while a rebase/merge is unresolved, so conflict markers are never
        committed.
        """
        ...

    def commit_handoff_note(self, issue_number: int, content: str) -> AttemptCommit:
        """Write the handoff note and commit only the note file."""
        ...

    def handoff_note_commit(
        self, prepared: PreparedAttempt, issue_number: int, *, attempt_id: str
    ) -> AttemptCommit | None:
        """The newest commit of this attempt touching the issue's handoff note path.

        Position-independent: a seal commit landing on top of a
        model-committed note does not hide the note. Notes committed by
        earlier rounds of a continued branch never count as this attempt's
        note.
        """
        ...

    def handoff_note_commits(
        self, prepared: PreparedAttempt, issue_number: int, *, attempt_id: str
    ) -> tuple[AttemptCommit, ...]:
        """Every commit of this attempt touching the issue's handoff note path.

        Newest first, under the same rules as :meth:`handoff_note_commit`:
        position-independent, and limited to this attempt (earlier rounds'
        note commits are excluded). The newest entry is what
        :meth:`handoff_note_commit` returns. Used by handoff-note validation
        together with :meth:`handoff_note_commit_carries_work` to skip the
        unbroken run of this attempt's note commits beneath the newest one
        when deriving the expected ``last_work_commit``.
        """
        ...

    def handoff_note_commit_carries_work(self, revision: str, issue_number: int) -> bool:
        """Whether a note-touching commit is really a work commit.

        True only when ``revision`` deletes the issue's handoff note while
        also changing other files -- a work commit that dropped a stale
        note as a "process artifact" alongside a real change. Such a
        commit ends the note run, so validation expects it as
        ``last_work_commit``. A commit that leaves the note in place
        (even with other files swept in) or that only touches the note
        (including deleting just the note) is still part of the run.
        """
        ...

    def cleanup(
        self, *, base_branch: str, prepared: PreparedAttempt, retain_branch: bool
    ) -> None:
        """Best-effort abort of an unresolved rebase and switch to base_branch.

        Never delete branches. Never switch to the base branch while the tree
        is dirty: the workspace stays on the attempt branch so the caller can
        hold intake for operator inspection. The one deliberate file deletion
        is scoped to reproducible artifacts: files ignored on the attempt
        branch but exposed as untracked on the base branch are removed after
        the checkout, so intake does not hold on them; anything still ignored
        on the base branch is kept, and any unexplained path aborts the whole
        sweep with everything preserved.
        """
        ...


class BranchPusher(Protocol):
    """The narrow branch-push operation the publisher depends on.

    Deliberately separate from :class:`AttemptWorkspace`: publication owns
    the push, orchestration owns the worktree. ``GitWorkspace`` satisfies
    both.
    """

    def push_attempt_branch(
        self, branch: str, *, max_retries: int, deadline: float | None = None
    ) -> str: ...
