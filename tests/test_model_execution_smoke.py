"""Opt-in live validation of the pinned Meta and skills execution contract."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from simple_coding_agent.config import load_runtime_config
from simple_coding_agent.model_execution import ModelExecutor, ModelExecutionStatus


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="Set RUN_MODEL_SMOKE=1 to make a paid Meta model request.",
)


def test_dispatches_the_upstream_skills_with_review_subagent_provenance() -> None:
    """Run only in the disposable fixture from the validated live probe (#8)."""

    repository = Path(os.environ["MODEL_SMOKE_REPOSITORY"])
    issue_body = os.environ["MODEL_SMOKE_ISSUE_BODY"]

    execution = asyncio.run(
        ModelExecutor(load_runtime_config()).execute(
            issue_body=issue_body,
            working_directory=repository,
        )
    )

    assert execution.status is ModelExecutionStatus.SUCCEEDED
    assert execution.observed_models == ("muse-spark-1.3-contributor",)
    assert any(event.name == "implement" for event in execution.skill_events)
    assert any(event.name == "code-review" for event in execution.skill_events)
    assert all(event.agent_id is None or isinstance(event.agent_id, str) for event in execution.skill_events)
