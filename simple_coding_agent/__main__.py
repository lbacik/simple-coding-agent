"""Runnable entry point for the unattended single-process agent."""

from __future__ import annotations

from simple_coding_agent.attempt_state import AttemptStateStore
from simple_coding_agent.command_runner import CommandRunner
from simple_coding_agent.completion import CompletionEvaluator, VerificationRunner
from simple_coding_agent.config import load_repository_profile, load_runtime_config
from simple_coding_agent.git_workspace import GitWorkspace
from simple_coding_agent.github_tracker import GitHubGraphQLTransport, GitHubTracker
from simple_coding_agent.lifecycle import AgentLifecycle, ModelAttemptRunner
from simple_coding_agent.model_execution import ModelExecutor
from simple_coding_agent.publication import Publisher


def main() -> None:
    """Run sequential attempts forever for the one operator-configured repository."""

    config = load_runtime_config()
    github = GitHubGraphQLTransport(config.github_token)
    workspace = GitWorkspace(
        config.clone_dir,
        f"https://github.com/{config.target_repo}.git",
        token_provider=lambda: config.github_token,
    )
    attempt_state = AttemptStateStore(config.data_dir)
    runner = ModelAttemptRunner(
        attempt_state=attempt_state,
        workspace=workspace,
        verifier=VerificationRunner(
            CommandRunner(redactions=(config.github_token, config.meta_api_key))
        ),
        evaluator=CompletionEvaluator(config.review_blocking_severities),
        model_executor=ModelExecutor(config),
    )
    lifecycle = AgentLifecycle(
        tracker=GitHubTracker(github, config.target_repo),
        attempt_state=attempt_state,
        workspace=workspace,
        profile_loader=load_repository_profile,
        publisher=Publisher(
            workspace,
            github,
            config.target_repo,
            max_retries=config.max_retries,
            attempt_state=attempt_state,
        ),
        attempt_runner=runner,
        poll_interval=config.poll_interval,
    )
    lifecycle.run_forever(stop=lambda: False)


if __name__ == "__main__":
    main()
