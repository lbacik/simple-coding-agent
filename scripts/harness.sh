#!/bin/sh
# Conformance harness entry point for the external producer suite.
#
# The suite invokes `<harness> <verb> '<request-json>'` as a single
# executable; this wrapper keeps that shape while the mode comes from the
# environment: set SCA_HARNESS_DOCKER=1 (and SCA_HARNESS_IMAGE, DOCKER) to
# proxy verbs into a harness container, or leave it unset for the
# process-local engine. Usage:
#
#   SCA_HARNESS_DOCKER=1 SCA_HARNESS_IMAGE=simple-coding-agent:harness \
#     python -m agent_operator_contract.producer_suite --harness scripts/harness.sh \
#       --capabilities stop-after,next-issue,operator-handoff,recovery-retry,recovery-release,errors-reset \
#       --require-all
exec python -m conformance_harness "$@"
