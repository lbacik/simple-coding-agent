"""Validate CLI envelopes against the pinned contract JSON Schemas.

Every contract ``_run`` helper routes its parsed envelope through
:func:`validate_producer_envelope`, every export-stream helper routes its
tar bytes through :func:`validate_export_stream`, and Contract Event tests
route single events through :func:`validate_contract_event`, so the in-repo
suite fails on any drift between our projections and agent-operator-contract
``v1.0.0`` --- not just on the local assertions.

The checks layer the shared validators from the pinned library
(:mod:`agent_operator_contract.documents`) under strict ``jsonschema``
validation against the same tag's ``schemas/`` payloads, plus the fixed
exit-code table (:func:`shared.exit_code_for`).
"""

from __future__ import annotations

import io
import json as _json
import tarfile

import jsonschema
from agent_operator_contract import documents, shared

_DRAFT = jsonschema.Draft202012Validator

#: Canonical operation -> (result schema name, result validator).
_RESULT_CHECKS = {
    "identity": ("identity", documents.validate_identity),
    "status": ("status", documents.validate_status),
    "command": ("command", documents.validate_command_result),
    "attempts-list": ("attempts-list", documents.validate_attempts_list),
    "attempts-show": ("attempts-show", documents.validate_attempts_show),
    "stop": ("mutation-result", documents.validate_mutation_result),
    "resume": ("mutation-result", documents.validate_mutation_result),
    "next-issue": ("mutation-result", documents.validate_mutation_result),
    "handoff": ("mutation-result", documents.validate_mutation_result),
    "recovery-retry": ("mutation-result", documents.validate_mutation_result),
    "recovery-release": ("mutation-result", documents.validate_mutation_result),
    "errors-reset": ("mutation-result", documents.validate_mutation_result),
}


def _operation_of(argv: list[str]) -> str | None:
    """Best-effort canonical operation for one contract argv, if it parses."""

    rest = argv[2:] if len(argv) >= 2 else []
    if rest[:1] == ["--offline"]:
        rest = rest[1:]
    if not rest:
        return None
    head, tail = rest[0], rest[1:]
    if head == "attempts":
        if tail[:1] == ["list"]:
            return "attempts-list"
        if tail[:1] == ["show"]:
            return "attempts-show"
        if tail[:1] == ["export"]:
            return "attempts-export"
        return None
    if head == "next":
        return "next-issue" if tail[:1] == ["issue"] else None
    if head == "recovery":
        if tail[:1] == ["retry"]:
            return "recovery-retry"
        if tail[:1] == ["release"]:
            return "recovery-release"
        return None
    if head == "errors":
        return "errors-reset" if tail[:1] == ["reset"] else None
    if head == "handoff":
        return "handoff" if tail[:1] == ["now"] else None
    if head in ("identity", "status", "command", "stop", "resume"):
        return head
    return None


def _schema_problems(name: str, document: object) -> list[str]:
    """Collect ``jsonschema`` violations of one pinned schema, if any."""

    validator = _DRAFT(documents.load_schema(name))
    return [
        f"schema {name!r}: {error.message}"
        for error in validator.iter_errors(document)
    ]


def validate_producer_envelope(
    envelope: dict, argv: list[str], exit_code: int
) -> None:
    """Assert one CLI envelope matches the pinned schemas and exit table.

    Raises :class:`AssertionError` describing every violation: the envelope
    shape (shared validator plus the ``envelope`` schema), the fixed exit
    code for error envelopes, and the per-operation result document for
    success envelopes.
    """

    problems = list(documents.validate_envelope(envelope))
    problems.extend(_schema_problems("envelope", envelope))
    if envelope.get("ok") is True:
        if exit_code != 0:
            problems.append(f"success envelope must exit 0, got {exit_code}")
        operation = _operation_of(argv)
        check = _RESULT_CHECKS.get(operation or "")
        if check is not None and isinstance(envelope.get("result"), dict):
            schema_name, validate = check
            result = envelope["result"]
            problems.extend(
                f"result: {problem}" for problem in validate(result)
            )
            problems.extend(_schema_problems(schema_name, result))
    else:
        error = envelope.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        if isinstance(code, str):
            try:
                expected = shared.exit_code_for(code)
            except KeyError:
                problems.append(f"unknown error code {code!r}")
            else:
                if expected != exit_code:
                    problems.append(
                        f"error {code!r} must exit {expected}, got {exit_code}"
                    )
        operation = _operation_of(argv)
        if operation in ("attempts-export",):
            pass  # tar-stream failures carry an envelope; nothing more to check
    assert not problems, "envelope violates the pinned contract schemas:\n" + "\n".join(
        f"  - {problem}" for problem in problems
    )


def validate_export_stream(payload: bytes) -> None:
    """Assert one ``attempts export`` tar stream opens with a valid manifest."""

    names: list[str] = []
    manifest: dict | None = None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r|") as archive:
        for member in archive:
            assert member.name is not None
            names.append(member.name)
            if member.name == "manifest.json":
                handle = archive.extractfile(member)
                assert handle is not None
                loaded = _json.loads(handle.read().decode("utf-8"))
                assert isinstance(loaded, dict)
                manifest = loaded
    assert names[:1] == ["manifest.json"], f"manifest.json must be first: {names[:2]}"
    assert manifest is not None, "export stream is missing manifest.json"
    problems = list(documents.validate_export_manifest(manifest))
    problems.extend(_schema_problems("export-manifest", manifest))
    assert not problems, (
        "export manifest violates the pinned contract schemas:\n"
        + "\n".join(f"  - {problem}" for problem in problems)
    )


def validate_contract_event(event: dict) -> None:
    """Assert one Contract Event matches the pinned ``contract-event`` schema."""

    problems = list(documents.validate_contract_event(event))
    problems.extend(_schema_problems("contract-event", event))
    assert not problems, (
        "contract event violates the pinned contract schemas:\n"
        + "\n".join(f"  - {problem}" for problem in problems)
    )
