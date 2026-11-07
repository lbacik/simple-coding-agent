"""Conformance harness for the operator contract (producer side).

The harness is an engine-supplied executable that runs on the host, outside
the production image. The producer suite invokes it as
``<harness> <verb>`` with a JSON document as the argument and reads one JSON
reply from stdout, in the same style as ``agentctl`` (operator contract
§15.1). It wires the real lifecycle to the shared test fakes (fake GitHub
and fake model execution from ``tests/fakes.py`` — never copies) and
implements the fourteen harness verbs so the suite can bring the instance
into every state it tests.

Local use::

    export SCA_HARNESS_STATE_DIR=/tmp/sca-harness
    python -m conformance_harness setup '{"verb": "setup", "params": {"repository": "octo/example"}}'

The production image carries none of this package and none of the fakes:
the ``harness`` Docker build stage adds both on top of the shared base.
"""
