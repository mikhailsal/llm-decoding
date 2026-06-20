"""Decoding Sandbox -- study LLM token probabilities and decoding strategies.

A console-first, library-backed tool. The ``core`` subpackage holds pure logic
(config, storage preflight, data model, backend protocol) with no UI deps so a
future web UI can reuse it. The ``cli`` subpackage is the terminal front-end.
"""

__version__ = "0.1.0"
