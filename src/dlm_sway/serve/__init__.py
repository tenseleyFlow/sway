"""``sway serve`` daemon: warm-backend HTTP API for iterative workflows.

Loading the HF backend takes 15s cold (model + adapter weights, KV cache
allocation, deterministic-mode setup). For interactive flows — notebook
exploration, the S34 ``sway watch`` loop, the S29 live HTML report —
that 15s startup is the dominant cost on every run.

This package exposes ``sway serve`` as a long-running daemon that loads
the backend once and serves a small HTTP API. First call: ~15s cold.
Every subsequent call against the same model: ~2s warm. Five-to-ten-X
DX win for users who iterate.

The package is gated behind the ``[serve]`` extra (FastAPI + uvicorn)
so users who only run one-shot ``sway run`` invocations don't pull
the daemon dependencies.

Public surface:

- :class:`dlm_sway.serve.client.ServeClient` — Python SDK for
  notebooks; one-liner ``ServeClient(url).run(spec)``.
- :func:`dlm_sway.serve.app.create_app` — FastAPI app factory used by
  the CLI's uvicorn launcher and unit tests' ``TestClient``.
- :class:`dlm_sway.serve.cache.BackendCache` — LRU backend cache the
  app uses to keep multiple loaded models warm; capped via the
  ``--max-loaded-models`` CLI flag.
"""

from __future__ import annotations
