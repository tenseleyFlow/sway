"""Suite runner.

Iterates the probe list, materializes each into a ``(Probe, Spec)`` via
the registry, executes it with a :class:`~dlm_sway.probes.base.RunContext`,
and assembles a :class:`~dlm_sway.core.result.SuiteResult`.

Runtime contract:

- Probes are executed in declaration order (not sorted, not parallelized).
  The null-adapter baseline has to run before any probe that needs z-scores,
  so authoring order is load-bearing.
- A probe that raises is recorded as
  :attr:`~dlm_sway.core.result.Verdict.ERROR` and the suite continues —
  one broken probe doesn't torch the whole report.
- The backend is the caller's responsibility: the runner does not build
  or close it, so callers can reuse a backend across multiple suites.
"""

from __future__ import annotations

import time
from typing import Any

from dlm_sway import __version__
from dlm_sway.core.errors import ProbeError
from dlm_sway.core.result import ProbeResult, SuiteResult, Verdict, utcnow
from dlm_sway.core.scoring import DifferentialBackend
from dlm_sway.probes.base import RunContext, build_probe
from dlm_sway.probes.null_adapter import NullAdapterSpec, get_null_stats
from dlm_sway.suite.spec import SwaySpec


def run(
    spec: SwaySpec,
    backend: DifferentialBackend,
    *,
    spec_path: str = "<memory>",
    doc_text: str | None = None,
    sections: tuple[Any, ...] | None = None,
) -> SuiteResult:
    """Execute every probe in ``spec`` against ``backend``."""
    started = utcnow()
    ctx = RunContext(
        backend=backend,
        seed=spec.defaults.seed,
        top_k=spec.defaults.top_k,
        sections=sections,
        doc_text=doc_text,
    )

    results: list[ProbeResult] = []
    null_stats: dict[str, dict[str, float]] = {}

    for raw in spec.suite:
        probe, probe_spec = build_probe(raw)
        if not probe_spec.enabled:
            results.append(
                ProbeResult(
                    name=probe_spec.name,
                    kind=probe_spec.kind,
                    verdict=Verdict.SKIP,
                    score=None,
                    message="disabled in spec",
                )
            )
            continue

        t0 = time.perf_counter()
        try:
            result = probe.run(probe_spec, ctx)
        except ProbeError as exc:
            result = ProbeResult(
                name=probe_spec.name,
                kind=probe_spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — probe impls may raise anything
            result = ProbeResult(
                name=probe_spec.name,
                kind=probe_spec.kind,
                verdict=Verdict.ERROR,
                score=None,
                message=f"{type(exc).__name__}: {exc}",
            )
        duration = time.perf_counter() - t0
        # Re-stamp duration (probes don't know their own wall time).
        result = _with_duration(result, duration)
        results.append(result)

        # Null-adapter result seeds ctx.null_stats for subsequent probes.
        if isinstance(probe_spec, NullAdapterSpec) and result.evidence.get("null_stats"):
            null_stats.update(result.evidence["null_stats"])
            # RunContext is frozen; swap in a fresh one so later probes
            # see the populated stats.
            ctx = RunContext(
                backend=ctx.backend,
                seed=ctx.seed,
                top_k=ctx.top_k,
                sections=ctx.sections,
                doc_text=ctx.doc_text,
                null_stats=null_stats,
            )

    finished = utcnow()
    return SuiteResult(
        spec_path=spec_path,
        started_at=started,
        finished_at=finished,
        base_model_id=spec.models.base.base,
        adapter_id=str(spec.models.ft.adapter) if spec.models.ft.adapter else "",
        sway_version=__version__,
        probes=tuple(results),
        null_stats=null_stats,
    )


def _with_duration(result: ProbeResult, duration: float) -> ProbeResult:
    """Return a copy of ``result`` with :attr:`ProbeResult.duration_s` set."""
    return ProbeResult(
        name=result.name,
        kind=result.kind,
        verdict=result.verdict,
        score=result.score,
        raw=result.raw,
        z_score=result.z_score,
        base_value=result.base_value,
        ft_value=result.ft_value,
        evidence=result.evidence,
        message=result.message,
        duration_s=duration,
    )


__all__ = ["run", "get_null_stats"]
