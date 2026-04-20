"""Optional matplotlib-based visualizations.

Behind the ``viz`` extra. Three functions cover the three plots that
make the sway report come alive in a notebook or saved PNG:

- :func:`plot_section_sis`: per-section bar chart of effective SIS
  (the flagship attribution view).
- :func:`plot_adapter_ablation`: the λ-scaled divergence curve — the
  sway signature plot.
- :func:`plot_kl_histogram`: distribution of per-prompt KL divergences
  (the raw data behind A1 DeltaKL).

Each function raises :class:`~dlm_sway.core.errors.BackendNotAvailableError`
with a pip hint when matplotlib isn't installed. No function writes to
disk on your behalf — the caller decides (``fig.savefig(...)``).
"""

from __future__ import annotations

from typing import Any

from dlm_sway.core.errors import BackendNotAvailableError
from dlm_sway.core.result import SuiteResult


def _require_mpl() -> Any:
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise BackendNotAvailableError(
            "visualize",
            extra="viz",
            hint="sway's visualization module needs matplotlib.",
        ) from exc


def plot_section_sis(suite: SuiteResult) -> Any:
    """Render a per-section ``effective_sis`` bar chart.

    Returns the matplotlib ``Figure``; the caller handles display / save.
    """
    plt = _require_mpl()

    probe = _find_probe(suite, "section_internalization")
    if probe is None or not probe.evidence.get("per_section"):
        raise ValueError("suite has no section_internalization evidence to plot")

    rows: list[dict[str, Any]] = list(probe.evidence["per_section"])
    labels = [f"{row['tag'] or row['section_id'][:8]}\n({row['kind']})" for row in rows]
    values = [float(row["effective_sis"]) for row in rows]
    colors = ["#2ca02c" if row["passed"] else "#d62728" for row in rows]

    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(rows)), 4.0))
    ax.bar(range(len(rows)), values, color=colors)
    ax.axhline(
        float(probe.evidence.get("per_section_threshold", 0.0)),
        color="gray",
        linestyle="--",
        linewidth=1,
        label="threshold",
    )
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("effective SIS")
    ax.set_title("Section Internalization Score")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_adapter_ablation(suite: SuiteResult) -> Any:
    """Render the signature λ-scaled divergence curve."""
    plt = _require_mpl()

    probe = _find_probe(suite, "adapter_ablation")
    if probe is None or not probe.evidence.get("lambdas"):
        raise ValueError("suite has no adapter_ablation evidence to plot")

    lambdas = list(probe.evidence["lambdas"])
    divs = list(probe.evidence["mean_divergence_per_lambda"])

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(lambdas, divs, marker="o", linewidth=2, color="#1f77b4")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, label="λ=1 (trained)")
    sat = probe.evidence.get("saturation_lambda")
    if sat is not None:
        ax.axvline(
            float(sat),
            color="#2ca02c",
            linestyle="--",
            linewidth=1,
            label=f"sat λ={float(sat):.2f}",
        )
    ax.set_xlabel("λ (adapter scale)")
    ax.set_ylabel("mean JS divergence vs λ=0")
    ax.set_title(
        f"Adapter Ablation (R²={float(probe.evidence.get('linearity', 0.0)):.2f}, "
        f"overshoot={float(probe.evidence.get('overshoot', 0.0)):.2f})"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_kl_histogram(suite: SuiteResult) -> Any:
    """Render the per-prompt KL distribution from a DeltaKL probe."""
    plt = _require_mpl()

    probe = _find_probe(suite, "delta_kl")
    if probe is None or not probe.evidence.get("per_prompt"):
        raise ValueError("suite has no delta_kl evidence to plot")

    values = list(probe.evidence["per_prompt"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.hist(values, bins=max(5, min(20, len(values) // 2)), color="#ff7f0e", edgecolor="white")
    ax.axvline(
        float(probe.raw or 0.0),
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"mean={float(probe.raw or 0.0):.3f}",
    )
    ax.set_xlabel(probe.evidence.get("divergence_kind", "divergence"))
    ax.set_ylabel("count")
    ax.set_title("DeltaKL — per-prompt distribution")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _find_probe(suite: SuiteResult, kind: str) -> Any:
    for p in suite.probes:
        if p.kind == kind:
            return p
    return None
