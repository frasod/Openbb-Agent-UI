from __future__ import annotations

"""Interactive chart utilities (Week 3).

These helpers allow callers to enrich an existing chart artifact with
technical indicators, annotations, or grid layouts.  The
implementation is intentionally lightweight: *chart* artifacts are
pure ``dict`` objects, so we mutate/extend them in-place and return the
same reference for fluent chaining.
"""

from typing import Any, List

from openbb_ai import chart

__all__ = [
    "add_technical_indicators",
    "add_annotations",
    "create_chart_grid",
]


def _ensure_layers(chart_artifact: dict) -> dict:
    """Guarantee that ``chart_artifact['layers']`` exists for overlays."""
    chart_artifact.setdefault("layers", [])
    return chart_artifact


def add_technical_indicators(chart_artifact: dict, indicators: List[dict]) -> dict:  # noqa: D401
    """Overlay indicator traces (e.g., RSI, MACD) on *chart_artifact*.

    Each *indicator* must include::
        {
          "type": "line",  # or "scatter"
          "name": "RSI 14",
          "data": [{"x": ..., "y": ...}, ...],
          "yAxisIndex": 1  # optional – defaults to secondary axis
        }
    """
    artifact = _ensure_layers(chart_artifact)
    for ind in indicators:
        artifact["layers"].append(ind)
    return artifact


def add_annotations(chart_artifact: dict, events: List[dict]) -> dict:  # noqa: D401
    """Add text/shape annotations to *chart_artifact*.

    Example *event*::
        {"x": "2024-01-15", "text": "Earnings", "yPosition": 0.95}
    """
    artifact = _ensure_layers(chart_artifact)
    artifact.setdefault("annotations", []).extend(events)
    return artifact


def create_chart_grid(charts: List[dict], layout: str = "2x2") -> dict:  # noqa: D401
    """Combine multiple *charts* into a grid container artifact."""
    rows, cols = (int(x) for x in layout.split("x")) if "x" in layout else (1, len(charts))

    grid_artifact = chart(
        "grid",
        charts=charts,
        layout={"rows": rows, "cols": cols},
        name="Chart Grid",
        description="Composite of multiple charts in a dashboard layout.",
    )
    return grid_artifact 