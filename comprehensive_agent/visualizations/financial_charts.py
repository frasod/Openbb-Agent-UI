from __future__ import annotations

"""Financial chart helpers (Week 3 – advanced chart capabilities).

Each helper returns an OpenBB `chart` artifact so callers can stream the
artifact back to the Workspace with minimal boiler-plate.  All heavy
Plotly logic stays client-side; here we only prepare a clean, validated
data payload plus the parameters expected by the front-end renderer.
"""

from typing import Any, Dict, List, Sequence

import pandas as pd
from openbb_ai import chart

from ..processors.data_validator import DataValidator  # Re-use existing validation utilities
from ..processors.error_handler import (
    ErrorHandler,
    VisualizationError,
    error_boundary,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_records(df_or_records: Sequence[dict] | pd.DataFrame) -> List[Dict[str, Any]]:
    """Return list-of-dict records regardless of input container."""
    if isinstance(df_or_records, pd.DataFrame):
        return df_or_records.to_dict("records")
    return list(df_or_records)


def _validate_numeric_fields(records: List[Dict[str, Any]], *fields: str) -> bool:
    """Ensure *fields* are present and numeric in *records*.

    This is intentionally strict: we bail out early if validation fails so that
    the caller can decide on fallback behaviour (e.g., render a status update
    instead of a chart).
    """
    if not records:
        return False

    for field in fields:
        for row in records:
            if field not in row or not DataValidator._is_valid_number(row[field]):
                return False
    return True


# ---------------------------------------------------------------------------
# Public chart helpers
# ---------------------------------------------------------------------------

@error_boundary(VisualizationError)
async def candlestick_chart(
    ohlc: Sequence[dict] | pd.DataFrame,
    *,
    time_key: str = "date",
    open_key: str = "open",
    high_key: str = "high",
    low_key: str = "low",
    close_key: str = "close",
    hide_rangeslider: bool = True,
    colors: dict[str, str] | None = None,
    name: str | None = None,
    description: str | None = None,
):
    """Create a candlestick chart artifact.

    Parameters
    ----------
    ohlc:
        Iterable of mappings or a *pandas* ``DataFrame`` with *open*/*high*/*low*/*close* columns plus a time column.
    time_key, open_key, high_key, low_key, close_key:
        Column names to map to the financial OHLC schema.
    hide_rangeslider:
        If *True*, the renderer will hide the default x-axis range-slider.
    colors:
        Optional mapping ``{"increasing": "#26a69a", "decreasing": "#ef5350"}``.
    name, description:
        Override default artifact metadata.
    """
    records = _to_records(ohlc)

    if not _validate_numeric_fields(records, open_key, high_key, low_key, close_key):
        raise VisualizationError("Provided OHLC data failed validation.")

    artifact = chart(
        "candlestick",
        data=records,
        x_key=time_key,
        ohlc_keys={
            "open": open_key,
            "high": high_key,
            "low": low_key,
            "close": close_key,
        },
        params={
            "hideRangeSlider": hide_rangeslider,
            "colors": colors or {},
        },
        name=name or "Candlestick Chart",
        description=description or "OHLC candlestick chart with optional volume overlay.",
    )
    return artifact


@error_boundary(VisualizationError)
async def correlation_heatmap(
    matrix: pd.DataFrame | Sequence[dict],
    *,
    x_key: str | None = None,
    y_key: str | None = None,
    name: str | None = None,
    description: str | None = None,
):
    """Return a heat-map artifact for a correlation matrix."""
    if isinstance(matrix, pd.DataFrame):
        records: List[Dict[str, Any]] = matrix.reset_index().to_dict("records")
        x_key = x_key or matrix.columns[0]
        y_key = y_key or "index"
    else:
        records = list(matrix)
        if not (x_key and y_key):
            raise VisualizationError("x_key and y_key must be provided when passing raw records.")

    artifact = chart(
        "heatmap",
        data=records,
        x_key=x_key,
        y_key=y_key,
        name=name or "Correlation Heat-map",
        description=description or "Asset correlation visualisation.",
    )
    return artifact


@error_boundary(VisualizationError)
async def treemap_chart(
    hierarchical_data: Sequence[dict],
    *,
    label_key: str = "label",
    parent_key: str = "parent",
    value_key: str = "value",
    name: str | None = None,
    description: str | None = None,
):
    """Return a treemap chart artifact."""
    artifact = chart(
        "treemap",
        data=_to_records(hierarchical_data),
        label_key=label_key,
        parent_key=parent_key,
        value_key=value_key,
        name=name or "Treemap Chart",
        description=description or "Portfolio composition / sector breakdown.",
    )
    return artifact


@error_boundary(VisualizationError)
async def waterfall_chart(
    breakdown_data: Sequence[dict],
    *,
    x_key: str = "label",
    y_key: str = "value",
    name: str | None = None,
    description: str | None = None,
):
    """Return a waterfall chart artifact."""
    artifact = chart(
        "waterfall",
        data=_to_records(breakdown_data),
        x_key=x_key,
        y_keys=[y_key],
        name=name or "Waterfall Chart",
        description=description or "Revenue / expense breakdown.",
    )
    return artifact


@error_boundary(VisualizationError)
async def dual_axis_chart(
    primary_data: Sequence[dict],
    secondary_data: Sequence[dict],
    *,
    x_key: str,
    primary_y_key: str,
    secondary_y_key: str,
    name: str | None = None,
    description: str | None = None,
):
    """Return a dual-axis line/column composite chart.

    The front-end renderer is expected to map ``yAxisIndex`` based on the
    supplied ``axis`` parameter embedded in the trace configs.
    """
    payload = [
        {
            **row,
            "series": 0,
            "axis": 0,
        }
        for row in _to_records(primary_data)
    ] + [
        {
            **row,
            "series": 1,
            "axis": 1,
        }
        for row in _to_records(secondary_data)
    ]

    artifact = chart(
        "dual_axis",
        data=payload,
        x_key=x_key,
        y_keys=[primary_y_key, secondary_y_key],
        name=name or "Dual-axis Chart",
        description=description or "Primary vs secondary metric over time.",
    )
    return artifact 