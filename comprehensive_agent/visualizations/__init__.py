"""Visualization modules for charts and tables."""

# Export public helpers for convenient import
from .charts import generate_charts, create_sample_charts
from .financial_charts import (
    candlestick_chart,
    correlation_heatmap,
    treemap_chart,
    waterfall_chart,
    dual_axis_chart,
)
from .interactive_charts import (
    add_technical_indicators,
    add_annotations,
    create_chart_grid,
)

__all__ = [
    # generic chart generation
    "generate_charts",
    "create_sample_charts",
    # advanced financial charts
    "candlestick_chart",
    "correlation_heatmap",
    "treemap_chart",
    "waterfall_chart",
    "dual_axis_chart",
    # interactive features
    "add_technical_indicators",
    "add_annotations",
    "create_chart_grid",
]