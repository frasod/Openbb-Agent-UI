from typing import List, Dict, Any
from datetime import datetime
import logging

from openbb_ai import cite, citations

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def create_citation_for_widget(
    widget: Any,
    input_args: Dict[str, Any],
    extra_details: Dict[str, Any] | None = None,  # noqa: ANN001
) -> Any:  # noqa: ANN401
    """Return a single citation object for *widget* using *input_args*.

    The helper always injects three standard metadata fields (widget name,
    data source, timestamp) and then merges *extra_details* (if supplied).
    """

    base_details: Dict[str, Any] = {
        "widget_name": getattr(widget, "name", "unknown"),
        "data_source": getattr(widget, "origin", "unknown"),
        "timestamp": datetime.now().isoformat(),
    }

    if extra_details:
        base_details.update(extra_details)

    return cite(widget=widget, input_arguments=input_args, extra_details=base_details)

# -----------------------------------------------------------------------------
# Internal utilities
# -----------------------------------------------------------------------------

def _extract_widget_parameters(widget: Any) -> Dict[str, Any]:
    if not hasattr(widget, 'params') or not widget.params:
        return {}
    
    return {
        param.name: param.current_value 
        for param in widget.params 
        if hasattr(param, 'name') and hasattr(param, 'current_value')
    }

def _generate_citation_details(result_data: Any, widget: Any) -> Dict[str, Any]:  # noqa: ANN401
    details = {"generated_at": datetime.now().isoformat()}
    
    if widget:
        if hasattr(widget, 'description'):
            details["widget_description"] = widget.description
        if hasattr(widget, 'name'):
            details["widget_name"] = widget.name
        if hasattr(widget, 'origin'):
            details["data_source"] = widget.origin
    
    if result_data:
        if hasattr(result_data, 'items'):
            details["data_items_count"] = len(result_data.items)
            if result_data.items and hasattr(result_data.items[0], 'data_format'):
                details["data_format"] = str(type(result_data.items[0].data_format).__name__)
        elif isinstance(result_data, dict) and "items" in result_data:
            details["data_items_count"] = len(result_data["items"])
    
    return details

async def generate_citations(widget_data: List[Any], widgets: List[Any]) -> List[Any]:  # noqa: ANN401
    """Generate citation collection SSEs for *widgets*.

    Returns a *list* containing a single ``CitationCollectionSSE`` (kept as a
    list for backward-compatibility with calling code).
    """

    if not widget_data or not widgets:
        return []

    citation_list = []

    for i, widget in enumerate(widgets):
        if not widget:
            continue

        try:
            input_arguments = _extract_widget_parameters(widget)

            # Widget- and data-specific details
            extra_details = _generate_citation_details(
                widget_data[i] if i < len(widget_data) else None, widget
            )

            citation_obj = create_citation_for_widget(
                widget, input_arguments, extra_details=extra_details
            )

            citation_list.append(citation_obj)
        except Exception as exc:  # pragma: no cover – runtime safety
            logger.error("Citation generation failed for widget %s: %s", i, exc)

    if not citation_list:
        return []

    try:
        return [citations(citation_list)]
    except Exception as exc:  # pragma: no cover
        logger.error("Citations object creation failed: %s", exc)
        return citation_list