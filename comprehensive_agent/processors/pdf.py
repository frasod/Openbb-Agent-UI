from typing import List, Optional, Dict, Any, Union
import base64
import httpx
import pdfplumber
import logging
from io import BytesIO
from .error_handler import ErrorHandler, DataProcessingError, error_boundary

logger = logging.getLogger(__name__)

@error_boundary(DataProcessingError)
async def process_pdf_data(widget_data: Union[List[Any], Any]) -> Optional[str]:
    """Process PDF data from OpenBB widgets with robust error handling.
    
    Handles both URL-based and base64-encoded PDF content following
    the DataContent format used by OpenBB Workspace.
    """
    if not widget_data:
        return None
    
    try:
        pdf_content = ""
        
        if isinstance(widget_data, list):
            for result in widget_data:
                content = await _process_single_pdf_source(result)
                if content:
                    pdf_content += content
        else:
            content = await _process_single_pdf_source(widget_data)
            if content:
                pdf_content += content
        
        return pdf_content if pdf_content else None
        
    except Exception as e:
        await ErrorHandler.log_error_with_context(
            DataProcessingError(f"PDF processing failed: {e}"), 
            widget_data
        )
        return None

async def _process_single_pdf_source(data_source: Any) -> str:
    """Process a single PDF data source (DataContent or dictionary)"""
    result = ""
    
    try:
        if hasattr(data_source, 'items'):
            for item in data_source.items:
                if _is_pdf_item(item):
                    content = await _extract_pdf_content_from_item(item)
                    if content:
                        filename = getattr(item.data_format, 'filename', 'PDF Document')
                        result += f"\n\n===== {filename} =====\n{content}\n"
        elif isinstance(data_source, dict):
            for item in data_source.get("items", []):
                if item.get("format") == "PdfDataFormat":
                    content = await extract_pdf_content(item)
                    if content:
                        result += f"\n\n--- PDF Content ---\n{content}\n"
    except Exception as e:
        logger.error(f"Error processing PDF source: {e}")
        
    return result

def _is_pdf_item(item: Any) -> bool:
    """Check if an item contains PDF data"""
    try:
        from openbb_ai.models import PdfDataFormat
        return isinstance(getattr(item, 'data_format', None), PdfDataFormat)
    except ImportError:
        return hasattr(item, 'data_format') and str(type(item.data_format).__name__) == 'PdfDataFormat'

async def _extract_pdf_content_from_item(item) -> Optional[str]:
    """Extract PDF content from OpenBB data item (URL or base64)"""
    try:
        # Handle URL-based PDF (SingleFileReference)
        if hasattr(item, 'url'):
            return await _download_and_extract_pdf(str(item.url))
        
        # Handle base64-encoded PDF (SingleDataContent)
        content = getattr(item, 'content', '')
        if content:
            return _extract_pdf_from_base64(content)
            
    except Exception as e:
        logger.error(f"PDF extraction from item failed: {e}")
        return None
    
    return None

async def _download_and_extract_pdf(url: str) -> Optional[str]:
    """Download PDF from URL and extract text"""
    try:
        logger.info(f"Downloading PDF from {url}")
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
            return _extract_text_from_pdf_bytes(response.content)
            
    except Exception as e:
        logger.error(f"Failed to download PDF from {url}: {e}")
        return None

async def extract_pdf_content(pdf_data: Dict[str, Any]) -> Optional[str]:
    try:
        content = pdf_data.get("content", {})
        if isinstance(content, dict):
            if "url" in content:
                return await extract_pdf_from_url(content["url"])
            elif "data" in content:
                return extract_pdf_from_base64(content["data"])
    except Exception as e:
        print(f"Error extracting PDF content: {e}")
        return None
    
    return None

async def extract_pdf_from_url(url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
            with pdfplumber.open(BytesIO(response.content)) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                return text.strip()
    except Exception as e:
        print(f"Error extracting PDF from URL {url}: {e}")
        return None

def _extract_pdf_from_base64(base64_data: str) -> Optional[str]:
    """Extract text from base64-encoded PDF data"""
    try:
        pdf_bytes = base64.b64decode(base64_data)
        return _extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        logger.error(f"Base64 PDF extraction failed: {e}")
        return None

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    """Extract text from PDF bytes using pdfplumber"""
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            document_text = ""
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        document_text += f"--- Page {page_num} ---\n{page_text}\n\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            return document_text.strip() if document_text else None
            
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return None

# Legacy method for backward compatibility
def extract_pdf_from_base64(base64_data: str) -> Optional[str]:
    """Legacy method - use _extract_pdf_from_base64 instead"""
    return _extract_pdf_from_base64(base64_data)

async def analyze_pdf_content(content: str) -> dict:
    analysis = {
        "word_count": len(content.split()),
        "character_count": len(content),
        "has_tables": "table" in content.lower() or "|" in content,
        "has_financial_terms": any(term in content.lower() for term in [
            "revenue", "profit", "earnings", "financial", "balance sheet", 
            "income statement", "cash flow", "assets", "liabilities"
        ]),
        "pages_estimated": max(1, len(content) // 3000)
    }
    
    return analysis