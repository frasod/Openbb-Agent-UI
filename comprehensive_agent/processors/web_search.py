import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class WebSearchProcessor:
    """Enhanced web search processor with multiple search engines and content extraction"""
    
    def __init__(self):
        self.max_results = 10
        self.timeout = 30
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def search_web(self, query: str, max_results: int = 5, region: str = "us-en") -> Dict[str, Any]:
        """
        Perform web search using DuckDuckGo
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Region for search (default: us-en)
            
        Returns:
            Dictionary with search results and metadata
        """
        if not DDGS:
            return {
                "error": "DuckDuckGo search not available. Please install: pip install duckduckgo-search",
                "results": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        
        try:
            logger.info(f"Searching web for: {query}")
            
            # Use async wrapper for DuckDuckGo search
            results = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._search_duckduckgo, 
                query, 
                max_results, 
                region
            )
            
            # Enhance results with content extraction
            enhanced_results = []
            for result in results[:max_results]:
                enhanced_result = await self._enhance_result(result)
                enhanced_results.append(enhanced_result)
            
            return {
                "results": enhanced_results,
                "query": query,
                "total_results": len(results),
                "timestamp": datetime.now().isoformat(),
                "search_engine": "DuckDuckGo"
            }
            
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return {
                "error": f"Web search failed: {str(e)}",
                "results": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
    
    def _search_duckduckgo(self, query: str, max_results: int, region: str) -> List[Dict]:
        """Synchronous DuckDuckGo search"""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(
                    query, 
                    max_results=max_results,
                    region=region,
                    safesearch="moderate",
                    timelimit="m"  # Recent results (last month)
                ))
                return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []
    
    async def _enhance_result(self, result: Dict) -> Dict:
        """Enhance search result with additional content extraction"""
        enhanced = {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", ""),
            "content_preview": "",
            "content_type": "webpage",
            "relevance_score": 0.0
        }
        
        # Try to extract more content from the webpage
        try:
            content = await self._extract_webpage_content(enhanced["url"])
            enhanced["content_preview"] = content[:500] + "..." if len(content) > 500 else content
            enhanced["content_type"] = self._detect_content_type(enhanced["url"], content)
            enhanced["relevance_score"] = self._calculate_relevance_score(enhanced)
        except Exception as e:
            logger.warning(f"Failed to extract content from {enhanced['url']}: {str(e)}")
        
        return enhanced
    
    async def _extract_webpage_content(self, url: str) -> str:
        """Extract clean text content from webpage"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
                response = await client.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    script.decompose()
                
                # Extract main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content') or soup.body
                
                if main_content:
                    text = main_content.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                
                # Clean up text
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                return ' '.join(lines)
                
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {str(e)}")
            return ""
    
    def _detect_content_type(self, url: str, content: str) -> str:
        """Detect the type of content based on URL and content"""
        url_lower = url.lower()
        
        if any(domain in url_lower for domain in ['news', 'reuters', 'bloomberg', 'cnbc', 'marketwatch']):
            return "news"
        elif any(domain in url_lower for domain in ['sec.gov', 'investor.gov', 'finra.org']):
            return "regulatory"
        elif any(domain in url_lower for domain in ['yahoo.com/finance', 'google.com/finance', 'investing.com']):
            return "financial_data"
        elif any(domain in url_lower for domain in ['research', 'analysis', 'report']):
            return "research"
        elif '.pdf' in url_lower:
            return "pdf"
        else:
            return "webpage"
    
    def _calculate_relevance_score(self, result: Dict) -> float:
        """Calculate relevance score based on various factors"""
        score = 0.0
        
        # Title relevance
        if result.get("title"):
            score += 0.3
        
        # Snippet quality
        if result.get("snippet") and len(result["snippet"]) > 50:
            score += 0.2
        
        # Content preview quality
        if result.get("content_preview") and len(result["content_preview"]) > 100:
            score += 0.3
        
        # Content type bonus
        content_type = result.get("content_type", "")
        if content_type in ["news", "research", "financial_data"]:
            score += 0.2
        
        return min(score, 1.0)
    
    async def search_financial_news(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search specifically for financial news"""
        financial_query = f"{query} site:reuters.com OR site:bloomberg.com OR site:cnbc.com OR site:marketwatch.com OR site:yahoo.com/finance"
        return await self.search_web(financial_query, max_results)
    
    async def search_company_info(self, company: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for company information"""
        company_query = f"{company} earnings financial results investor relations"
        return await self.search_web(company_query, max_results)
    
    async def search_market_data(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search for market data and analysis"""
        market_query = f"{query} market analysis financial data"
        return await self.search_web(market_query, max_results)

# Utility function for easy integration
async def process_web_search(query: str, search_type: str = "general") -> Optional[Dict[str, Any]]:
    """
    Process web search request
    
    Args:
        query: Search query
        search_type: Type of search ("general", "financial", "company", "market")
    
    Returns:
        Search results or None if failed
    """
    try:
        processor = WebSearchProcessor()
        
        if search_type == "financial":
            return await processor.search_financial_news(query)
        elif search_type == "company":
            return await processor.search_company_info(query)
        elif search_type == "market":
            return await processor.search_market_data(query)
        else:
            return await processor.search_web(query)
            
    except Exception as e:
        logger.error(f"Web search processing failed: {str(e)}")
        return None

# Function to detect if a message contains web search request
def detect_web_search_request(message: str) -> Optional[Dict[str, str]]:
    """
    Detect if a message contains a web search request
    
    Returns:
        Dictionary with query and search_type if detected, None otherwise
    """
    message_lower = message.lower()
    
    # Check for @web mention
    if "@web" in message_lower:
        # Extract query after @web
        parts = message.split("@web")
        if len(parts) > 1:
            query = parts[1].strip()
            
            # Determine search type based on keywords
            if any(keyword in query.lower() for keyword in ["earnings", "financial", "company", "stock", "revenue"]):
                search_type = "financial"
            elif any(keyword in query.lower() for keyword in ["market", "economy", "trading", "analysis"]):
                search_type = "market"
            else:
                search_type = "general"
            
            return {
                "query": query,
                "search_type": search_type
            }
    
    # Check for other web search indicators
    web_indicators = [
        "search web for", "search the internet", "look up online", 
        "find information about", "web search", "internet search"
    ]
    
    for indicator in web_indicators:
        if indicator in message_lower:
            # Extract query after the indicator
            parts = message_lower.split(indicator)
            if len(parts) > 1:
                query = parts[1].strip()
                return {
                    "query": query,
                    "search_type": "general"
                }
    
    return None