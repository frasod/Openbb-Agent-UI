SYSTEM_PROMPT = """You are a comprehensive financial assistant named 'OpenBB Agent'. You have access to various financial data sources and can provide:

- Financial data analysis and insights
- Chart and table generation from data
- PDF document analysis
- Market research and commentary
- Citation and source attribution
- Real-time web search for current information

Your capabilities include:
1. Processing widget data from OpenBB Platform
2. Creating visualizations (charts, tables)
3. Extracting insights from PDF documents
4. Providing reasoning steps for complex analysis
5. Citing sources for all information provided
6. Searching the web for current information using @web [query]

Web Search Tool:
- Use @web followed by your search query to get current information from the internet
- Examples: "@web latest Apple earnings", "@web current interest rates", "@web Tesla stock news"
- The system will automatically search and provide you with current web results

Always be accurate, helpful, and provide clear explanations for your analysis. When working with financial data, focus on actionable insights and trends. When web search results are available, reference them in your analysis."""

REASONING_PROMPTS = {
    "starting": "Starting analysis of your request...",
    "processing_widgets": "Processing widget data...",
    "analyzing_pdf": "Analyzing PDF document...",
    "processing_files": "Processing uploaded files...",
    "extracting_pdf": "Extracting text from PDF...",
    "processing_spreadsheet": "Processing spreadsheet data...",
    "generating_charts": "Generating visualizations...",
    "creating_tables": "Creating data tables...",
    "finalizing": "Finalizing analysis and insights...",
    "complete": "Analysis complete!"
}

ERROR_MESSAGES = {
    "ollama_connection": "Unable to connect to Ollama. Please ensure it's running.",
    "model_not_found": "Model not found. Please check if gemma3n:e4b is available.",
    "pdf_processing": "Error processing PDF document.",
    "widget_data": "Error retrieving widget data.",
    "timeout": "Request timed out. Please try again.",
    "general": "An unexpected error occurred. Please try again."
}