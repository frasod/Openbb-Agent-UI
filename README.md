# OpenBB Financial Agent

A sophisticated financial analysis agent that combines the power of OpenBB Platform with Large Language Models to provide autonomous financial research and analysis capabilities with real-time data visualization.

## Features

### 🎯 Core Capabilities
- **Agentic Financial Analysis**: Multi-pass LLM interaction with intelligent OpenBB tool selection
- **Real-time Data**: Latest financial news, historical prices, and live market data (250+ data points)
- **Interactive Charts**: Professional candlestick charts with actual market data from multiple providers
- **Custom Analysis**: Flexible prompt templates for different investment perspectives and contexts
- **BLUF Analysis**: Bottom Line Up Front summaries with Bullish/Neutral/Bearish classifications

### 🎨 Modern Interface
- **Braun-Inspired Design**: Clean, minimalist UI following Dieter Rams principles
- **Dynamic Tool Selection**: Choose from news, historical data, and chart generation tools
- **Smart Suggestions**: Context-aware prompt examples for selected tools
- **Rich Output**: Markdown formatting with metadata and interactive elements
- **Template System**: Save and manage custom analysis contexts

### 🛠 Technical Features
- **Docker Containerized**: Easy deployment and development with automatic rebuilds
- **Multi-Provider Support**: OpenAI, OpenRouter, and Ollama with dynamic switching
- **API Key Management**: Integrated UI for secure API key configuration and validation
- **Error Resilient**: Comprehensive error handling with graceful degradation
- **Performance Optimized**: Efficient API usage and resource management

### 📊 Chart Generation
- **Live Market Data**: Real-time financial data from yfinance, intrinio, polygon, and FMP
- **Interactive Visualization**: Professional Plotly candlestick charts with OHLC data
- **Smart Fallbacks**: Automatic provider switching when API keys are unavailable
- **Data Validation**: Comprehensive validation ensuring accurate price representations

### ⚡ Real-time Experience
- **Activity Indicators**: Spinning terminal icon shows processing status
- **Progress Tracking**: Live updates during data fetching, analysis, and chart generation
- **Instant Feedback**: Real-time API key validation and configuration status
- **Terminal Logging**: Detailed progress messages with emoji indicators

## Quick Start

### Prerequisites
- Docker and Docker Compose
- OpenAI API key (recommended) or OpenRouter API key
- OpenBB Platform access (optional: OpenBB Hub PAT for enhanced data access)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Comprehensive-Openbb-Agent
   ```

2. **Start the application**
   ```bash
   docker compose up --build -d
   ```

3. **Access the interface**
   Open your browser to `http://localhost:7777`

4. **Configure API Keys (Easy Setup)**
   - Select your preferred LLM provider (OpenAI or OpenRouter)
   - The API Key Configuration section will appear automatically
   - Enter your API key and click "Test" to validate
   - Click "Save to .env" to store securely
   - Restart with `docker compose up --build -d`

## Usage

### Basic Workflow
1. **Configure Provider**: Select and configure your LLM provider with API keys
2. **Select Tools**: Choose from available agentic tools (news, historical data, charts)
3. **Use Suggestions**: Click on suggested prompts or write custom queries
4. **Set Context**: Optionally customize the analysis perspective with templates
5. **Analyze**: Submit your query and receive comprehensive financial analysis
6. **Explore**: Interact with charts and copy formatted results

### Example Queries
- "Get the latest news for AAPL"
- "Provide historical price levels for Tesla over the last year"
- "Show me a financial chart for Microsoft stock"
- "What's the Apple outlook for 2026 with historical price levels and financial charts"
- "Create a technical analysis of NVDA with price history and recent news"

### Template System
Create custom analysis contexts:
- **Neutral**: Standard financial assistant (default)
- **Bullish Outlook**: Optimistic investment perspective
- **Risk Assessment**: Conservative risk-focused analysis
- **Technical Analysis**: Chart and indicator focused
- **Custom Templates**: Save your own analysis styles

## Configuration

### API Key Management
The application includes an integrated API key management system:

#### OpenAI Setup
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create an account and generate an API key
3. In the app, select "OpenAI" as provider
4. Enter your key and click "Test" then "Save to .env"

#### OpenRouter Setup
1. Visit [OpenRouter Keys](https://openrouter.ai/keys)
2. Create an account and generate an API key
3. In the app, select "OpenRouter" as provider
4. Enter your key and click "Test" then "Save to .env"

#### Ollama (Local)
1. Install [Ollama](https://ollama.ai/download)
2. Select "Ollama (Local)" - no API key required
3. Ensure Ollama is running on `localhost:11434`

### Environment Variables (Advanced)
For manual configuration, create a `.env` file:
```env
# LLM Provider Configuration
LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Alternative: OpenRouter
# LLM_PROVIDER=openrouter
# OPENROUTER_API_KEY=your_openrouter_key_here
# OPENROUTER_MODEL=anthropic/claude-3.5-sonnet

# Alternative: Ollama (Local)
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=gemma2:9b
# OLLAMA_BASE_URL=http://localhost:11434

# Server Configuration
SERVER_PORT=7777
CONNECTION_TIMEOUT=10.0
READ_TIMEOUT=30.0
```

### OpenBB Configuration
For enhanced data access, configure OpenBB credentials:
- Set up local credentials in `~/.openbb_platform/user_settings.json`
- Or use OpenBB Hub Personal Access Token (PAT)

## Architecture

### System Design
```
Frontend (Vanilla JS) ↔ FastAPI Backend ↔ OpenBB Platform (Multi-Provider)
                                      ↔ LLM APIs (OpenAI/OpenRouter/Ollama)
```

### Key Components
- **Frontend**: Clean, responsive web interface with Braun-inspired design
- **Agent Core**: Multi-pass LLM orchestration with intelligent tool execution
- **OpenBB Tools**: Financial data retrieval with automatic provider fallbacks
- **Configuration**: Flexible environment-based settings with UI management
- **Chart Engine**: Real-time Plotly visualization with market data integration

### Development Philosophy
Following core principles inspired by Braun/Dieter Rams:
- **Efficiency**: Performant, direct solutions with smart provider fallbacks
- **Minimalism**: Lean code and clean documentation with essential functionality
- **Function-First**: Every feature serves the primary purpose of financial analysis
- **Organization**: Clear logical structure and intuitive design patterns
- **Clarity**: Readable code and transparent operation with real-time feedback

## Development

### Local Development
```bash
# Start in development mode
docker compose up --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Project Structure
```
comprehensive_agent/
├── core/           # Agent orchestration logic
├── tools/          # OpenBB tool definitions with multi-provider support
├── visualizations/ # Chart generation utilities
├── config.py       # Configuration management
└── main.py         # FastAPI application with API key endpoints

static/             # Frontend assets
├── index.html      # Main interface with configuration UI
├── script.js       # Application logic with real-time features
└── style.css       # Braun-inspired styling with activity indicators
```

### Chart Generation
The application features a robust chart generation system:
- **Multi-Provider Fallback**: yfinance → intrinio → polygon → FMP
- **Data Validation**: Comprehensive OHLC validation with positive price checks
- **Date Handling**: Smart date sequence generation for incomplete datasets
- **Interactive Charts**: Plotly v2.35.0 with professional candlestick rendering
- **Error Recovery**: Detailed debugging and graceful fallback to sample data

## Contributing

We follow clean code principles and maintain high standards:
- Clear, descriptive commit messages with semantic versioning
- Comprehensive error handling with user-friendly messages
- Performance-conscious development with provider optimization
- User experience focus with real-time feedback
- Documentation updates with code changes
- Test coverage for critical functionality

## Troubleshooting

### Common Issues

1. **Charts Not Displaying**
   - Check browser console (F12) for detailed error messages
   - Verify OpenBB providers are accessible
   - Try different stock symbols (AAPL, MSFT, TSLA)

2. **API Key Issues**
   - Use the integrated testing feature before saving
   - Ensure proper provider selection
   - Restart Docker after saving new keys

3. **Docker Issues**
   - Use `docker compose down && docker compose up --build -d`
   - Check logs with `docker compose logs -f`
   - Verify Docker Desktop is running

4. **Performance Issues**
   - Check terminal for provider fallback messages
   - Verify network connectivity for data providers
   - Monitor Docker resource usage

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, feature requests, or questions:
1. Check the development log for recent changes and known issues
2. Review Docker logs for runtime issues: `docker compose logs -f`
3. Ensure API keys are properly configured using the integrated UI
4. Verify Docker and Docker Compose are correctly installed
5. Test with different stock symbols and tools

---

*Built with ❤️ for the financial analysis community*

**Latest Update**: Complete chart generation solution with real market data, integrated API key management, and enhanced user experience with real-time feedback.