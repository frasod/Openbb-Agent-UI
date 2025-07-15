@echo off
REM Comprehensive OpenBB Agent Startup Script for Windows
REM This script sets up the environment and starts the agent

echo 🚀 Starting Comprehensive OpenBB Agent...

REM Navigate to the script directory
cd /d "%~dp0"

REM Set custom Ollama models directory
set OLLAMA_MODELS=D:\Ollama\manifests\blobs
echo 🔧 Setting OLLAMA_MODELS to: %OLLAMA_MODELS%

REM Activate the virtual environment
if exist "..\..\venv\Scripts\activate.bat" (
    echo 🔧 Activating virtual environment...
    call "..\..\venv\Scripts\activate.bat"
) else (
    echo ❌ Virtual environment not found at ..\..\venv\
    echo Please create the virtual environment first
    pause
    exit /b 1
)

echo 🔍 Checking Ollama connection...
curl -s http://localhost:11434/api/version >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Warning: Ollama is not running on localhost:11434
    echo    Please start Ollama server: ollama serve
    echo    With custom models directory: set OLLAMA_MODELS=%OLLAMA_MODELS%
    echo    And ensure you have the model: ollama pull gemma3n:e2b
    echo.
) else (
    echo ✅ Ollama server is running
    echo 🔍 Checking available models...
    curl -s http://localhost:11434/api/tags
    echo.
)

echo 🚀 Starting agent server on http://localhost:7777...
echo    - Use Ctrl+C to stop the server
echo    - Add http://localhost:7777 to OpenBB Workspace as agent endpoint
echo.

python -m comprehensive_agent.main