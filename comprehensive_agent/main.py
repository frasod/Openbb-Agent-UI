"""
Main API entry point for the OpenBB Financial Agent.

This module follows a minimalist and function-first design philosophy.
It is responsible for handling web requests and delegating the core
agent logic to the appropriate modules.
"""
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from openbb import obb # Used by tools
import json
import os

from .config import settings, openai_client, get_current_model
from .core.model_engine import run_agent_loop
from .tools.openbb_tools import get_tool_definitions

app = FastAPI(
    title="OpenBB LLM Agent",
    description="An agent that uses LLMs and OpenBB Platform to provide autonomous financial research.",
)

# Serve static files for the frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/tools")
async def get_tools():
    """Returns a list of available tools for the frontend to display."""
    tools = get_tool_definitions()
    return [{"name": tool["function"]["name"], "description": tool["function"]["description"]} for tool in tools]

@app.get("/providers")
async def get_providers():
    """Returns available LLM providers and models."""
    return {
        "current_provider": settings.llm_provider,
        "current_model": get_current_model(),
        "providers": {
            "openai": {
                "name": "OpenAI",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
                "requires_key": True,
                "key_name": "OPENAI_API_KEY",
                "help_url": "https://platform.openai.com/api-keys"
            },
            "openrouter": {
                "name": "OpenRouter",
                "models": [
                    "anthropic/claude-3.5-sonnet",
                    "openai/gpt-4o",
                    "meta-llama/llama-3.1-70b-instruct",
                    "google/gemini-pro-1.5",
                    "mistralai/mistral-large"
                ],
                "requires_key": True,
                "key_name": "OPENROUTER_API_KEY",
                "help_url": "https://openrouter.ai/keys"
            },
            "ollama": {
                "name": "Ollama (Local)",
                "models": ["llama3:8b", "gemma2:9b", "mistral:7b", "codellama:13b"],
                "requires_key": False,
                "key_name": None,
                "help_url": "https://ollama.ai/download"
            }
        }
    }

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    selected_tools: list[str] | None = None
    system_prompt: str | None = None

class AgentResponse(BaseModel):
    content: str
    tools_used: list[str]
    llm_provider: str
    model: str
    token_usage: int
    chart: dict | None = None

class ApiKeyRequest(BaseModel):
    provider: str
    api_key: str

@app.post("/api/save-key")
async def save_api_key(request: ApiKeyRequest):
    """Save API key to .env file."""
    try:
        provider_info = {
            "openai": "OPENAI_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"
        }
        
        if request.provider not in provider_info:
            raise HTTPException(status_code=400, detail="Invalid provider")
        
        env_key = provider_info[request.provider]
        
        # Read existing .env file or create new one
        env_path = ".env"
        env_vars = {}
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        
        # Update the API key
        env_vars[env_key] = request.api_key
        
        # Ensure other necessary env vars exist
        if 'LLM_PROVIDER' not in env_vars:
            env_vars['LLM_PROVIDER'] = request.provider
        if 'OPENAI_MODEL' not in env_vars:
            env_vars['OPENAI_MODEL'] = 'gpt-4o-mini'
        if 'OPENROUTER_MODEL' not in env_vars:
            env_vars['OPENROUTER_MODEL'] = 'anthropic/claude-3.5-sonnet'
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        return {"status": "success", "message": f"API key saved for {request.provider}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save API key: {str(e)}")

@app.post("/api/test-key")
async def test_api_key(request: ApiKeyRequest):
    """Test if an API key is valid."""
    try:
        if request.provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=request.api_key)
            # Test with a simple completion
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            return {"status": "success", "message": "OpenAI API key is valid"}
            
        elif request.provider == "openrouter":
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {request.api_key}"}
                )
                if response.status_code == 200:
                    return {"status": "success", "message": "OpenRouter API key is valid"}
                else:
                    return {"status": "error", "message": "OpenRouter API key is invalid"}
        
        return {"status": "error", "message": "Provider not supported for testing"}
        
    except Exception as e:
        return {"status": "error", "message": f"API key test failed: {str(e)}"}

# --- Core Agent Logic Endpoint (Streaming) ---
@app.post("/query/stream")
async def query_agent_stream(request: QueryRequest):
    """
    Streams the agent's progress in real-time for the terminal sidebar.
    """
    async def event_stream():
        try:
            async for update in run_agent_loop(
                query=request.query,
                selected_tools=request.selected_tools,
                system_prompt=request.system_prompt
            ):
                yield {
                    "event": update["type"],
                    "data": json.dumps(update)
                }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"type": "error", "message": f"Stream error: {str(e)}"})
            }
    
    return EventSourceResponse(event_stream())

# --- Core Agent Logic Endpoint (Non-streaming for backward compatibility) ---
@app.post("/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    """
    Receives a user query and returns the final synthesized response.
    """
    try:
        final_response = None
        async for update in run_agent_loop(
            query=request.query,
            selected_tools=request.selected_tools,
            system_prompt=request.system_prompt
        ):
            if update["type"] == "final":
                final_response = update
                break
        
        if not final_response:
            raise HTTPException(status_code=500, detail="No final response received from agent")
        
        return AgentResponse(
            content=final_response["content"],
            tools_used=final_response["tools_used"],
            llm_provider=final_response["llm_provider"],
            model=final_response["model"],
            token_usage=final_response["token_usage"],
            chart=final_response["chart"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return 204 for favicon to avoid 404 errors in browser."""
    return Response(status_code=204)