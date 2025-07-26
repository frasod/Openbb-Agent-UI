"""
This module contains the core of the OpenBB Agent's logic.
It manages the interaction with the language model, including
tool selection, execution, and response synthesis. This separation
of concerns ensures the agent's logic is clear and maintainable.
"""
import json
import httpx
from typing import AsyncGenerator, Dict, Any
from ..config import settings, openai_client, get_current_model
from ..tools.openbb_tools import AVAILABLE_TOOLS, get_tool_definitions

async def run_agent_loop(
    query: str, 
    selected_tools: list[str] | None = None, 
    system_prompt: str | None = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Runs the main agent loop with real-time progress updates.
    Yields progress updates for the terminal sidebar.
    """
    # Start with the system prompt, then the user query
    messages = [
        {"role": "system", "content": system_prompt or "You are a helpful financial assistant."},
        {"role": "user", "content": query}
    ]
    
    # Filter tools based on user selection
    if selected_tools:
        tools = [
            tool for tool in get_tool_definitions()
            if tool["function"]["name"] in selected_tools
        ]
    else:
        tools = []

    yield {"type": "progress", "message": f"🚀 Initializing {settings.llm_provider} ({get_current_model()})..."}
    
    # --- First Pass: Decide which tool to use (if any) ---
    yield {"type": "progress", "message": "🧠 Analyzing query and selecting tools..."}
    
    try:
        if settings.llm_provider == "ollama":
            response_data = await _call_ollama_api(messages, tools)
        else:
            # Both OpenAI and OpenRouter use OpenAI-compatible API
            response = openai_client.chat.completions.create(
                model=get_current_model(),
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            response_data = {
                "message": response.choices[0].message,
                "usage": response.usage
            }
    except Exception as e:
        yield {"type": "error", "message": f"❌ LLM API Error: {str(e)}"}
        return

    response_message = response_data["message"]
    usage_data = response_data.get("usage")

    tool_calls = getattr(response_message, 'tool_calls', None)
    if not tool_calls:
        yield {
            "type": "final",
            "content": response_message.content or "I don't have a tool for that yet.",
            "tools_used": [],
            "llm_provider": settings.llm_provider,
            "model": get_current_model(),
            "token_usage": usage_data.total_tokens if usage_data else 0,
            "chart": None,
        }
        return

    # --- Second Pass: Execute tools ---
    yield {"type": "progress", "message": f"🔧 Executing {len(tool_calls)} tool(s)..."}
    
    messages.append(response_message)
    tools_used = []
    chart_data = None
    
    for i, tool_call in enumerate(tool_calls):
        function_name = tool_call.function.name
        tools_used.append(function_name)
        
        yield {"type": "progress", "message": f"⚡ Running {function_name} ({i+1}/{len(tool_calls)})..."}
        
        function_to_call = AVAILABLE_TOOLS.get(function_name, {}).get("function")
        
        if not function_to_call:
            tool_response = f"Error: Tool '{function_name}' not found."
        else:
            try:
                function_args = json.loads(tool_call.function.arguments)
                
                # Check if the function is async
                import inspect
                if inspect.iscoroutinefunction(function_to_call):
                    tool_response = await function_to_call(**function_args)
                else:
                    tool_response = function_to_call(**function_args)
                
                # If a chart was generated, store it
                if function_name == "generate_financial_chart":
                    try:
                        chart_data = json.loads(tool_response)
                        yield {"type": "progress", "message": "📊 Chart generated successfully!"}
                    except json.JSONDecodeError as e:
                        yield {"type": "progress", "message": f"⚠️ Chart generation warning: {str(e)}"}
                        chart_data = None
                        
            except Exception as e:
                tool_response = f"Error executing {function_name}: {str(e)}"
                yield {"type": "progress", "message": f"❌ Tool error: {str(e)}"}

        messages.append(
            {"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": tool_response}
        )

    # --- Third Pass: Synthesize the final answer ---
    yield {"type": "progress", "message": "📝 Synthesizing final analysis..."}
    
    messages.append({
        "role": "user",
        "content": """Based on the information provided, please synthesize a final answer. 
                    It is crucial that your analysis is based on the most current data available from the tools.
                    Start your response with a 'Bottom Line Up Front' (BLUF) section of at least 100 words. 
                    In this BLUF, you must classify the investment opportunity as one of the following: 'Bullish', 'Neutral', or 'Bearish'.
                    Then, provide the detailed analysis and supporting data."""
    })

    try:
        if settings.llm_provider == "ollama":
            final_response_data = await _call_ollama_api(messages, [])
        else:
            final_response = openai_client.chat.completions.create(
                model=get_current_model(), messages=messages
            )
            final_response_data = {
                "message": final_response.choices[0].message,
                "usage": final_response.usage
            }
    except Exception as e:
        yield {"type": "error", "message": f"❌ Final synthesis error: {str(e)}"}
        return

    final_answer = final_response_data["message"].content
    final_usage = final_response_data.get("usage")

    yield {"type": "progress", "message": "✅ Analysis complete!"}

    yield {
        "type": "final",
        "content": final_answer,
        "tools_used": tools_used,
        "llm_provider": settings.llm_provider,
        "model": get_current_model(),
        "token_usage": (usage_data.total_tokens if usage_data else 0) + (final_usage.total_tokens if final_usage else 0),
        "chart": chart_data,
    }

async def _call_ollama_api(messages: list, tools: list) -> dict:
    """
    Call Ollama API directly using httpx since it doesn't use OpenAI format.
    """
    async with httpx.AsyncClient() as client:
        payload = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False
        }
        
        # Note: Ollama might not support tools/function calling yet
        # This is a simplified implementation
        response = await client.post(
            f"{settings.ollama_base_url}/v1/chat/completions",
            json=payload,
            timeout=settings.read_timeout
        )
        response.raise_for_status()
        data = response.json()
        
        # Convert to OpenAI-like format
        return {
            "message": type('obj', (object,), {
                'content': data['choices'][0]['message']['content'],
                'tool_calls': None  # Ollama might not support this yet
            }),
            "usage": data.get('usage', {"total_tokens": 0})
        }