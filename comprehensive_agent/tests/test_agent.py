import pytest
from fastapi.testclient import TestClient
from comprehensive_agent.main import app

client = TestClient(app)

def test_agents_json_endpoint():
    response = client.get("/agents.json")
    assert response.status_code == 200
    
    data = response.json()
    assert "comprehensive_agent" in data
    
    agent_config = data["comprehensive_agent"]
    assert agent_config["name"] == "Comprehensive Financial Agent"
    assert "streaming" in agent_config["features"]
    assert agent_config["features"]["streaming"] is True

def test_query_endpoint_exists():
    response = client.post("/v1/query", json={
        "messages": [
            {"role": "human", "content": "Hello"}
        ]
    })
    assert response.status_code in [200, 422]

@pytest.mark.asyncio
async def test_widget_data_processing():
    from comprehensive_agent.processors.widgets import process_widget_data
    
    mock_data = []
    result = await process_widget_data(mock_data)
    assert isinstance(result, str)

@pytest.mark.asyncio
async def test_chart_generation():
    from comprehensive_agent.visualizations.charts import create_sample_charts
    
    charts = create_sample_charts()
    assert isinstance(charts, list)

@pytest.mark.asyncio
async def test_table_generation():
    from comprehensive_agent.visualizations.tables import create_sample_table
    
    table = create_sample_table()
    assert table is not None

def test_config_loading():
    from comprehensive_agent.config import settings
    
    assert settings.ollama_model == "gemma3n:e4b"
    assert settings.server_port == 7777
    assert settings.ollama_base_url == "http://localhost:11434"

def test_prompts_loading():
    from comprehensive_agent.prompts import SYSTEM_PROMPT, REASONING_PROMPTS
    
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 0
    assert isinstance(REASONING_PROMPTS, dict)
    assert "starting" in REASONING_PROMPTS