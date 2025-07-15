import pytest
from typing import List, Dict
from unittest.mock import patch
from comprehensive_agent.processors.financial_web_search import FinancialWebSearcher

@pytest.fixture
def searcher():
    return FinancialWebSearcher()

def test_search_and_analyze_success(searcher, mocker):
    mock_results = [{'title': 'Test', 'body': 'positive news', 'href': 'bloomberg.com/test', 'date': '2023-10-01'}]
    mocker.patch('duckduckgo_search.DDGS.text', return_value=mock_results)
    results = searcher.search_and_analyze('test query')
    assert len(results) > 0
    assert 'sentiment_score' in results[0]

def test_no_results(searcher, mocker):
    mocker.patch('duckduckgo_search.DDGS.text', return_value=[])
    assert searcher.search_and_analyze('empty') == []

def test_api_failure(searcher, mocker):
    mocker.patch('duckduckgo_search.DDGS.text', side_effect=Exception('API error'))
    with pytest.raises(Exception):
        searcher.search_and_analyze('fail') 