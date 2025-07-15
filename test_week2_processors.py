#!/usr/bin/env python3
"""
Test script for Week 2 enhanced processors
"""

import asyncio
import json
import pandas as pd
from io import BytesIO
from datetime import datetime, timedelta
import sys
import os

# Add the comprehensive_agent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'comprehensive_agent'))

from comprehensive_agent.processors.spreadsheet_processor import SpreadsheetProcessor
from comprehensive_agent.processors.api_data_processor import APIDataProcessor

def create_sample_excel_data():
    """Create sample Excel data for testing."""
    # Create sample financial data
    income_data = {
        'Metric': ['Revenue', 'Gross Profit', 'Operating Expenses', 'Net Income', 'EBITDA'],
        '2023': [1000000, 600000, 350000, 250000, 300000],
        '2022': [900000, 520000, 320000, 200000, 250000],
        '2021': [800000, 480000, 300000, 180000, 220000]
    }
    
    balance_data = {
        'Metric': ['Total Assets', 'Current Assets', 'Current Liabilities', 'Total Debt', 'Shareholders Equity'],
        '2023': [2000000, 800000, 300000, 500000, 1200000],
        '2022': [1800000, 720000, 280000, 450000, 1070000],
        '2021': [1600000, 640000, 260000, 400000, 940000]
    }
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        pd.DataFrame(income_data).to_excel(writer, sheet_name='Income Statement', index=False)
        pd.DataFrame(balance_data).to_excel(writer, sheet_name='Balance Sheet', index=False)
    
    return output.getvalue()

def create_sample_api_data():
    """Create sample API data for testing."""
    return {
        'data': [
            {
                'symbol': 'AAPL',
                'price': '$150.25',
                'volume': '1,234,567',
                'market_cap': '2.5T',
                'pe_ratio': 25.5,
                'date': '2024-01-15'
            },
            {
                'ticker': 'MSFT',
                'closing_price': 380.50,
                'trading_volume': 987654,
                'marketcap': '2.8T',
                'p_e_ratio': 28.2,
                'trading_date': '2024-01-15T16:00:00'
            },
            {
                'instrument': 'GOOGL',
                'last_price': 142.75,
                'vol': '2.1M',
                'market_capitalization': '1.8b',
                'price_earnings_ratio': 22.1,
                'timestamp': '2024-01-15 16:00:00'
            }
        ]
    }

def create_time_series_datasets():
    """Create multiple time series datasets for merge testing."""
    base_date = datetime(2024, 1, 1)
    
    dataset1 = {
        'data': [
            {
                'date': (base_date + timedelta(days=i)).isoformat(),
                'price': 100 + i * 0.5,
                'volume': 1000000 + i * 10000,
                'symbol': 'AAPL'
            }
            for i in range(10)
        ]
    }
    
    dataset2 = {
        'data': [
            {
                'timestamp': (base_date + timedelta(days=i + 5)).isoformat(),
                'closing_price': 105 + i * 0.3,
                'trading_volume': 1050000 + i * 8000,
                'ticker': 'AAPL',
                'pe_ratio': 25.0 + i * 0.1
            }
            for i in range(10)
        ]
    }
    
    return [dataset1, dataset2]

async def test_spreadsheet_processor():
    """Test the SpreadsheetProcessor."""
    print("Testing SpreadsheetProcessor...")
    
    processor = SpreadsheetProcessor()
    
    # Test Excel parsing
    excel_data = create_sample_excel_data()
    result = await processor.parse_excel_file(excel_data)
    
    print(f"✓ Parsed Excel file with {result['metadata']['total_sheets']} sheets")
    print(f"✓ Sheet names: {result['metadata']['sheet_names']}")
    
    # Test financial statement detection
    if result['financial_statements']:
        statements = result['financial_statements']
        print(f"✓ Detected financial statements:")
        for stmt_type, stmt_data in statements.items():
            if stmt_data and stmt_type != 'other_financial_data':
                print(f"  - {stmt_type}: {stmt_data['sheet_name']} (confidence: {stmt_data['confidence_score']:.2f})")
        
        # Test financial metrics extraction
        metrics = await processor.extract_financial_metrics(statements)
        if metrics:
            print(f"✓ Extracted financial metrics:")
            for category, ratios in metrics.items():
                if ratios and category != 'calculated_at':
                    print(f"  - {category}: {list(ratios.keys())}")
    
    # Test CSV processing
    csv_data = "symbol,price,volume,date\nAAPL,150.25,1234567,2024-01-15\nMSFT,380.50,987654,2024-01-15"
    csv_result = await processor.handle_csv_data(csv_data)
    
    print(f"✓ Processed CSV with {len(csv_result['data'])} records")
    
    return True

async def test_api_data_processor():
    """Test the APIDataProcessor."""
    print("\nTesting APIDataProcessor...")
    
    processor = APIDataProcessor()
    
    # Test data normalization
    sample_data = create_sample_api_data()
    normalized = await processor.normalize_financial_data(sample_data)
    
    print(f"✓ Normalized {len(normalized['data'])} records")
    print(f"✓ Field mappings: {len(normalized['schema']['field_mappings'])}")
    print(f"✓ Standardized fields: {list(normalized['schema']['standardized_fields'].keys())}")
    
    # Test time series merging
    datasets = create_time_series_datasets()
    merged = await processor.merge_time_series(datasets)
    
    print(f"✓ Merged {merged['metadata']['source_datasets']} datasets")
    print(f"✓ Total records after merge: {merged['metadata']['total_records']}")
    
    if merged['merge_info']['overlapping_periods']:
        print(f"✓ Detected {len(merged['merge_info']['overlapping_periods'])} overlapping periods")
    
    # Test missing data filling
    # Create data with missing values
    data_with_missing = {
        'data': [
            {'price': 100.0, 'volume': 1000000, 'pe_ratio': None},
            {'price': None, 'volume': 1100000, 'pe_ratio': 25.5},
            {'price': 102.0, 'volume': None, 'pe_ratio': 26.0},
            {'price': 103.0, 'volume': 1200000, 'pe_ratio': 26.5}
        ]
    }
    
    filled = await processor.fill_missing_data(data_with_missing, method="interpolate")
    
    print(f"✓ Filled missing data using interpolation")
    print(f"✓ Values filled: {filled['fill_info']['values_filled']}")
    print(f"✓ Quality improvement: {filled['fill_info']['quality_improvement']['completeness_after']:.2f}")
    
    return True

async def main():
    """Run all tests."""
    print("=== Week 2 Enhanced Processors Test ===\n")
    
    try:
        # Test SpreadsheetProcessor
        await test_spreadsheet_processor()
        
        # Test APIDataProcessor
        await test_api_data_processor()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)