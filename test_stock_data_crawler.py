import pytest
import pandas as pd
import datetime
from unittest.mock import patch
from io import StringIO
import os

from ticker_data_crawler import StockDataCrawler


@pytest.fixture
def mock_yf_download_data():
    """Generate fake stock data for testing."""
    dates = pd.date_range(end=datetime.datetime.now(), periods=5, freq='h')
    return pd.DataFrame({
        'Open': [100, 101, 102, 103, 104],
        'Close': [101, 102, 103, 104, 105],
    }, index=dates)


@patch('yfinance.download')
def test_fetch_data_creates_expected_dataframe(mock_download, mock_yf_download_data, tmp_path):
    # --- SETUP ---
    mock_download.return_value = mock_yf_download_data
    crawler = StockDataCrawler(
        stocks=['AAPL'],
        etfs=[],
        interval='1h'
    )
    crawler.OUTPUT_CSV = str(tmp_path / 'test_output.csv')

    # --- ACT ---
    crawler.fetch_data()

    # --- ASSERT ---
    df = crawler.changes_df
    assert isinstance(df, pd.DataFrame)
    assert 'AAPL' in df.columns
    assert 'target' in df.columns
    # assert 'AAPL_prev' in df.columns
    # assert 'AAPL_prev2' in df.columns

    # target should be 0/1 only
    assert df['target'].isin([0, 1]).all()

    # --- SAVE TEST ---
    crawler.save_to_csv()
    assert os.path.exists(crawler.OUTPUT_CSV)

    # CSV content check
    saved_df = pd.read_csv(crawler.OUTPUT_CSV, index_col=0)
    assert not saved_df.empty
    # assert 'AAPL_prev2' in saved_df.columns


@patch('yfinance.download')
def test_handles_empty_downloads_gracefully(mock_download, tmp_path):
    mock_download.return_value = pd.DataFrame()  # Empty dataframe

    crawler = StockDataCrawler(stocks=['AAPL'], etfs=[], interval='1h')
    crawler.OUTPUT_CSV = str(tmp_path / 'empty_test.csv')

    # Should not crash
    crawler.fetch_data()
    crawler.save_to_csv()

    assert os.path.exists(crawler.OUTPUT_CSV)
