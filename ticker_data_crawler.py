import datetime
import yfinance as yf
import pandas as pd
import os

class StockDataCrawler:
    """Fetch price data and compute raw changes (Close - Open) for a list of tickers.

    This class is generic: pass source tickers and a single target ticker to fetch.
    """
    OUTPUT_CSV = 'raw_ticker_changes.csv'

    def __init__(self, stocks, etfs, interval='1h', target: str = 'target'):
        self.stocks = stocks
        self.etfs = etfs
        self.interval = interval
        self.target = target
        self.changes_df = pd.DataFrame()

    def fetch_data(self):
        """Fetch price data for tickers and compute raw changes (Close - Open).
        Returns True if any data collected, False otherwise.
        """
        tickers = self.stocks + self.etfs
        days_per_chunk = 60

        changes_list = []

        # always include the configured target ticker so it can be used as the label
        for ticker in tickers + [self.target]:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=720)  # ~2 years
            dfs = []
            while start_date < end_date:
                chunk_end = min(start_date + datetime.timedelta(days=days_per_chunk), end_date)
                try:
                    df_chunk = yf.download(
                        ticker,
                        interval=self.interval,
                        start=start_date,
                        end=chunk_end,
                        progress=False,
                        auto_adjust=False,
                    )
                except Exception as e:
                    print(f"Warning: download failed for {ticker} chunk {start_date} -> {chunk_end}: {e}")
                    start_date += datetime.timedelta(days=days_per_chunk)
                    continue

                if not df_chunk.empty:
                    dfs.append(df_chunk)
                start_date += datetime.timedelta(days=days_per_chunk)

            if len(dfs) == 0:
                print(f"No data for {ticker}, skipping.")
                continue

            df = pd.concat(dfs)
            print(f"Data for {ticker} has {len(df)} rows.")

            # Price change = Close - Open (raw change, not engineered)
            if not df.empty:
                changes = df['Close'] - df['Open']
                # normalize the label column name to the configured target identifier
                changes.name = "target" if ticker == self.target else ticker
                changes_list.append(changes)

        if len(changes_list) == 0:
            print("No changes collected for any ticker.")
            self.changes_df = pd.DataFrame()
            return False

        # Align by datetime automatically and drop rows with missing data
        self.changes_df = pd.concat(changes_list, axis=1)
        self.changes_df.dropna(inplace=True)

        print(f"Total rows collected after alignment and dropping NaNs: {len(self.changes_df)}")
        return True

    def save_to_csv(self):
        """Save the raw changes DataFrame to OUTPUT_CSV. Always writes a file (empty if no data)."""
        out_dir = os.path.dirname(self.OUTPUT_CSV)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        if self.changes_df is None or self.changes_df.empty:
            # write an empty CSV so callers/tests can rely on file existence
            pd.DataFrame().to_csv(self.OUTPUT_CSV, index=False)
            print(f"No data to save — wrote empty CSV to {self.OUTPUT_CSV}")
            return

        self.changes_df.to_csv(self.OUTPUT_CSV, index=True)
        print(f"CSV saved to {self.OUTPUT_CSV}")


# Example usage guarded to avoid network calls on import (keeps tests/imports safe)
if __name__ == "__main__":
    # --- SETTINGS ---
    source_tickers = ['SCHB']
    other_efts = []

    crawler = StockDataCrawler(
        stocks=source_tickers,
        etfs=other_efts,
        interval='1h',
        target='target',
    )

    if crawler.fetch_data():
        crawler.save_to_csv()
    else:
        crawler.save_to_csv()
