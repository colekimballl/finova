# cardano scripts usage guide

## data collection
fetch recent data (4 weeks):
	python cardano_data.py

fetched data from 2021
	python fast-cardano-data-fetcher.py --from-date 2021-03-19 --merge

specific time period
	python cardano_data.py --weeks 12  # 12 weeks of data
	python fast-cardano-data-fetcher.py --years 1 --merge  # 1 year of data

data processing
process hourly data with basic features:

	python data_processor.py --timeframe 1h --features
	python data_processor.py --all-timeframes --all-features
	python data_processor.py --timeframe 1d --indicators --patterns
	python data_processor.py --stats

available feature options:

--features: basic price features
--indicators: technical indicators
--patterns: candlestick patterns
--regimes: market regime features
--ml: machine learning features


data visualization

	python indicator_visualizer.py --timeframe 1h --candlestick --show
	python indicator_visualizer.py --timeframe 1d --candlestick --output ../data/processed/visualizations
	python indicator_visualizer.py --timeframe 1h --days 90 --candlestick --show  # 90 days
	python indicator_visualizer.py --timeframe 1d --days 1500 --candlestick --show 
	entire^

common timeframe options:

1m: minute-by-minute (very detailed, short periods only)
5m: 5-minute (detailed intraday)
15m: 15-minute (intraday patterns)
1h: hourly (good balance for short/medium analysis)
4h: 4-hour (medium-term trends)
1d: daily (long-term analysis)

typical workflow
python fast-cardano-data-fetcher.py --from-date 2021-03-19 --merge
bash ../cron_setup.sh
python data_processor.py --all-timeframes --all-features
python indicator_visualizer.py --timeframe 1d --days 365 --candlestick --show
python data_processor.py --all-timeframes


