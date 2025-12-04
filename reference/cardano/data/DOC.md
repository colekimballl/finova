Data Overview

all data is stored in the data/cardano/ 

data/
└── cardano/
    ├── ADAUSD-1m-data.csv     # 1-minute data (master dataset)
    ├── ADAUSD-5m.csv          # 5-minute data (derived from 1m)
    ├── ADAUSD-15m.csv         # 15-minute data (derived from 1m)
    ├── ADAUSD-1h.csv          # 1-hour data (derived from 1m)
    ├── ADAUSD-4h.csv          # 4-hour data (derived from 1m)
    └── ADAUSD-1d.csv          # Daily data (derived from 1m)

files follow the naming convention: {SYMBOL}-{TIMEFRAME}.csv

sample data
datetime,timestamp,open,high,low,close,volume
2025-02-12 04:00:00,1739392800,0.46331,0.46408,0.46251,0.46329,21587.35
2025-02-12 05:00:00,1739396400,0.46329,0.46339,0.46203,0.46236,23584.31
2025-02-12 06:00:00,1739400000,0.46236,0.46374,0.46227,0.46301,19754.14
...

	first few rows from 1h data ^^


w/o specifcation will gather last 28 days, but can we changed with --weeks arguement parameter

daily cron job runs at 1:30 AM (configured in cron_setup.sh)
executes cardano_data.py --weeks 1 to fetch the most recent week of data


