# Data Directory

This directory stores all data files related to the project, organized into subdirectories.

## Subdirectories

- **raw/**: Raw data files collected from exchanges or other sources.
- **processed/**: Data that has been cleaned or processed and is ready for analysis.
- **historical/**: Historical market data files.

## Data Management

- **Raw Data**: Place all unprocessed data files here. Use consistent naming conventions, such as `exchange_symbol_data.csv`.
- **Processed Data**: Store data that has been processed by your scripts in this directory.
- **Historical Data**: Keep historical data files for backtesting and analysis.

## Notes

- Large data files should not be committed to version control.
- Use data versioning tools if necessary.
- Document any data processing steps in the `/docs/` directory or within scripts.


