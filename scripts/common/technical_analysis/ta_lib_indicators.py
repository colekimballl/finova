# ta_lib_indicators.py

import pandas as pd
import talib as ta
import logging

logger = logging.getLogger(__name__)

def calculate_ta_lib_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate technical indicators using TA-Lib"""
    logger.info("Calculating TA-Lib technical indicators...")
    c = config['trend_indicators']
    df['sma'] = ta.SMA(df['Close'], timeperiod=c['sma_period'])
    df['ema'] = ta.EMA(df['Close'], timeperiod=c['ema_period'])
    df['tema'] = ta.TEMA(df['Close'], timeperiod=c['tema_period'])
    df['wma'] = ta.WMA(df['Close'], timeperiod=c['wma_period'])
    df['kama'] = ta.KAMA(df['Close'], timeperiod=c['kama_period'])
    df['trima'] = ta.TRIMA(df['Close'], timeperiod=c['trima_period'])
    
    c = config['momentum_indicators']
    df['rsi'] = ta.RSI(df['Close'], timeperiod=c['rsi_period'])
    df['macd'], df['macdsignal'], df['macdhist'] = ta.MACD(
        df['Close'], 
        fastperiod=c['macd_fast'],
        slowperiod=c['macd_slow'],
        signalperiod=c['macd_signal']
    )
    df['mom'] = ta.MOM(df['Close'], timeperiod=c['mom_period'])
    df['roc'] = ta.ROC(df['Close'], timeperiod=c['roc_period'])
    df['willr'] = ta.WILLR(df['High'], df['Low'], df['Close'], timeperiod=c['willr_period'])
    
    c = config['volatility_indicators']
    df['atr'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=c['atr_period'])
    df['upperband'], df['middleband'], df['lowerband'] = ta.BBANDS(
        df['Close'],
        timeperiod=c['bbands_period'],
        nbdevup=c['bbands_dev'],
        nbdevdn=c['bbands_dev']
    )
    
    c = config['volume_indicators']
    df['obv'] = ta.OBV(df['Close'], df['Volume'])
    df['mfi'] = ta.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=c['mfi_period'])
    
    logger.info("TA-Lib technical indicators calculated successfully.")
    return df
