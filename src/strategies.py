import pandas as pd
import numpy as np

#######################################################
# Standard moving average crossover strategy
#######################################################
def ma_strategy(inv_amt, prices_df, short_ma, long_ma, 
                transaction_costs = 0.001, # as a percent
                slippage = 0.0005 # as a percent
                ):
    # Returns a df with signals, positions, and returns
    df = prices_df.copy()
    df = df.drop(columns=['Dividends', "Capital Gains"], errors="ignore")  

    # calculate moving averages
    df["MA_short"] =  df["Close"].rolling(window = short_ma).mean()
    df["MA_long"] =  df["Close"].rolling(window = long_ma).mean()

    # generate signals
    df["signal"] = 0
    df.loc[df["MA_short"] > df["MA_long"], "signal"] = 1
        # 1 if MA short is above MA long, 0 otherwise

    # identify crossovers
    total_cost = transaction_costs + slippage
    df["position"] = df["signal"].diff()
    df["trade_cost"] = 0.0
    df.loc[df['position'] != 0, 'trade_cost'] = total_cost

    # calculate returns
    df['market_ret'] = df['Close'].pct_change()
    df['strategy_ret'] = df['market_ret'] * df['signal'].shift(1)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade_cost']
    
    # Market portfolio - buy at beginning, hold for the whole time
    df['market_ret_net'] = df['market_ret'].copy()
    df.loc[df.index[0], 'market_ret_net'] = df.loc[df.index[0], 'market_ret'] - total_cost

    # Calculate cumulative returns
    df['cumulative_market'] = (1 + df['market_ret_net']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_ret_net']).cumprod()
    
    # Portfolio value
    df['portfolio_value'] = inv_amt * df['cumulative_strategy']

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index("Date", drop=True)
        else:
            df.index = pd.to_datetime(df.index)
   
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    
    return df

#######################################################
# MA crossover strategy with volatility filters
#######################################################
def ma_volatility_filter(inv_amt, prices_df, short_ma, long_ma, 
                vol_threshold=0.70,
                transaction_costs=0.001,
                slippage=0.0005):
    df = prices_df.copy()
    df = df.drop(columns=['Dividends', "Capital Gains", "Stock Splits"], errors="ignore") 

    df["MA_short"] = df["Close"].rolling(window=short_ma).mean()
    df["MA_long"] = df["Close"].rolling(window=long_ma).mean()

    # Calculate True Range properly
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))
        ),
        np.abs(df['Close'].shift(1) - df['Low'])
    )
    
    # Use EMA (Wilder's smoothing)
    df['ATR'] = df['TR'].ewm(span=14, adjust=False).mean()
    df['ATR_pct'] = df['ATR'] / df['Close']

    # Calculate threshold
    atr_cutoff = df['ATR_pct'].quantile(vol_threshold)
    df['high_volatility'] = df['ATR_pct'] > atr_cutoff

    # Signal for trading
    df['signal'] = 0
    df["ma_signal"] = (df["MA_short"] > df["MA_long"]).astype(int)
    df['is_entry'] = (df['ma_signal'] > df['ma_signal'].shift(1)).fillna(False)
    # Detect MA crossovers down (exits)
    df['is_exit'] = (df['ma_signal'] < df['ma_signal'].shift(1)).fillna(False)
    # Valid entries (not during high vol)
    df['valid_entry'] = df['is_entry'] & (~df['high_volatility'])
    in_session = False
    
    for i in range(len(df)):
        if df.iloc[i]['is_exit']:
            # Exit signal - close position
            in_session = False
            df.iloc[i, df.columns.get_loc('signal')] = 0
        elif df.iloc[i]['valid_entry']:
            # Valid entry - open position
            in_session = True
            df.iloc[i, df.columns.get_loc('signal')] = 1
        elif in_session and df.iloc[i]['ma_signal'] == 1:
            # Still in session and MA still bullish - stay in
            df.iloc[i, df.columns.get_loc('signal')] = 1
        else:
            # Not in session
            df.iloc[i, df.columns.get_loc('signal')] = 0
    
    # Clean up
    df = df.drop(['is_entry', 'is_exit', 'valid_entry'], axis=1)
    
    # Cost calculations
    total_cost = transaction_costs + slippage
    df["position"] = df["signal"].diff()
    df["trade_cost"] = 0.0
    df.loc[df['position'] != 0, 'trade_cost'] = total_cost

    # Calculate returns
    df['market_ret'] = df['Close'].pct_change()
    df['strategy_ret'] = df['market_ret'] * df['signal'].shift(1)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade_cost']
    
    # Market portfolio
    df['market_ret_net'] = df['market_ret'].copy()
    df.loc[df.index[0], 'market_ret_net'] = df.loc[df.index[0], 'market_ret'] - total_cost

    # Cumulative returns
    df['cumulative_market'] = (1 + df['market_ret_net']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_ret_net']).cumprod()
    
    # Portfolio value
    df['portfolio_value'] = inv_amt * df['cumulative_strategy']
    df['market_portfolio_value'] = inv_amt * df['cumulative_market']

    # Index handling
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index("Date", drop=True)
        else:
            df.index = pd.to_datetime(df.index)
   
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    return df

#######################################################
# MA crossover strategy with regime detection
#######################################################
def ma_regime_detection(inv_amt, prices_df, short_ma, long_ma, 
                adx_cutoff,
                transaction_costs=0.001,  # as a percent
                slippage=0.0005,  # as a percent
                  # Changed parameter name
                ):
    df = prices_df.copy()
    df = df.drop(columns=['Dividends', "Capital Gains", "Stock Splits"], errors="ignore") 

    df["MA_short"] = df["Close"].rolling(window=short_ma).mean()
    df["MA_long"] = df["Close"].rolling(window=long_ma).mean()

    # find ADX
    Window = 14
    df['+DM'] = np.where(
      (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
      np.maximum(df['High'] - df['High'].shift(1), 0),
      0
    )
    df['-DM'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )

    # ATR - for finding ADX
    df['TR'] = np.maximum(
    np.maximum(
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift(1))  
    ),
    np.abs(df['Close'].shift(1) - df['Low'])
    )
    df['ATR'] = df['TR'].rolling(window=Window).mean()
    
    df['+DM_EMA'] = df['+DM'].ewm(span=Window, adjust=False).mean()
    df['-DM_EMA'] = df['-DM'].ewm(span=Window, adjust=False).mean()

    df['+DI'] = 100 * df['+DM_EMA'] / df['ATR']
    df['-DI'] = 100 * df['-DM_EMA'] / df['ATR']
    df['DX'] = np.abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']) * 100
    df['ADX'] = df['DX'].ewm(span=Window, adjust=False).mean()

    # Calculate threshold from data (renamed variable)
    df['trending'] = df['ADX'] > adx_cutoff

    # signal for trading
    df["ma_signal"] = (df["MA_short"] > df["MA_long"]).astype(int)
    df['is_entry'] = (df['ma_signal'] > df['ma_signal'].shift(1)).fillna(False)
    df['is_exit'] = (df['ma_signal'] < df['ma_signal'].shift(1)).fillna(False)
    df['valid_entry'] = df['is_entry'] & (df['trending'])

    df['signal'] = 0
    in_session = False

    for i in range(len(df)):
        if df.iloc[i]['is_exit']:
            in_session = False
            df.iloc[i, df.columns.get_loc('signal')] = 0
        elif df.iloc[i]['valid_entry']:
            in_session = True
            df.iloc[i, df.columns.get_loc('signal')] = 1
        elif in_session and df.iloc[i]['ma_signal'] == 1:
            df.iloc[i, df.columns.get_loc('signal')] = 1
        else:
            df.iloc[i, df.columns.get_loc('signal')] = 0

    # cleanup
    df = df.drop(['is_entry', 'is_exit', 'valid_entry'], axis=1)
    
    # cost calculations
    total_cost = transaction_costs + slippage
    df["position"] = df["signal"].diff()
    df["trade_cost"] = 0.0
    df.loc[df['position'] != 0, 'trade_cost'] = total_cost

    # calculate returns
    df['market_ret'] = df['Close'].pct_change()
    df['strategy_ret'] = df['market_ret'] * df['signal'].shift(1)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade_cost']
    
    # Market portfolio - buy at beginning, hold for the whole time
    df['market_ret_net'] = df['market_ret'].copy()
    df.loc[df.index[0], 'market_ret_net'] = df.loc[df.index[0], 'market_ret'] - total_cost

    # Calculate cumulative returns
    df['cumulative_market'] = (1 + df['market_ret_net']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_ret_net']).cumprod()
    
    # Portfolio value
    df['portfolio_value'] = inv_amt * df['cumulative_strategy']
    df['market_portfolio_value'] = inv_amt * df['cumulative_market']  # Add this for consistency metrics

    df = df.set_index("Date", inplace=False, drop=True, append=False)
   
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    return df

###############################################################################
# MA crossover strategy that adjusts position sizes according to volatility
###############################################################################
def ma_position_resize(inv_amt, prices_df, short_ma, long_ma, 
                atr_cutoff=0.75,
                min_position=0.25,
                max_position=1.00,
                transaction_costs=0.001,
                slippage=0.0005):
    df = prices_df.copy()
    df = df.drop(columns=['Dividends', "Capital Gains", "Stock Splits"], errors="ignore") 

    df["MA_short"] = df["Close"].rolling(window=short_ma).mean()
    df["MA_long"] = df["Close"].rolling(window=long_ma).mean()

    window = 14

    # Calculate ATR (use EMA for consistency with V1)
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))  
        ),
        np.abs(df['Close'].shift(1) - df['Low'])
    )
    df['ATR'] = df['TR'].ewm(span=window, adjust=False).mean()  # FIXED: EMA
    df['ATR_pct'] = df['ATR'] / df['Close']

    # Target volatility
    target_vol = df['ATR_pct'].quantile(atr_cutoff)

    # Position sizing
    df['position_size_ratio'] = np.clip(
        target_vol / df['ATR_pct'],
        min_position,
        max_position
    )

    # Generate signals
    df["signal"] = 0
    df.loc[(df["MA_short"] > df["MA_long"]) & (df['ATR'].notna()), "signal"] = 1

    # FIXED: Track actual position size (only set at entry, held until exit)
    df['actual_position_size'] = 0.0
    
    for i in range(len(df)):
        if i == 0:
            continue
            
        prev_signal = df.iloc[i-1]['signal']
        curr_signal = df.iloc[i]['signal']
        
        if curr_signal == 0:
            # Not in position
            df.iloc[i, df.columns.get_loc('actual_position_size')] = 0.0
        elif prev_signal == 0 and curr_signal == 1:
            # Entry - set position size based on current volatility
            df.iloc[i, df.columns.get_loc('actual_position_size')] = df.iloc[i]['position_size_ratio']
        else:
            # Holding - maintain same position size
            df.iloc[i, df.columns.get_loc('actual_position_size')] = df.iloc[i-1]['actual_position_size']

    # Calculate position changes
    df["position"] = df["signal"].diff()
    
    # Transaction costs
    total_cost = transaction_costs + slippage
    df["trade_cost"] = 0.0
    df.loc[df['position'] != 0, 'trade_cost'] = total_cost

    # Returns scaled by actual position size
    df['market_ret'] = df['Close'].pct_change()
    df['strategy_ret'] = df['market_ret'] * df['signal'].shift(1) * df['actual_position_size'].shift(1)
    df['strategy_ret_net'] = df['strategy_ret'] - df['trade_cost']
    
    # Market portfolio
    df['market_ret_net'] = df['market_ret'].copy()
    df.loc[df.index[0], 'market_ret_net'] = df.loc[df.index[0], 'market_ret'] - total_cost

    # Cumulative returns
    df['cumulative_market'] = (1 + df['market_ret_net']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_ret_net']).cumprod()
    
    # Portfolio value
    df['portfolio_value'] = inv_amt * df['cumulative_strategy']
    df['market_portfolio_value'] = inv_amt * df['cumulative_market']

    # Index handling
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index("Date", drop=True)
        else:
            df.index = pd.to_datetime(df.index)
   
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

    return df
