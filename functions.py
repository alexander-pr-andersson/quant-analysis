import pandas as pd
import numpy as np

def create_new_multilevel_index(name, cols):
    return pd.MultiIndex.from_product([[name], cols])

def flatten_df(df):
    df = df.loc[:, ('Close')].reset_index()
    df.columns = ['Date', 'Close']
    df = df.set_index('Date')
    return df

def calculate_drawdown_full(df):
    df = flatten_df(df)
    df['log_return'] = np.log(df.div(df.shift(1)))
    df = df.dropna()
    df.loc[:, 'cum_return'] = df['log_return'].cumsum().apply(np.exp)
    df['cum_max'] = df['cum_return'].cummax()
    df['drawdown'] = df['cum_max'] - df['cum_return']
    df['drawdown_pct'] = (df['cum_max'] - df['cum_return']) / df['cum_max']

    return df

def calculate_normalized_close(df):
    df = df['Close'].div(df['Close'].iloc[0]).mul(100)
    df.columns = create_new_multilevel_index('Norm_close', df.columns)
    return df

def calculate_pct_change(df):
    df = df.pct_change()
    df.columns = create_new_multilevel_index('Pct_change', df.columns)
    return df

def calculate_daily_return(df):
    df = df.diff()
    df.columns = pd.MultiIndex.from_product([['Change'], df.columns])
    return df

def calculate_log_return(df):
    df = np.log(df.div(df.shift(1)))
    df.columns = pd.MultiIndex.from_product([['log_return'], df.columns])
    return df

def calculate_cum_return(df):
    df.columns = df.columns.droplevel()
    df = df.cumsum().apply(np.exp)
    df.columns = pd.MultiIndex.from_product([['cum_return'], df.columns])
    return df

def calculate_cum_max(df):
    df.columns = df.columns.droplevel()
    df = df.cummax()
    df.columns = pd.MultiIndex.from_product([['cum_max'], df.columns])
    return df

def calculate_drawdown(df1, df2):
    df = df1['cum_max'] - df2['cum_return']
    df.columns = pd.MultiIndex.from_product([['drawdown'], df.columns])
    return df

def calculate_drawdown_percentage(df1, df2):
    df = (df1['cum_max'] - df2['cum_return']) / df1['cum_max']
    df.columns = pd.MultiIndex.from_product([['drawdown_percent'], df.columns])
    return df

def transform_df(df):
    normalized_close_df = calculate_normalized_close(df)
    pct_change_df = calculate_pct_change(df['Close'])
    daily_change_df = calculate_daily_return(df['Close'])
    log_return_df = calculate_log_return(df['Close'])
    cum_return_df = calculate_cum_return(log_return_df.copy())
    cum_max_df = calculate_cum_max(cum_return_df.copy())
    drawdown_df = calculate_drawdown(cum_max_df, cum_return_df)
    drawdown_percentage_df = calculate_drawdown_percentage(cum_max_df, cum_return_df)
    # Merge
    df = df.join(normalized_close_df).join(pct_change_df).join(daily_change_df).join(log_return_df).join(cum_return_df).join(cum_max_df).join(drawdown_df).join(drawdown_percentage_df)
    return df

def retrive_single_ticker(df, ticker):
    return df.xs(ticker, level=1, axis=1)