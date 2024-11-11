import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def compound_returns(s, start=100):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compound_returns, start=start)
    elif isinstance(s, pd.Series):
        return start * (1 + s).cumprod()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def compute_returns(s):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compute_returns)
    elif isinstance(s, pd.Series):
        return s / s.shift(1) - 1
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def annualize_returns(s, periods_per_year):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_rets, periods_per_year=periods_per_year)
    elif isinstance(s, pd.Series):
        return ((1 + s).prod()) ** (periods_per_year / s.shape[0]) - 1

def annualize_volatility(s, periods_per_year, ddof=1):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(annualize_volatility, periods_per_year=periods_per_year)
    elif isinstance(s, pd.Series):
        return s.std(ddof=ddof) * (periods_per_year)**(0.5)
    elif isinstance(s, list):
        return np.std(s, ddof=ddof) * (periods_per_year)**(0.5)
    elif isinstance(s, (int, float)):
        return s * (periods_per_year)**(0.5)

def drawdown(s):
    peaks = s.cummax()
    drawdown = (s - peaks) / peaks
    return pd.DataFrame({"Value": s, "Peaks": peaks, "Drawdown": drawdown})

def annualized_sharpe_ratio(s, risk_free_rate, periods_per_year, volatility=None):
    if isinstance(s, pd.DataFrame):
        s.aggregate(annualized_sharpe_ratio, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year, volatility=volatility)
    elif isinstance(s, pd.Series):
        rfr_to_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        ann_ex_ret = annualize_returns(s - rfr_to_period, periods_per_year)
        ann_vol = annualize_volatility(s, periods_per_year)
        return ann_ex_ret / ann_vol
    elif isinstance(s, (int, float)) and volatility is not None:
        return (s - risk_free_rate) / volatility