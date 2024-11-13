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

def semivolatility(s):
    return s[s < 0].std(ddof=0) 

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

def var_historic(s, level=0.5):
    if isinstance(s, pd.DataFrame):
        s.aggregate(var_historic, level=level)
    elif isinstance(s, pd.Series):
        return -np.percentile(s, 100 * level)
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

def var_gaussian(s, level=0.05, CornishFisher=False):
    za = scipy.stats.norm.ppf(level, 0, 1)
    if CornishFisher:
        S, K = scipy.stats.skew(s), (scipy.stats.kurtosis(s) + 3)
        za = za + (za**2 - 1) * S/6 + (za**3 - 3*za) * (K-3) / 24 - (2*za**3 - 5*za) * (S**2) / 36 
    return -(s.mean() + za * s.std(ddof=0))

def cvar_historic(s, level=0.05):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(cvar_historic, level=level)
    elif isinstance(s, pd.Series):
        mask = s < -var_historic(s, level=level)
        return -s[mask].mean()
    else:
        raise TypeError("Expected pd.DataFrame or pd.Series")

# ---------------------------------------------------------------------------------
# Modern Portfolio Theory 
# ---------------------------------------------------------------------------------

def cppi(risky_rets, safe_rets=None, start_value=1000, floor=0.8, m=3, drawdown=None, risk_free_rate=0.03, periods_per_year=12):
    account_value = start_value
    floor_value = floor * account_value

    if isinstance(risky_rets, pd.Series):
        risky_rets = pd.DataFrame(risky_rets, columns="Risky return")

    if safe_rets is None:
        safe_rets = pd.DataFrame().reindex_like(risky_rets)
        safe_rets[:] = risk_free_rate / periods_per_year

    account_history = pd.DataFrame().reindex_like(risky_rets)
    cushion_history = pd.DataFrame().reindex_like(risky_rets)
    risky_w_history = pd.DataFrame().reindex_like(risky_rets)

    if drawdown is not None:
        peak_history  = pd.DataFrame().reindex_like(risky_rets)
        floor_history = pd.DataFrame().reindex_like(risky_rets)
        peak = start_value
        m = 1 / drawdown

    for step in range(len(risky_rets.index)):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak * (1 - drawdown)
            floor_history.iloc[step] = floor_value

        cushion = (account_value - floor_value) / account_value
        risky_w = m * cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w  = 1 - risky_w
        risky_allocation = risky_w * account_value
        safe_allocation  = safe_w  * account_value

        account_value = risky_allocation * (1 + risky_rets.iloc[step]) + safe_allocation * (1 + safe_rets.iloc[step])

        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion 
        risky_w_history.iloc[step] = risky_w

    cppi_rets = (account_history / account_history.shift(1) - 1).dropna()

    backtest_result = {
        "Risky wealth"    : risky_wealth, 
        "CPPI wealth"     : account_history, 
        "CPPI returns"    : cppi_rets, 
        "Cushions"        : cushion_history,
        "Risky allocation": risky_w_history,
        "Safe returns"    : safe_rets
    }

    if drawdown is not None:
        backtest_result.update({
            "Floor value": floor_history,
            "Peaks"      : peak_history,
            "m"          : m
        })

    return backtest_result