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

def compute_logreturns(s):
    if isinstance(s, pd.DataFrame):
        return s.aggregate(compute_logreturns)
    elif isinstance(s, pd.Series):
        return np.log(s / s.shift(1))
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

# ---------------------------------------------------------------------------------
# Random walks
# ---------------------------------------------------------------------------------

def simulate_gbm_from_returns(n_years=10, n_scenarios=20, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0):
    '''
    (S_{t+dt} - S_t) / S_t = mu * dt + sigma * sqrt(dt) * xi
    where xi is a normal random variable in N(0, 1)
    '''
    dt = 1 / periods_per_year
    n_steps = int(n_years * periods_per_year)
    rets = pd.DataFrame( np.random.normal(loc=mu*dt, scale=sigma*(dt**0.5), size=(n_steps, n_scenarios)) )
    prices = compound_returns(rets, start=start)
    prices = insert_first_row_df(prices, start)

    return prices, rets

def simulate_gbm_from_prices(n_years=10, n_scenarios=20, mu=0.07, sigma=0.15, periods_per_year=12, start=100.0):
    '''
    S_t = S_0 * exp{ (mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * xi }
    where xi is a normal random variable in N(0, 1)
    '''
    dt = 1 / periods_per_year
    n_steps = int(n_years * periods_per_year)
    factor_dt = np.exp( np.random.normal(loc=(mu - 0.5*sigma**2)*dt, scale=sigma*(dt**(0.5)), size=(n_steps, n_scenarios)) )
    prices = start * pd.DataFrame(factor_dt).cumprod()
    prices = insert_first_row_df(prices, start)
    rets = compute_logreturns(prices).dropna()

    return prices, rets

# ---------------------------------------------------------------------------------
# CIR model
# ---------------------------------------------------------------------------------

def discount(t, r):
    if not isinstance(t, pd. Series):
        t = pd.Series(t)
    if not isinstance(r, list):
        r = np.array([r])
    df = pd.DataFrame(1 / (1 + r) ** t)
    df.index = t
    return df

def present_value(s, r):
    if not isinstance(s, pd.DataFrame):
        raise TypeError("Expected pd.DataFrame")
    dates = pd.Series(s.index)
    return (discount(dates, r) * s).sum()

def funding_ratio(asset_value, liabilities, r):
    return asset_value / present_value(liabilities, r)

def compounding_rate(r, periods_per_year=None):
    '''
    Given a nominal rate r, it returns the continuously compounded rate R = e^r - 1 if periods_per_year is None.
    If periods_per_year is not None, then returns the discrete compounded rate R = (1 + r / N) ** N - 1.
    '''
    if periods_per_year is None:
        return np.exp(r) - 1
    else:
        return (1 + r / periods_per_year) ** periods_per_year - 1

def compounding_rate_inv(R, periods_per_year=None):
    '''
    Given a compounded rate R, it returns the nominal rate r from continuously compounding 
    r = log(1 + R) if periods_per_year is None
    If periods_per_year is not None, then returns the nominal rate from discrete 
    compounding r = N * [(1 + R) ^ (1 / N) - 1]
    '''
    if periods_per_year is None:
        return np.log(1 + R)
    else:
        return periods_per_year * ( (1 + R) ** (1 / periods_per_year) - 1 )

def simulate_cir(n_years=10, n_scenarios=10, a=0.05, b=0.03, sigma=0.05, periods_per_year=12, r0=None):
    if r0 is None:
        r0 = b
    
    def zcbprice(ttm, r, h):
        A = ( 2*h*np.exp((a+h)*ttm/2) / (2*h + (a+h)*(np.exp(h*ttm)-1)) ) ** (2 * a * b / sigma ** 2)
        B = 2*(np.exp(h*ttm)-1) / (2*h + (a+h)*(np.exp(h*ttm)-1)) 
        return A * np.exp(-B * r)

    dt = 1 / periods_per_year
    n_steps = int(n_years * periods_per_year) + 1

    r0 = compounding_rate_inv(r0)

    xi = np.random.normal(loc=0, scale=dt**0.5, size=(n_steps, n_scenarios))

    rates = np.zeros_like(xi)
    rates[0] = r0

    zcb_prices = np.zeros_like(xi)
    h = np.sqrt(a**2 + 2 * sigma**2)
    zcb_prices[0] = zcbprice(n_years, r0, h)

    for step in range(1, n_steps):
        r_t = rates[step - 1]
        rates[step] = r_t + a * (b - r_t) + sigma * np.sqrt(r_t) * xi[step]
        zcb_prices[step] = zcbprice(n_years - dt * step, r_t, h)

    return rates, zcb_prices


# ---------------------------------------------------------------------------------
# Mortgage
# ---------------------------------------------------------------------------------

def pmt(r, n, pv, fv):
    """
    Calculates the payment for a loan based on constant payments and a constant interest rate.
    See https://support.microsoft.com/en-us/office/pmt-function-f30c80b4-7710-4959-b10b-498c3a5a8a55

    Input:
    r: interest rate for the loan
    n: total number of payments for the loan
    pv: present value, or principal. the total amount that a series of future payments is worth now
    fv: future value. a cash balance you want to attain after the last payment is made. It is assumed to be 0

    Output:
    Monthly payment
    """
    return r / ((1 + r)^n - 1) * (pv * (1 + r)^n - fv)


# ---------------------------------------------------------------------------------
# Pandas methods
# ---------------------------------------------------------------------------------

def insert_first_row_df(df, row):
    df.loc[-1] = row
    df.index = df.index + 1
    return df.sort_index()

def mac_duration(cash_flows, discount_rate):
    pass

def ldi_mixer(psp_rets, lhp_rets, allocator):
    if psp_rets.shape == lhp_rets.shape:
        raise ValueError

def ldi_fixed_allocator():
    pass

def ldi_glidepath_allocator():
    pass

def ldi_floor_allocator():
    pass

def ldi_drawdown_allocator():
    pass