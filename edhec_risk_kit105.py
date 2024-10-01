import pandas as pd

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns
    computes and returns a data frame that contains:
    the wealth index
    previous peaks
    percent drawdowns
    """
    wealth_index = 1000*(1+ return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks" : previous_peaks,
        "Drawdown" : drawdowns
    })

def get_ffme_returns():
    """
    load the fama french dataset for the returns of the top and bottom deciles by marketcap
    """
    me_m = pd.read_csv("Files/data/Portfolios_Formed_on_ME_monthly_EW.csv",
                    header=0, 
                    index_col=0, 
                    na_values = -99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['smallcap','largecap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format ="%Y%m").to_period('M')
    return rets

def get_hfi_returns():
    """
    load and format the EDHEC hedge fund index returns
    """
    hfi = pd.read_csv("Files/data/edhec-hedgefundindices.csv",
                    header=0, 
                    index_col=0)
    hfi = hfi/100
    hfi.index = pd.to_datetime(hfi.index, format="%d/%m/%Y").to_period('M')
    return hfi

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3


def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(is_normal)
    else:
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level