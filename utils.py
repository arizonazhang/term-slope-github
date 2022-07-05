import math
from scipy.stats import norm
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import mysql.connector
import time

def bs(S, K, T, r, sigma, q, op_type):
    """
    calculate option price using Black-Scholes formula
    :param S: spot price of underlying
    :param K: strike price
    :param T: time to maturity, unit in years
    :param r: risk-free rate, continuously compounded
    :param sigma: volatility, annual
    :param q: dividend yield, continuously compounded
    :param op_type: "C" or "P"
    :return: option price (in float)
    """
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** (0.5))
    d2 = d1 - sigma * T ** (0.5)
    if op_type == "C":
        return norm.cdf(d1) * S * math.exp(-q * T) - norm.cdf(d2) * K * math.exp(-r * T)
    else:
        return norm.cdf(-d2) * K * math.exp(-r * T) - norm.cdf(-d1) * S * math.exp(-q * T)


def iv_bisection(S, K, T, r, q, p, op_type, verbose=False):
    """
    solve for implied volatility using bi-section method, Not in use
    :param S: spot price of underlying
    :param K: strike price
    :param T: time to maturity, unit in years
    :param r: risk-free rate, continuously compounded
    :param q: dividend yield, continuously compounded
    :param p: option price
    :param op_type: "C" or "P"
    :param verbose: whether to print volatility range in each round
    :return:
    """
    rng = (0.0001, 1)
    mid = bs(S, K, T, r, (rng[0] + rng[1]) / 2, q, op_type)
    i = 0
    while abs(p - mid) >= 10 ** -7:
        if p < mid:
            rng = (rng[0], (rng[0] + rng[1]) / 2)
        else:
            rng = ((rng[0] + rng[1]) / 2, rng[1])
        mid = bs(S, K, T, r, (rng[0] + rng[1]) / 2, q, op_type)
        if verbose:
            print("range is {:.2f} and {:.2f}".format(rng[0], rng[1]))
        i += 1
    return (rng[0] + rng[1]) / 2, i


def _vega(S, K, T, r, sigma, q):
    """
    compute vega of an option. This is later used for newton's approach in finding implied vol
    :param S: spot price of underlying
    :param K: strike price
    :param T: time to maturity, unit in years
    :param r: risk-free rate, continuously compounded
    :param sigma: volatility, annual
    :param q: dividend yield, continuously compounded
    :return:
    """
    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * T ** (0.5))
    return math.exp(-q * T) * S * math.sqrt(T) * norm.cdf(d1)


def iv_newton(S, K, T, r, q, p, op_type, verbose=False, max_iter=1000):
    """
    solve for implied volatility using newton's method
    :param S: spot price of underlying
    :param K: strike price
    :param T: time to maturity, unit in years
    :param r: risk-free rate, continuously compounded
    :param q: dividend yield, continuously compounded
    :param p: option price
    :param op_type: "C" or "P"
    :param verbose: whether to print volatility range in each round
    :param max_iter: maximum iterations before stopping the search
    """
    if op_type == "C":
        rng = np.sqrt(2 * math.pi / T) * p / S
    else:
        c = S + p - K
        rng = np.sqrt(2 * math.pi / T) * c / S

    mid = bs(S, K, T, r, rng, q, op_type)
    i = 0
    while abs(p - mid) >= 10 ** -7 and i < max_iter:
        v = _vega(S, K, T, r, rng, q)
        rng = rng - (mid - p) / v
        i += 1
        mid = bs(S, K, T, r, rng, q, op_type)
        if verbose:
            print("solution: {} with vega: {}".format(rng, v))
    return rng, i


def pc(sigma_c, sigma_p, T):
    """
    calculate put-call ratio
    :param sigma_c: volatility of call
    :param sigma_p: volatility of put
    :param T: time to maturity
    :return:
    """
    d1 = sigma_c / 2 * math.sqrt(T)
    d2 = sigma_p / 2 * math.sqrt(T)
    v1 = norm.cdf(d1) - norm.cdf(-d1)
    v2 = norm.cdf(d2) - norm.cdf(-d2)
    return v1 / v2 - 1


def getData(code, start_dt="2006-01-03", end_dt="2021-12-31"):
    """get HSI price data from jydb"""
    query = """
    select ClosePrice as close, LowPrice as low, HighPrice as high, OpenPrice as open, TurnoverValue as volume, 0 as openinterest, TradingDay as datetime
    from jydb.QT_OSIndexQuote where indexCode = %s and tradingday between %s and %s;
    """
    engine = create_engine("mysql+mysqlconnector://infoport:HKaift-123@192.168.2.81/AlternativeData")
    # read in index quotes
    data = pd.read_sql(query, engine, params=[code, start_dt, end_dt])
    data = data.set_index("datetime")
    data = data.fillna(0)
    return data


def replace_into_mysql(query, table_name, data):
    """run the query to insert/replace data in the table"""
    config = {
        'user': 'infoport',
        'password': 'HKaift-123',
        'host': '192.168.2.81',
        'database': 'AlternativeData',
        'raise_on_warnings': False
    }

    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor()
    cnt = 0
    for i in range(len(data.index)):
        data_query = data.iloc[i,:].to_dict()
        try:
            cursor.execute(query, data_query)
        except Exception as e:
            print(e)
            cnt += 1
    cnx.commit()
    cursor.close()
    cnx.close()
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Uploaded {len(data.index)-cnt}/{len(data.index)} records into table [{table_name}]. ", end="")
    print_local_time()


def print_local_time():
    """print local time, formatted"""
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"Time: {t}")