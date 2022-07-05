import numpy as np
import pandas as pd

from pandas.tseries.offsets import BMonthEnd
from utils import iv_newton, pc, getData, print_local_time, replace_into_mysql
import pymongo
import pandas_market_calendars as mcal
import eikon as ek

from apscheduler.schedulers.blocking import BlockingScheduler

def get_month_code(x, code_ls):
    """find month code given a list of specifications in x, used in dataframe apply"""
    if x['month'] <= 12 - x['forward']:
        return code_ls[int(x['month'] + x['forward']) - 1]
    else:
        return code_ls[int(x['month'] + x['forward']) - 13]


class Options:
    call_code = "ABCDEFGHIJKL"
    put_code = "MNOPQRSTUVWX"

    def __init__(self, start_date, end_date):
        self.nrequest = 0  # request counter
        self.ndata = 0 # data point counter
        self.start_dt = start_date
        self.end_dt = end_date
        self.codes = None # main dataframe containing option prices and parameters for computation
        self.dailyspc = None

    def get_api(self, func, rics, **args):
        """
        query data from api, add number of request and data points downloaded
        :param func: ek query function to use (ek.get_data or ek.get_timeseries)
        :param rics: tickers of requested securities
        :param args: other arguments to pass into the func
        :return:
        """
        try:
            data = func(rics, **args)
            if func == ek.get_data:
                data, err = data
                if err:
                    print(err)
            elif func == ek.get_timeseries:
                data.index = pd.to_datetime(data.index)

                if len(rics) == 1:
                    data.columns = rics
                if len(data.columns) != len(rics):
                    raise KeyError

            self.nrequest += 1
            self.ndata += data.shape[0]*data.shape[1]
            return data

        except KeyError:
            print("Error: some indices not available")
        except ValueError:
            print("Error: a parameter type or value is wrong")
        except Exception:
            print("Error: request failed")

    def get_tickers(self):
        """get a list of ATM option tickers (RIC tickers) covering different maturity"""
        hsi = getData("1001098", self.start_dt, self.end_dt)
        hsi['mid'] = hsi.loc[:, ['close', 'low', 'high', 'open']].mean(axis=1)
        hsi['strike'] = hsi['mid']
        hsi['month'] = hsi.index.month.astype(int)

        # add a "forward" column: denote no. of month before expiry
        codes = hsi.groupby(['datetime']).apply(lambda x: pd.Series(range(1, 13))).unstack()
        codes.name = "forward"
        codes = pd.merge(codes, hsi[['month', 'strike', 'mid', 'close']], left_index=True, right_index=True)

        # add a "strike" column: strike price of atm option = mid price round to the nearest 200
        codes['strike'] = codes['mid'].apply(lambda x: x - x % 200 if x % 200 < 100 else x + 200 - x % 200)

        # add a "strike" column: strike price of option second closest to atm: mid price round to other side of 200
        codes['strike_rev'] = codes['mid'].apply(lambda x: x - x % 200 if x % 200 >= 100 else x + 200 - x % 200)

        # add "month_code" column for call and put
        # (the single alphabetic number denoting expiry month in contract ticker)
        codes['month_code_call'] = codes.apply(get_month_code, args=(self.call_code,), axis=1)
        codes['month_code_put'] = codes.apply(get_month_code, args=(self.put_code,), axis=1)

        # add "year_code" column
        # (single number denoting expiry year in contract ticker)
        codes['year_code'] = codes.apply(lambda x: "2" if x['month'] <= 12 - x['forward'] else "3", axis=1)

        # construct four contract tickers
        codes['ric_call'] = codes.apply(
            lambda x: "HSI" + f"{x['strike']:.0f}" + x['month_code_call'] + x['year_code'] + ".HF", axis=1)
        codes['ric_put'] = codes.apply(
            lambda x: "HSI" + f"{x['strike']:.0f}" + x['month_code_put'] + x['year_code'] + ".HF", axis=1)
        codes['ric_call_rev'] = codes.apply(
            lambda x: "HSI" + f"{x['strike_rev']:.0f}" + x['month_code_call'] + x['year_code'] + ".HF", axis=1)
        codes['ric_put_rev'] = codes.apply(
            lambda x: "HSI" + f"{x['strike_rev']:.0f}" + x['month_code_put'] + x['year_code'] + ".HF", axis=1)

        # remove tickers whose corresponding contract do not exist
        # option contracts are only available for the next 3 calendar month and quarter months (3, 6, 9, 12)
        codes = codes.drop(codes[(codes.month % 3 == 0) & (codes.forward.isin([4, 5, 7, 8, 10, 11]))].index)
        codes = codes.drop(codes[(codes.month % 3 == 1) & (codes.forward.isin([3, 4, 6, 7, 9, 10, 12]))].index)
        codes = codes.drop(codes[(codes.month % 3 == 2) & (codes.forward.isin([3, 5, 6, 8, 9, 11, 12]))].index)
        codes = codes.reset_index(0)

        del codes['level_0']
        del codes['year_code']
        del codes['month_code_call']
        del codes['month_code_put']

        self.codes = codes

    def add_delivery(self):
        # find delivery date of the options - business month end of the expiry month
        monthend = lambda x: pd.to_datetime(x.name.strftime("%Y%m"), format='%Y%m') + BMonthEnd(x['forward'] + 1)
        self.codes['delivery_date'] = self.codes.apply(monthend, axis=1)

        # load trading calendar to determine trading day before expiry
        hkex = mcal.get_calendar('HKEX')
        tradingdays = pd.date_range(start=self.start_dt, end=self.codes['delivery_date'].max(), freq=hkex.holidays())

        # calculate T (in unit of one year)
        countT = lambda x: np.sum((tradingdays > x.name) & (tradingdays <= x['delivery_date']))
        self.codes['T'] = self.codes.apply(countT, axis=1) / 252

        # filter out expired options (no historical data available) 
        self.codes = self.codes.loc[self.codes.delivery_date >= pd.to_datetime('today')]

    def add_risk_free(self):
        # get hibor data from Refinitiv
        rics = ["HIHKD1MD=", "HIHKD3MD=", "HIHKD6MD=", "HIHKD1YD="]
        rf = self.get_api(ek.get_timeseries, rics,
                          start_date=self.start_dt, end_date=self.end_dt, fields='CLOSE', debug=True)
        rf.columns = [1, 3, 6, 12]

        # build yield curve using cubic interpolation
        rf.loc[:, [2, 4, 5, 7, 8, 9, 10, 11]] = np.nan
        rf = rf.apply(lambda x: pd.DataFrame(x.astype(float), index=rf.columns).interpolate('cubic').iloc[:, 0], axis=1)

        rf = rf.unstack().reset_index()
        rf.columns = ['rfm', 'datetime', 'r']

        # convert to continuous compounding
        rf['r'] = np.log(1 + rf['r'] / 100)

        # join codes df with risk free rate
        self.codes['rfm'] = self.codes['forward']
        self.codes = self.codes.reset_index().set_index(['datetime', 'rfm']).join(rf.set_index(['datetime', 'rfm']))
        self.codes = self.codes.reset_index()

        del self.codes['rfm']

    def add_option_price(self):
        date_list = self.codes.datetime.unique()
        for key in ['ric_call', 'ric_put', 'ric_call_rev', 'ric_put_rev']:
            ls = []
            # for each date, get option settlment price
            for spot_date in date_list:
                rics = self.codes.loc[self.codes.datetime == spot_date, key].to_list()
                spot_date_str = np.datetime_as_string(spot_date, unit='D')
                close = self.get_api(ek.get_data, rics,
                                fields=['TR.SETTLEMENTPRICE.date', 'TR.SETTLEMENTPRICE.value'],
                                parameters={'SDate': spot_date_str, 'EDate': spot_date_str, 'Frq': 'D', 'FILL': 'PREVIOUS'})
                close['Date'] = spot_date
                ls.append(close)

            # join with the dataframe
            close = pd.concat(ls)
            new_col_name = "c" if "call" in key else "p"
            new_col_name += "_rev" if "rev" in key else ""
            close.columns = [key, 'datetime', new_col_name]
            self.codes = pd.merge(self.codes, close, on=[key, 'datetime'], how='inner')

    def get_dividend(self):
        # compute implied dividend yield from options second closest to atm
        div1 = self.codes.apply(lambda x: -1 / x['T'] * np.log(
            (x['c_rev'] - x['p_rev'] + x['strike_rev'] * np.exp(-x['r'] * x['T'])) / x['close']), axis=1)

        # compute implied dividend yield from atm options
        div2 = self.codes.apply(lambda x: -1 / x['T'] * np.log(
            (x['c'] - x['p'] + x['strike'] * np.exp(-x['r'] * x['T'])) / x['close']), axis=1)

        # take average the above two
        self.codes['q'] = (div1 + div2) / 2

    def get_iv(self):
        # compute implied volatility using newtons methods
        self.codes['iv_call'] = self.codes.apply(
            lambda x: iv_newton(x['close'], x['strike'], x['T'], x['r'], x['q'], x['c'], op_type="C")[0], axis=1)
        self.codes['iv_put'] = self.codes.apply(
            lambda x: iv_newton(x['close'], x['strike'], x['T'], x['r'], x['q'], x['p'], op_type="P")[0], axis=1)

    def get_spc(self):
        # compute put-call ratio of each pair
        self.codes['pc'] = self.codes.apply(lambda x: pc(x['iv_call'], x['iv_put'], x['T']), axis=1)

        # compute spc of each day
        grouped = self.codes.groupby('datetime')
        spc_ratio = grouped.apply(lambda s: np.cov(s[['pc', 'T']], rowvar=False, bias=True)[1, 0] / np.var(s['T']))
        self.dailyspc = spc_ratio

    def save_daily_spc(self):
        # reformat the dataframe
        spc_ratio = self.dailyspc.reset_index()
        spc_ratio.columns = ['date', 'spc']
        spc_ratio['date'] = spc_ratio['date'].dt.strftime('%Y-%m-%d')

        # upload to database
        query = 'replace into DailySPC (date, spc) values (%(date)s, %(spc)s)'
        replace_into_mysql(query, 'DailySPC', spc_ratio)

    def save_weekly_spc(self):
        # reformat the dataframe
        _spc = self.dailyspc.reset_index()
        _spc.columns = ['date', 'ratio']
        _spc = _spc.sort_values('date')

        # calculate weekly average
        idyear = _spc.date.dt.isocalendar().year
        idweek = _spc.date.dt.isocalendar().week
        weekly_ratio = _spc.groupby([idyear, idweek]).agg({'date': lambda x: x.tail(1), 'ratio': 'mean'})

        # convert each timestamp to the friday to the respective week
        weekly_ratio['date'] = weekly_ratio['date'].apply(lambda x: x + np.timedelta64(4 - x.weekday(), 'D'))
        weekly_ratio['date'] = weekly_ratio['date'].dt.strftime("%Y-%m-%d")

        # upload to mongodb database
        myclient = pymongo.MongoClient("mongodb://app_developer:hkaift123@192.168.2.85:4010/")
        db = myclient["app_data"]
        coll = db["WeeklySPC"]
        data_dict = weekly_ratio.to_dict('records')
        try:
            query = coll.insert_many(data_dict)
            print(f"Upload [weekly_spc] successful. {len(data_dict)} records", end="")
        except:
            print("Upload [weekly_spc] failed. (Probably due to duplicates or connection error) ", end="")
        print_local_time()

    def get_vol_curve(self):
        # compile the vol curve dataframe
        call = self.codes.pivot(index='datetime', values='iv_call', columns='forward')
        put = self.codes.pivot(index='datetime', values='iv_put', columns='forward')
        call['type'] = "c"
        put['type'] = "p"
        data = pd.concat([call, put]).sort_index().reset_index()

        data.columns = [f"{col:.0f}M" if col not in ['datetime', 'type'] else col for col in data.columns]
        data['datetime'] = data['datetime'].dt.strftime("%Y-%m-%d")
        return data

    def save_vol_curve(self):
        data = self.get_vol_curve()

        # save to sql
        query = "replace into HSIVolCurve (date, {c}) values (%(datetime)s, {d})"
        query = query.format(c=",".join(data.columns[1:]), d=",".join([f"%({col})s" for col in data.columns[1:]]))
        replace_into_mysql(query, 'HSIVolCurve', data)

    def print_data(self):
        print("-------Overview of all data-------")
        print(self.codes.head())
        print("")

    def add_request(self):
        self.nrequest += 1

    def add_data(self, points):
        self.ndata += points

    def print_counter(self):
        print("-------API download counts--------")
        print(f"No. of requests made: {self.nrequest}")
        print(f"No. of data points downloaded: {self.ndata}")
        print("")


