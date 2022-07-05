import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import errorcode
from scipy.stats import norm
import backtrader as bt
import backtrader.indicators as btind


def connectDB():
    config = {
        'user': 'infoport',
        'password': 'HKaift-123',
        'host': '192.168.2.81',
        'database': 'AlternativeData',
        'raise_on_warnings': False
    }
    try:
        cnx = mysql.connector.connect(**config)
        return cnx
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(err)
        return 0


def notify_order_common(obj, order):
    """print order info"""
    if order.status in [order.Submitted, order.Accepted]:
        return
    elif order.status in [order.Completed]:
        if order.isbuy():
            obj.log("-- BUY Executed, name: {}, price: {:.2f}, cost: {:.2f}, comm: {:.2f}".format(order.data._name,
                                                                                                   order.executed.price,
                                                                                                   order.executed.value,
                                                                                                   order.executed.comm))
            value = obj.getpositionbyname(order.data._name).size * obj.getpositionbyname(order.data._name).price
            print("order size: {:.2f}".format(order.size))
        else:
            obj.log("-- SELL Executed, name: {}, price: {:.2f}, cost: {:.2f}, comm: {:.2f}".format(order.data._name,
                                                                                                    order.executed.price,
                                                                                                    order.executed.value,
                                                                                                    order.executed.comm))
            value = obj.getpositionbyname(order.data._name).size * obj.getpositionbyname(order.data._name).price
            print("order size: {:.2f}".format(order.size))
    elif order.status in [order.Margin, order.Cancelled]:
        print("Order failed. Error Code: {:.0f}".format(order.status))
        

def getData(code, start_dt="2006-01-03", end_dt="2021-12-31", add_spc=True, add_rsq=False, maturity=365):
    """this is annotations"""
    query = """
    select ClosePrice as close, LowPrice as low, HighPrice as high, OpenPrice as open, TurnoverValue as volume, 0 as openinterest, TradingDay as datetime
    from jydb.QT_OSIndexQuote where indexCode = %s and tradingday between %s and %s;
    """
    cnx = connectDB()

    # read in index quotes
    data = pd.read_sql(query, cnx, params=[code, start_dt, end_dt])
    data = data.set_index("datetime")

    if add_spc:
        # read in spc ratio
        data_spec = pd.read_csv('impvol_data.csv')
        data_spec = data_spec[['Date', 'Days', 'ImpliedVol', 'CallPut', 'Delta']]

        def process(df):
            df = df.copy()
            df.loc[:, 'Maturity'] = df.loc[:, 'Days'] / 365
            df.loc[:, 'Date'] = pd.to_datetime(df.loc[:, 'Date'])
            df = df.set_index("Date")
            return df

        call = process(data_spec.loc[(data_spec['Delta'] == 50) & (data_spec['Days'] <= maturity), :])
        put = process(data_spec.loc[(data_spec['Delta'] == -50) & (data_spec['Days'] <= maturity), :])
        call['d1'] = call['ImpliedVol'] / 2 * call["Maturity"] ** 0.5
        call['v'] = call.d1.apply(lambda x: norm.cdf(x) - norm.cdf(-x))

        put['d1'] = put['ImpliedVol'] / 2 * put["Maturity"] ** 0.5
        put['v'] = put.d1.apply(lambda x: norm.cdf(x)-norm.cdf(-x))

        ratio = pd.concat([put['v'] / call['v'] - 1, put['Maturity']], axis=1)
        spc = ratio.groupby("Date").apply(lambda s: np.cov(s[['v', 'Maturity']], rowvar=False, bias=True)[1, 0] / np.var(
            s['Maturity']))  # bias = True so that cov and var are all divided by N and matched the OLS formula
        spc.name = "spc"
        data = data.join(spc)
        data.spc = data.spc.fillna(method="ffill")

        if add_rsq:
            import scipy.stats as sp
            rsq = ratio.groupby("Date").apply(lambda s: sp.linregress(s['Maturity'], s['v']).rvalue ** 2)
            rsq.name = "rsq"
            data = data.join(rsq)

    data = data.fillna(0)

    return data


def wchange_gen(neg_pos, name):
    def func(obj):
        if len(obj.data0) % obj.p.n == obj.p.lag:
            signal = 1 if obj.ma[0] < obj.ma[-obj.p.n] else neg_pos  # long-only
            if obj.broker.getposition(obj.data0).size != 0:
                position = obj.broker.getposition(obj.data0).size / abs(obj.broker.getposition(obj.data0).size)
            else:
                position = 0
            print('===== Date: {}, Signal: {} ====='.format(obj.data.datetime.date(0), signal))
            print('Total cash: {:.2f}; value: {:.2f}; '.format(obj.broker.getcash(), obj.broker.getvalue()))
            print(
                'holding HSI of {:.2f} at {:.2f}'.format(obj.broker.getposition(obj.data0).size, obj.data0.close[0]))
            if signal != position or len(obj.data0) == 1:
                s = obj.broker.getvalue() * obj.p.cash_buf
                print("Trade target value: {:.2f}".format(s * signal))
                obj.order = obj.order_target_value(data=obj.data0, target=signal * s)
    func.__name__ = name
    return func


wchange = wchange_gen(neg_pos=0, name='wchange')
wchange_ls = wchange_gen(neg_pos=-1, name='wchange_ls')


class PandasData_extend(bt.feeds.PandasData):
    lines = ('spc',)
    params = (('spc', -1), )


class Vanila_Strat(bt.Strategy):
    params = (('n', 5), ('lag', 0), ('cash_buf', 0.95),)

    def log(self, txt, dt=None, isprint=True):
        if isprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt}, {txt}")

    def notify_order(self, order):
        notify_order_common(self, order)

    def __init__(self):
        print("this is the start of the strat.")
        self.ma = btind.SimpleMovingAverage(self.data0.spc, period=self.p.n)

    def next(self):
        pass


################## CLASS below are NOT IN USE #######################################

class stampDutyCommissionScheme(bt.CommInfoBase): ##TODO: inherit CommInfoBase Class?
    params = (
        ('stamp_duty', 0.001),
        ('commission', 0.002),
        ('stocklike', True),
        ('percabs', True),
        ('commtype', bt.CommInfoBase.COMM_PERC)
    )
    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return size * price * self.p.commission
        elif size < 0:
            return -size * price * (self.p.stamp_duty + self.p.commission)
        else:
            return 0
    def getsize(self, price, cash):
        return self.p.leverage * (cash/price)

class Strat_Simple(bt.Strategy):
    params = (('n', 1), ('lag', 0), ('cash_buf', 0.95),)
    def log(self, txt, dt = None, isprint = True):
        if isprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt}, {txt}")

    def notify_order(self, order):
        notify_order_common(self, order)

    def __init__(self):
        print("this is the start of the strat.")
        self.ma = btind.SimpleMovingAverage(self.data0.spc, period=self.p.n)
        self.signal = btind.If(self.data0.spc<0, 1, -1)

    def next(self):
        if len(self.data0) % self.p.n == self.p.lag:
            print('===== Date: {}, Signal: {} ====='.format(self.data.datetime.date(0), self.signal[0]))
            print('Total cash: {:.2f}; value: {:.2f}; '.format(self.broker.getcash(), self.broker.getvalue()))
            print(
                'holding HSI of {:.2f} at {:.2f}'.format(self.broker.getposition(self.data0).size, self.data0.close[0]))
            if self.signal[0] != self.signal[-self.p.n] or len(self.data0) == 1:
                s = self.broker.getvalue() * self.p.cash_buf
                print("Trade target value: {:.2f}".format(s*self.signal[0]))
                self.order = self.order_target_value(data=self.data0, target=self.signal[0] * s)


class Strat_MA(bt.Strategy):
    params = (('n', 20), ('lag', 0), ('cash_buf', 0.90),)
    def log(self, txt, dt = None, isprint = True):
        if isprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt}, {txt}")

    def notify_order(self, order):
        notify_order_common(self, order)

    def __init__(self):
        print("this is the start of the strat.")
        self.ma = btind.SimpleMovingAverage(self.data0.spc, period=20)
        self.signal = btind.If(self.ma<0, 1, -1)

    def next(self):
        if len(self.data0) % self.p.n == self.p.lag:
            print('===== Date: {}, Signal: {} ====='.format(self.data.datetime.date(0), self.signal[0]))
            print('Total cash: {:.2f}; value: {:.2f}; '.format(self.broker.getcash(), self.broker.getvalue()))
            print(
                'holding HSI of {:.2f} at {:.2f}'.format(self.broker.getposition(self.data0).size, self.data0.close[0]))
            if self.signal[0] != self.signal[-self.p.n] or len(self.data0) == 1:
                s = self.broker.getvalue() * self.p.cash_buf
                print("Trade target value: {:.2f}".format(s*self.signal[0]))
                self.order = self.order_target_value(data=self.data0, target=self.signal[0] * s)

class Strat_Change(bt.Strategy):
    params = (('n', 1), ('lag', 0), ('cash_buf', 0.95),)
    def log(self, txt, dt = None, isprint = True):
        if isprint:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"{dt}, {txt}")

    def notify_order(self, order):
        notify_order_common(self, order)

    def __init__(self):
        print("this is the start of the strat.")

    def next(self):
        if len(self.data0) % self.p.n == self.p.lag:
            signal = 1 if self.data0.spc[0] > self.data0.spc[-1] else 0
            if self.broker.getposition(self.data0).size != 0:
                position = self.broker.getposition(self.data0).size / abs(self.broker.getposition(self.data0).size)
            else:
                position = 0
            print('===== Date: {}, Signal: {} ====='.format(self.data.datetime.date(0), signal))
            print('Total cash: {:.2f}; value: {:.2f}; '.format(self.broker.getcash(), self.broker.getvalue()))
            print(
                'holding HSI of {:.2f} at {:.2f}'.format(self.broker.getposition(self.data0).size, self.data0.close[0]))
            if signal != position or len(self.data0) == 1:
                s = self.broker.getvalue() * self.p.cash_buf
                print("Trade target value: {:.2f}".format(s*signal))
                self.order = self.order_target_value(data=self.data0, target=signal * s)

