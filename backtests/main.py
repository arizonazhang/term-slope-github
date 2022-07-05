from strat import *
import pandas as pd

if __name__ == '__main__':
    # load data
    maturity = 365
    hsi = getData("1001098", maturity=maturity)
    # print(hsi)

    # initialization
    for func in [wchange]:
        # input data
        cerebro = bt.Cerebro()
        datafeed1 = PandasData_extend(dataname=hsi)
        cerebro.adddata(datafeed1, name="hsi")

        # commission & broker setting
        cerebro.broker.setcash(100000.0)
        cerebro.broker.addcommissioninfo(stampDutyCommissionScheme(stamp_duty=0.0, commission=0.0))
        cerebro.broker.set_coc(True) # for checking purpose

        # add traders
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="_Trades")
        cerebro.addanalyzer(bt.analyzers.Transactions)

        # run the strategy
        Vanila_Strat.next = func
        cerebro.addstrategy(Vanila_Strat)
        res = cerebro.run()

        # save return
        ret = pd.Series(res[0].analyzers._TimeReturn.get_analysis())
        ret.to_csv('pnl_{}_{}.csv'.format(func.__name__, maturity))

        tran = res[0].analyzers.transactions.get_analysis()
        tran_df = pd.DataFrame.from_dict({key: values[0] for key, values in tran.items()},
                                      orient='index', columns=['amount', 'price', 'sid', 'symbol', 'value'])

