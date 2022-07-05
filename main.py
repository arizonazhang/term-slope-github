import numpy as np
import pandas as pd

import eikon as ek
from option import Options
from apscheduler.schedulers.blocking import BlockingScheduler


def weekly_update():
    today = pd.to_datetime("today").normalize()
    dayofweek = today.weekday()

    # obtain the last friday and monday correspondingly
    if 4 < dayofweek <= 6:
        # if the program is run on weekends, it would select the friday on the same week
        friday = today - np.timedelta64(dayofweek-4, 'D')
    else:
        # if the program is run on weekdays, it would select the friday on the previous week
        friday = today - np.timedelta64(dayofweek+1+2, 'D')
    monday = friday - np.timedelta64(4, 'D')

    # prepare data
    op = Options(start_date=str(monday)[:10], end_date=str(friday)[:10])
    op.get_tickers() # get option tickers
    op.add_delivery() # get option delivery date
    op.add_risk_free() # get risk free rate
    op.add_option_price() # get option price
    op.get_dividend() # compute dividend yield
    op.get_iv() # compute implied vol
    op.print_data()
    op.print_counter()

    # save spc data
    op.get_spc()
    op.save_daily_spc()
    op.save_weekly_spc()

    # save order book table
    op.save_vol_curve()

    return op


if __name__ == "__main__":
    # make sure refinitiv is logged in before running this job
    ek.set_app_key('51dd084a4cea45dab6ac09cdbfe75cd83d17caa8')

    weekly_update()

    # running every friday midnight
    scheduler = BlockingScheduler()
    try:
        print("Scheduler starts...")
        scheduler.add_job(weekly_update, "cron", day_of_week='sat', hour=1, minute=1, timezone="Asia/Shanghai")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


