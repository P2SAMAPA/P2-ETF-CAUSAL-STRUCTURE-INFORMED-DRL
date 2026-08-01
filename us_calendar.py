import pandas as pd
from pandas_market_calendars import get_calendar


def get_us_trading_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    nyse = get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    return schedule.index
