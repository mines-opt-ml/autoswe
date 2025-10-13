import pandas as pd

def compute_day_of_snow_year(date: pd.Timestamp, beginning_of_snow_year: str) -> int:
    """
    Convert a calendar date to a continuous DayOfSnowYear (DOSY)
    given the snow-year start (e.g. '10-01' for October 1st).

    """
    start_month, start_day = map(int, beginning_of_snow_year.split("-"))
    start_of_snow_year = pd.Timestamp(year=date.year, month=start_month, day=start_day)

    if date < start_of_snow_year:
        start_of_snow_year = start_of_snow_year - pd.DateOffset(years=1)

    return (date - start_of_snow_year).days + 1