# Move roughly up to line 74 of dataset.py to here.
# Many small functions few 2--3 inputs and informative names.
# Most of these functions should take config as input, which will direct functions
# to correct data files.
# The output should be two pandas DataFrames with column names that we want
# dynamic_features of shape (num_dynamic_fields, num_stations, num_days)
# static_features of shape (num_static_fields, num_stations)
# fields = {Elevation, Slope, ... , SWE, Station_name}
