import csv
import arrow

import numpy as np
import scipy.stats.distributions as ssd

import moresque.ustructures as us

def main(val, t, start_date=None, datafile=None):
    date_format = 'YYYY-MM-DD HH:mm:ss'
    start_date = arrow.get(start_date, date_format)
    f = open(datafile)
    reader = csv.reader(f, delimiter=',')
    curr_row = next(reader)
    curr_date = arrow.get(curr_row[0], date_format)
    while curr_date < start_date:
        curr_row = next(reader)
        curr_date = arrow.get(curr_row[0], date_format)

    err_val = float(curr_row[1])
    max_val = val + (err_val * val)
    min_val = val - (err_val * val)

    f.close()

    return us.Interval(minimum=min_val, maximum=max_val, num=100)
