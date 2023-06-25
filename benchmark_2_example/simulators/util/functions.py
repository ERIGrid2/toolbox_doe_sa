# Copyright (c) 2021 by ERIGrid 2.0. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

def clamp(a, x, b):
    """
    Ensures x lies in the closed interval [a, b]
    :param a:
    :param x:
    :param b:
    :return:
    """
    if x > a:
        if x < b:
            return x
        else:
            return b
    else:
        return a

def clamp2(a, x, b):
    """
    Ensures x lies in the closed interval [a, b],
    :param a:
    :param x:
    :param b:
    :returns: tuple (result, -1/0/1)
    """
    if x > a:
        if x < b:
            return (x, 0)
        else:
            return (b, 1)
    else:
        return (a, -1)

def interpolator(x):
    return 3*x*x - 2*x**3


def lininterp(Tlo, T, Thi):
    '''
        returns y \in [0,1] such that
        T < Tlo => y=1
        Thi < T => y=0
        with the interpolated value for
        Tlo <= T <= Thi
    '''
    return clamp(0, (Thi-T)/(Thi-Tlo), 1)


def log_mean(T_hi, T_lo, exact=False):
    if exact:
        from numpy import log
        return (T_hi-T_lo)/log(T_hi/T_lo)
    else:
        d = T_hi - T_lo
        return T_hi - d/2*(1 + d/6/T_hi*(1 + d/2/T_hi))  # third order taylor expansion


def safediv(a, b, tol=1e-6):
    if b == 0:
        return 0
    else:
        return a/b


def get_electricity_price_at_time(t):
    from math import sin, pi, exp, cos
    from numpy import cos
    # TODO: implement with regression on prices, possibly optimize later
    return 40*(1.0 + 0.2*cos(2*pi*t/(24*60*60)))
