#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 09:37:28 2021

@author: jtobin
"""


import numpy as np
import pandas as pd
import ctypes
import platform

# create clib object; include fix for alleged bug in ctypes
mode = dict(winmode=0) if platform.python_version() >= '3.8' else dict()  
clib = ctypes.CDLL('montecarlo_cpp/mcmc.dll', **mode)

# reference for setting up Python/C++ bridge:
# from https://stackoverflow.com/questions/602580/how-can-i-use-c-class-in-python
# https://stackoverflow.com/questions/29015606/ctypes-return-array
# https://docs.python.org/3/library/ctypes.html


class MarkovChain:
    """
    A Markov-Chain Monte Carlo return generation object. A Markov-Chain Monte
    Carlo process is a type of Monte Carlo simulation that produces an array
    of simulated asset returns that are randomly selected from two underlying
    Gaussian return distributions, one representing high-volatility regimes,
    the other representing low-volatility regimes. When these two overlapping
    return distributions are combined through the Markov-Chain process, the
    resulting distribution follows a negatively-skewed return distribution.
    """
    
    def __init__(self, exp_ret, exp_vol, n_trials, n_years, periodicity):
        """
        Creates an instance of class MarkovChain.
        """
        
        # core variables
        self.exp_ret = exp_ret
        self.exp_vol = exp_vol
        self.n_trials = n_trials
        self.n_years = n_years
        self.periodicity = periodicity
        
        # get number of periods per year; get number of periods
        self.n_periods = self.n_years * self.periodicity
        
        # meta-object
        self.obj = None
        
        # empty returns object
        self.returns = None
        
        # types for init method
        clib.init.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int]
        clib.init.restype = ctypes.c_void_p
        
        # set types for generateReturns method 
        clib.generateReturns.argtypes = [ctypes.c_void_p]
        clib.generateReturns.restype = ctypes.c_void_p
                   
        # types for getReturns method. This is a double pointer object
        self.doublePtr = ctypes.POINTER(ctypes.c_double * self.n_periods)
        self.doublePtrPtr = ctypes.POINTER(self.doublePtr * self.n_trials)
        clib.getReturns.argtypes = [ctypes.c_void_p]
        clib.getReturns.restype = self.doublePtrPtr
        
        # initiate object from c library
        self.obj = clib.init(exp_ret, exp_vol, n_trials, n_years, periodicity)
    
    def generate_returns(self):
        """
        This function runs the generateReturns method from the C++ library.
        This stores the returns in the clib.returns object. The function
        then collects the returns object from clib and returns it
        to the Python environment. Note that the contents of clib.returns must
        be processed before they are usable in Python.

        Returns:
            self.returns: np.array of returns, size n_trials x n_periods
        """

        # generate returns data
        clib.generateReturns(self.obj)
        
        # collect returns data
        rets = clib.getReturns(self.obj)
        
        # clean up returns prior to release
        self.returns = [[a for a in x.contents] for x in rets.contents]
        self.returns = np.array(self.returns)
        
        return self.returns
    