#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 09:29:34 2021

@author: jtobin
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# load in markov chain class
from markov_chain import MarkovChain


class Simulation:
    
    def __init__(self, init_wealth, exp_ret, exp_vol, n_trials, n_years):
        """
        Initializes Simulation Class

        Parameters
        ----------
        init_wealth : int
            starting wealth i.e. £100.
        exp_ret : float
            Expected Annual Return .
        exp_vol : float
            Expected Volatility.
        n_trials : int
            Number of Trials, use 10000.
        n_years : int
            Number of Years.
        """

        # input paramters
        self.init_wealth = init_wealth
        self.exp_ret = exp_ret
        self.exp_vol = exp_vol
        self.n_trials = n_trials
        self.n_years = n_years
        
        # additional params
        self.days_per_year = 252
        self.n_days = self.n_years * self.days_per_year
        
        # empty matrices
        self.daily_returns = np.zeros(shape=(self.n_trials, self.n_days))
        self.returns = np.zeros(shape=(self.n_trials, self.n_years))
        self.values = np.zeros(shape=(self.n_trials, self.n_years + 1))
        
    def simulate_values(self):
        """
        Runs the simulation. Calls in the MarkovChain class and uses it
        to generate a set of returns. Then computes annual returns and finally
        portfolio values based on those annual returns.
        """
        
        # instantiate Markov Chain Returns process
        mc = MarkovChain(self.exp_ret, self.exp_vol, self.n_trials,
                         self.n_years, self.days_per_year)
        
        # generate returns based on regime switching algorithm
        self.daily_returns = mc.generate_returns()
        
        # collapse into annual returns
        for i, j in enumerate(range(0, self.n_days, self.days_per_year)):
            self.returns[:, i] = np.prod(1 + self.daily_returns[:, j:(j + self.days_per_year - 1)], axis=1) - 1
        
        # set starting portfolio value
        self.values[:, 0] = self.init_wealth
        
        # apply to portfolio value over time
        for i in range(self.n_years):
            self.values[:, i+1] = self.values[:, i] * (1 + self.returns[:, i])
            
        return self
