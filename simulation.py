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

os.chdir(os.path.expanduser('~/Documents/projects/montecarlo'))

# load in markov chain class
from markov_chain import MarkovChain


class Simulation:
    
    def __init__(self, init_wealth, exp_ret, exp_vol, n_trials, n_years):
        """
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
        self.daily_ret = self.exp_ret / self.days_per_year
        self.daily_geo_ret = (1 + self.exp_ret) ** (1 / self.days_per_year) - 1
        self.daily_vol = self.exp_vol / np.sqrt(self.days_per_year)
        
        # empty matrices
        self.daily_returns = np.zeros(shape=(self.n_trials, self.n_days))
        self.returns = np.zeros(shape=(self.n_trials, self.n_years))
        self.values = np.zeros(shape=(self.n_trials, self.n_years + 1))
        
    def simulate_values(self):
        """
        """
        
        # instantiate Markov Chain Returns process
        mc = MarkovChain(self.daily_geo_ret, self.daily_vol, self.n_trials,
                         self.n_days)
        
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
    

# testing
init_w = 100
exp_ret = 0.09
exp_vol = 0.18
n_trials = 500
n_years = 30

# initiate
sim = Simulation(init_w, exp_ret, exp_vol, n_trials, n_years)
sim.simulate_values()

# histogram
x = pd.Series(sim.values.reshape(-1, 1).ravel())
sns.histplot(np.log(x))
    