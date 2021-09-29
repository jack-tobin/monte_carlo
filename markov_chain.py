#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 21 09:37:28 2021

@author: jtobin
"""


import numpy as np


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
    
    def __init__(self, exp_ret, exp_vol, n_trials, n_per):
        """
        Creates an instance of class MarkovChain.

        Parameters
        ----------
        exp_ret : float
            Expected return. Daily periodicity. Geometric.
        exp_vol : float
            Expected volatility. Daily periodicity.
        n_trials : int
            Number of trials.
        n_per : int
            Number of periods.

        Returns
        -------
        None.

        """
        
        # core variables
        self.exp_ret = exp_ret
        self.exp_vol = exp_vol
        self.n_trials = n_trials
        self.n_per = n_per
        
        # probabilities
        self.prob_low = 0.77
        self.prob_high = 1 - self.prob_low
        
        # transition matrix
            # interpretation
            # x[0, 0] = prob of low vol given previously low vol
            # x[1, 0] = prob of high vol given previously low vol
            # x[0, 1] = prob of low vol given previosuly high vol
            # x[1, 1] = prob of high vol given previously high vol
        self.trans_mat = np.array([[0.97, 0.09],
                                   [0.03, 0.91]])
        
        # shift matrix
            # interpration
            # x[0, 0] = mean loc shift from avg during low vol
            # x[1, 0] = stdev scale shift from avg during low vol
            # x[0, 1] = mean loc shift from avg during high vol
            # x[1, 1] = stdev scale shift from avg during high vol
        self.shift_mat = np.array([[0.0006, -0.002],
                                   [0.66, 1.67]])
        
        # empty matrices
        self.states = np.zeros(shape=(self.n_trials, self.n_per))
        self.returns = np.zeros(shape=(self.n_trials, self.n_per))
        
    @staticmethod
    def regime_returns(m, n, r, v, shift_matrix, regime):
        """
        A static method that produces an array of simulated random returns
        based on the selected regime. The regime selected informs how to
        modify the 'average' return/volatility statistics so that they
        represent the underlying overlapping distributions.

        Parameters
        ----------
        m : int
            Number of trials.
        n : int
            Number of periods.
        r : float
            Expected daily return.
        v : float
            Expected daily volatility.
        shift_matrix : Numpy array
            Matrix detailing the relative shift in regime-specific return
            distributions from the average of all regimes.
        regime : int
            0 for low volatility, 1 for high volatility.

        Returns
        -------
        rets : Numpy array
            An array of random returns that follow a normal distribution for
            the specified regime.

        """
        
        # determine params
        ret = r + shift_matrix[0, regime]
        vol = v * shift_matrix[1, regime]
        
        # generate returns
        rets = np.random.normal(loc=ret, scale=vol, size=(m, n))
        
        return rets
        
    def generate_states(self):
        """
        Generates regime states based on the probability of moving between
        volatiltiy regimes. Probabilities are based on Bayesian conditional
        probabilities of moving between regimes based on prior state.

        Returns
        -------
        self : MarkovChain
            Returns self.

        """
        
        # generate random probabilities
        self.probs = np.random.uniform(size=(self.n_trials, self.n_per))
        
        # determine state of period 0
        self.states[self.probs[:, 0] < self.prob_high, 0] = 1
        
        # determine state of subsequent periods
        for i in range(1, self.n_per):
            prbs = [self.trans_mat[1, int(x)] for x in self.states[:, i-1]]
            self.states[self.probs[:, i] < prbs, i] = 1
                
        return self
    
    def select_returns(self):
        """
        Selects which returns to use from the two underlying regime-specific
        return distributions based on the regime states determined by the
        conditional probability process.

        Returns
        -------
        self : MarkovChain
            Returns self.

        """
        
        # two return distributions, one for each vol regime
        # low vol
        lv_rets = self.regime_returns(m=self.n_trials,
                                      n=self.n_per,
                                      r=self.exp_ret,
                                      v=self.exp_vol,
                                      shift_matrix=self.shift_mat, 
                                      regime=0)
        # high vol
        hv_rets = self.regime_returns(m=self.n_trials,
                                      n=self.n_per,
                                      r=self.exp_ret,
                                      v=self.exp_vol,
                                      shift_matrix=self.shift_mat, 
                                      regime=1)
        
        # select which returns based on state
        self.returns[self.states == 0] = lv_rets[self.states == 0]
        self.returns[self.states == 1] = hv_rets[self.states == 1]
        
        return self
    
    def generate_returns(self):
        """
        Generates an array of final returns that reflect the regime-switching
        effects of the Markov Chain process. Distribution of final returns
        should reflect moderate negative skew.

        Returns
        -------
        self.returns : Numpy array
            Returns the final returns array as a result of the entire Markov
            Chain process.

        """
        
        # run all elements of this process
        self.generate_states()
        self.select_returns()
        
        return self.returns
    