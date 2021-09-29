#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:00:04 2021

@author: jtobin
"""

import pandas as pd
import yfinance as yf
from sklearn.cluster import KMeans
import seaborn as sns
sns.set_style('white')


def download_data():
    """
    Downloads SPX and VIX daily data from 1990 to present (earliest VIX is
    available). Returns Date, SPX and VIX into a single dataframe without
    Date as index.
    
    returns :: all_data, DataFrame of Date, SPX, and VIX prices. Daily.
    """
    
    # download SPX daily data since beginning of time
    spx = yf.Ticker('^GSPC')
    spx_prices = spx.history(period='max')['Close']
    
    # calc returns
    spx_returns = spx_prices.pct_change()
    spx_returns.dropna(inplace=True)
    
    # un-index dates
    spx_returns = pd.DataFrame(spx_returns)
    spx_returns.reset_index(inplace=True)
    
    # gather VIX data
    vix = yf.Ticker('^VIX')
    vix_prices = vix.history(period='max')['Close']
    vix_prices = pd.DataFrame(vix_prices)
    vix_prices.reset_index(inplace=True)
    
    # merge vix onto spx_prices
    all_data = spx_returns.merge(vix_prices, how='left', on='Date')
    all_data.columns = ['Date', 'SPX', 'VIX']
    all_data.dropna(inplace=True)
    
    return all_data
    

def regime_clustering(vol_data):
    """
    Clusters the VIX data into two regimes based on K-Means clustering along
    two clusters
    
    vol_data :: Series, VIX price data.
    returns :: lab_df, Dataframe of each date's volatiltiy regime number
        and associated label
    """
    
    # subset data
    X = vol_data.to_numpy().reshape(-1, 1)
    
    # init kmeans
    km = KMeans(n_clusters=2)
    km.fit(X)
    
    # get data
    labs = km.labels_
    cc = km.cluster_centers_
    
    # get index of max cluster center
    highvol_idx = cc.argmax(axis=0)[0]
    lowvol_idx = 0 if highvol_idx == 1 else 1
    
    # assemble into dict
    assignments = {highvol_idx: 'High Vol',
                   lowvol_idx: 'Low Vol'}
    
    # map dict to labels
    lab_df = pd.DataFrame(labs)
    lab_df.columns = ['Number']
    lab_df['Label'] = lab_df['Number'].map(assignments)
    
    return lab_df
