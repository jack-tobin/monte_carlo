/**
 * @file main.cpp
 * @author Jack Tobin
 * @brief Main file for Markov Chain Monte Carlo modelling.
 * @version 0.1
 * @date 2022-06-24
 */

#include <iostream>
#include <string>
#include <random>
#include <cmath>

// include MCMC class
#include "MarkovChainMonteCarlo.hpp"

using namespace std;

int main() {
  // set random seed
  srand(time(NULL));

  // simulation parameters
  const int n_simulations = 10000;
  const int n_years = 30;
  char periodicity = 'd';

  // normal distribtion parameters
  double ann_mean = 0.10;
  double ann_std = 0.20;

  // init class MarkovChain and get returns data.
  MarkovChainMonteCarlo mcmc(ann_mean, ann_std, n_simulations, n_years, periodicity);

  // generate returns from mcmc class
  mcmc.generateReturns();

  // get periods
  const int n_periods = mcmc.getNumPeriods();

  // initiate empty prices matrix
  double** prices = new double*[n_simulations];
  for (int i = 0; i < n_simulations; i++) {
    prices[i] = new double[n_periods+1];
  }

  // populate starting values of $1
  for (int i = 0; i < n_simulations; i++) {
    prices[i][0] = 1.0;
  }

  // compute prices
  for (int i = 0; i < n_simulations; i++) {
    for (int j = 0; j < n_periods; j++) {
      prices[i][j+1] = prices[i][j] * (1 + mcmc.returns[i][j]);
    }
  }

  // print a few values
  cout << prices[5][10] << endl;
  cout << prices[23][42] << endl;

  return 0;
};
