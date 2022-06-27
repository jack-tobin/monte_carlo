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
#include <fstream>

// include MCMC class
#include "MarkovChainMonteCarlo.hpp"

using namespace std;

int main() {
  // set random seed
  srand(time(NULL));

  // simulation parameters
  const int n_simulations = 10000;
  const int n_years = 30;
  int periodicity = 1;

  // normal distribtion parameters
  double ann_mean = 0.10;
  double ann_std = 0.20;

  // init class MarkovChain and get returns data.
  MarkovChainMonteCarlo mcmc(ann_mean, ann_std, n_simulations, n_years, periodicity);

  // generate returns from mcmc class
  mcmc.generateReturns();

  // get numbers
  const int n_periods = mcmc.getNumPeriods();

  // write some to csv
  std::ofstream myFile("data.csv");
  for (int i = 0; i < 100; i++) {
    for (int j = 0; j < n_periods; j++) {
      myFile << mcmc.returns[i][j] << "\n";
    }
    //myFile << "\n";
  }
  
  // Close the file
  myFile.close();
  
  return 0;
};
