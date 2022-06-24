/**
 * @file MarkovChainMonteCarlo.cpp
 * @author Jack Tobin
 * @brief Definitions for objects in MarkovChainMonteCarlo.h
 * @version 0.1
 * @date 2022-06-24
 */

#include <iostream>
#include <random>
#include "MarkovChainMonteCarlo.hpp"

using namespace std;

/**
 * @brief Construct a new Markov Chain Monte Carlo object
 * 
 * @param new_annual_return       double annual return estimate 
 * @param new_annual_volatility   double annual volatility estimate
 * @param new_num_trials          int number of trials (recomm. 10k)
 * @param new_num_years           int number of years to simulate
 * @param new_periodicity         char simulation granularity
 */
MarkovChainMonteCarlo::MarkovChainMonteCarlo(double new_annual_return, double new_annual_volatility, 
  int new_num_trials, int new_num_years, char new_periodicity) {
  
  // set input attributes
  annual_return = new_annual_return;
  annual_volatility = new_annual_volatility;
  num_trials = new_num_trials;
  num_years = new_num_years;
  periodicity = new_periodicity;

  // compute total periods
  periods_per_year = convertPeriodicityToPeriods(new_periodicity);
  num_periods = periods_per_year * num_years;

  // compute de-annualised return and volatility
  periodic_return = deAnnualiseReturn(annual_return, periods_per_year);
  periodic_volatility = deAnnualiseVolatility(annual_volatility, periods_per_year);

  // empty arrays for states and returns
  rands = new double*[num_trials];
  states = new double*[num_trials];
  returns = new double*[num_trials];
  for (int i = 0; i < num_trials; i++) {
    rands[i] = new double[num_periods];
    states[i] = new double[num_periods];
    returns[i] = new double[num_periods];
  }  
}

/**
 * @brief Converts periodicity specified as char to number of periods per year
 * specified as int.
 * 
 * @param periodicity   char Periodicity i.e. 'd', 'm', 'q' etc.
 * @return int n_pers   the number of periods per year.
 */
int MarkovChainMonteCarlo::convertPeriodicityToPeriods(char periodicity) {
    int n_pers;
    // determine number of periods based on stated periodicity
    switch (periodicity) {
      case 'd':
        n_pers = 252; // daily
        break;
      case 'w':
        n_pers = 52;  // weekly
        break;
      case 'm':
        n_pers =12;   // monthly
        break;
      case 'q':
        n_pers = 4;   // quarterly
        break;
      case 'y':
        n_pers = 1;   // weekly
        break;
      default:
        n_pers = 1;   // default annual
        break;
    }

    return n_pers;
  }

/**
 * @brief Generates the matrix of periodic returns from the markov chain
 * switching process.
 */
void MarkovChainMonteCarlo::generateReturns() {

  // compute regime dist params
  regimeDists();

  // generate probs
  fillRandUnifArray();

  // generate states
  fillStatesArray();

  // initialise random number generators
  default_random_engine generator;
  normal_distribution<double> low_vol_dist(low_vol_mean, low_vol_vol);
  normal_distribution<double> high_vol_dist(high_vol_mean, high_vol_vol);

  // populate regime specific returns
  for (int i = 0; i < num_trials; i++) {
    for (int j = 0; j < num_periods; j++) {
      if (states[i][j] == 1) {
        // populate high vol
        double high_vol_r = high_vol_dist(generator);
        returns[i][j] = high_vol_r;
      }
      else {
        // populate low vol
        double low_vol_r = low_vol_dist(generator);
        returns[i][j] = low_vol_r;
      }
    }
  }

  // clean up memory
  delete[] states;
}

/**
 * @brief De-annualises an annual return figure based on a given frequency.
 * 
 * @param annual_return   double Annual Return.
 * @param frequency       int Frequency of returns desired.
 * @return double         The periodic return.
 */
double MarkovChainMonteCarlo::deAnnualiseReturn(double annual_return, 
  int frequency) {
  double one_div_freq = 1.0/frequency;
  double one_plus_ann_ret = 1.0 + annual_return;
  double periodic_return = pow(one_plus_ann_ret, one_div_freq) - 1.0;

  return periodic_return;
}

/**
 * @brief  De-annualiss an annual volatility figure based on a given frequency.
 * 
 * @param annual_vol  double Annual Volatility.
 * @param frequency   int Frequency of returns desired.
 * @return double     The periodic volatility.
 */
double MarkovChainMonteCarlo::deAnnualiseVolatility(double annual_vol, 
  int frequency) {
  double sqrt_freq = pow(frequency, 0.5);
  double periodic_vol = annual_vol / sqrt_freq;

  return periodic_vol;
}

/**
 * @brief Compute first two moments of the distributions of the two volatility
 * regimes for the markov chain process. Regime mean returns are shifted
 * additively from the unconditional mean; regime volatilities are scaled
 * multiplicatively from the unconditional volatility.
 */
void MarkovChainMonteCarlo::regimeDists() {
  // compute volatility regime distributions
  low_vol_mean = periodic_return + low_mean_shift;
  high_vol_mean = periodic_return + high_mean_shift;
  low_vol_vol = periodic_volatility * low_vol_shift;
  high_vol_vol = periodic_volatility * high_vol_shift;
}

/**
 * @brief Produces a uniformly-distributed psuedo-random number between 0 and 1
 * 
 * @return double Random number between 0 and 1.
 */
double MarkovChainMonteCarlo::randUniform() {
  return ((double) rand() / (RAND_MAX));
}

/**
 * @brief Fills the rands array with random numbers.
 */
void MarkovChainMonteCarlo::fillRandUnifArray() {
  // assign random numbers to the rands array
  for (int i = 0; i < num_trials; i++) {
    for (int j = 0; j < num_periods; j++) {
      rands[i][j] = randUniform();
    }
  }
}

/**
 * @brief Fills states array with state information. A high volatiltiy state
 * is represented by a value of 1; a low volatility state is represented
 * by a value of 0. States are assigned high volatility if the random number
 * for that period is below the threshold probability level.
 * 
 * The first period uses the unconditional probability of a high volatility state;
 * subsequent states use conditional probabilities based on the previous state
 * value.
 */
void MarkovChainMonteCarlo::fillStatesArray() {
  // determine states of period 0
  for (int i = 0; i < num_trials; i++) {
    if (rands[i][0] < prob_high) {
      states[i][0] = 1;
    }
    else {
      states[i][0] = 0;
    }
  }

  // determine subsequent states
  for (int i = 0; i < num_trials; i++) {
    for (int j = 1; j < num_periods; j++) {

      // if previous state was low
      if (states[i][j-1] == 0) { 
        if (rands[i][j] < prob_low_high) {
          states[i][j] = 1;
        }
        else {
          states[i][j] = 0;
        }
      }

      // if previous state was high
      else if (states[i][j-1] == 1) { 
        if (rands[i][j] < prob_high_high) {
          states[i][j] = 1;
        }
        else {
          states[i][j] = 0;
        }
      }
    }
  }

  // clean up memory
  delete[] rands;
}

// getters

/**
 * @brief Gets Num Periods object.
 * 
 * @return int num_periods
 */
int MarkovChainMonteCarlo::getNumPeriods() {
  return num_periods;
}

/**
 * @brief Gets Prob High object.
 * 
 * @return double prob_high
 */
double MarkovChainMonteCarlo::getProbHigh() {
  return prob_high;
}

/**
 * @brief Gets Prob Low object.
 * 
 * @return double prob_low
 */
double MarkovChainMonteCarlo::getProbLow() {
  return prob_low;
}

/**
 * @brief Gets Prob low Low object
 * 
 * Note this reads: "Probability current regime is low given previous is low"
 * 
 * @return double prob_low_low
 */
double MarkovChainMonteCarlo::getProbLowLow() {
  return prob_low_low;
}

/**
 * @brief Gets Prob Low High object
 * 
 * @return double prob_low_high
 */
double MarkovChainMonteCarlo::getProbLowHigh() {
  return prob_low_high;
}

/**
 * @brief Gets Prob High Low object
 * 
 * @return double prob_high_low
 */
double MarkovChainMonteCarlo::getProbHighLow() {
  return prob_high_low;
}

/**
 * @brief Gets Prob High High object
 * 
 * @return double prob_high_low
 */
double MarkovChainMonteCarlo::getProbHighHigh() {
  return prob_high_low;
}

/**
 * @brief Gets Low Mean Shift object
 * 
 * This is the amount by which the unconditional mean is adjusted additively
 * to arrive at the conditional low-volatility regime mean return.
 * 
 * @return double low_mean_shift
 */
double MarkovChainMonteCarlo::getLowMeanShift() {
  return low_mean_shift;
}

/**
 * @brief Gets High Mean Shift object.
 * 
 * This is the amount by which the unconditional mean is adjusted additively
 * to arrive at the conditional high-volatility regime mean return.
 * 
 * @return double high_mean_shift
 */
double MarkovChainMonteCarlo::getHighMeanShift() {
  return high_mean_shift;
}

/**
 * @brief Gets Low Vol Shift object
 * 
 * This is the amount by which the unconditional volatility is adjusted
 * multiplicatively to arrive at the conditional low volatiltiy regime volatility.
 * 
 * @return double low_vol_shift
 */
double MarkovChainMonteCarlo::getLowVolShift() {
  return low_vol_shift;
}

/**
 * @brief Gets High Vol Shift object
 * 
 * This is the amount by which the unconditional volatility is adjusted
 * multiplicatively to arrive at the conditoinal high volatility regime volatility.
 * 
 * @return double high_vol_shift
 */
double MarkovChainMonteCarlo::getHighVolShift() {
  return high_vol_shift;
}

// setters

/**
 * @brief Sets Prob High object.
 * 
 * Note this also adjusts prob_low to be consistent.
 * 
 * @param new_prob_high double New value for prob_high
 */
void MarkovChainMonteCarlo::setProbHigh(double new_prob_high) {
  prob_high = new_prob_high;
  prob_low = 1 - prob_high;
}

/**
 * @brief Sets Prob Low object.
 * 
 * Note this also adjusts prob_high to be consistent.
 * 
 * @param new_prob_low double New value for prob_low
 */
void MarkovChainMonteCarlo::setProbLow(double new_prob_low) {
  prob_low = new_prob_low;
  prob_high = 1 - prob_low;
}

/**
 * @brief Sets Prob Low Low object
 * 
 * Note this also adjusts prob_low_high to be consistent since the
 * probability of low and high conditional on low must sum to 1.
 * 
 * @param new_prob_low_low double New value for prob_low_low
 */
void MarkovChainMonteCarlo::setProbLowLow(double new_prob_low_low) {
  prob_low_low = new_prob_low_low;
  prob_low_high = 1 - prob_low_low;
}

/**
 * @brief Sets Prob Low High object
 * 
 * Note this also adjusts the prob_low_low object to be consistent since the
 * probability of low and high conditional on low must sum to 1.
 * 
 * @param new_prob_low_high double New value for prob_low_high
 */
void MarkovChainMonteCarlo::setProbLowHigh(double new_prob_low_high) {
  prob_low_high = new_prob_low_high;
  prob_low_low = 1 - prob_low_high;
}

/**
 * @brief Sets Prob High Low object
 * 
 * Note this also adjusts prob_high_high to be consistent since the probability
 * of low and high conditional on high must sum to 1.
 * 
 * @param new_prob_high_low double New value for prob_high_low
 */
void MarkovChainMonteCarlo::setProbHighLow(double new_prob_high_low) {
  prob_high_low = new_prob_high_low;
  prob_high_high = 1 - prob_high_low;
}

/**
 * @brief Sets Prob High High object
 * 
 * Note this also adjusts prob_high_low to be consistent since the probability
 * of low and high conditional on high must sum to 1.
 * 
 * @param new_prob_high_high double New value for prob_high_high
 */
void MarkovChainMonteCarlo::setProbHighHigh(double new_prob_high_high) {
  prob_high_high = new_prob_high_high;
  prob_high_low = 1 - prob_high_high;
}

/**
 * @brief Sets Low Mean Shift object
 * 
 * @param new_low_mean_shift double New value for low_mean_shift
 */
void MarkovChainMonteCarlo::setLowMeanShift(double new_low_mean_shift) {
  low_mean_shift = new_low_mean_shift;
}

/**
 * @brief Sets High Mean Shift object.
 * 
 * @param new_high_mean_shift double New value for high_mean_shift
 */
void MarkovChainMonteCarlo::setHighMeanShift(double new_high_mean_shift) {
  high_mean_shift = new_high_mean_shift;
}

/**
 * @brief Sets Low Vol Shift object
 * 
 * @param new_low_vol_shift double New value for low_vol_shift
 */
void MarkovChainMonteCarlo::setLowVolShift(double new_low_vol_shift) {
  low_vol_shift = new_low_vol_shift;
}

/**
 * @brief Sets High Vol Shift object
 * 
 * @param new_high_vol_shift double New value for high_vol_shift
 */
void MarkovChainMonteCarlo::setHighVolShift(double new_high_vol_shift) {
  high_vol_shift = new_high_vol_shift;
}

/**
 * @brief Sets Annual Return object
 * 
 * Note this also adjusts periodic_return to be consistent.
 * 
 * @param new_annual_return double New value for annual_return
 */
void MarkovChainMonteCarlo::setAnnualReturn(double new_annual_return) {
  annual_return = new_annual_return;
  periodic_return = deAnnualiseReturn(annual_return, periods_per_year);
}

/**
 * @brief Sets Annual Volatility object
 * 
 * This aslo adjusts periodic_volatility to be consistent.
 * 
 * @param new_annual_volatility double New value for annual_volatility
 */
void MarkovChainMonteCarlo::setAnnualVolatility(double new_annual_volatility) {
  annual_volatility = new_annual_volatility;
  periodic_volatility = deAnnualiseVolatility(annual_volatility, periods_per_year);
}

/**
 * @brief Sets Num Trials object
 * 
 * @param new_num_trials int New value for num_trials
 */
void MarkovChainMonteCarlo::setNumTrials(int new_num_trials) {
  num_trials = new_num_trials;
}

/**
 * @brief Sets Num Years object
 * 
 * Note this also adjusts num_periods to be consistent.
 * 
 * @param new_num_years int New value for num_years.
 */
void MarkovChainMonteCarlo::setNumYears(int new_num_years) {
  num_years = new_num_years;
  num_periods = periods_per_year * num_years;
}

/**
 * @brief Sets Periodicity object
 * 
 * Note this also adjusts periods_per_year and num_periods to be consistent.
 * 
 * @param new_periodicity char New value of periodicity.
 */
void MarkovChainMonteCarlo::setPeriodicity(char new_periodicity) {
  periodicity = new_periodicity;
  periods_per_year = convertPeriodicityToPeriods(periodicity);
  num_periods = periods_per_year * num_years;
}
