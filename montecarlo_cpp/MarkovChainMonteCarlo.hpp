/**
 * @file MarkovChainMonteCarlo.hpp
 * @author Jack Tobin
 * @brief Header file for Markov Chain Monte Carlo modelling.
 * @version 0.1
 * @date 2022-06-24
 */

#include <random>
#include <cmath>
#include <iostream>

class MarkovChainMonteCarlo {

private:
  // baseline attributes
  double annual_return;
  double annual_volatility;
  double periodic_return;
  double periodic_volatility;
  int num_trials;
  int num_years;
  int periodicity;
  int num_periods;
  int periods_per_year;

  // regime switching parameters
  double prob_low = 0.77;
  double prob_high = 1 - prob_low;
  double prob_low_low = 0.97;
  double prob_low_high = 1 - prob_low_low;
  double prob_high_low = 0.09;
  double prob_high_high = 1 - prob_high_low;

  // changing distribution parameters
  double low_mean_shift = 0.0006;
  double high_mean_shift = -0.0020;
  double low_vol_shift = 0.66;
  double high_vol_shift = 1.67;

  // empty regime dist params
  double low_vol_mean;
  double high_vol_mean;
  double low_vol_vol;
  double high_vol_vol;

  // empty arrays for states and returns
  double** rands;
  double** states;
  
  // private method declarations

  /**
   * @brief De-annualise an annual return into desired periodicity.
   * 
   * @param annual_return   double Annual Return.
   * @param frequency       int Periods per year.
   * @return double De-annualised periodic return.
   */
  double deAnnualiseReturn(double annual_return, int frequency);

  /**
   * @brief De-annualise an annual volatility into desired periodicity.
   * 
   * @param annual_vol  double Annual Vol.
   * @param frequency   int Periods per year. 
   * @return double De-annualised periodic volatility.
   */
  double deAnnualiseVolatility(double annual_vol, int frequency);

  /**
   * @brief Compute regime distributions.
   */
  void regimeDists();

  /**
   * @brief Obtain a pseudo-random number between 0 and 1.
   * 
   * @return double random number between 0 and 1.
   */
  double randUniform();

  /**
   * @brief Fills empty 2d array with random numbers.
   */
  void fillRandUnifArray();

  /**
   * @brief Populates regimes array with 1 if high volatiltiy and 0 if 
   * low volatility. See definition for more detail.
   */
  void fillStatesArray();

public:
  // public attributes
  double** returns;

  /**
   * @brief Construct a new Markov Chain Monte Carlo object
   * 
   * @param new_annual_return       double Annual return.
   * @param new_annual_volatility   double Annual volatility.
   * @param new_num_trials          int Number of simulations.
   * @param new_num_years           int Number of years.
   * @param new_periodicity         int Periodicity.
   */
  MarkovChainMonteCarlo(double new_annual_return, double new_annual_volatility, 
    int new_num_trials, int new_num_years, int new_periodicity);

  // public method declarations

  /**
   * @brief Generates random returns from the MCMC process.
   */
  void generateReturns();
  
  // getters

  /**
   * @brief Get the Returns object
   * 
   * @return double** returns
   */
  double** getReturns();

  /**
   * @brief Get the Num Periods object
   * 
   * @return int num_periods
   */
  int getNumPeriods();

  /**
   * @brief Get the Prob High object
   * 
   * @return double prob_high
   */
  double getProbHigh();

  /**
   * @brief Get the Prob Low object
   * 
   * @return double prob_low
   */
  double getProbLow();

  /**
   * @brief Get the Prob Low Low object
   * 
   * @return double prob_low_low
   */
  double getProbLowLow();

  /**
   * @brief Get the Prob Low High object
   * 
   * @return double prob_low_high
   */
  double getProbLowHigh();

  /**
   * @brief Get the Prob High Low object
   * 
   * @return double prob_high_low
   */
  double getProbHighLow();

  /**
   * @brief Get the Prob High High object
   * 
   * @return double prob_high_high
   */
  double getProbHighHigh();

  /**
   * @brief Get the Low Mean Shift object
   * 
   * @return double low_mean_shift
   */
  double getLowMeanShift();

  /**
   * @brief Get the High Mean Shift object
   * 
   * @return double high_mean_shift
   */
  double getHighMeanShift();

  /**
   * @brief Get the Low Vol Shift object
   * 
   * @return double low_vol_shift
   */
  double getLowVolShift();

  /**
   * @brief Get the High Vol Shift object
   * 
   * @return double high_vol_shift
   */
  double getHighVolShift();

  // setters

  /**
   * @brief Set the Prob High object
   * 
   * @param new_prob_high double New prob_high
   */
  void setProbHigh(double new_prob_high);

  /**
   * @brief Set the Prob Low object
   * 
   * @param new_prob_low double New prob_log
   */
  void setProbLow(double new_prob_low);

  /**
   * @brief Set the Prob Low Low object
   * 
   * @param new_prob_low_low double New prob_low_low
   */
  void setProbLowLow(double new_prob_low_low);

  /**
   * @brief Set the Prob Low High object
   * 
   * @param new_prob_low_high double New prob_low_high
   */
  void setProbLowHigh(double new_prob_low_high);

  /**
   * @brief Set the Prob High Low object
   * 
   * @param new_prob_high_low double New prob_high_low
   */
  void setProbHighLow(double new_prob_high_low);

  /**
   * @brief Set the Prob High High object
   * 
   * @param new_prob_high_high double New prob_high_high
   */
  void setProbHighHigh(double new_prob_high_high);

  /**
   * @brief Set the Low Mean Shift object
   * 
   * @param new_low_mean_shift double New low_mean_shift
   */
  void setLowMeanShift(double new_low_mean_shift);

  /**
   * @brief Set the High Mean Shift object
   * 
   * @param new_high_mean_shift double New high_mean_shift
   */
  void setHighMeanShift(double new_high_mean_shift);

  /**
   * @brief Set the Low Vol Shift object
   * 
   * @param new_low_vol_shift double New low_vol_shift
   */
  void setLowVolShift(double new_low_vol_shift);

  /**
   * @brief Set the High Vol Shift object
   * 
   * @param new_high_vol_shift double New high_vol_shift
   */
  void setHighVolShift(double new_high_vol_shift);

  /**
   * @brief Set the Annual Return object
   * 
   * @param new_annual_return double New annual_return
   */
  void setAnnualReturn(double new_annual_return);

  /**
   * @brief Set the Annual Volatility object
   * 
   * @param new_annual_volatility double New annual_volatility
   */
  void setAnnualVolatility(double new_annual_volatility);

  /**
   * @brief Set the Num Trials object
   * 
   * @param new_num_trials int New num_trials
   */
  void setNumTrials(int new_num_trials);

  /**
   * @brief Set the Num Years object
   * 
   * @param new_num_years int New num_years
   */
  void setNumYears(int new_num_years);

  /**
   * @brief Set the Periodicity object
   * 
   * @param new_periodicity int New periodicity
   */
  void setPeriodicity(int new_periodicity);
};
