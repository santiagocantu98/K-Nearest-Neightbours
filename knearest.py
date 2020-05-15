
"""
    This Python script was done for the second practical exam of 
    Artificial Intelligence class, the exam consists of 
    creating functions in order to train the algorithmn
    so it can find the optimal w0 and w1 of the data set

    Author: Santiago Cantu
    email: santiago.cantu@udem.edu
    Institution: Universidad de Monterrey
    First created: March 29, 2020
"""

import numpy as np
import math
import pandas as pd
import utilityKNN as ai

def main():
  """
    function that runs the program calling all the functions
    to get the minimum square error for the testing data
    after finding the minimum values of w0 and w1
  """
  # Initializing learning rate
  learning_rate = 0.0005
  # Initializing stopping criteria
  stopping_criteria = 0.01
  # load the data training data from a csv file with an url
  training_x,testing_x, training_y, testing_y,mean,sd= ai.store_data("https://github.com/santiagocantu98/K-Nearest-Neightbours/raw/master/diabetes.csv","training")
  normal_testing = np.copy(testing_x)

  # scalates the features of the testing data
  testing_data_scaled,mean,sd = ai.scale_features(testing_x,mean,sd)
  ai.print_scaled_data(testing_data_scaled,"testing")
  ai.calculate_euclidean_distance(training_x, training_y , testing_data_scaled, testing_y,normal_testing)



# calls the main function
main();

