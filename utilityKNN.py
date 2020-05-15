import numpy as np
import math
import pandas as pd
import time

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

def store_data(url,data):
  """
    function that reads a csv file from a github url
    and then stores it into x and y data for training
    and testing

    Inputs
    :param url = string type
    :param data = string type

    Output
    :return x: numpy matrix
    :return y: numpy array
    :return mean: numpy array
    :return sd: numpy array
    :return w: numpy array
  """
  if(data == "training"):
    # load data from an url
    training_data = pd.read_csv(url)
    #amount of samples and features
    numberSamples, numberFeatures = training_data.shape
    # remove headers from features and separates data in x and y
    x = pd.DataFrame.to_numpy(training_data.iloc[:,0:numberFeatures-1])
    y = pd.DataFrame.to_numpy(training_data.iloc[:,-1]).reshape(numberSamples,1)
    
    # array for means of every feature
    mean = []
    # array for standard deviation of every feature
    sd = []
    
    # 95% of the data for training
    training_size = int(len(x)*.95)
    # 5% of the data for testing
    testing_size = len(x) - training_size
    # declaration of testing and training data arrays
    testing_data_x = np.zeros([testing_size,8])
    testing_data_y = np.zeros([testing_size,1])
    training_data_x = np.zeros([training_size,8])
    training_data_y = np.zeros([training_size,1])
    # size of data
    data_size = len(x)
    
    # 5% of the data for testing 
    for size in range(testing_size):
        testing_data_x[testing_size-1-size] = x[data_size-1-size]
        testing_data_y[testing_size-1-size] = y[data_size-1-size]
    # 95% of the data for training
    for size in range(training_size):
        training_data_x[size] = x[size]
        training_data_y[size] = y[size]
        
    # prints training data    
    print_data(training_data_x,"training")
    # prints testing data
    print_data(testing_data_x,"testing")
    #amount of samples and features of training data
    trainingSamples, trainingFeatures = training_data_x.shape
    # scale features so when returned, the data is already scalated stores x,mean and sd
    training_data_x,mean,sd = scale_features(training_data_x,mean,sd)
    # prints scaled training data
    print_scaled_data(training_data_x,"training")
    #amount of samples and features of training data
    trainingFeatures, trainingSamples = training_data_x.shape

    return training_data_x, testing_data_x, training_data_y,testing_data_y,mean,sd


def scale_features(x,mean,sd):
  """
    function that scalates the x features from the
    training data and testing data with the mean and
    standard deviation

    Input
    :param x: numpy matrix
    :param mean: numpy array
    :param sd: numpy array
    :param data: string type

    Output
    :return x: numpy matrix with scalated values
    :return mean: numpy array of mean
    :return sd: numpy array of standard deviation

  """
  # scalates data
  for size in range(x.shape[1]):
      x_data = x[:,size]
      m_data = np.mean(x_data)
      sd_data = np.std(x_data)
      mean.append(m_data)
      sd.append(sd_data)
      x[:,size] = (x_data - m_data)/ sd_data
  return x,mean,sd

def calculate_euclidean_distance(training_x, training_y , testing_data_scaled, testing_y, testing_x):
    """ 
        function that calculates the euclidean distance between
        the testing points and the training data
        
        INPUTS
        param training_x: numpy array
        param training_y: numpy array
        param testing_data_scalated: numpy array
        param testing_y: numpy array
        
        OUTPUT
        Covariance matrix of k = 5, 10 and 20
        
    """
    
    testing_size = testing_data_scaled.shape[0]
    training_size = training_x.shape[0]
    prob = np.arange(testing_size)
    testingArray5 = np.arange(testing_size)
    testingArray10 = np.arange(testing_size)
    testingArray20 = np.arange(testing_size)
    for i in range(testing_size):
        array = np.arange(training_size, dtype=float)
        for j in range(training_size):
            distance = math.sqrt((training_x[j][0] - testing_data_scaled[i][0])**2 +
                        (training_x[j][1] - testing_data_scaled[i][1])**2 +
                        (training_x[j][2] - testing_data_scaled[i][2])**2 +
                        (training_x[j][3] - testing_data_scaled[i][3])**2 +
                        (training_x[j][4] - testing_data_scaled[i][4])**2 +
                        (training_x[j][5] - testing_data_scaled[i][5])**2 +
                        (training_x[j][6] - testing_data_scaled[i][6])**2 +
                        (training_x[j][7] - testing_data_scaled[i][7])**2) 
            array[j] = distance
        aux = np.argsort(array)
        diabetes, notDiabetes = compute_conditional_probabilities(aux,5, training_y)
        prob[i] = diabetes
        prediction = predict(diabetes, notDiabetes)
        testingArray5[i] = prediction
        diabetes, notDiabetes = compute_conditional_probabilities(aux,10, training_y)
        prediction = predict(diabetes, notDiabetes)
        testingArray10[i] = prediction
        diabetes, notDiabetes = compute_conditional_probabilities(aux,20, training_y)
        prediction = predict(diabetes, notDiabetes)
        testingArray20[i] = prediction
    accuracy5,precision5,recall5,specificity5,f1_score5 = covariance_matrix(testingArray5, testing_y,5)
    accuracy10,precision10,recall10,specificity10,f1_score10 = covariance_matrix(testingArray10, testing_y,10)
    accuracy20,precision20,recall20,specificity20,f1_score20 = covariance_matrix(testingArray20, testing_y,20)
    
    print('------------------------------------------------')
    print(' Performance metrics ')
    print('------------------------------------------------')
    print('K    Accuracy    Precision   Recall   Specificity    F1-score')
    print("5       %.3f         %.3f      %.3f        %.3f         %.3f" %  (accuracy5,precision5,recall5,specificity5,f1_score5))
    print("10      %.3f         %.3f      %.3f        %.3f         %.3f" %  (accuracy10,precision10,recall10,specificity10,f1_score10))
    print("20      %.3f         %.3f      %.3f        %.3f         %.3f" %  (accuracy20,precision20,recall20,specificity20,f1_score20))
    print('------------------------------------------------')
    print(' Testing point (features ')
    print('------------------------------------------------')   
    print("{}         {}         {}         {}         {}         {}         {}         {}         {}         {}".format('Preg.','Gluc.','BloodPr.','SkinThick.','Insulin','BMI','DiabetesPed.','Age','Prob. Diabetes','Prob. No Diabetes'))
    for i in range(testing_x.shape[0]):
        diabetes = prob[i]/5
        print(diabetes)
        noDiabetes = 1 - prob[i]/5
        print("{a:1.2f}         {b:1.2f}         {c:1.2f}              {d:1.2f}             {e:1.2f}             {f:1.2f}          {g:1.2f}                {h:1.2f}            {m:1.2f}                 {k:1.2f}".format(a = testing_x[i][0], b = testing_x[i][1], c = testing_x[i][2],d = testing_x[i][3],e = testing_x[i][4],f = testing_x[i][5],g = testing_x[i][6],h = testing_x[i][7], m = diabetes, k = noDiabetes))
    
    
    

def compute_conditional_probabilities(indexArray, k, training_y):
    """
        function that calculates the probability of having or not diabetes
        in women patients with the training and testing data
        
        Input
        param indexArray: numpy array
        param training_y: numpy array
        
        Output
        returns the number of distances inside the k of diabetes and 
        notDiabetes
    """
    
    diabetes = 0
    notDiabetes = 0
    for n in range(k):
        result = np.where(indexArray == n)
        if(training_y[result] == 1):
            diabetes = diabetes + 1
        if(training_y[result] == 0):
            notDiabetes = notDiabetes + 1   
    return diabetes, notDiabetes
    
def predict(diabetes, notDiabetes):
    """
        function that predicts if the patient has diabetes or no
        
        input
        param diabetes: int type
        param notDiabetes: int type
        
        output
        returns 1 if the patient has diabetes, 0 if not and -1 if equal
    """
    if(diabetes > notDiabetes):
        return 1
    elif (diabetes < notDiabetes):
        return 0
    elif (diabetes == notDiabetes):
        return -1
    
    
  
def print_data(sample,data):
  """
    function to print the training and testing data

    input
    :param sample: numpy matrix with data
    :param data: string type variable

    output
    prints the testing and training data
  """
  if(data == "testing"):
    print('------------------------------------------------')
    print('Testing data')
    print('------------------------------------------------')
    print(sample)
  if(data == "training"):
    print('------------------------------------------------')
    print('Training data')
    print('------------------------------------------------')
    print(sample)


def print_scaled_data(scaled_data,data):
  """
    function to print the training and testing data scalated

    input
    :param sample: numpy matrix with data
    :param data: string type variable

    output
    prints the testing and training data scalated
  """
  if(data == "testing"):
    print('------------------------------------------------')
    print('Testing data scaled')
    print('------------------------------------------------')
    print(scaled_data)
  if(data == "training"):
    print('------------------------------------------------')
    print('Training data scaled')
    print('------------------------------------------------')
    print(scaled_data)
  

def covariance_matrix(testingArray, testing_y,k):
    """
        function that calculated the true positives, false positives,
        false negatives and true negatives with the predicted values 
        and the actual values
        
        input
        :param predicted: numpy array type
        :param y: numpy array type
        
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    for i in range(testingArray.shape[0]):
        if(testingArray[i] == 1 and testing_y[i] == 1):
            true_positive += 1
        elif(testingArray[i] == 1 and testing_y[i] == 0):
            false_positive += 1
        elif(testingArray[i] == 0 and testing_y[i] == 1):
            false_negative += 1
        elif(testingArray[i] == 0 and testing_y[i] == 0):
            true_negative += 1
    
    # calculates the accuracy
    accuracy = (true_positive + true_negative)/(true_positive + true_negative
               + false_negative + false_positive)
    # calculates the precision
    precision = (true_positive)/(true_positive + false_positive)
    #calculates the recall
    recall = (true_positive)/(true_positive + false_negative)
     # calculates the specificity
    specificity = (true_negative)/(true_negative + false_positive)
     # calculates the f1_score
    f1_score = (2*precision*recall)/(precision+recall)
    print_confusion_matrix(true_positive,false_positive,false_negative,
                            true_negative, accuracy, precision, recall, specificity, f1_score,k)
    return accuracy,precision,recall,specificity,f1_score
    
    
def print_confusion_matrix(tp,fp,fn,tn, accuracy, precision, recall, specificity, f1_score, k):
    """
        function that prints the covariance matrix
        
        input
        :param tp: int type
        :param fp: int type
        :param fn: int type
        :param tn: int type
        
        output
        :prints the covariance matrix, accuracy, 
    """
    print('----------------------------------------------------------------------------------')
    print('Confusion Matrix of K = ',k)
    print('----------------------------------------------------------------------------------')
    print('                                                                                  ')
    print('                                                        Actual Class              ')
    print('                                                                                  ')
    print('                                             Granted(1)                Refused(0) ')
    print('                                                                                  ')
    print('Predicted Class            Granted(1)     True Positives:',tp, '     False Positives: ',fp)
    print('                           Refused(0)     False Negatives:',fn,'     True Negatives:',tn)
    print('                                                                                  ')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recal: ',recall)
    print('Specificity: ',specificity)
    print('F1 score: ',f1_score)
    



    
