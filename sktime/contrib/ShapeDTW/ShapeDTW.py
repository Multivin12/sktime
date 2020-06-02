
#utilities
import numpy as np
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe as load_ts

#Transforms
from sktime.transformers.dictionary_based import PAA

#Classifiers
from sktime.classifiers.base import BaseClassifier
from sktime.classifiers.distance_based import KNeighborsTimeSeriesClassifier

#Write ShapeDTW as a classifier (that implements BaseClassifier).
#Wrapper for KNeighborsTimeSeriesClassifier that uses DTW as a distance metric

class ShapeDTW(BaseClassifier):

    """
    Parameters
    ----------
    subsequence_length          : int, defines the length of the subsequences.
    shape_descriptor_function   : string, defines the function to describe the set of subsequences (default = 'raw')
    
    The possible shape descriptor functions are as follows:
        - 'raw'                 : use the raw subsequence as the shape descriptor function.
        - 'paa'                 : use PAA as the shape descriptor function.
        - 'dwt'                 : use DWT (Discrete Wavelet Transform) as the shape descriptor function.
        - 'slope'               : use the gradient of each subsequence fitted by a total least squares regression as the shape descriptor function.
        - 'derivative'          : use the derivative of each subsequence as the shape descriptor function.
        - 'hog1d'               : use a histogram of gradients in one dimension as the shape desciptor function.
        - 'compound'            : use a combination of two or more shape descriptors simultaneously.
    shape_descriptor_functions  : string list, only applicable when the shape_descriptor_function is set to 'compound'. Use a list of
                                  shape descriptor functions at the same time.
    """
    def __init__(self,subsequence_length=5,shape_descriptor_function='raw',shape_descriptor_functions=None):
        pass


    """
    Parameters
    ----------
    X - training data.
    y - list of class labels.
    
    Returns
    -------
    self : object
    """
    def fit(self, X, y, input_checks=True):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("ShapeDTW cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0,0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects (ShapeDTW cannot yet handle multivariate problems")
                
        n_samps, self.series_length = X.shape
        
        raise NotImplementedError('this is an abstract method')

    """
    Find probability estimates for each class for all cases in X.
    Parameters
    ----------
    X : The training input samples. array-like or sparse matrix of shape = [n_test_instances, series_length]
        If a Pandas data frame is passed (sktime format) a check is performed that it only has one column.
        If not, an exception is thrown, since this classifier does not yet have
        multivariate capability.

    Returns
    -------
    output : array of shape = [n_test_instances, num_classes] of probabilities
    """
    def predict_proba(self, X, input_checks=True):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("ShapeDTW cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0,0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects (ShapeDTW cannot yet handle multivariate problems")
                
        n_samps, self.series_length = X.shape
        raise NotImplementedError('this is an abstract method')
        
    """
    Find predictions for all cases in X. Built on top of predict_proba
    Parameters
    ----------
    X : The training input samples. array-like or pandas data frame.
    If a Pandas data frame is passed, a check is performed that it only has one column.
    If not, an exception is thrown, since this classifier does not yet have
    multivariate capability.

    Returns
    -------
    output : array of shape = [n_test_instances]
    """
    def predict(self, X, input_checks=True):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] > 1:
                raise TypeError("ShapeDTW cannot handle multivariate problems yet")
            elif isinstance(X.iloc[0,0], pd.Series):
                X = np.asarray([a.values for a in X.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects (ShapeDTW cannot yet handle multivariate problems")
                
        n_samps, self.series_length = X.shape
        pass 
        
if __name__ == "__main__":
    testPath="C:\\Users\\Vince\\Documents\\Dissertation Repositories\\datasets\\Univariate2018_ts\\Chinatown\\Chinatown_TRAIN.ts"
    trainData,trainDataClass =  load_ts(testPath)

    num_atts = trainData.shape[1]
    num_insts = trainData.shape[0]
    
    if isinstance(trainData, pd.DataFrame):
            if trainData.shape[1] > 1:
                raise TypeError("ShapeDTW cannot handle multivariate problems yet")
            elif isinstance(trainData.iloc[0,0], pd.Series):
                trainData = np.asarray([a.values for a in trainData.iloc[:,0]])
            else:
                raise TypeError("Input should either be a 2d numpy array, or a pandas dataframe with a single column of Series objects (ShapeDTW cannot yet handle multivariate problems")


    first = trainData[0,:]
    second = trainData[1,:]
    print(first)
    