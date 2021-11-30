# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:54:45 2020
@author: Donovan
@modified
"""
import json
from json import JSONEncoder

class ML_Model:
    """
    This class creates a machine learning model based on the data sent,
    data preprocessing, and type of ml classifier.

    """

    def __init__(self, train_data, ml_classifier, preprocess):
        """
        This function controls the initial creation of the machine learning model.

        Parameters
        ----------
        train_data : pandas DataFrame
            The data the machine learning model will be built on.
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.

        Attributes
        -------
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        X : pandas DataFrame
            The features in the train set.
        y : pandas Series
            The responce variable.
        ml_model : fitted machine learning classifier
            The machine learning model created using the training data.
        """
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess

        self.X = train_data.iloc[:,: -2].values
        self.y = train_data.iloc[:, -1].values
               
        self.X = self.preprocess.fit_transform(self.X)

        self.ml_model = ml_classifier.fit(self.X, self.y)

    def predict_train_image(self):
        """
        This function predicts the labels for a new set of train data that contains user labels.
        It returns these predictions and the probability.
        
        Attributes
        ----------
        X : pandas DataFrame
            The features in the train set.
        ml_model : Trained RandomForrest model.

        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        y_prediction = self.ml_model.predict(self.X)
        y_probabilities = self.ml_model.predict_proba(self.X)#Get probabilites for each label
        y_probabilities = [max(prob) for prob in y_probabilities]#Choose the max probability out of 2 label
        return y_prediction, max(y_probabilities) # Return prediction along with prediction
    
    def predict_test_image(self,test_images):
        """
        This function predicts the labels for a new set of test data that does not contains labels.
        It returns these predictions and the probability.
        Parameters
        ----------
        test_images : pandas DataFrame
            The new data to be labeled.
        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """      
        test_images = test_images.iloc[:,: -1].values   
        test_images = self.preprocess.fit_transform(test_images)
        y_prediction = self.ml_model.predict(test_images)
        y_probabilities = self.ml_model.predict_proba(test_images)#Get probabilites for each label
        y_probabilities = [max(prob) for prob in y_probabilities]#Choose the max probability out of 2 label
        return y_prediction, y_probabilities # Return prediction along with prediction
        