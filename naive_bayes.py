import pandas as pd
from typing import List, Set

from classes.naive_bayes_likelihoods import NaiveBayesLikelihoods
from classes.naive_bayes_prior_probabilities import NaiveBayesPriorProbabilities


class NaiveBayes:
    """
    A Naive Bayes classifier.
    """

    def __init__(self):
        # The target attribute to predict
        self.target_attribute: str = None

        # A full set of possible class labels ( = values of the target attribute)
        self.class_labels: Set[str | int | float] = None

        # The likelihoods of the classifier
        self.likelihoods: NaiveBayesLikelihoods = None

        # The prior probabilities of the classifier
        self.prior_probabilities: NaiveBayesPriorProbabilities = None

    def fit(self, dataset: pd.DataFrame, target_attribute: str):
        """
        Fit the Naive Bayes classifier to the training dataset.
        Sets the target attribute and the class labels.
        Calculates the prior probabilities, and the likelihoods.

        Parameters:
        dataset (pd.DataFrame): The training dataset
        target_attribute (str): The target attribute to predict
        """
        # Make sure that the target_attribute is in the dataset
        if target_attribute not in dataset.columns:
            raise ValueError(f"Target attribute '{target_attribute}' not in dataset.")

        # TODO

    def _calculate_prior_probabilities(
        self, dataset: pd.DataFrame
    ) -> NaiveBayesPriorProbabilities:
        """
        Calculate the prior probability for each class.
        (The target attribute has to be set before calling this method.)

        Parameters:
        dataset (pd.DataFrame): The training dataset

        Returns:
        NaiveBayesPriorProbabilities: The prior probabilities for each class
        """
        # Make sure that the target_attribute is set
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # TODO

    def _calculate_likelihoods(self, dataset: pd.DataFrame) -> NaiveBayesLikelihoods:
        """
        Calculate the likelihoods for each attribute and class.
        (The target attribute has to be set before calling this method.)

        Parameters:
        dataset (pd.DataFrame): The training dataset

        Returns:
        NaiveBayesLikelihoods: The likelihoods for each attribute and class
        """
        # Make sure that the target_attribute is set
        if self.target_attribute is None:
            raise ValueError("Target attribute not set.")

        # TODO

    def predict(self, dataset: pd.DataFrame) -> List[str | int | float]:
        """
        Predict the target attribute for a given dataset.

        Parameters:
        dataset (pd.DataFrame): The dataset to predict the target attribute for

        Returns:
        List[str | int | float]: A list of predicted class labels
        """

        # If the likelihoods or/and the prior probabilities are not calculated yet, raise an error
        if self.likelihoods is None or self.prior_probabilities is None:
            raise ValueError("Model not trained yet.")

        # TODO

    def _predict_tuple(self, tuple: pd.Series) -> str | int | float:
        """
        Predict the target attribute for a given row in the dataset.

        Parameters:
        tuple (pd.Series): The row in the dataset to predict the target attribute for

        Returns:
        str | int | float: The predicted class label
        """
        # TODO
