import pandas as pd

from naive_bayes import NaiveBayes

from classes.naive_bayes_likelihoods import NaiveBayesLikelihoods
from classes.naive_bayes_prior_probabilities import NaiveBayesPriorProbabilities

#####
# Tests with likelihoods and prior probabilities inspired by the small student dataset
#####


def test_with_values_inspired_by_the_small_student_dataset():
    """
    Test predict() with likelihoods and prior probabilities inspired by the small student dataset.

    The biggest difference to the likelihoods and prior probabilities in the small student dataset is that the likelihoods for "High" and "Low" are not 0.0 in this test, to avoid multiplying by 0. This is also a problem in practice, and can be solved by using Laplacian correction or similar techniques. In this submission, we however keep it simple and avoid test cases with 0.0 likelihoods.
    """

    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Set the class labels
    naive_bayes.class_labels = {"Yes", "No"}

    # Set the prior probabilities
    naive_bayes.prior_probabilities = NaiveBayesPriorProbabilities()
    naive_bayes.prior_probabilities.add_prior_probability("Yes", 0.5)
    naive_bayes.prior_probabilities.add_prior_probability("No", 0.5)

    # Set the likelihoods
    naive_bayes.likelihoods = NaiveBayesLikelihoods()
    naive_bayes.likelihoods.add_continuous_likelihood(
        "Age", "Yes", 24.333333333333332, 1.5275252316519468
    )
    naive_bayes.likelihoods.add_continuous_likelihood(
        "Age", "No", 25.0, 1.7320508075688772
    )

    naive_bayes.likelihoods.add_categorical_likelihood(
        "Major", "CS", "Yes", 0.3333333333333333
    )
    naive_bayes.likelihoods.add_categorical_likelihood("Major", "CS", "No", 0.0)
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Major", "DS", "Yes", 0.6666666666666666
    )
    naive_bayes.likelihoods.add_categorical_likelihood("Major", "DS", "No", 1.0)

    naive_bayes.likelihoods.add_categorical_likelihood(
        "Participation", "High", "Yes", 0.6666666666666666
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Participation", "High", "No", 0.01
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Participation", "Low", "Yes", 0.01
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Participation", "Low", "No", 0.6666666666666666
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Participation", "Medium", "Yes", 0.3333333333333333
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Participation", "Medium", "No", 0.3333333333333333
    )

    # Create a dataset to predict
    data = pd.DataFrame(
        {
            "Age": [21, 25, 11],
            "Major": ["CS", "DS", "DS"],
            "Participation": ["High", "Low", "Medium"],
        }
    )

    # Predict the dataset
    predictions = naive_bayes.predict(data)

    # Check if the predictions are correct
    assert predictions == ["Yes", "No", "No"]


#####
# Tests with likelihoods and prior probabilities inspired by the small submission dataset
#####


def test_with_values_inspired_by_the_small_submission_dataset():
    """
    Test predict() with likelihoods and prior probabilities inspired by the small submission dataset.
    """

    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Set the class labels
    naive_bayes.class_labels = {"Yes", "No"}

    # Set the prior probabilities
    naive_bayes.prior_probabilities = NaiveBayesPriorProbabilities()
    naive_bayes.prior_probabilities.add_prior_probability("Yes", 0.7)
    naive_bayes.prior_probabilities.add_prior_probability("No", 0.3)

    # Set the likelihoods
    naive_bayes.likelihoods = NaiveBayesLikelihoods()
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Topic", "Classification", "Yes", 0.2857142857142857
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Topic", "Classification", "No", 0.3333333333333333
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Topic", "Clustering", "Yes", 0.42857142857142855
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Topic", "Clustering", "No", 0.3333333333333333
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Topic", "Frequent Patterns", "Yes", 0.2857142857142857
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Topic", "Frequent Patterns", "No", 0.3333333333333333
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Knowledge", "High", "Yes", 0.2857142857142857
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Knowledge", "High", "No", 0.6666666666666666
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Knowledge", "Low", "Yes", 0.2857142857142857
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Knowledge", "Low", "No", 0.3333333333333333
    )
    naive_bayes.likelihoods.add_categorical_likelihood(
        "Knowledge", "Medium", "Yes", 0.42857142857142855
    )
    naive_bayes.likelihoods.add_categorical_likelihood("Knowledge", "Medium", "No", 0.0)
    naive_bayes.likelihoods.add_continuous_likelihood(
        "Hours", "Yes", 4.428571428571429, 1.1338934190276817
    )
    naive_bayes.likelihoods.add_continuous_likelihood(
        "Hours", "No", 2.3333333333333335, 1.5275252316519468
    )

    # Create a dataset to predict
    data = pd.DataFrame(
        {
            "Topic": ["Clustering", "Classification", "Frequent Patterns"],
            "Knowledge": ["Medium", "High", "Low"],
            "Hours": [4, 3, 6.8],
        }
    )

    # Predict the dataset
    predictions = naive_bayes.predict(data)

    assert predictions == ["Yes", "No", "Yes"]
