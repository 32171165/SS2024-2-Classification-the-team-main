import pandas as pd

from naive_bayes import NaiveBayes

from classes.naive_bayes_likelihoods import NaiveBayesLikelihoods
from classes.naive_bayes_prior_probabilities import NaiveBayesPriorProbabilities

#####
# Tests with likelihoods and prior probabilities inspired by the small student dataset
#####


def test_with_values_inspired_by_the_small_student_dataset():
    """
    Test _predict_tuple() with likelihoods and prior probabilities inspired by the small student dataset.

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

    # Create a tuple to predict
    tuple_to_predict = pd.Series(
        {
            "Age": 20.5,
            "Major": "CS",
            "Participation": "High",
        }
    )

    # Predict the tuple
    prediction = naive_bayes._predict_tuple(tuple=tuple_to_predict)

    # Check if the prediction is correct
    assert prediction == "Yes"

    # Create a second tuple to predict
    tuple_to_predict = pd.Series(
        {
            "Age": 30.5,
            "Major": "DS",
            "Participation": "Low",
        }
    )

    # Predict the second tuple
    prediction = naive_bayes._predict_tuple(tuple=tuple_to_predict)

    # Check if the prediction is correct
    assert prediction == "No"


#####
# Tests with likelihoods and prior probabilities inspired by the small submission dataset
#####


def test_with_values_inspired_by_the_small_submission_dataset():
    """
    Test _predict_tuple() with likelihoods and prior probabilities inspired by the small submission dataset.
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

    # Create a tuple to predict
    tuple_to_predict = pd.Series(
        {
            "Topic": "Classification",
            "Knowledge": "Medium",
            "Hours": 4,
        }
    )

    # Predict the tuple
    prediction = naive_bayes._predict_tuple(tuple=tuple_to_predict)

    # Check if the prediction is correct
    assert prediction == "Yes"

    # Create a second tuple to predict
    tuple_to_predict = pd.Series(
        {
            "Topic": "Clustering",
            "Knowledge": "Low",
            "Hours": 1,
        }
    )

    # Predict the second tuple
    prediction = naive_bayes._predict_tuple(tuple=tuple_to_predict)

    # Check if the prediction is correct
    assert prediction == "No"
