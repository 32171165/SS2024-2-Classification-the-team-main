import pytest

from naive_bayes import NaiveBayes

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset(small_student_dataset):
    """
    Test the calculation of the prior probabilities with the small student dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Calculate the prior probabilities
    prior_probabilities = naive_bayes._calculate_prior_probabilities(
        small_student_dataset
    )

    # Check if the prior probabilities are correct
    assert prior_probabilities.get_prior_probability("Yes") == pytest.approx(0.5)
    assert prior_probabilities.get_prior_probability("No") == pytest.approx(0.5)


#####
# Test with the small submission dataset
#####


def test_with_small_submission_dataset(small_submission_dataset):
    """
    Test the calculation of the prior probabilities with the small submission dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Calculate the prior probabilities
    prior_probabilities = naive_bayes._calculate_prior_probabilities(
        small_submission_dataset
    )

    # Check if the prior probabilities are correct
    assert prior_probabilities.get_prior_probability("Yes") == pytest.approx(0.7)
    assert prior_probabilities.get_prior_probability("No") == pytest.approx(0.3)
