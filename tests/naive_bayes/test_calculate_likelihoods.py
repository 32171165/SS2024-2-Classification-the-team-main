import pytest

from naive_bayes import NaiveBayes

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset(small_student_dataset):
    """
    Test the calculation of the likelihoods with the small student dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Calculate the likelihoods
    likelihoods = naive_bayes._calculate_likelihoods(small_student_dataset)

    # Check if the likelihoods are correct
    # Attribute 'Age'
    assert likelihoods.likelihoods["Age"]["Yes"]["mean"] == pytest.approx(
        24.333333333333332
    )
    assert likelihoods.likelihoods["Age"]["Yes"]["std"] == pytest.approx(
        1.5275252316519468
    )
    assert likelihoods.likelihoods["Age"]["No"]["mean"] == pytest.approx(25.0)
    assert likelihoods.likelihoods["Age"]["No"]["std"] == pytest.approx(
        1.7320508075688772
    )

    # Attribute 'Major'
    assert likelihoods.likelihoods["Major"]["CS"]["Yes"] == pytest.approx(
        0.3333333333333333
    )
    assert likelihoods.likelihoods["Major"]["CS"]["No"] == pytest.approx(0.0)
    assert likelihoods.likelihoods["Major"]["DS"]["Yes"] == pytest.approx(
        0.6666666666666666
    )
    assert likelihoods.likelihoods["Major"]["DS"]["No"] == pytest.approx(1.0)

    # Attribute 'Participation'
    assert likelihoods.likelihoods["Participation"]["High"]["Yes"] == pytest.approx(
        0.6666666666666666
    )
    assert likelihoods.likelihoods["Participation"]["High"]["No"] == pytest.approx(0.0)
    assert likelihoods.likelihoods["Participation"]["Low"]["Yes"] == pytest.approx(0.0)
    assert likelihoods.likelihoods["Participation"]["Low"]["No"] == pytest.approx(
        0.6666666666666666
    )
    assert likelihoods.likelihoods["Participation"]["Medium"]["Yes"] == pytest.approx(
        0.3333333333333333
    )
    assert likelihoods.likelihoods["Participation"]["Medium"]["No"] == pytest.approx(
        0.3333333333333333
    )


#####
# Test with the small submission dataset
#####


def test_with_small_submission_dataset(small_submission_dataset):
    """
    Test the calculation of the likelihoods with the small submission dataset.
    """
    # Create a NaiveBayes object
    naive_bayes = NaiveBayes()

    # Set the target attribute
    naive_bayes.target_attribute = "Passed"

    # Calculate the likelihoods
    likelihoods = naive_bayes._calculate_likelihoods(small_submission_dataset)

    # Check if the likelihoods are correct
    # Attribute "Topic"
    assert likelihoods.likelihoods["Topic"]["Classification"]["Yes"] == pytest.approx(
        0.2857142857142857
    )
    assert likelihoods.likelihoods["Topic"]["Classification"]["No"] == pytest.approx(
        0.3333333333333333
    )
    assert likelihoods.likelihoods["Topic"]["Clustering"]["Yes"] == pytest.approx(
        0.42857142857142855
    )
    assert likelihoods.likelihoods["Topic"]["Clustering"]["No"] == pytest.approx(
        0.3333333333333333
    )
    assert likelihoods.likelihoods["Topic"]["Frequent Patterns"][
        "Yes"
    ] == pytest.approx(0.2857142857142857)
    assert likelihoods.likelihoods["Topic"]["Frequent Patterns"]["No"] == pytest.approx(
        0.3333333333333333
    )

    # Attribute "Knowledge"
    assert likelihoods.likelihoods["Knowledge"]["High"]["Yes"] == pytest.approx(
        0.2857142857142857
    )
    assert likelihoods.likelihoods["Knowledge"]["High"]["No"] == pytest.approx(
        0.6666666666666666
    )
    assert likelihoods.likelihoods["Knowledge"]["Low"]["Yes"] == pytest.approx(
        0.2857142857142857
    )
    assert likelihoods.likelihoods["Knowledge"]["Low"]["No"] == pytest.approx(
        0.3333333333333333
    )
    assert likelihoods.likelihoods["Knowledge"]["Medium"]["Yes"] == pytest.approx(
        0.42857142857142855
    )
    assert likelihoods.likelihoods["Knowledge"]["Medium"]["No"] == pytest.approx(0.0)

    # Attribute "Hours"
    assert likelihoods.likelihoods["Hours"]["Yes"]["mean"] == pytest.approx(
        4.428571428571429
    )
    assert likelihoods.likelihoods["Hours"]["Yes"]["std"] == pytest.approx(
        1.1338934190276817
    )
    assert likelihoods.likelihoods["Hours"]["No"]["mean"] == pytest.approx(
        2.3333333333333335
    )
    assert likelihoods.likelihoods["Hours"]["No"]["std"] == pytest.approx(
        1.5275252316519468
    )
