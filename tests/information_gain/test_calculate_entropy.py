import pytest

import information_gain

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset(small_student_dataset):
    """
    Test the calculate_entropy function with the small student dataset (using the "Passed" attribute as the target attribute as intended).
    """

    # Calculate the entropy of the dataset
    entropy = information_gain.calculate_entropy(
        dataset=small_student_dataset, target_attribute="Passed"
    )

    # Check if the calculated entropy is correct
    assert entropy == pytest.approx(1)
