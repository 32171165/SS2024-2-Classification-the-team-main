import pytest

import gini_index

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset(small_student_dataset):
    """
    Test the calculate_impurity function with the small student dataset (using the "Passed" attribute as the target attribute as intended).
    """
    # Calculate the entropy of the dataset
    entropy = gini_index.calculate_impurity(
        dataset=small_student_dataset, target_attribute="Passed"
    )

    # Check if the calculated entropy is correct
    assert entropy == pytest.approx(0.5)
