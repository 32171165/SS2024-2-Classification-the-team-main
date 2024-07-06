import pytest

import gini_index


#####
# Test with the small student dataset
#####


def test_with_small_student_dataset_major_as_partition_attribute_and_cs_as_split(
    small_student_dataset,
):
    """
    Test the calculate_impurity_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the discrete-valued "Major" attribute as the partition attribute and the split "CS".
    """

    # Calculate the impurity for the partitioned dataset
    impurity_partitioned = gini_index.calculate_impurity_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Major",
        split={"CS"},
    )

    # Check if the calculated impurity is correct
    assert impurity_partitioned == pytest.approx(0.4)


def test_with_small_student_dataset_major_as_partition_attribute_and_ds_as_split(
    small_student_dataset,
):
    """
    Test the calculate_impurity_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the discrete-valued "Major" attribute as the partition attribute and the split "DS" (is basically the same split as with "CS", but this should be tested to make sure the function works correctly with different splits as well)
    """

    # Calculate the impurity for the partitioned dataset
    impurity_partitioned = gini_index.calculate_impurity_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Major",
        split={"DS"},
    )

    # Check if the calculated impurity is correct
    assert impurity_partitioned == pytest.approx(0.4)


def test_with_small_student_dataset_participation_as_partition_attribute_and_medium_high_as_split(
    small_student_dataset,
):
    """
    Test the calculate_impurity_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the discrete-valued "Participation" attribute as the partition attribute and the split {"Medium", "High"}.
    """

    # Calculate the impurity for the partitioned dataset
    impurity_partitioned = gini_index.calculate_impurity_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Participation",
        split={"Medium", "High"},
    )

    # Check if the calculated impurity is correct
    assert impurity_partitioned == pytest.approx(0.25)


def test_with_small_student_dataset_participation_as_partition_attribute_and_medium_as_split(
    small_student_dataset,
):
    """
    Test the calculate_impurity_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the discrete-valued "Participation" attribute as the partition attribute and the split {"Medium"}.
    """

    # Calculate the impurity for the partitioned dataset
    impurity_partitioned = gini_index.calculate_impurity_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Participation",
        split={"Medium"},
    )

    # Check if the calculated impurity is correct
    assert impurity_partitioned == pytest.approx(0.5)


def test_with_small_student_dataset_age_as_partition_attribute_and_split_23_5(
    small_student_dataset,
):
    """
    Test the calculate_impurity_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the continuous-valued "Age" attribute as the partition attribute and the split 23.5.
    """

    # Calculate the impurity for the partitioned dataset
    impurity_partitioned = gini_index.calculate_impurity_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Age",
        split=23.5,
    )

    # Check if the calculated impurity is correct
    assert impurity_partitioned == pytest.approx(0.5)


def test_with_small_student_dataset_age_as_partition_attribute_and_split_25_5(
    small_student_dataset,
):
    """
    Test the calculate_impurity_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the continuous-valued "Age" attribute as the partition attribute and the split 25.5.
    """

    # Calculate the impurity for the partitioned dataset
    impurity_partitioned = gini_index.calculate_impurity_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Age",
        split=25.5,
    )

    # Check if the calculated impurity is correct
    assert impurity_partitioned == pytest.approx(0.4444444444444444)
