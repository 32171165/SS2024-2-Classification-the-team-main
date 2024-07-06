import pytest

import information_gain

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset_and_major_as_partition_attribute(
    small_student_dataset,
):
    """
    Test the calculate_information_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended) and the discrete-valued "Major" attribute as the partition attribute.
    """

    # Calculate the information for the partitioned dataset
    information = information_gain.calculate_information_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Major",
    )

    # Check if the calculated information is correct
    assert information == pytest.approx(0.8091254953788906)


def test_with_small_student_dataset_and_participation_as_partition_attribute(
    small_student_dataset,
):
    """
    Test the calculate_information_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended) and the discrete-valued "Participation" attribute as the partition attribute.
    """

    # Calculate the information for the partitioned dataset
    information = information_gain.calculate_information_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Participation",
    )

    # Check if the calculated information is correct
    assert information == pytest.approx(0.3333333333333333)


def test_with_small_student_dataset_age_as_partition_attribute_and_a_split_point_at_23_5(
    small_student_dataset,
):
    """
    Test the calculate_information_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the continuous-valued "Age" attribute as the partition attribute and a split point at 23.5.
    """

    # Calculate the information for the partitioned dataset
    information = information_gain.calculate_information_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Age",
        split_value=23.5,
    )

    # Check if the calculated information is correct
    assert information == pytest.approx(1)


def test_with_small_student_dataset_age_as_partition_attribute_and_a_split_point_at_25_5(
    small_student_dataset,
):
    """
    Test the calculate_information_partitioned function with the small student dataset (using the "Passed" attribute as the target attribute as intended), the continuous-valued "Age" attribute as the partition attribute and a split point at 25.5.
    """

    # Calculate the information for the partitioned dataset
    information = information_gain.calculate_information_partitioned(
        dataset=small_student_dataset,
        target_attribute="Passed",
        partition_attribute="Age",
        split_value=25.5,
    )

    # Check if the calculated information is correct
    assert information == pytest.approx(0.9182958340544896)
