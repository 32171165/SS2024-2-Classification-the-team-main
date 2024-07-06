import pytest

from decision_tree import DecisionTree

from classes.decision_tree_decision_outcome_above import (
    DecisionTreeDecisionOutcomeAbove,
)
from classes.decision_tree_decision_outcome_below_equal import (
    DecisionTreeDecisionOutcomeBelowEqual,
)
from classes.decision_tree_decision_outcome_equals import (
    DecisionTreeDecisionOutcomeEquals,
)

#####
# Test with the small student dataset
#####


def test_with_small_student_dataset_attribute_age(small_student_dataset):
    """
    Test with the small student dataset (using the "Passed" attribute as the target attribute as
    intended) and the continuous-valued "Age" attribute as the partition attribute.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Calculate the information gain for the partitioned dataset
    information_gain_value, outcomes = decision_tree._calculate_information_gain(
        data=small_student_dataset,
        attribute="Age",
    )

    # Check if the calculated information gain is correct
    assert information_gain_value == pytest.approx(0.08170416594551044)

    # Check if the outcomes are correct (25.0 has the highest information gain)
    assert len(outcomes) == 2
    assert DecisionTreeDecisionOutcomeAbove(value=25.0) in outcomes
    assert DecisionTreeDecisionOutcomeBelowEqual(value=25.0) in outcomes


def test_with_small_student_dataset_attribute_major(small_student_dataset):
    """
    Test with the small student dataset (using the "Passed" attribute as the target attribute as
    intended) and the discrete-valued "Major" attribute as the partition attribute.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Calculate the information gain for the partitioned dataset
    information_gain_value, outcomes = decision_tree._calculate_information_gain(
        data=small_student_dataset,
        attribute="Major",
    )

    # Check if the calculated information gain is correct
    assert information_gain_value == pytest.approx(0.19087450462110944)

    # Check if the outcomes are correct
    assert len(outcomes) == 2
    assert DecisionTreeDecisionOutcomeEquals(value="DS") in outcomes
    assert DecisionTreeDecisionOutcomeEquals(value="CS") in outcomes


def test_with_small_student_dataset_attribute_participation(small_student_dataset):
    """
    Test with the small student dataset (using the "Passed" attribute as the target attribute as
    intended) and the discrete-valued "Participation" attribute as the partition attribute.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Calculate the information gain for the partitioned dataset
    information_gain_value, outcomes = decision_tree._calculate_information_gain(
        data=small_student_dataset,
        attribute="Participation",
    )

    # Check if the calculated information gain is correct
    assert information_gain_value == 0.6666666666666667

    # Check if the outcomes are correct
    assert len(outcomes) == 3
    assert DecisionTreeDecisionOutcomeEquals(value="High") in outcomes
    assert DecisionTreeDecisionOutcomeEquals(value="Medium") in outcomes
    assert DecisionTreeDecisionOutcomeEquals(value="Low") in outcomes
