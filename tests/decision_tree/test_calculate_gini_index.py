import pytest

from decision_tree import DecisionTree

from classes.decision_tree_decision_outcome_above import (
    DecisionTreeDecisionOutcomeAbove,
)
from classes.decision_tree_decision_outcome_below_equal import (
    DecisionTreeDecisionOutcomeBelowEqual,
)
from classes.decision_tree_decision_outcome_in_list import (
    DecisionTreeDecisionOutcomeInList,
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

    # Calculate the gini index (reduction of impurity) for the partitioned dataset
    gini_index_value, outcomes = decision_tree._calculate_gini_index(
        data=small_student_dataset,
        attribute="Age",
    )

    # Check if the calculated gini index (reduction of impurity) is correct
    assert gini_index_value == pytest.approx(0.05555555555555558)

    # Check if the outcomes are correct (25.0 has the lowest gini index)
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

    # Calculate the gini index (reduction of impurity) for the partitioned dataset
    gini_index_value, outcomes = decision_tree._calculate_gini_index(
        data=small_student_dataset,
        attribute="Major",
    )

    # Check if the calculated gini index (reduction of impurity) is correct
    assert gini_index_value == pytest.approx(0.1)

    # Check if the outcomes are correct
    assert len(outcomes) == 2
    assert DecisionTreeDecisionOutcomeInList(values=["CS"]) in outcomes
    assert DecisionTreeDecisionOutcomeInList(values=["DS"]) in outcomes


def test_with_small_student_dataset_attribute_participation(small_student_dataset):
    """
    Test with the small student dataset (using the "Passed" attribute as the target attribute as
    intended) and the discrete-valued "Participation" attribute as the partition attribute.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Calculate the gini index (reduction of impurity) for the partitioned dataset
    gini_index_value, outcomes = decision_tree._calculate_gini_index(
        data=small_student_dataset,
        attribute="Participation",
    )

    # Check if the calculated gini index (reduction of impurity) is correct
    assert gini_index_value == pytest.approx(0.25)

    # Check if the outcomes are correct
    # (The split {"Low"}/{"Medium", "High"} and {"High"}/{"Medium", "Low"} have the lowest gini index)
    assert len(outcomes) == 2
    assert (
        DecisionTreeDecisionOutcomeInList(values=["Medium", "High"]) in outcomes
        and DecisionTreeDecisionOutcomeInList(values=["Low"]) in outcomes
    ) or (
        DecisionTreeDecisionOutcomeInList(values=["Low", "Medium"]) in outcomes
        and DecisionTreeDecisionOutcomeInList(values=["High"]) in outcomes
    )


# There is no full recursive test for gini_index, as the tree can be built in too many different ways (since the gini index is often exactly the same for multiple attributes/splits)
