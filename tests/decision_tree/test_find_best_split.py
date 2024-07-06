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
from classes.decision_tree_decision_outcome_in_list import (
    DecisionTreeDecisionOutcomeInList,
)

#####
# Test with the complete small student dataset
#####


def test_with_complete_small_student_dataset_and_information_gain(
    small_student_dataset,
):
    """
    Test with the complete small student dataset (using the "Passed" attribute as the
    target attribute as intended) and information gain as splitting criterion.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Find the best split
    split_attribute, outcomes = decision_tree._find_best_split(
        data=small_student_dataset,
        attribute_list=["Age", "Major", "Participation"],
        attribute_selection_method="information_gain",
    )

    # Check if the best split is correct (it should be "Participation")
    assert split_attribute == "Participation"

    # Check if the outcomes are correct
    assert len(outcomes) == 3
    assert DecisionTreeDecisionOutcomeEquals(value="High") in outcomes
    assert DecisionTreeDecisionOutcomeEquals(value="Medium") in outcomes
    assert DecisionTreeDecisionOutcomeEquals(value="Low") in outcomes


def test_with_complete_small_student_dataset_and_gini_index(small_student_dataset):
    """
    Test with the complete small student dataset (using the "Passed" attribute as the
    target attribute as intended) and the Gini index as splitting criterion.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Find the best split
    split_attribute, outcomes = decision_tree._find_best_split(
        data=small_student_dataset,
        attribute_list=["Age", "Major", "Participation"],
        attribute_selection_method="gini_index",
    )

    # Check if the best split is correct (it should be "Participation")
    assert split_attribute == "Participation"

    # Check if the outcomes are correct
    assert len(outcomes) == 2
    assert (
        DecisionTreeDecisionOutcomeInList(values=["Medium", "High"]) in outcomes
        and DecisionTreeDecisionOutcomeInList(values=["Low"]) in outcomes
    ) or (
        DecisionTreeDecisionOutcomeInList(values=["Low", "Medium"]) in outcomes
        and DecisionTreeDecisionOutcomeInList(values=["High"]) in outcomes
    )


#####
# Test with the small student dataset (participation = "Medium")
#####


def test_with_small_student_dataset_participation_medium_and_information_gain(
    small_student_dataset,
):
    """
    Test with the small student dataset (using the "Passed" attribute as the target attribute as
    intended), only "Medium" in Participation (first split) and information gain as splitting criterion.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Set the target attribute
    decision_tree.target_attribute = "Passed"

    # Only select tuples with "Medium" in the "Participation" attribute
    small_student_dataset_medium_participation = small_student_dataset[
        small_student_dataset["Participation"] == "Medium"
    ]

    # Find the best split
    split_attribute, outcomes = decision_tree._find_best_split(
        data=small_student_dataset_medium_participation,
        attribute_list=["Age", "Major"],
        attribute_selection_method="information_gain",
    )

    # Check if the best split is correct (it should be "Age")
    assert split_attribute == "Age"

    # Check if the outcomes are correct
    assert len(outcomes) == 2
    assert DecisionTreeDecisionOutcomeAbove(value=25.0) in outcomes
    assert DecisionTreeDecisionOutcomeBelowEqual(value=25.0) in outcomes
