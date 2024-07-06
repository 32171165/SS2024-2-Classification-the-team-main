import pytest

from decision_tree import DecisionTree

from classes.decision_tree_internal_node import DecisionTreeInternalNode
from classes.decision_tree_leaf_node import DecisionTreeLeafNode
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
# Test with the small student dataset (full recursion)
#####


def test_with_small_student_dataset_and_information_gain(small_student_dataset):
    """
    Test with the small student dataset (using the "Passed" attribute as the target attribute as
    intended) and information gain as splitting criterion.
    """
    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Fit the decision tree
    decision_tree.fit(
        dataset=small_student_dataset,
        target_attribute="Passed",
        attribute_selection_method="information_gain",
    )

    # Get the decision tree root node
    root_node = decision_tree.tree

    # Check if the decision tree is correct

    # Check the root node
    assert isinstance(root_node, DecisionTreeInternalNode)
    assert root_node.get_label() == "Participation"

    # Check the root nodes' branches
    # There has to be 3 branches
    assert len(root_node.get_branches()) == 3

    # Get the branch with get_label() == "High"
    high_branches = [
        branch
        for branch in root_node.get_branches()
        if branch.get_label().value == "High"
    ]
    assert (
        len(high_branches) == 1
    )  # There should be exactly one branch with the label "High"
    high_branch = high_branches[0]
    assert isinstance(high_branch.get_label(), DecisionTreeDecisionOutcomeEquals)

    # Check the node the branch with label "High" leads to
    assert isinstance(high_branch.get_branch_node(), DecisionTreeLeafNode)
    assert high_branch.get_branch_node().get_label() == "Yes"

    # Get the branch with get_label() == "Low"
    low_branches = [
        branch
        for branch in root_node.get_branches()
        if branch.get_label().value == "Low"
    ]
    assert (
        len(low_branches) == 1
    )  # There should be exactly one branch with the label "Low"
    low_branch = low_branches[0]

    # Check the node the branch with label "Low" leads to
    assert isinstance(low_branch.get_branch_node(), DecisionTreeLeafNode)
    assert low_branch.get_branch_node().get_label() == "No"

    # Get the branch with get_label() == "Medium"
    medium_branches = [
        branch
        for branch in root_node.get_branches()
        if branch.get_label().value == "Medium"
    ]
    assert (
        len(medium_branches) == 1
    )  # There should be exactly one branch with the label "Medium"
    medium_branch = medium_branches[0]

    # Check the node the branch with label "Medium" leads to
    assert isinstance(medium_branch.get_branch_node(), DecisionTreeInternalNode)
    assert medium_branch.get_branch_node().get_label() == "Age"

    # Go one level deeper (starting at the age node)
    age_node = medium_branch.get_branch_node()

    # Check the branches of the age node
    # There has to be 2 branches
    assert len(age_node.get_branches()) == 2

    # Get the branch with class DecisionTreeDecisionOutcomeAbove
    above_branches = [
        branch
        for branch in age_node.get_branches()
        if isinstance(branch.get_label(), DecisionTreeDecisionOutcomeAbove)
    ]
    assert (
        len(above_branches) == 1
    )  # There should be exactly one branch with the class DecisionTreeDecisionOutcomeAbove
    above_branch = above_branches[0]
    assert above_branch.get_label().value == pytest.approx(25.0)

    # Check the node the branch with class DecisionTreeDecisionOutcomeAbove leads to
    assert isinstance(above_branch.get_branch_node(), DecisionTreeLeafNode)
    assert above_branch.get_branch_node().get_label() == "No"

    # Get the branch with class DecisionTreeDecisionOutcomeBelowEqual
    below_equal_branches = [
        branch
        for branch in age_node.get_branches()
        if isinstance(branch.get_label(), DecisionTreeDecisionOutcomeBelowEqual)
    ]
    assert (
        len(below_equal_branches) == 1
    )  # There should be exactly one branch with the class DecisionTreeDecisionOutcomeBelowEqual
    below_equal_branch = below_equal_branches[0]
    assert below_equal_branch.get_label().value == pytest.approx(25.0)

    # Check the node the branch with class DecisionTreeDecisionOutcomeBelowEqual leads to
    assert isinstance(below_equal_branch.get_branch_node(), DecisionTreeLeafNode)
    assert below_equal_branch.get_branch_node().get_label() == "Yes"


# There is no test for the gini index as the splitting criterion because with the small student dataset, the gini index can lead to many different decision trees. We might add a test for the gini index as the splitting criterion in the future with an other dataset but for now the test of the private helper functions shall be enough.
