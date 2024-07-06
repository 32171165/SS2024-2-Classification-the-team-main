from decision_tree import DecisionTree

from classes.decision_tree_leaf_node import DecisionTreeLeafNode
from classes.decision_tree_internal_node import DecisionTreeInternalNode
from classes.decision_tree_branch import DecisionTreeBranch
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
# Tests with a decision tree inspired by the small student dataset
# Tree structure:
# Participation
# ├── High: Yes
# ├── Medium: Age
# │   ├── <= 25: Yes
# │   └── > 25: No
# └── Low: No
#####


def test_with_decision_tree_inspired_by_small_student_dataset_and_hit_on_first_layer():
    """
    Test _predict_tuple() with a decision tree inspired by the small student dataset.
    The leaf node is hit on the first layer.
    """

    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Create the decision tree
    decision_tree.tree = DecisionTreeInternalNode(
        attribute_label="Participation",
        branches=[
            DecisionTreeBranch(
                label=DecisionTreeDecisionOutcomeEquals(value="High"),
                branch_node=DecisionTreeLeafNode(class_label="Yes"),
            ),
            DecisionTreeBranch(
                label=DecisionTreeDecisionOutcomeEquals(value="Medium"),
                branch_node=DecisionTreeInternalNode(
                    attribute_label="Age",
                    branches=[
                        DecisionTreeBranch(
                            label=DecisionTreeDecisionOutcomeBelowEqual(value=25),
                            branch_node=DecisionTreeLeafNode(class_label="Yes"),
                        ),
                        DecisionTreeBranch(
                            label=DecisionTreeDecisionOutcomeAbove(value=25),
                            branch_node=DecisionTreeLeafNode(class_label="No"),
                        ),
                    ],
                ),
            ),
            DecisionTreeBranch(
                label=DecisionTreeDecisionOutcomeEquals(value="Low"),
                branch_node=DecisionTreeLeafNode(class_label="No"),
            ),
        ],
    )

    # Create a tuple to predict
    tuple_to_predict = {
        "Age": 20.5,
        "Major": "CS",
        "Participation": "High",
    }

    # Predict the tuple
    prediction = decision_tree._predict_tuple(
        tuple=tuple_to_predict,
        node=decision_tree.tree,
    )

    # Check if the prediction is correct
    assert prediction == "Yes"

    # Create a second tuple to predict
    tuple_to_predict = {
        "Age": 30.5,
        "Major": "CS",
        "Participation": "Low",
    }

    # Predict the second tuple
    prediction = decision_tree._predict_tuple(
        tuple=tuple_to_predict,
        node=decision_tree.tree,
    )

    # Check if the prediction is correct
    assert prediction == "No"


def test_with_decision_tree_inspired_by_small_student_dataset_and_hit_on_second_layer():
    """
    Test _predict_tuple() with a decision tree inspired by the small student dataset.
    The leaf node is hit on the second layer.
    """

    # Create a DecisionTree object
    decision_tree = DecisionTree()

    # Create the decision tree
    decision_tree.tree = DecisionTreeInternalNode(
        attribute_label="Participation",
        branches=[
            DecisionTreeBranch(
                label=DecisionTreeDecisionOutcomeEquals(value="High"),
                branch_node=DecisionTreeLeafNode(class_label="Yes"),
            ),
            DecisionTreeBranch(
                label=DecisionTreeDecisionOutcomeEquals(value="Medium"),
                branch_node=DecisionTreeInternalNode(
                    attribute_label="Age",
                    branches=[
                        DecisionTreeBranch(
                            label=DecisionTreeDecisionOutcomeBelowEqual(value=25),
                            branch_node=DecisionTreeLeafNode(class_label="Yes"),
                        ),
                        DecisionTreeBranch(
                            label=DecisionTreeDecisionOutcomeAbove(value=25),
                            branch_node=DecisionTreeLeafNode(class_label="No"),
                        ),
                    ],
                ),
            ),
            DecisionTreeBranch(
                label=DecisionTreeDecisionOutcomeEquals(value="Low"),
                branch_node=DecisionTreeLeafNode(class_label="No"),
            ),
        ],
    )

    # Create a tuple to predict
    tuple_to_predict = {
        "Age": 25,
        "Major": "CS",
        "Participation": "Medium",
    }

    # Predict the tuple
    prediction = decision_tree._predict_tuple(
        tuple=tuple_to_predict,
        node=decision_tree.tree,
    )

    # Check if the prediction is correct
    assert prediction == "Yes"

    # Create a second tuple to predict
    tuple_to_predict = {
        "Age": 26,
        "Major": "CS",
        "Participation": "Medium",
    }

    # Predict the second tuple
    prediction = decision_tree._predict_tuple(
        tuple=tuple_to_predict,
        node=decision_tree.tree,
    )

    # Check if the prediction is correct
    assert prediction == "No"
