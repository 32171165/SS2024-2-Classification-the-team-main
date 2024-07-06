import pandas as pd

from decision_tree import DecisionTree

from classes.decision_tree_node import DecisionTreeNode
from classes.decision_tree_leaf_node import DecisionTreeLeafNode
from classes.decision_tree_internal_node import DecisionTreeInternalNode
from classes.decision_tree_branch import DecisionTreeBranch
from classes.decision_tree_decision_outcome import DecisionTreeDecisionOutcome
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
# Tests with a decision tree inspired by the small student dataset
# Tree structure:
# Participation
# ├── High: Yes
# ├── Medium: Age
# │   ├── <= 25: Yes
# │   └── > 25: No
# └── Low: No
#####


def test_with_decision_tree_inspired_by_small_student_dataset():
    """
    Test _predict_tuple() with a decision tree inspired by the small student dataset.
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

    # Create the tuples to predict (pandas DataFrame)
    tuples_to_predict = pd.DataFrame(
        {
            "Age": [20, 30, 40],
            "Major": ["CS", "DS", "DS"],
            "Participation": ["High", "Medium", "Low"],
        }
    )

    # Predict the tuples
    predictions = decision_tree.predict(dataset=tuples_to_predict)

    # Check if the predictions are correct
    assert predictions == ["Yes", "No", "No"]
