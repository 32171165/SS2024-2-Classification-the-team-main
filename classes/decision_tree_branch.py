from classes.decision_tree_decision_outcome import DecisionTreeDecisionOutcome
from classes.decision_tree_node import DecisionTreeNode


class DecisionTreeBranch:
    """A class representing a branch in a decision tree"""

    def __init__(
        self, label: DecisionTreeDecisionOutcome, branch_node: DecisionTreeNode
    ):
        """
        Initialize the branch

        Parameters:
        label (DecisionTreeDecision): The label of the branch
        """
        # The label of the branch
        self.label = label

        # The node the branch leads to
        self.branch_node = branch_node

    def get_label(self) -> DecisionTreeDecisionOutcome:
        """
        Get the label of the branch

        Returns:
        DecisionTreeDecision: The label of the branch
        """

        return self.label

    def get_branch_node(self) -> DecisionTreeNode:
        """
        Get the node the branch leads to

        Returns:
        DecisionTreeNode: The node the branch leads to
        """

        return self.branch_node

    def value_matches(self, value: str | int | float) -> bool:
        """
        Check if the given value matches/complies with the branch label

        Parameters:
        value (str|int|float): The value to check

        Returns:
        bool: True if the value matches/complies with the branch label, False otherwise
        """

        return self.label.value_matches(value)

    def __str__(self, level: int = 0) -> str:
        """
        Get a string representation of the branch

        Parameters:
        level (int): The level of the branch in the decision tree (starting from 0 for the branches starting at the root node)

        Returns:
        str: The string representation of the branch (including the node the branch leads to)
        """
        # Indentation for the current level
        indentation = "  " * level

        # Return the string representation of the branch
        return (
            f"{indentation}\\ {self.label} \n{self.branch_node.__str__(level=level+1)}"
        )
