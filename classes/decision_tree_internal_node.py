from typing import List

from classes.decision_tree_node import DecisionTreeNode
from classes.decision_tree_branch import DecisionTreeBranch


class DecisionTreeInternalNode(DecisionTreeNode):
    """A class representing an interal node in a decision tree"""

    def __init__(self, attribute_label: str, branches: List[DecisionTreeBranch]):
        """
        Initialize the internal node

        Parameters:
        label (str): The attribute the internal node is based on
        branches (List[DecisionTreeBranch]): The branches starting at the internal node
        """
        # Call the constructor of the superclass
        super().__init__()

        # The attribute the internal node is based on
        self.attribute_label = attribute_label

        # The branches of the internal node
        self.branches = branches

    def get_label(self) -> str:
        """
        Get the attribute the internal node is based on

        Returns:
        str: The attribute the internal node is based on
        """
        return self.attribute_label

    def get_branches(self) -> List[DecisionTreeBranch]:
        """
        Get the branches of the internal node

        Returns:
        List[DecisionTreeBranch]: The branches of the internal node
        """
        return self.branches

    def __str__(self, level: int = 0):
        """
        Get a string representation of the internal node

        Parameters:
        level (int): The level of the node in the tree

        Returns:
        str: The string representation of the internal node (including the branches)
        """
        # Indentation for the current level
        indentation = "  " * level

        # Return the string representation of the internal node
        return f"{indentation} | {self.attribute_label}\n" + "\n".join(
            [f"  {branch.__str__(level=level)}" for branch in self.branches]
        )
