from classes.decision_tree_node import DecisionTreeNode


class DecisionTreeLeafNode(DecisionTreeNode):
    """A class representing a leaf node in a decision tree"""

    def __init__(self, class_label: str | int | float):
        """
        Initialize the leaf node

        Parameters:
        class_label (str|int|float): The class the leaf node represents
        """
        # Call the constructor of the superclass
        super().__init__()

        # The class the leaf node represents
        self.class_label = class_label

    def get_label(self) -> str | int | float:
        """
        Get the class label of the leaf node

        Returns:
        str|int|float: The class label of the leaf node
        """
        return self.class_label

    def __str__(self, level: int = 0):
        """
        Get a string representation of the leaf node

        Parameters:
        level (int): The level of the node in the tree

        Returns:
        str: The string representation of the leaf node
        """
        # Indentation for the current level
        indentation = "  " * level

        # Return the string representation of the leaf node
        return f"{indentation} ↳ {str(self.class_label)}"
