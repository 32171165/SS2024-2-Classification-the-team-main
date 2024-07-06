class DecisionTreeDecisionOutcome:
    """
    Abstract superclass for all possible decision outcomes in a decision tree
    """

    def value_matches(self, value: str | int | float) -> bool:
        """
        Check if the given value matches the criteria to fall into this decision outcome

        Parameters:
        value (str|int|float): The value to check

        Returns:
        bool: True if the value matches the criteria, False otherwise
        """
        raise NotImplementedError()

    def __str__(self) -> str:
        """
        Return a string representation of the decision outcome

        Returns:
        str: A string representation of the decision outcome
        """
        raise NotImplementedError()
