from classes.decision_tree_decision_outcome import DecisionTreeDecisionOutcome


class DecisionTreeDecisionOutcomeEquals(DecisionTreeDecisionOutcome):
    """
    A class representing the outcome of a decision with only one value being part of the outcome
    (e.g. If only "high" is part of this specific outcome)
    """

    def __init__(self, value: str | int | float):
        """
        Initialize the decision outcome

        Parameters:
        value (str|int|float): The value that is considered a valid outcome
        """
        self.value = value

    def value_matches(self, value: str | int | float) -> bool:
        """
        Check if the given value matches the criteria to fall into this decision outcome

        Parameters:
        value (str|int|float): The value to check

        Returns:
        bool: True if the value matches the criteria, False otherwise
        """
        return self.value == value

    def __eq__(self, decisionOutcome: object) -> bool:
        """
        Check if the given value is equal to this decision outcome

        Parameters:
        value (object): The value to check

        Returns:
        bool: True if the given value is equal to this decision outcome, False otherwise
        """
        if not isinstance(decisionOutcome, DecisionTreeDecisionOutcomeEquals):
            return False

        return self.value == decisionOutcome.value

    def __str__(self) -> str:
        """
        Return a string representation of the decision outcome

        Returns:
        str: A string representation of the decision outcome
        """
        return str(self.value)
