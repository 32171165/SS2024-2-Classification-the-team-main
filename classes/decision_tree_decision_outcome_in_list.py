from typing import List

from classes.decision_tree_decision_outcome import DecisionTreeDecisionOutcome


class DecisionTreeDecisionOutcomeInList(DecisionTreeDecisionOutcome):
    """
    A class representing the outcome of a decision with multiple values being part of the outcome
    (e.g. If both "high" and "medium" are part of this specific outcome)
    """

    def __init__(self, values: List[str | int | float]):
        """
        Initialize the decision outcome

        Parameters:
        values (List[str|int|float]): A list of values are considered possible outcomes
        """
        self.value = values

    def value_matches(self, value: str | int | float) -> bool:
        """
        Check if the given value matches the criteria to fall into this decision outcome

        Parameters:
        value (str|int|float): The value to check

        Returns:
        bool: True if the value matches the criteria, False otherwise
        """
        return value in self.value

    def __eq__(self, decisionOutcome: object) -> bool:
        """
        Check if the given value is equal to this decision outcome

        Parameters:
        value (object): The value to check

        Returns:
        bool: True if the given value is equal to this decision outcome, False otherwise
        """
        if not isinstance(decisionOutcome, DecisionTreeDecisionOutcomeInList):
            return False

        # It is enough if both lists contain the same values, the order does not matter
        return set(self.value) == set(decisionOutcome.value)

    def __str__(self) -> str:
        """
        Return a string representation of the decision outcome

        Returns:
        str: A string representation of the decision outcome
        """
        return "{" + ", ".join(str(value) for value in self.value) + "}"
