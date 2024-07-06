import warnings


class NaiveBayesPriorProbabilities:
    """
    A class to store all prior probabilities of a trained Naive Bayes classifier
    """

    def __init__(self):
        """
        Initialize the object
        """
        self.prior_probabilities = dict()

    def add_prior_probability(self, class_label: str | int | float, probability: float):
        """
        Add a prior probability for a class label

        Parameters:
        class_label (str|int|float): The class label
        probability (float): The prior probability of the class label
        """
        # Check if the class label is already in the dictionary
        if class_label in self.prior_probabilities:
            warnings.warn(
                f"Prior probability for class label '{class_label}' already exists. Overwriting the prior probability."
            )

        # Add the prior probability to the dictionary
        self.prior_probabilities[class_label] = probability

    def get_prior_probability(self, class_label: str | int | float) -> float:
        """
        Get the prior probability of a class label

        Parameters:
        class_label (str|int|float): The class label

        Returns:
        float: The prior probability of the class label
        """
        # Check if the class label is in the dictionary
        if class_label not in self.prior_probabilities:
            raise ValueError(
                f"Prior probability for class label '{class_label}' not found."
            )

        # Return the prior probability
        return self.prior_probabilities[class_label]

    def __str__(self) -> str:
        """
        Get a string representation of the prior probabilities

        Returns:
        str: A string representation of the prior probabilities
        """
        return str(self.prior_probabilities)
