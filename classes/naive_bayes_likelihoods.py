import warnings
import math


class NaiveBayesLikelihoods:
    """
    A class to store all likelihoods of a trained Naive Bayes classifier.
    """

    def __init__(self):
        """
        Initialize the objects.
        """
        self.likelihoods = dict()

    def add_categorical_likelihood(
        self,
        attribute: str,
        value: str,
        class_label: str | int | float,
        likelihood: float,
    ):
        """
        Add the likelihood of a categorical attribute to the collection.

        Parameters:
        attribute (str): The attribute the likelihood is for
        value (str): The value of the attribute the likelihood is for
        class_label (str | int | float): The class label the likelihood is for
        likelihood (float): The likelihood of the value given the class label
        """
        # Check if the attribute is already in the dictionary
        if attribute not in self.likelihoods:
            self.likelihoods[attribute] = dict()

        # Check if the value is already in the dictionary
        if value not in self.likelihoods[attribute]:
            self.likelihoods[attribute][value] = dict()

        # Check if the class label is already in the dictionary
        if class_label in self.likelihoods[attribute][value]:
            warnings.warn(
                f"Likelihood for attribute '{attribute}', value '{value}', and class label '{class_label}' already exists. Overwriting the likelihood."
            )

        # Add the likelihood to the dictionary
        self.likelihoods[attribute][value][class_label] = likelihood

    def add_continuous_likelihood(
        self, attribute: str, class_label: str | int | float, mean: float, std: float
    ):
        """
        Add the likelihood of a continuous attribute to the collection.

        To be more precise:
        Add the mean and standard deviation to the collection.
        This will be used to dynamically calculate the likelihood for a given value in the get likelihood method.

        Parameters:
        attribute (str): The attribute the likelihood is for
        class_label (str | int | float): The class label the likelihood is for
        mean (float): The mean of the attribute given the class label
        std (float): The standard deviation of the attribute given the class label
        """
        # Check if the attribute is already in the dictionary
        if attribute not in self.likelihoods:
            self.likelihoods[attribute] = dict()

        # Check if the class label is already in the dictionary
        if class_label in self.likelihoods[attribute]:
            warnings.warn(
                f"Likelihood for attribute '{attribute}' and class label '{class_label}' already exists. Overwriting the likelihood."
            )

        # Add the likelihood to the dictionary
        self.likelihoods[attribute][class_label] = {"mean": mean, "std": std}

    def get_likelihood(
        self, attribute: str, value: str | int | float, class_label: str | int | float
    ) -> float:
        """
        Get the likelihood for a given attribute and value.

        Parameters:
        attribute (str): The attribute the likelihood is for
        value (str | int | float): The value of the attribute the likelihood is for
        class_label (str | int | float): The class label the likelihood is for

        Returns:
        float: The likelihood of the value given the class label
        """
        # Check if the attribute is in the dictionary
        if attribute not in self.likelihoods:
            raise ValueError(f"Attribute '{attribute}' not found in the likelihoods.")

        # Check if the value is continuous or categorical
        # (in our case all attributes with numerical values are considered continuous)
        if isinstance(value, (int, float)):
            # Check if the class label is in the dictionary
            if class_label not in self.likelihoods[attribute]:
                raise ValueError(
                    f"Class label '{class_label}' not found in the likelihoods for attribute '{attribute}'."
                )

            # Get the mean and standard deviation
            mean = self.likelihoods[attribute][class_label]["mean"]
            std = self.likelihoods[attribute][class_label]["std"]

            # Calculate the likelihood using the gaussian distribution
            likelihood = (1 / (std * (2 * math.pi) ** 0.5)) * math.e ** (
                -0.5 * ((value - mean) / std) ** 2
            )

            return likelihood
        else:
            # Check if the value is in the dictionary
            if value not in self.likelihoods[attribute]:
                raise ValueError(
                    f"Value '{value}' not found in the likelihoods for attribute '{attribute}'."
                )

            # Check if the class label is in the dictionary
            if class_label not in self.likelihoods[attribute][value]:
                raise ValueError(
                    f"Class label '{class_label}' not found in the likelihoods for attribute '{attribute}' and value '{value}'."
                )

            # Get the likelihood
            likelihood = self.likelihoods[attribute][value][class_label]

            return likelihood

    def __str__(self) -> str:
        """
        Get a string representation of the likelihoods.

        Returns:
        str: The string representation of the likelihoods
        """
        return str(self.likelihoods)
