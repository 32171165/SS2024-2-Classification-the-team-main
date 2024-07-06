import pandas as pd
from math import log
from typing import List

"""
Collection of functions to calculate the entropy, information and 
information gain of attributes in a dataset.
"""


def calculate_entropy(dataset: pd.DataFrame, target_attribute: str) -> float:
    """
    Calculate the entropy for a given target attribute in a dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the entropy for
    target_attribute (str): The target attribute used as the class label

    Returns:
    float: The calculated entropy (= expected information)
    """
    # TODO
    # Get the frequency of each class in the target attribute
    class_counts = dataset[target_attribute].value_counts()
    
    # Calculate the total number of instances
    total_instances = len(dataset)
    
    # Calculate the proportion of each class
    probabilities = class_counts / total_instances
    
    # Compute the entropy
    entropy = -sum(prob * log(prob, 2) for prob in probabilities)
    
    return float(entropy)


def calculate_information_partitioned(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split_value: int | float = None,
) -> float:
    """
    Calculate the information for a given target attribute in a dataset if the dataset is partitioned by a given attribute.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the information for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split_value (int|float), default None: The value to split the partition attribute on. If set to None, the function will calculate the information for a discrete-valued partition attribute. If set to a value, the function will calculate the information for a continuous-valued partition attribute.
    """
    # TODO
    total_instances = len(dataset)
    
    if split_value is None:
        # Categorical attribute
        unique_values = dataset[partition_attribute].unique()
        partitions = [dataset[dataset[partition_attribute] == value] for value in unique_values]
    else:
        # Continuous attribute
        partitions = [
            dataset[dataset[partition_attribute] <= split_value],
            dataset[dataset[partition_attribute] > split_value]
        ]
    
    # Calculate the weighted entropy of the partitions
    weighted_entropy = 0.0
    for partition in partitions:
        partition_entropy = calculate_entropy(partition, target_attribute)
        partition_weight = len(partition) / total_instances
        weighted_entropy += partition_weight * partition_entropy
    
    return weighted_entropy


def calculate_information_gain(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split_value: int | float = None,
) -> float:
    """
    Calculate the information gain for a given target attribute in a dataset if the dataset is partitioned by a given attribute.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the information gain for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split_value (int|float), default None: The value to split the partition attribute on. If set to None, the function will calculate the information gain for a discrete-valued partition attribute. If set to a value, the function will calculate the information gain for a continuous-valued partition attribute.

    Returns:
    float: The calculated information gain
    """
    # TODO
        # Calculate entropy before the split
    entropy_before_split = calculate_entropy(dataset, target_attribute)
    
    # Calculate entropy after the split
    entropy_after_split = calculate_information_partitioned(dataset, target_attribute, partition_attribute, split_value)
    
    # Calculate information gain
    information_gain = entropy_before_split - entropy_after_split
    
    return information_gain
