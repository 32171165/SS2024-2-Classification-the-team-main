import pandas as pd
from math import log
from typing import List, Set

"""
Collection of functions to calculate the impurity and the gini index of attributes in a dataset.
"""


def calculate_impurity(dataset: pd.DataFrame, target_attribute: str) -> float:
    """
    Calculate the impurity for a given target attribute in a dataset.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the impurity for
    target_attribute (str): The target attribute used as the class label

    Returns:
    float: The calculated impurity
    """
    # TODO
    # Get the frequency of each class in the target attribute
    class_counts = dataset[target_attribute].value_counts()
    
    # Calculate the total number of instances
    total_instances = len(dataset)
    
    # Calculate the proportion of each class
    probabilities = class_counts / total_instances
    
    # Compute the Gini Index
    gini_index = 1 - sum(prob ** 2 for prob in probabilities)
    
    return float(gini_index)


def calculate_impurity_partitioned(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split: int | float | Set[str],
) -> float:
    """
    Calculate the impurity for a given target attribute in a dataset if the dataset is partitioned by a given attribute and split.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the impurity for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split (int|float|Set[str]): The split used to partition the partition attribute. If the partition attribute is discrete-valued, the split is a set of strings (Set[str]). If the partition attribute is continuous-valued, the split is a single value (int or float).
    """
    # TODO
    total_instances = len(dataset)
    
    if isinstance(split, set):
        # Categorical attribute
        partition1 = dataset[dataset[partition_attribute].isin(split)]
        partition2 = dataset[~dataset[partition_attribute].isin(split)]
    else:
        # Continuous attribute
        partition1 = dataset[dataset[partition_attribute] <= split]
        partition2 = dataset[dataset[partition_attribute] > split]
    
    # Calculate the weighted Gini Index of the partitions
    weighted_gini_index = 0.0
    for partition in [partition1, partition2]:
        partition_size = len(partition)
        if partition_size > 0:
            partition_impurity = calculate_impurity(partition, target_attribute)
            partition_weight = partition_size / total_instances
            weighted_gini_index += partition_weight * partition_impurity
    
    return float(weighted_gini_index)

def calculate_gini_index(
    dataset: pd.DataFrame,
    target_attribute: str,
    partition_attribute: str,
    split: int | float | Set[str],
) -> float:
    """
    Calculate the Gini index (= reduction of impurity) for a given target attribute in a dataset if the dataset is partitioned by a given attribute and split.

    Parameters:
    dataset (pd.DataFrame): The dataset to calculate the Gini index for
    target_attribute (str): The target attribute used as the class label
    partition_attribute (str): The attribute that is used to partition the dataset
    split (int|float|Set[str]): The split used to partition the partition attribute. If the partition attribute is discrete-valued, the split is a set of strings (Set[str]). If the partition attribute is continuous-valued, the split is a single value (int or float).

    Returns:
    float: The calculated Gini index
    """
    # TODO
        # Calculate the initial impurity of the dataset
    initial_impurity = calculate_impurity(dataset, target_attribute)
    
    # Calculate the impurity after partitioning the dataset
    partitioned_impurity = calculate_impurity_partitioned(dataset, target_attribute, partition_attribute, split)
    
    # Calculate the Gini Index (reduction in impurity)
    gini_index = initial_impurity - partitioned_impurity
    
    return float(gini_index)
