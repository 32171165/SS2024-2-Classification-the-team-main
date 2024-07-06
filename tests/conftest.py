import pytest

import pandas as pd


@pytest.fixture
def small_student_dataset():
    """
    Create a small dataset with basic student information.

    Also used in Exercise Sheet 4: Classification - Exercise 1.
    """

    # Create the dataset
    dataset = pd.DataFrame(
        {
            "Age": [23, 23, 26, 24, 26, 26],
            "Major": ["CS", "DS", "DS", "DS", "DS", "DS"],
            "Participation": ["High", "Low", "High", "Medium", "Medium", "Low"],
            "Passed": ["Yes", "No", "Yes", "Yes", "No", "No"],
        }
    )

    return dataset


@pytest.fixture
def small_submission_dataset():
    """
    Create a small dataset with some data about submissions.

    Also used in Exercise Sheet 4: Classification - Exercise 2.
    """

    # Create the dataset
    dataset = pd.DataFrame(
        {
            "Topic": [
                "Classification",
                "Clustering",
                "Frequent Patterns",
                "Clustering",
                "Frequent Patterns",
                "Frequent Patterns",
                "Classification",
                "Clustering",
                "Clustering",
                "Classification",
            ],
            "Knowledge": [
                "High",
                "Low",
                "High",
                "Medium",
                "High",
                "Medium",
                "Low",
                "Low",
                "High",
                "Medium",
            ],
            "Hours": [1, 4, 5, 5, 2, 3, 6, 5, 3, 4],
            "Passed": [
                "No",
                "No",
                "Yes",
                "Yes",
                "No",
                "Yes",
                "Yes",
                "Yes",
                "Yes",
                "Yes",
            ],
        }
    )

    return dataset
