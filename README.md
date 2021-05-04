# PhysioNet/CinC Challenge 2021 Evaluation Metrics

This repository contains the Python and MATLAB evaluation code for the PhysioNet/Computing in Cardiology Challenge 2021.

The `evaluate_model` script evaluates the output of your classifier using the evaluation metric that is described on the [webpage](https://physionetchallenges.org/2021/) for the PhysioNet/CinC Challenge 2021. While this script reports multiple evaluation metrics, we use the last score (`Challenge Metric`) to rank your algorithm.

The `weights.csv` file describes a table that defines the Challenge evaluation metric. The rows and columns of this table correspond to classes or diagnoses that are scored for the Challenge, and the entries define the value or weight give to each classifier output for each label. Some rows or columns contain multiple classes that are separated with a pipe (`|`), and these classes are treated equivalently by the evaluation metric.

## Python

You can run the Python evaluation code by installing the NumPy Python package and running the following command in your terminal:

    python evaluate_model.py labels outputs scores.csv class_scores.csv

where `labels` is a directory containing files with one or more labels for each ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; `scores.csv` (optional) is a collection of scores for your algorithm; and `class_scores.csv` (optional) is a collection of per-class scores for your algorithm.

## MATLAB

You can run the MATLAB evaluation code by installing Python and the NumPy Python package and running the following command in MATLAB:

    evaluate_model('labels', 'outputs', 'scores.csv', 'class_scores.csv')

where `labels` is a directory containing files with one or more labels for each ECG recording, such as the training database on the PhysioNet webpage; `outputs` is a directory containing files with outputs produced by your algorithm for those recordings; `scores.csv` (optional) is a collection of scores for your algorithm; and `class_scores.csv` (optional) is a collection of per-class scores for your algorithm.

## Troubleshooting

Unable to run this code with your code? Try one of the [example classifiers](https://physionetchallenges.org/2021/#submissions) on the [training data](https://physionetchallenges.org/2021/#data). Unable to install or run Python? Try [Python](https://www.python.org/downloads/), [Anaconda](https://www.anaconda.com/products/individual), or your package manager.
