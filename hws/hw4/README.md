# Homework 3 overview & requirements

This project's folder contains the following files:

- pipeline_library.py: general functions for a machine learning pipeline (reading data, preprocessing data, generating features, building models, etc.)
- predict_fudning.py: specific functions for applying the functions in pipeline_library to the DonorsChoose Data
- projects_2012-2013.csv: the DonorsChoose dataset
- projects_1000.csv: a sample of 1000 DonorsChoose projects
- projects_sample.csv: a sample of 10000 DonorsChoose projects
- assignment.md: the assignment statement

The project was developed using Python 3.7.3 on MacOS Mojave 10.14.4. It requires the following libraries:

| Package        | Version     |
| :------------: | :---------: |
| graphviz       | 0.8.4      |
| pandas         | 0.24.2      |
| matplotlib     | 3.0.3       |
| numpy          | 1.16.2      |
| seaborn        | 0.9.0       |
| scikit-learn   | 0.20.3      |

Helpful documentation and references are cited throughout the docstrings of the code.

To run the program, use the following command:
```
python3 predict_funding.py -d <path to dataset> -f <path to features config JSON file>
-m <path to models config JSON file> [-p <path to optional preprocessing config file>]
[-s <optional random seed>] [--savefigs (denotes that figures should be saved instead of displayed)]
```