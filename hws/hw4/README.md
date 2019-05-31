# Homework 5 overview & requirements

This project's folder contains the following files:

- writeup.pdf: the explanatory writeup detailed in the assingment
- demonstratrion.ipynb: jupyter notebook demonstrating some of the pipeline's functionality
- pipeline_library.py: general functions for a machine learning pipeline (reading data, preprocessing data, generating features, building models, etc.)
- predict_fudning.py: specific functions for applying the functions in pipeline_library to the DonorsChoose Data

- output/: text and figure output from running the software with the full dataset

- configs/features.json: details features parameters to run the models with
- configs/preprocessing.json: details preprocess features to run the models with
- configs/models.json: a full list of models to run
- configs/flipped_models.json: a full list of models to run in reverse order
- configs/mini_models.json: a shorter list of models to run

- data/projects_2012-2013.csv: the DonorsChoose dataset
- data/projects_1000.csv: a sample of 1000 DonorsChoose projects
- data/projects_sample.csv: a sample of 10000 DonorsChoose projects

- assignment.md: the assignment statement

The project was developed using Python 3.7.3 on MacOS Mojave 10.14.4. It requires the following libraries:

| Package        | Version     |
| :------------: | :---------: |
| dateutil       | 2.8.0.      |
| graphviz       | 0.8.4       |
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