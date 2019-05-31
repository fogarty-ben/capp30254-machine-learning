# Homework 5 overview & requirements

This project's folder contains the following files:

- writeup.pdf: the explanatory writeup detailed in the assignment
- demonstratrion.ipynb: jupyter notebook demonstrating some of the pipeline's functionality
- assignment.md: the assignment statement  


- pipeline_library.py: general functions for a machine learning pipeline (reading data, preprocessing data, generating features, building models, etc.)
- predict_fudning.py: specific functions for applying pipeline_library to the DonorsChoose Data  


- output/: some sample figure output from running the software with the full dataset  


- configs/features.json: details features parameters to run the models with
- configs/preprocessing.json: details preprocess features to run the models with
- configs/models.json: a list of models to run  


- data/projects_2012-2013.csv: the DonorsChoose dataset
- data/projects_1000.csv: a sample of 1,000 DonorsChoose projects
- data/projects_sample.csv: a sample of 10,000 DonorsChoose projects  



The project was developed using Python 3.7.3 on MacOS Mojave 10.14.4, and results were obtained by running the project on a compute node of the Research Computing Center at the University of Chicago and on a c4.8xlarge AWS EC2 virtual machine. It requires the following libraries:

| Package        | Version     |
| :------------: | :---------: |
| dateutil       | 2.8.0      |
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