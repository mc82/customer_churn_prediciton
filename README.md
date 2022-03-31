# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description

The intention of this project is to evaluate multiple kinds of models to predict the likelihood of churn based on given bank data.

## Prerequisites

In order to run the code a virtual environment which contains all the Python requirements which are specified in [requirements.txt](requirements.txt) is recommended. In order to install this environment run this statement:

```bash
 make setup-environment
```

## Execution

In order to run to full process of EDA, training, prediction, evaluation run:

```bash
make run
```

## Output

<!-- # TODO describe output structure  -->

After executing the process you can find the resulting plots in the [plot]{.plot/} directory.

## Tests

Unit test are implemented of the most important classes/methods. We use **pytest** as a testing framework.
In order to run the test execute to following command:

```bash
make test
```

The progress of the test will be shown in the terminal. However, there is also a log file in **output/test.log** to lookup the test results.
