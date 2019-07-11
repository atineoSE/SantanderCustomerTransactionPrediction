# SantaderCustomerTransactionPrediction
Project for ML on the Santander Customer Transaction Prediction dataset.

## Getting the data

The original datasets can be obtained from the [corresponding Kaggle competition](https://www.kaggle.com/c/santander-customer-transaction-prediction/overview), by accepting terms and conditions, then downloading, and adding to the `data` subdirectory over the root of this repo to be found by the python scripts.

The original training set is split into 75% for training, 5% for validation and 20% for testing, maintaining class proportion with [this python script](python/pre.py).

## Repo structure

The following projects are included in this repo:

- [SantanderCreateML](SantanderCreateML): a Xcode macOS project with a script to create predictive models by using the Create ML API in code (most importantly `MLBoostedTreeRegressor`).
- [SantanderInference](SantanderInference): a Xcode iOS project to test performance of inference on device. This one uses a subset of the available data, to avoid very long execution times.
- [python](python): python scripts to 
  - [generate data sets](python/pre.py),
  - create classifier models using 5-stratification as per the solution by [Vladislav Bogorod](https://www.kaggle.com/bogorodvo) in [Kaggle](https://www.kaggle.com/bogorodvo/starter-code-saving-and-loading-lgb-xgb-cb),
  - script for [converting models](convert_classifiers), 
  - plus some experimental scripts dealing with model quantization.