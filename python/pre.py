import pandas as pd, numpy as np
from random import randrange
from math import floor

# Read input files
input = pd.read_csv('../data/train.csv')

# Split 75% for training, 5% for validation and 20% for testing
# maintaining class proportion

# Class 1
input_class_one = input.query("target == 1")
split_idx1_class_one = int(floor(input_class_one.shape[0] * 0.75))
split_idx2_class_one = int(floor(input_class_one.shape[0] * 0.8))
training_class_one = input_class_one[:split_idx1_class_one]
validation_class_one = input_class_one[split_idx1_class_one:split_idx2_class_one]
test_class_one = input_class_one[split_idx2_class_one:]

# Class 0
input_class_zero = input.query("target == 0")
split_idx1_class_zero = int(floor(input_class_zero.shape[0] * 0.75))
split_idx2_class_zero = int(floor(input_class_zero.shape[0] * 0.8))
training_class_zero = input_class_zero[:split_idx1_class_zero]
validation_class_zero = input_class_zero[split_idx1_class_zero:split_idx2_class_zero]
test_class_zero = input_class_zero[split_idx2_class_zero:]

# Training data
train = pd.concat([training_class_one, training_class_zero])
train = train.reindex(np.random.permutation(train.index))

# Validation data
validation = pd.concat([validation_class_one, validation_class_zero])
validation = validation.reindex(np.random.permutation(validation.index))

# Test data
test = pd.concat([test_class_one, test_class_zero])
test = test.reindex(np.random.permutation(test.index))

print("Training data")
print(train.describe())
print("Validation data")
print(validation.describe())
print("Test data")
print(test.describe())

# Export to CSV file
train.to_csv('../SantanderCreateML/SantanderCreateML/Data/train_pre.csv', sep=",", index=False)
validation.to_csv('../SantanderCreateML/SantanderCreateML/Data/validation_pre.csv', sep=",", index=False)
test.to_csv('../SantanderCreateML/SantanderCreateML/Data/test_pre.csv', sep=",", index=False)
