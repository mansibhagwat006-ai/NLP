# Section D – Step 2 Test & Debug Note

The original AI code used `train_test_split` without stratification. I changed the split to use
`stratify=labels` so that the Positive and Negative classes are represented more reliably in the
test set, especially when the dataset is small or imbalanced. I also added a minimum-data check,
empty-review validation, whitespace cleanup, and a fixed random state. These changes make testing
more reliable and prevent invalid/empty user input from causing unnecessary problems.
