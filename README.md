AFSIS-Soil-Property-Prediction
==============================

Description:
Neural network for the Kaggle AFSIS Soil Property Prediction Challenge. Requires Theano and Sci-kit Learn libraries. Scores 0.50984 on the private leaderboard by itself. Results averaged with those from an SVM (not included) improve the score to 0.48421 (11th place).

How to Run:
Before starting the program you will need to download the files "training.csv" and "sorted_test.csv" from http://www.kaggle.com/c/afsis-soil-properties/data and place them in the proper directory. Currently it is set up to have the datasets one directory above where these files are located, but you can change that. If you want to create a valid submission file you will also need to download "sample_submission.csv". 

Run "mlp_regression_dropout.py" to begin prediction. It will build 50 different models and then average the predictions from the 25 best performing. Runtime is quite long without a GPU, so maybe reduce the amount of folds before starting. While running the program will generate a new csv file for the predictions from each new model. Once all models have been created the results of the 25 best will be averaged and saved in their own csv file. To get this into a valid submission format for Kaggle, simply copy paste these results into the sample_submission.csv file.
