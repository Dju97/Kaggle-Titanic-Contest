# Kaggle-Titanic-Contest
This repository hosts our code for the Kaggle Titanic Competition : we had to predict whether or not a passenger had survived to the Titanic Sinking.

The code finally used is 'code_Titanic_prediction_final.py'. We reached the 500th rank over 10000 competitors. You can see a complete report about our work in the directory

## Feature engineering

We spent much time on feature engineering, since it's a key point on being able to make the most of the information that features contains. The features "SibSp" and "Parch" needed to be linked to the name of passenger to know with whom passenger had a familial relation.
We also gathered people by tickets.

## Our model
We firstly used LDA to reduce dimensionality of data, then a SVM algorithm which led us to a pretty good score. To improve this one, we built manually a classifier taking into account family bound.
