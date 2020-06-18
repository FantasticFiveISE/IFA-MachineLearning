from time import time
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, plot_precision_recall_curve, \
    confusion_matrix, plot_confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize


def classification_model(model_model, data, predictors, outcome):
    model_model.fit(data[predictors], data[outcome])

    predictions = model_model.predict(data[predictors])

    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Training accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(n_splits=10)
    accuracy = []
    for train, test in kf.split(data):
        train_predictors = (data[predictors].iloc[train, :])
        train_target = data[outcome].iloc[train]
        model_model.fit(train_predictors, train_target)
        accuracy.append(model_model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(accuracy)))
    model_model.fit(data[predictors], data[outcome])


start = time()

# read csv files
print("Loading matches_data_2008_2015.csv")
matches_data_2008_2015 = pd.read_csv("matches_data_2008_2015.csv")
print("Done!")
print("Loading matches_data_2016.csv")
matches_data_2016 = pd.read_csv("matches_data_2016.csv")
print("Done!")

# get features to train as df
x_train = matches_data_2008_2015[['home_player_1', 'home_player_2',
                                  'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
                                  'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                                  'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
                                  'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
                                  'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                                  'LBA',
                                  'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']]

# get target to train as series
y_train = matches_data_2008_2015['winner_id_label']

# get features to test as df
x_test = matches_data_2016[['home_player_1', 'home_player_2',
                            'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
                            'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                            'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
                            'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
                            'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA',
                            'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']]

# get target to test as series
y_test = matches_data_2016['winner_id_label']

# svm model
# print("Starting SVM")
# clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
# print(clf.score(x_test, y_test))

# Random Forest model
print("Starting Random Forest")
# clf = GaussianNB().fit(x_train, y_train)

# Random Forest Algorithm
outcome_var = "winner_id_label"
clf = RandomForestClassifier(max_depth=15, random_state=0)
predictor_var = ['home_player_1', 'home_player_2',
                 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
                 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
                 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
                 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA',
                 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
classification_model(clf, matches_data_2008_2015, predictor_var, outcome_var)

clf.fit(x_train, y_train)

# clf = GradientBoostingClassifier(random_state=0).fit(x_train, y_train)
pred = clf.predict(x_train)

# draw confusion matrix
plot_confusion_matrix(clf, x_test, y_test)
plt.show()

# get importance of each feature
# feature_imp = pd.Series(clf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
# print(feature_imp)feature_imp
# Creating a bar plot
# sns.barplot(x=feature_imp, y=feature_imp.index)
# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()

# print scores
print("Training: " + str(clf.score(x_train, y_train)))
print("Test: " + str(clf.score(x_test, y_test)))
print("Precision score: " + str(precision_score(y_train, pred, average='macro')))
print("Accuracy score: " + str(accuracy_score(y_train, pred)))
print("Recall score: " + str(recall_score(y_train, pred, average='macro')))
print("F1 score: " + str(f1_score(y_train, pred, average='macro')))
print("Confusion matrix: " + str(confusion_matrix(y_train, pred)))

end = time()
print("Program run in {:.1f} minutes".format((end - start) / 60))
