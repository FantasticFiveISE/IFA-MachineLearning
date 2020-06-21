from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics, svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, confusion_matrix, \
    plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


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


def get_results(clf, x_train, y_train, x_test, y_test, pred):
    # draw confusion matrix
    plot_confusion_matrix(clf, x_test, y_test)
    plt.show()

    # # get importance of each feature
    # feature_imp = pd.Series(clf.feature_importances_, index=x_train.columns).sort_values(ascending=False)
    # # print(feature_imp)
    # # Creating a bar plot
    # sns.barplot(x=feature_imp, y=feature_imp.index)
    # # Add labels to your graph
    # plt.xlabel('Feature Importance Score')
    # plt.ylabel('Features')
    # plt.title("Visualizing Important Features")
    # plt.legend()
    # plt.show()

    # print scores
    print("2015/2016 season Test: %s " % "{0:.3%}".format(clf.score(x_test, y_test)))
    print("Precision score: %s " % "{0:.3%}".format(precision_score(y_train, pred, average='macro')))
    print("Recall score:  %s " % "{0:.3%}".format(recall_score(y_train, pred, average='macro')))
    print("F1 score:  %s " % "{0:.3%}".format(f1_score(y_train, pred, average='macro')))


start = time()

# read csv files
print("Loading matches_data_2008_2015.csv")
matches_data_2008_2015 = pd.read_csv("matches_data_2008_2015.csv")
print("Done!")
print("Loading matches_data_2016.csv")
matches_data_2016 = pd.read_csv("matches_data_2016.csv")
print("Done!")

# get features to train as df
x_tr = matches_data_2008_2015[['home_player_1', 'home_player_2',
                               'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
                               'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                               'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
                               'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
                               'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                               'LBA',
                               'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']]

# get target to train as series
y_tr = matches_data_2008_2015['winner_id_label']

# get features to test as df
x_te = matches_data_2016[['home_player_1', 'home_player_2',
                          'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
                          'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                          'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
                          'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
                          'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA',
                          'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']]

# get target to test as series
y_te = matches_data_2016['winner_id_label']

# target
outcome_var = "winner_id_label"
# features
predictor_var = ['home_player_1', 'home_player_2',
                 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
                 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
                 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
                 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
                 'B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD', 'LBA',
                 'PSH', 'PSD', 'PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']


# Gaussian naive Bayes, SVM, Random forest, Gradient boosting
# classifier = GaussianNB()
# classifier = svm.SVC()
# classifier = GradientBoostingClassifier(random_state=0)
# classifier = MLPClassifier(random_state=0, max_iter=300)
classifier = RandomForestClassifier(max_depth=10, random_state=0)

# get cross-validation score of a model
classification_model(classifier, matches_data_2008_2015, predictor_var, outcome_var)

# fit data to classifier
classifier.fit(x_tr, y_tr)

# get prediction
prediction = classifier.predict(x_tr)

# get results about
get_results(classifier, x_tr, y_tr, x_te, y_te, prediction)

end = time()
print("Program run in {:.1f} minutes".format((end - start) / 60))
