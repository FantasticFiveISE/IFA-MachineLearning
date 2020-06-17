from datetime import time
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

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
print("Starting SVM")
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
print(clf.score(x_test, y_test))

# Random Forest model
print("Starting Random Forest")
clf = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
print(clf.score(x_test, y_test))

# TODO: graphs and visualizations of the models, get the best model according to literature review. Good night!

end = time()
print("Program run in {:.1f} minutes".format((end - start)/60))
