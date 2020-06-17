import sqlite3
from datetime import time

import pandas as pd

start = time()


# clean nans
def clean_nan(data):
    # replace nan in totalizators with mean value of column
    data['B365H'].fillna(data['B365H'].mean(), inplace=True)
    data['B365D'].fillna(data['B365D'].mean(), inplace=True)
    data['B365A'].fillna(data['B365A'].mean(), inplace=True)
    data['BWH'].fillna(data['BWH'].mean(), inplace=True)
    data['BWD'].fillna(data['BWD'].mean(), inplace=True)
    data['BWA'].fillna(data['BWA'].mean(), inplace=True)
    data['IWH'].fillna(data['IWH'].mean(), inplace=True)
    data['IWD'].fillna(data['IWD'].mean(), inplace=True)
    data['IWA'].fillna(data['IWA'].mean(), inplace=True)
    data['LBH'].fillna(data['LBH'].mean(), inplace=True)
    data['LBD'].fillna(data['LBD'].mean(), inplace=True)
    data['LBA'].fillna(data['LBA'].mean(), inplace=True)
    data['PSH'].fillna(data['PSH'].mean(), inplace=True)
    data['PSD'].fillna(data['PSD'].mean(), inplace=True)
    data['PSA'].fillna(data['PSA'].mean(), inplace=True)
    data['WHH'].fillna(data['WHH'].mean(), inplace=True)
    data['WHD'].fillna(data['WHD'].mean(), inplace=True)
    data['WHA'].fillna(data['WHA'].mean(), inplace=True)
    data['VCH'].fillna(data['VCH'].mean(), inplace=True)
    data['VCD'].fillna(data['VCD'].mean(), inplace=True)
    data['VCA'].fillna(data['VCA'].mean(), inplace=True)

    # clean columns with nan
    data.dropna(inplace=True)
    return data


# get overall rating instead of players ids
def add_winner_label(matches):
    # array of winner ids
    labels = []

    # iterate the matches and get the winner for each match
    for index, row in matches.iterrows():
        home_goals = row['home_team_goal']
        away_goals = row['away_team_goal']

        if home_goals > away_goals:
            labels.append(1)
        elif home_goals < away_goals:
            labels.append(-1)
        else:
            labels.append(0)

    # create new column in matches df
    matches['winner_id_label'] = labels
    return matches


# get overall rating instead of players ids
def get_player_overall_rating(matches, player_stats):
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]

    # iterate all rows in matches df
    for index, row in matches.iterrows():
        # get match date
        date = row['date']

        # Loop through all players in match
        for player in players:
            # Get player ID
            player_id = row[player]

            # Get player stats
            stats = player_stats[player_stats.player_api_id == player_id]

            # Identify current stats
            current_stats = stats[stats.date < date].sort_values(by='date', ascending=False)[:1]

            # set overall_rating in match instead of player_id
            matches.at[index, player] = current_stats['overall_rating']

    # rename player feature to player + _overall_rating
    # for player in players:
    #     matches = matches.rename(columns={player: player + '_overall_rating'})

    return matches


# Connection to DB
database = 'database.sqlite'
conn = sqlite3.connect(database)

# get initial features to df
players_data = pd.read_sql("SELECT * FROM Player;", conn)
players_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
teams_data = pd.read_sql("SELECT * FROM Team;", conn)

matches_data_2016 = pd.read_sql("SELECT match_api_id, season, [date], home_team_api_id,"
                                "away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2,"
                                "home_player_3, home_player_4, home_player_5, home_player_6, home_player_7,"
                                "home_player_8, home_player_9, home_player_10, home_player_11, away_player_1,"
                                "away_player_2, away_player_3, away_player_4, away_player_5, away_player_6,"
                                "away_player_7, away_player_8, away_player_9, away_player_10, away_player_11,"
                                "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA,	LBH, LBD, LBA, PSH, PSD, PSA,"
                                "WHH, WHD, WHA, VCH, VCD, VCA "
                                "FROM Match where season like '%2015/2016%';", conn)

matches_data_2008_2015 = pd.read_sql("SELECT match_api_id, season, [date], home_team_api_id,"
                                     "away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2,"
                                     "home_player_3, home_player_4, home_player_5, home_player_6, home_player_7,"
                                     "home_player_8, home_player_9, home_player_10, home_player_11, away_player_1,"
                                     "away_player_2, away_player_3, away_player_4, away_player_5, away_player_6,"
                                     "away_player_7, away_player_8, away_player_9, away_player_10, away_player_11,"
                                     "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA, PSH, PSD, PSA,"
                                     "WHH, WHD, WHA, VCH, VCD, VCA "
                                     "FROM Match where season not like '%2015/2016%';", conn)

# clean nans
print("Cleaning nans...")
matches_data_2016 = clean_nan(matches_data_2016)
matches_data_2008_2015 = clean_nan(matches_data_2008_2015)
print("Cleaning done!")

# matches_data_2008_2015 = matches_data_2008_2015.tail(10)
# matches_data_2016 = matches_data_2016.tail(10)

# text_file = open("Output1.txt", "w")
# text_file.write(matches_data.to_string())
# text_file.close()

# get overall rating instead of players ids
print("Getting overall rating instead of players ids for 2015/2016...")
matches_data_2016 = get_player_overall_rating(matches_data_2016, players_stats_data)
print("Overall for 2015/2016 done!")
print("Getting overall rating instead of players ids for 2008-2015...")
matches_data_2008_2015 = get_player_overall_rating(matches_data_2008_2015, players_stats_data)
print("Overall for 2008-2015 done!")

# get winner team, home: 1, draw: 0, away: -1
print("Getting winner for 2015/2016...")
matches_data_2016 = add_winner_label(matches_data_2016)
print("Winner for 2015/2016 done!")
print("Getting winner for 2008-2015...")
matches_data_2008_2015 = add_winner_label(matches_data_2008_2015)
print("Winner for 2008-2015 done!")

# save df to csv
print("Saving 2015/2016 to CSV...")
matches_data_2016.to_csv("matches_data_2016.csv")
print("Saved 2015/2016!")
print("Saving 2008-2015 to CSV...")
matches_data_2008_2015.to_csv("matches_data_2008_2015.csv")
print("Saved 2008-2015!")

end = time()
print("Program run in {:.1f} minutes".format((end - start)/60))