import sqlite3
import pandas as pd
import numpy as np
import sklearn as skl


def add_winner_label(matches):
    # array of winner ids
    labels = []

    # iterate the matches and get the winner for each match
    for index, row in matches.iterrows():
        home_goals = row['home_team_goal']
        away_goals = row['away_team_goal']

        if home_goals > away_goals:
            labels.append(row['home_team_api_id'])
        elif home_goals < away_goals:
            labels.append(row['away_team_api_id'])
        else:
            labels.append(-1)

    # create new column in matches df
    matches_data['winner_id_label'] = labels
    return matches


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
    for player in players:
        matches = matches.rename(columns={player: player + '_overall_rating'})

    return matches


# Connection to DB
database = 'database.sqlite'
conn = sqlite3.connect(database)

players_data = pd.read_sql("SELECT * FROM Player;", conn)
players_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
teams_data = pd.read_sql("SELECT * FROM Team;", conn)
matches_data = pd.read_sql("SELECT match_api_id, [date], home_team_api_id,"
                           "away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2,"
                           "home_player_3, home_player_4, home_player_5, home_player_6, home_player_7,"
                           "home_player_8, home_player_9, home_player_10, home_player_11, away_player_1,"
                           "away_player_2, away_player_3, away_player_4, away_player_5, away_player_6,"
                           "away_player_7, away_player_8, away_player_9, away_player_10, away_player_11,"
                           "B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA,	LBH, LBD, LBA, PSH, PSD, PSA,"
                           "WHH, WHD, WHA, SJH, SJD, SJA, VCH, VCD, VCA, GBH, GBD, GBA, BSH, BSD, BSA "
                           "FROM Match;", conn)

matches_data.dropna(inplace=True)
matches_data = matches_data.tail(15)

# get overall rating instead of players ids
matches_data = get_player_overall_rating(matches_data, players_stats_data)
# calculate and add id of winner team, if draw: id = -1
matches_data = add_winner_label(matches_data)

print(matches_data.to_string())
