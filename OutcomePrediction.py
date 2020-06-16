import sqlite3
import pandas as pd
import numpy as np
import sklearn as skl

# Connection to DB
database = 'database.sqlite'
conn = sqlite3.connect(database)

player_data = pd.read_sql("SELECT * FROM Player;", conn)
player_stats_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)
team_data = pd.read_sql("SELECT * FROM Team;", conn)
match_data = pd.read_sql("SELECT * FROM Match;", conn)


