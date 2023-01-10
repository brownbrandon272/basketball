#%%
from ExploreData import ExploreData
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%
import os
print(os.getcwd())
data_path = '../data/nba_games/'

#%%
df_game_detail = pd.read_csv(data_path + 'games_details.csv')
display(df_game_detail.sample(5))

# %%
explore = ExploreData(df_game_detail)
explore.var_dict

# %%
explore.summary()

## Drop
# - GAME_ID
# - TEAM_ID - Maybe useful later for finding teammates
# - TEAM_ABBREVIATION
# - PLAYER_ID
# - PLAYER_NAME
# - NICKNAME

## Feature Engineering Required - Cat
# - COMMENT
    # FUTURE UPDATE: Text processing - case, lemmatizing, removing special characters

## Feature Engineering Required - Num
# - MIN

## Categorical
# - START_POSITION

## Numeric
# - 'FGM',
# - 'FGA',
# - 'FG_PCT',
# - 'FG3M',
# - 'FG3A',
# - 'FG3_PCT',
# - 'FTM',
# - 'FTA',
# - 'FT_PCT',
# - 'OREB',
# - 'DREB',
# - 'REB',
# - 'AST',
# - 'STL',
# - 'BLK',
# - 'TO',
# - 'PTS',
# - 'PLUS_MINUS'
# - 'PF'


# %%
df2 = df_game_detail.copy()
df2.drop(['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME'], axis=1, inplace=True)


# %%
def get_min(time_str):
    """Get minutes from time (MM:SS)"""
    try:
        m, s = time_str.split(':')
        mins = float(m) + float(s)/60
    except:
        mins = float(time_str)
    return mins

df2['MIN'] = df2['MIN'].apply(get_min)
print(df2['MIN'].dtype)

# %%
pd.set_option("display.max_rows", None)
print(df2['COMMENT'].value_counts())
pd.set_option("display.max_rows", 30)

#%%
def comment_cat(comment_str):
    try:
        code, reason = comment_str.split('-', maxsplit=1)
        return code.strip(), reason.strip()
    except:
        return comment_str, comment_str

df2['comment_code'], df2['comment_reason'] = zip(*df2['COMMENT'].apply(comment_cat))

#%%
pd.set_option("display.max_rows", None)
print(df2['comment_code'].value_counts(), "\n")
print(df2['comment_reason'].value_counts())
pd.set_option("display.max_rows", 30)

#%%
