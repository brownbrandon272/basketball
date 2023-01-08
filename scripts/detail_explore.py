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

# %%
explore.target_variable(target="PTS")

# %%
explore.var_dict