import logging
import os
import json
import pandas as pd

from pathlib import Path

data_dir = 'data'
file_list_games = [f for f in os.listdir(data_dir) if f.endswith('.json')]

game_data = {}
for g in file_list_games:
    with open(os.path.join(data_dir, g), 'r') as f:
        game_data.update(json.load(f))
        
df_game_data = pd.DataFrame.from_dict(game_data, orient='index')

df_game_data.reset_index(inplace=True)

df_game_data.rename({'index' : 'appid'}, axis=1, inplace=True)

Path('processed_data').mkdir(parents=True, exist_ok=True)
df_game_data.to_csv('processed_data/df_game_data.csv', index=False)
