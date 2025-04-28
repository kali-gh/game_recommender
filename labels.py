import logging
import os
import json
import random
import pandas as pd

import libs

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

from parameters import Params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params = Params('input_data/params.json')


def get_labels():
    logger.info("Building labels...")

    random.seed(42)

    logger.info("Initializing directories..")
    Path(params['output_data_dir']).mkdir(parents=True, exist_ok=True)
    Path(params['model_dir']).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(params['input_data_dir']) or not os.path.exists(params['user_input_data_dir'])  :
        raise AssertionError("input_data or user_input_data directory not found.")
    else:
        pass

    logger.info("Building the base df_game_data frame...")
    input_data_dir = params['input_data_dir']
    file_list_games = [f for f in os.listdir(input_data_dir) if f.endswith('.json')]

    game_data = {}
    for g in file_list_games:
        with open(os.path.join(input_data_dir, g), 'r') as f:
            game_data.update(json.load(f))

    df_game_data = pd.DataFrame.from_dict(game_data, orient='index')
    df_game_data.reset_index(inplace=True)

    df_game_data.rename({'index': 'appid'}, axis=1, inplace=True)

    columns = params['meta_cols']
    columns.extend(params['x_cols'])

    df_game_data = df_game_data.loc[:, columns].copy()

    logger.info(f"Loaded game data into data frame with columns {','.join(df_game_data.columns)}")

    logger.info("Building the embeddings...")
    documents = df_game_data['about_the_game'].tolist()
    df_embeddings = libs.build_df_embeddings(
        documents = documents,
        model_dir = params['model_dir'],
        stoplist = params['stoplist'],
        num_topics = params['num_topics'],
        min_word_length = params["min_word_length"],
        parse = True
    )

    df_game_data.appid = df_game_data.appid.astype(int)

    logger.info("Cleaning up feature data and adding 1-hot encodings for tags")
    df_game_data['tags'].apply(lambda x: [tag.strip() for tag in x])

    tag_data = df_game_data.tags
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(
        mlb.fit_transform(tag_data),
        columns=mlb.classes_,
       index=tag_data.index
    )

    # filter to ~>1% representation in the sample
    tag_means = res.mean().sort_values(ascending=False)
    tag_means_group = tag_means < params['tag_pct_threshold']

    cols_group = tag_means[tag_means_group].index.values

    res['tag_other'] = res[cols_group].sum(axis=1)
    res = res.drop(cols_group, axis=1)

    cols = res.columns

    def update_col_name(col):
        col = col.lower()
        col = col.replace(' ', '_')
        col = 'genre_' + col
        return col

    new_cols = [update_col_name(c) for c in cols]
    res.columns = new_cols

    logger.info("Combining data frames - game data, embeddings, one-hot encodings")
    df_game_data_one_hot = pd.concat([df_game_data, df_embeddings, res], axis=1)

    df_ratings = pd.read_csv(f'{params["user_input_data_dir"]}/ratings.csv')

    logger.info(f"original length : {len(df_game_data_one_hot)}")
    df_final = df_game_data_one_hot.merge(
        df_ratings[['appid', 'rating']],
        on='appid',
        how='left')
    logger.info(f"new length : {len(df_final)}")

    assert(len(df_final) == len(df_game_data_one_hot))

    df_final.drop(
        params['x_cols_drop']
        , axis=1
        , inplace=True
    )
    other_cols = [c for c in df_final.columns if c != params['y_col']]

    new_col_order = other_cols.copy()
    new_col_order.append(params['y_col'])

    df_final = df_final.loc[:, new_col_order].copy()

    logger.info(f"Loaded data with column order : {','.join(df_final.columns)}")

    return df_final

if __name__ == '__main__':

    df_labels = get_labels()
    df_labels.to_csv(f"{params['output_data_dir']}/df_labels.csv", index=False)