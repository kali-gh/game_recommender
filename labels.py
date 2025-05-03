import logging
import os
import json
import random
import pandas as pd
import numpy as np

import libs

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from parameters import Params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

params = Params('input_data/params.json')
output_data_dir_labels = os.path.join(params['output_data_dir'], params['output_data_subdir_labels'])
Path(output_data_dir_labels).mkdir(parents=True, exist_ok=True)

SEED = 1
random.seed(SEED)
np.random.seed(SEED)


def load_data():
    """
    Loads data from input_data directory
    Returns: train, validation and test dataframes + x columns
    """

    logger.info("Loading data from input_data directory")

    logger.info(f"Loading data from {output_data_dir_labels}")

    df = pd.read_csv(f"{output_data_dir_labels}/df_labels.csv")

    logger.info("Imputing mean review and min review counts")
    if 'review_pct_overall' in df.columns:
        df.loc[:, 'review_pct_overall'] = df.loc[:, 'review_pct_overall'].fillna(0)
    else:
        pass

    if 'review_count_overall' in df.columns:
        df.loc[:, 'review_count_overall'] =  df.loc[:, 'review_count_overall'].fillna(0)
        df.loc[:, 'review_count_overall'] = np.log(df.loc[:, 'review_count_overall']+1)
    else:
        pass

    df.to_csv(f"{output_data_dir_labels}/df_labels_model_ready.csv", index=False)

    logger.info("Building train, validation and test dataframes")
    cond_train_val = ~df.rating.isnull()

    df_train_val_local = df[cond_train_val].copy()
    df_test_local = df[~cond_train_val].copy()
    df_train_local, df_val_local = train_test_split(df_train_val_local, train_size=0.75, random_state = SEED)

    logger.info(f"Train dataset size : {len(df_train_val_local)}, validation dataset size : {len(df_val_local)},"
                f" test dataset size {len(df_test_local)}")

    x_cols_selected = [c for c in df.columns if c not in [params['y_col']]]
    x_cols_selected = [c for c in x_cols_selected if c not in params['meta_cols']]

    assert (params['y_col'] not in x_cols_selected)

    logger.info(f"Using columns {','.join(x_cols_selected)} for training")

    return df_train_val_local, df_train_local, df_val_local, df_test_local, x_cols_selected

def get_labels(pca_dim=5):
    logger.info("Building labels...")

    logger.info("Initializing directories..")
    Path(output_data_dir_labels).mkdir(parents=True, exist_ok=True)
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
            ginfo=json.load(f)
            game_data.update(ginfo)

    df_game_data = pd.DataFrame.from_dict(game_data, orient='index')
    df_game_data.reset_index(inplace=True)

    df_game_data.rename({'index': 'appid'}, axis=1, inplace=True)

    columns = params['meta_cols'].copy()
    columns.extend(params['x_cols'])

    logger.info("COLUMN COUNT")
    logger.info(len(columns))
    logger.info('meta cols')
    logger.info(params['meta_cols'])
    logger.info('xcols')
    logger.info(params['x_cols'])

    df_game_data = df_game_data.loc[:, columns].copy()

    logger.info(f"Loaded game data into data frame with columns {','.join(df_game_data.columns)}")

    logger.info("Building the embeddings...")
    logger.info(df_game_data.head())
    df_game_data["full_description"] = df_game_data["description"].astype(str) + df_game_data["about_the_game"]

    df_game_data.to_csv(os.path.join(output_data_dir_labels, 'df_game_data.csv'), index=False)

    logger.info("dtypes df_game_data: ")
    logger.info(df_game_data.dtypes)

    documents = df_game_data['full_description'].tolist()
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

    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_dim)
    res_transformed = pca.fit_transform(res.values)
    df_res_transformed = pd.DataFrame(res_transformed)
    df_res_transformed.columns = [f'x_genre_{i}' for i in df_res_transformed.columns]

    logger.info(df_res_transformed.head())

    logger.info("Combining data frames - game data, embeddings, one-hot encodings")
    df_game_data_one_hot = pd.concat([df_game_data, df_embeddings, df_res_transformed], axis=1)

    filename_ratings = f'{params["user_input_data_dir"]}/ratings.json'
    with open(filename_ratings, 'r') as f:
        ratings_json = json.load(f)
    df_ratings = pd.DataFrame.from_dict(ratings_json, orient='index')
    df_ratings.reset_index(inplace=True)
    df_ratings.rename({'index': 'appid'}, axis=1, inplace=True)
    df_ratings.appid = df_ratings.appid.astype(int)
    logger.info("df ratings head")
    logger.info(df_ratings.head())

    logger.info(f"original length : {len(df_game_data_one_hot)}")
    df_final = df_game_data_one_hot.merge(
        df_ratings[['appid', 'rating']],
        on='appid',
        how='left')
    logger.info(f"new length : {len(df_final)}")

    assert(len(df_final) == len(df_game_data_one_hot))

    logger.info("Filtering basesd on  is released status")
    logger.info(f"initial length {len(df_final)}")
    if params['release_status'] == 'released':
        cond = (df_final.is_released == True)
        df_final = df_final[cond].copy()
    else:
        pass
    logger.info(f"new length {len(df_final)}")

    df_final.to_csv(f"{output_data_dir_labels}/df_labels_all_cols.csv", index=False)

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

    stats = {
        'num_rows': len(df_final),
        'num_cols': len(df_final.columns),
        'num_unique' : df_final.appid.nunique(),
        'game_data_files' : len(game_data)
    }
    logger.info('Stats')
    logger.info(json.dumps(stats,indent=4))

    df_final.to_csv(f"{output_data_dir_labels}/df_labels.csv", index=False)

    return df_final

if __name__ == '__main__':

    df_labels = get_labels()
