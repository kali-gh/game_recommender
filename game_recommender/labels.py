import logging
import os
import json
import random
import pandas as pd
import numpy as np
import psutil

from .libs import build_df_embeddings

from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize

from .parameters import Params

##logging
import logging
import sys

message_format_string = "%(asctime)s;%(levelname)s;%(message)s"
date_format_string = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO,
                    format=message_format_string,
                    datefmt=date_format_string,
                    filename='logger-recommender.log', filemode='a')
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(message_format_string, date_format_string)
sh.setFormatter(formatter)
logging.getLogger('').addHandler(sh)

logger = logging.getLogger(__name__)
logger.info("starting up")
###

SEED = 1
random.seed(SEED)
np.random.seed(SEED)

def print_memory_usage():
    process = psutil.Process(os.getpid())
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")


def get_active_genres(df):
    # Filter columns that start with 'genre_'
    genre_columns = [col for col in df.columns if col.startswith('genre_') and col != 'genre_+']

    def process_row(row):
        # Get active genres where value is 1
        active_genres = [col for col in genre_columns if row[col] == 1]
        # Clean up genre names by removing 'genre_' prefix
        return [genre.replace('genre_', '') for genre in active_genres]

    return df.apply(process_row, axis=1)

def build_df_model_ready(params, df = None):
    """
    Builds the model read dataframe using the consolidated labels file

        - Imputes missing review scores with 0.0
        - Imputes counts with 0
        - Logs the review counts

    Returns: dataframe with model ready data
    After : writes a _SUCCESS statement to signal the data has been swapped out
    """

    logger.info("Loading data from input_data directory")

    print_memory_usage()

    if df is None:
        logger.info(f"Loading data from {params['output_data_dir_labels']}")

        df = pd.read_csv(f"{params['output_data_dir_labels']}/df_labels.csv")
    else:
        logger.info("using passed df labels")

    logger.info("Imputing mean review and min review counts")
    if 'review_pct_overall' in df.columns:
        df.loc[:, 'review_pct_overall'] = df.loc[:, 'review_pct_overall'].fillna(0)
    else:
        pass

    if 'review_count_overall' in df.columns:
        df.loc[:, 'review_count_overall'] =  df.loc[:, 'review_count_overall'].fillna(0)
        vals = df.loc[:, 'review_count_overall'].astype('Float64')
        df.loc[:, 'review_count_overall'] = None
        df.loc[:, 'review_count_overall'] = np.log(vals + 1.0)
    else:
        pass

    logger.info("adding list of genres")
    df['genres'] = get_active_genres(df)

    # winsorization
    winsorization_cols = [c for c in df.columns if c.startswith('x_genre_') or c.startswith('x_content_')]
    for c in winsorization_cols:
        logger.info(f"Winsorizing {c} to 0.01, removing top and bottom 1% extreme values")
        df.loc[:, c] = winsorize(df.loc[:, c].values, limits=[0.01,0.01])

    df.to_csv(f"{params['output_data_dir_labels']}/df_labels_model_ready.csv", index=False)
    with open(f"{params['output_data_dir_labels']}/_SUCCESS", 'w') as f:
        f.write('done')

    logger.info("writing out genre columns")
    genre_columns = [c for c in df.columns if c.startswith('genre_')]
    with open(f"{params['output_data_dir_labels']}/genre_columns.json", 'w') as f:
        json.dump(genre_columns, f)

    print_memory_usage()

    logger.info("done building df model ready")

    return df


def load_data(params):
    """
    Builds the model ready dataframe and splits the data for estimation

    Returns:
        df_train_val_local - dataframe with train and validation data
        df_train_local - dataframe with train data
        df_val_local - dataframe with validation data
        df_test_local - dataframe with test data
        x_cols_selected - x columns to use
    """

    df = build_df_model_ready(params)

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

def get_labels(
        pca_dim = 5,
        params = None):
    """
    Builds the labels dataframe by
        1. Loading all the parsed data for the games
        2. Combines "about the game" and "description" fields into one text field
        3. Builds Latent Semantic Analysis embeddings with the text data from the previous step with key topic vectors
        4. OneHotEncodes the tag data
        5. Runs the one hot's through PCA to get the pricnipal dimensions of tags and represents the tags data in the transformed space
        6. Joins the rating data onto the labels
        7. Filters to the released status
        8. Outputs data to f"{output_data_dir_labels}/df_labels.csv"

    Args:
        pca_dim : number of PCA components to use
        params: params for model

    Returns:
        pd  .DataFrame - dataframe after all the operations above
    After :
        writes the labels dataframe to output_data_dir_labels
    """

    logger.info("Building labels...")

    print_memory_usage()

    if params is None:
        logger.info("using params file")
        params = Params('input_data/params.json')
    else:
        logger.info("Using params passed in")

    logger.info("Initializing directories..")
    Path(params['output_data_dir_labels']).mkdir(parents=True, exist_ok=True)
    Path(params['model_dir']).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(params['input_data_dir']) or not os.path.exists(params['user_input_data_dir'])  :
        raise AssertionError("input_data or user_input_data directory not found.")
    else:
        pass

    logger.info("Building the base df_game_data frame...")
    input_data_dir = params['input_data_dir']
    file_list_games = [f for f in os.listdir(input_data_dir) if f.endswith('.json')]

    logger.info(f"Found {len(file_list_games)} games in the input_data directory")
    logger.info(f"First 10 games : {file_list_games[:10]}")

    game_data = {}
    for g in file_list_games:
        with open(os.path.join(input_data_dir, g), 'r') as f:
            ginfo=json.load(f)

            for k in params['x_cols_never_needed']:
                if k in ginfo.keys():
                    del ginfo[k]

            game_data.update(ginfo)

    print_memory_usage()

    df_game_data = pd.DataFrame.from_dict(game_data, orient='index')
    df_game_data.reset_index(inplace=True)

    del game_data
    import gc
    gc.collect()

    print_memory_usage()

    df_game_data.rename({'index': 'appid'}, axis=1, inplace=True)

    columns = params['meta_cols'].copy()
    columns.extend(params['x_cols'])

    logger.info("COLUMN COUNT")
    logger.info(len(columns))
    logger.info('meta cols')
    logger.info(params['meta_cols'])
    logger.info('xcols')
    logger.info(params['x_cols'])

    #df_game_data = df_game_data.loc[:, columns].copy()

    logger.info(f"Loaded game data into data frame with columns {','.join(df_game_data.columns)}")

    logger.info("Building the embeddings...")
    logger.info(df_game_data.head())

    logger.info("dtypes df_game_data: ")
    logger.info(df_game_data.dtypes)

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
    if 'x_content' in params['x_cols_model_prefixes']:
        logger.info("including embeddings")
        documents = df_game_data['words'].values
        logger.info("documents contents before embeddings")
        logger.info([d[:5] for d in documents[:2]])
        df_embeddings = build_df_embeddings(
            documents = documents,
            model_dir = params['model_dir'],
            stoplist = params['stoplist'],
            num_topics = params['num_topics'],
            min_word_length = params["min_word_length"],
            parse = True
        )

        df_game_data_one_hot = pd.concat([df_game_data, df_embeddings, res, df_res_transformed], axis=1)
        del df_embeddings
    else:
        logger.info("not running embeddings")
        df_game_data_one_hot = pd.concat([df_game_data, res, df_res_transformed], axis=1)

    del df_game_data
    del res
    del df_res_transformed
    gc.collect()

    print_memory_usage()

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

    df_final.to_csv(f"{params['output_data_dir_labels']}/df_labels_all_cols.csv", index=False)

    df_final.drop(
        params['x_cols_drop']
        , axis=1
        , inplace=True
    )
    other_cols = [c for c in df_final.columns if c != params['y_col']]

    new_col_order = other_cols.copy()
    new_col_order.append(params['y_col'])

    df_final = df_final.loc[:, new_col_order]

    logger.info(f"Loaded data with column order : {','.join(df_final.columns)}")

    stats = {
        'num_rows': len(df_final),
        'num_cols': len(df_final.columns),
        'num_unique' : df_final.appid.nunique(),
        'game_data_files' : len(df_final)
    }
    logger.info('Stats')
    logger.info(json.dumps(stats,indent=4))

    df_final['review_pct_overall'] = df_final['review_pct_overall'].astype('Int64')
    df_final['review_count_overall'] = df_final['review_count_overall'].astype('Int64')

    df_final.to_csv(f"{params['output_data_dir_labels']}/df_labels.csv", index=False)

    print_memory_usage()

    return df_final

if __name__ == '__main__':

    df_labels = get_labels()
