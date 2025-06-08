import torch
import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path

from game_recommender.labels import get_labels, load_data
from game_recommender.parameters import Params
from bayesian_normal.bayes_normal_known_var import BayesNormalEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("initializing")

SEED = 42
torch.manual_seed(SEED)



def get_x_cols(
        df,
        params):
    """
    Updates the x_cols list based on the columns in the dataframe

    Args:
        df : dataframe with data
        params: params to use

    Returns:
        x_cols : updated list of x columns
    """

    logger.info(f"Adding columns to x_cols {params['x_cols_model']}")
    x_cols = params['x_cols_model']

    for prefix in params['x_cols_model_prefixes']:
        logger.info(f"Adding prefix {prefix} to x_cols")
        extra_genre_x_cols = [c for c in df.columns if (c.startswith(prefix))]
        x_cols.extend(extra_genre_x_cols)

    logger.info(f"Number of x_cols {len(x_cols)}")
    logger.info(f"Final x_cols: {x_cols}")

    return x_cols

def run_inference_test(
        df,
        x_cols):
    """
    Runs inference on the test data, i.e. games with no rating

    Args:
        df : dataframe with test data
        x_cols : x columns for model

    Returns:
        df : the modified dataframe
    After:
        inferences written to df_predict.csv
    """

    cond_train_val = ~df.rating.isnull()
    df_train_val_local = df[cond_train_val].copy()
    cond_test = df.rating.isnull()
    df_test_local = df[cond_test].copy()

    logger.info(x_cols)
    X = df_train_val_local[x_cols].values

    y = df_train_val_local.rating.values
    y = y.reshape(-1, 1)

    n_param = len(x_cols)
    prior_alpha = 2.0
    prior_mean = np.zeros(shape=(n_param, 1))
    prior_cov = (1 / prior_alpha) * np.identity(n_param)

    logger.info(f"shape of x {X.shape}, shape of y {y.shape}")

    bn = BayesNormalEstimator(variance=1, prior_mean=prior_mean, prior_cov=prior_cov, predict_thompson_sample=True)
    fitted = bn.fit(X, y)
    df_train_val_local.loc[:, 'score'] = fitted.predict(X)

    X_test = df_test_local[x_cols].values
    logger.info(f"shape of x TEST {X_test.shape},")

    df_test_local.loc[:, 'score'] = fitted.predict(X_test)

    df_out = pd.concat([df_train_val_local, df_test_local], axis=0)
    logger.info(f"len of df is {len(df_out)}")

    df_out.sort_values(by='score', ascending=False, inplace=True)
    df_out.to_csv('df_predict.csv')

    return df_out


def train(pca_dim = 5, params=None):

    if params is None:
        params = Params('input_data/params.json')

        output_data_dir_scores = os.path.join(params['output_data_dir'], params['output_data_subdir_scores'])
        Path(output_data_dir_scores).mkdir(parents=True, exist_ok=True)
    else:
        logger.info("using supplied params")

    df_labels = get_labels(pca_dim, params=params)

    from game_recommender.labels import build_df_model_ready
    df = build_df_model_ready(params=params)

    cond_train_val = ~df.rating.isnull()
    df_train_val_local = df[cond_train_val].copy()

    x_cols = get_x_cols(df, params)

    df_train_val, df_test = train_test_split(df_train_val_local, test_size=0.2)

    X_train_val = df_train_val[x_cols].values
    y_train_val = df_train_val['rating'].values.reshape(-1,1)

    X_test = df_test[x_cols].values
    y_test = df_test['rating'].values.reshape(-1,1)

    n_param = len(x_cols)
    prior_alpha = 2.0
    prior_mean = np.zeros(shape=(n_param, 1))
    prior_cov = (1 / prior_alpha) * np.identity(n_param)

    logger.info(f"shape of x {X_train_val.shape}, shape of y {y_train_val.shape}")

    bn = BayesNormalEstimator(
        variance=1,
        prior_mean=prior_mean,
        prior_cov=prior_cov,
        predict_thompson_sample=True)

    scores = np.sqrt(-1*
        cross_val_score(
            bn,
            X_train_val,
            y_train_val,
            cv=10,
            scoring='neg_mean_squared_error')
    )

    logger.info(f"Scores : {scores}")
    logger.info(f"Mean score : {scores.mean()}")

    run_inference_test(df, x_cols)

    return scores.mean()

if __name__ == '__main__':
    train()