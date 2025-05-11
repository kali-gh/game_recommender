import torch
import json
import logging
import numpy as np
import os

from pathlib import Path
from scipy.stats import multivariate_normal

from bayesian_normal_regression.bayes_normal import BayesNormal
from bayesian_normal_regression.dist import Normal

from game_recommender.libs_torch import create_loader
from game_recommender.labels import get_labels, load_data
from game_recommender.parameters import Params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("initializing")

SEED = 42
torch.manual_seed(SEED)

params = Params('input_data/params.json')
df_labels = get_labels(params['pca_dim'])

output_data_dir_scores = os.path.join(params['output_data_dir'], params['output_data_subdir_scores'])
Path(output_data_dir_scores).mkdir(parents=True, exist_ok=True)

def train():

    logger.info("Loading data")
    df_train_val, df_train, df_val, df_test, x_cols = load_data()
    input_size = len(x_cols)

    batch_size = params['batch_size']

    if params['use_all_data']:
        logger.info("Using all data for training")
        train_loader = create_loader(df_train_val, x_cols, batch_size)
    else:
        logger.info("only using training")
        train_loader = create_loader(df_train, x_cols, batch_size)

    val_loader = create_loader(df_val, x_cols, batch_size)

    logger.info("Training model")

    logger.info("Running inference on test data")
    logger.info(len(df_test))
    test_loader = create_loader(df_test, x_cols, batch_size, shuffle=False)

    #### parameters
    n_param = input_size
    prior_alpha = 2.0
    prior_mean = np.zeros(shape=(n_param, 1))
    prior_cov = (1 / prior_alpha) * np.identity(n_param)

    w0_true = -0.7
    w1_true = 0.5
    precision_y = 25
    std_dev_y = np.sqrt(1 / precision_y)

    logger.debug("PARAMETERS")
    logger.debug(
        f"{prior_alpha=}\n"
        f"{prior_mean=}\n"
        f"{prior_cov=}\n"

        f"{w0_true=}\n"
        f"{w1_true=}\n"

        f"{precision_y=}\n"
        f"{std_dev_y=}\n"
    )

    ##### main script
    logger.debug("Initializing prior")
    prior = Normal(mean=prior_mean, cov=prior_cov)
    logger.debug(prior)


    def model(W_array, X_array):
        return W_array[0] + W_array[1] * X_array

    bn = BayesNormal(variance=1, prior_mean=prior_mean, prior_cov=prior_cov)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        bn.add_observations(X=inputs.numpy(), y=targets.numpy())

        logger.debug('posterior')
        logger.debug(bn.posterior)

    X_val, y_val = df_val.loc[:, x_cols], df_val.loc[:, 'rating']
    y_pred = X_val @ bn.posterior.mean

    from sklearn.metrics import root_mean_squared_error

    rmse = root_mean_squared_error(y_true=y_val, y_pred=y_pred)
    logger.info('rmse on validation')
    logger.info(rmse)

    logger.info("Running inference on test data")
    logger.info(len(df_test))
    test_loader = create_loader(df_test, x_cols, batch_size, shuffle=False)
    X_test = df_test.loc[:, x_cols]
    inferences = X_test @ bn.posterior.mean
    df_test['score'] = inferences

    logger.info("Outputing scores")
    cols_score = ['appid', 'title', 'score']
    df_score_only = df_test[cols_score].copy()
    df_test.to_csv(f"{output_data_dir_scores}/df_test.csv")
    df_score_only.sort_values(by='score', ascending=False, inplace=True)

    with open(f"{params['user_input_data_dir']}/sample_ids_recent_releases.json", 'r') as f:
        recent_release_ids = json.load(f)

    df_score_only.loc[:, 'recent_release'] = df_score_only.appid.isin(recent_release_ids)

    df_score_only.to_csv(
        f"{output_data_dir_scores}/df_score_only_bayesian_mean.csv",
        index=False)

    logger.info("Running inference on test data")
    logger.info(len(df_test))
    test_loader = create_loader(df_test, x_cols, batch_size, shuffle=False)
    X_test = df_test.loc[:, x_cols]
    sample_posterior = multivariate_normal(mean=bn.posterior.mean.flatten(), cov=bn.posterior.cov).rvs(size=1)
    sample_posterior = np.reshape(sample_posterior, (-1, 1))
    inferences = X_test @ sample_posterior
    df_test['score'] = inferences

    logger.info("Outputing scores")
    cols_score = ['appid', 'title', 'score']
    df_score_only = df_test[cols_score].copy()
    df_test.to_csv(f"{output_data_dir_scores}/df_test.csv")
    df_score_only.sort_values(by='score', ascending=False, inplace=True)

    with open(f"{params['user_input_data_dir']}/sample_ids_recent_releases.json", 'r') as f:
        recent_release_ids = json.load(f)

    df_score_only.loc[:, 'recent_release'] = df_score_only.appid.isin(recent_release_ids)

    df_score_only.to_csv(
        f"{output_data_dir_scores}/df_score_only_bayesian_thompson.csv",
        index=False)

    return rmse

if __name__ == '__main__':
    train()