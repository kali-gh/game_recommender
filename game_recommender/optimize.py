pca_dim = [1, 3, 5, 10]

from labels import get_labels
from train_bayesian import train

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from parameters import Params
params = Params('input_data/params.json')

rmse = {}
for pca in pca_dim:
    labels = get_labels(pca_dim=pca)
    df_labels = get_labels(pca_dim=pca)
    df_labels.to_csv(f"{params['output_data_dir']}/df_labels.csv", index=False)

    r = train()
    rmse[pca] = r

logger.info(rmse)

