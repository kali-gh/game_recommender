import logging
import torch
import random
import json
import torch.nn as nn
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from parameters import Params

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

params = Params('input_data/params.json')


def load_data():
    """
    Loads data from input_data directory
    Returns: train, validation and test dataframes + x columns
    """

    logger.info("Loading data from input_data directory")

    logger.info(f"Loading data from {params['output_data_dir']}")

    df = pd.read_csv(f"{params['output_data_dir']}/df_labels.csv")

    logger.info("Imputing mean review and min review counts")
    mean_review_pct_overall = df.loc[:, 'review_pct_overall'].dropna().mean()
    min_review_count_overall = 1

    df.loc[:, 'review_pct_overall'] = df.loc[:, 'review_pct_overall'].fillna(mean_review_pct_overall)
    df.loc[:, 'review_count_overall'] = min_review_count_overall

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

    return df_train_local, df_val_local, df_test_local, x_cols_selected

class NN(nn.Module):
    """
    Simple FC NN with some dropout
    """
    def __init__(self):

        super(NN, self).__init__()

        layers = []

        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_model():
    """
    Runs training

    Returns: nothing
    After: model trained
    """

    model.train()
    for epoch in range(num_epochs):

        running_loss = 0.0
        running_val_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        for val_batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
            model.eval() # block
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            running_val_loss += val_loss.item()
            model.train()

        epoch_loss = running_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, ValLoss: {epoch_val_loss}')

def inference():
    """
    Runs inference on the model on the test loader

    Returns: array of scores
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)

class CustomDataset(Dataset):
    """
    Defines a dataset for our processing
    """

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_loader(df, shuffle=True):
    """
    Builds a loader to use for training / validation / test

    Args:
        df: dataframe to pull loader from
    Returns:
        the loader to use
    """

    X = torch.FloatTensor(df[x_cols].values)
    y = torch.FloatTensor(df.rating.values).reshape(-1, 1)

    dataset = CustomDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader

logger.info("Loading data")
df_train, df_val, df_test, x_cols = load_data()
input_size =  len(x_cols)

logger.info("Defining parameters")
hidden_sizes = params['hidden_sizes']
output_size = params['output_size']

learning_rate = params['learning_rate']
num_epochs = params['num_epochs']
batch_size = params['batch_size']
dropout_rate = params['dropout_rate']

logger.info("Defining model")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

logger.info("Building train and validation loader")
train_loader = create_loader(df_train)
val_loader = create_loader(df_val)

logger.info("Training model")

model_path = f"{params['model_dir']}/model.pt"
if params['train']:
    train_model()
    torch.save(
        model.state_dict(),
        model_path)
else:
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

logger.info("Running inference on test data")
logger.info(len(df_test))
test_loader = create_loader(df_test, shuffle=False)
inferences = inference()
df_test['score'] = inferences

logger.info("Outputing scores")
cols_score = ['appid', 'title', 'score']
df_score_only = df_test[cols_score].copy()
df_test.to_csv(f"{params['output_data_dir']}/df_test.csv")
df_score_only.sort_values(by='score', ascending=False, inplace=True)

with open(f"{params['user_input_data_dir']}/sample_ids_recent_releases.json", 'r') as f:
    recent_release_ids = json.load(f)

df_score_only.loc[:, 'recent_release'] = df_score_only.appid.isin(recent_release_ids)

df_score_only.to_csv(
    f"{params['output_data_dir']}/df_score_only.csv",
    index=False)

