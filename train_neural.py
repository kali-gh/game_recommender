import logging
import torch
import json
import torch.nn as nn
import numpy as np
import os

from pathlib import Path
from game_recommender.parameters import Params
from game_recommender.libs_torch import CustomDataset, create_loader
from game_recommender.labels import get_labels, load_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
torch.manual_seed(SEED)

params = Params('input_data/params.json')

output_data_dir_scores = os.path.join(params['output_data_dir'], params['output_data_subdir_scores'])
Path(output_data_dir_scores).mkdir(parents=True, exist_ok=True)

df_labels = get_labels(params['pca_dim'])

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


logger.info("Loading data")
df_train_val, df_train, df_val, df_test, x_cols = load_data()
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
train_loader = create_loader(df_train, x_cols, batch_size, shuffle=False)
val_loader = create_loader(df_val, x_cols, batch_size, shuffle=False)

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
test_loader = create_loader(df_test, x_cols, batch_size, shuffle=False)
inferences = inference()
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
    f"{output_data_dir_scores}/df_score_only.csv",
    index=False)

