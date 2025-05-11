### Overview

We build a simple game recommendation engine based on a simple content based approach. 
The user inputs their preferred games, by rating games from 1 to 10. Feature data is then used to predict this score. 
The feature data contains (1) basic attributes about the game and 
(2) simple topic model embeddings which describe the game.
We build different models to predict the score based on these features and run inference on recent releases. The user can
also give a input game and the system will output the most similar game along with that games estimated rating.
Our model validation uses a train/validation/test approach and we measure the performance on validation for this and
then run inference on the test set. 

### Motivation
Game recommendations on popular platforms are usually surfaced in a similar fashion as other entertainment applications 
like for music (Spotify, Apple Music) or streaming (Netflix, HBO Max) and come typically in two forms:

1. A carousel of recommended titles "you might enjoy" or "more like this" based on the game that you are currently viewing 
2. Top rated or "most popular titles" based on some user given criteria

Neither of these two will, based on the user's preferences:

1. Proactively highlight new releases that might be of interest to a customer.
2. Recommend a list of titles in the overall catalog that might be of interest.
3. Find an exhaustive list of similar titles, that are also likely to be highly rated by the user.
4. React to new user preferences in a sequential, online fashion as the user interacts with the system.

We develop a system that can solve 1-4 above. This repository contains the modelling portion for the system (see left half
of diagram below).


### Diagram
![alt text](assets/game_recommender_v2.png)


### Data and features

1. The rating scale of the user ranges from 1 (very bad) to 10 (excellent).
2. We use title tags and meta information as is with some minimal imputation.
3. We use derived transforms to represent genres and content in an embedding space using principal components (PCA) and 
Latent Semantic Analysis (LSA) respectively. 

### Results

The cross validated root mean squared error on an input sample of 128 user rated titles gives a cross validated 
root mean squared error of 2.27. 

The performance is comparable to a deep neural network baseline (implemented in train_neural.py).  


### Example outputs

Given a user profile that has preferences for:
1. Preference for action-adventure titles & Soul's like titles
2. Preference for strategy & tactics games

The system recommends new titles as of 2025.05.10 as shown below:

> <a href="https://store.steampowered.com/app/2680010">The First Berserker: Khazan</a>, Score = 7.65
>
> <a href="https://store.steampowered.com/app/3373660/Look_Outside">Look Outside</a>, Score = 7.49
> 
> <a href="https://store.steampowered.com/app/1486920/Tempest_Rising/">Tempest Rising</a>, Score = 7.27
> 
> <a href="https://store.steampowered.com/app/1903340">Clair Obscur: Expedition 33</a>, Score = 7.25
> 
> <a href="https://store.steampowered.com/app/2481670/Dino_Path_Trail/">Dino Path Trail</a>, Score = 7.20

which are broadly a good match to the user profile, with the possible exception of Look Outside which has horror elements.

### Running

After setting up the input data (which is not provided in this repo), run :

```
python train_bayesian.py
```


### Environment set up

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


### Building as a wheel
```
python setup.py bdist_wheel
```

