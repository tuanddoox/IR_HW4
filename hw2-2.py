#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from argparse import Namespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm.notebook import trange

from ltr.utils import seed


# # Chapter 2: Counterfactual LTR (130 points)

# Loading the dataset:

# In[ ]:


from ltr.dataset import load_data

data = load_data()


# We assume that there is a logging policy that shows the results for each query to the users and logs the user clicks.
# For that, we provide a logging policy simulator `LoggingPolicy`.
# Our logging policy only shows top 20 documents to the users.
# You can use this simulator to:
# - Get the position of the documents for a query in the SERP: `query_positions`.
# - Gather the (simulated) clicks of users for a query: `gather_clicks`.
# 

# In[ ]:


from ltr.logging_policy import LoggingPolicy

logging_policy = LoggingPolicy()

# Gather the clicks on the SERP for query 20
for i in range(10):
    clicked_docs = np.where(logging_policy.gather_clicks(20))[0]
    clicked_positions = logging_policy.query_positions(20)[clicked_docs]
    print(f'clicks for session {i+1} on documents', clicked_docs, 'on positions', clicked_positions)



# ---
# 
# ## Utils (10 points)
# 
# ### Click data loader (10 points)
# First, we need to have a data loader that feeds the model with features and click data.
# In this data loader, you have to select `topk=20` items for each query, and return three tensors:
# - Feature vectors of the selected documents,
# - One instance of the clicks over the selected documents, using the `gather_clicks(qid)` function, and
# - The positions of the selected documents in the SERP.
# 
# **IMPORTANT** Here you *should not* use the `labels` for training. It is assumed that we cannot observe the real labels and want to use the `clicks` to train our LTR model instead.
# 

# In[ ]:


from ltr.dataset import ClickLTRData

train_dl = DataLoader(ClickLTRData(data, logging_policy), batch_size=1, shuffle=True)

for features, clicks, positions in train_dl:
    print(features.shape, clicks.shape, positions.shape)

    assert positions.dtype == torch.long
    print(features.shape, clicks.shape, positions.shape)
    print('clicks:', clicks)
    print('positions:', positions)
    break


# ### LTR model (0 points!)
# Further, let's modify the `LTRModel` from previous chapter and take the width of the middle layer as an argument:

# In[ ]:


from ltr.model import LTRModel

net = LTRModel(data.num_features, width=20)
print(net)


# ---
# 
# ## ListNet (40 points)
# 
# In the previous chapter, you have implemented different loss functions for LTR.
# Here we use another well known listwise loss funtion, called `ListNet`, and will use it for our unbiased LTR model.
# The idea behind ListNet is very simple:
# To solve the discontinuity issue of NDCG, in **ListNet**, the loss function is based on probability distribution on permutations.
# 
# Define a family of distributions on permutation of scores $z$, $P_z(\pi)$, s.t. $\sum_{\pi\in\Omega} P_z(\pi)=1$, where $\Omega$ is the set of all $n!$ permutations.
# Ideally, we want the scores of our LTR model lead to the same permutation distribution as the labels $y$, i.e.,
# 
# $$
# \min KL(P_y,P_z)=-\sum_{\pi\in\Omega} P_y(\pi) \log P_z(\pi)
# $$
# 
# Plackett-Luce distribution gives a general formula for calculating the permutation distribution:
# 
# $$
# P_z(\pi) = \prod_{j=1}^{n} \frac{\exp(z_{\pi(j)})}{\sum_{k=j}^{n} \exp(z_{\pi(k)})}
# $$
# In ListNet, instead of calculating $n!$ permutation probabilities, the top one probability of each document is calculated:
# 
# $$
# P_z(j) = \sum_{\pi(1)=j, \pi\in\Omega} P_z(\pi) = \frac{\exp(z_{j})}{\sum_{k=1}^{n} \exp(z_{k})},
# $$
# which is the softmax function.
# 
# Then, the loss is defined as follows:
# 
# $$
# \mathcal{L}_{\text{ListNet}}=-\sum_{j=1}^{n} P_y(j) \log P_z(j),
# $$
# where the softmax function is used to calculate $P_y(j)$ and $P_z(j)$ from the labels and predictions, respectively.

# ### ListNet loss function (20 points)
# Implement the ListNet loss function.

# In[ ]:


from ltr.loss import listNet_loss
 
biased_net = LTRModel(data.num_features, width=20)

for features, clicks, positions in train_dl:
    print(features.shape, clicks.shape, positions.shape)
    output = biased_net(features)
    print(output.shape, clicks.shape)
    loss = listNet_loss(output, clicks)
    print(loss)
    break


# ### Biased ListNet training (10 points)
# Now use `listNet_loss` to train an LTR model. Since we use `clicks` instead of `relevance`, and do not correct for the bias, this would be a biased model.

# In[ ]:


from ltr.train import train_biased_listNet

params = Namespace(epochs=1, 
                    lr=1e-4,
                    batch_size=1,
                    metrics={"ndcg@10", "precision@10", "recall@10"})

biased_net = LTRModel(15, width=20)
train_biased_listNet(biased_net, params, data)


# ### Saving the results (10 points - no implementation!)
# Since we randomly simulate clicks and use them to train our model, for the evaluation we train and save 10 different models and inspect the average and std over them.
# 
# **IMPORTANT** Run the following cell to store your models and results. After it finishes, make sure to push the results to the git repo.
# 
# _Estimated time on Codespaces_: 5m

# In[ ]:


from ltr.utils import create_results
from ltr.train import train_biased_listNet

seed(42)
params = Namespace(epochs=20, 
                    lr=1e-4,
                    batch_size=1,
                    metrics={"ndcg@10", "precision@10", "recall@10"})


for i in range(10):
    print('Training Model', i)
    biased_net = LTRModel(15, width=20)
    create_results(data, biased_net, 
                train_biased_listNet, 
                biased_net,
                f"./outputs/biased_listNet_{i}.json",
                params)

    torch.save(biased_net.state_dict(), f"./outputs/biased_listNet_{i}")


# ---
# 
# ## Unbiased ListNet (30 points)
# 
# ### Unbiased ListNet loss function (10 points)
# 
# Now, we use IPS to have an unbiased ListNet:

# In[ ]:


from ltr.loss import unbiased_listNet_loss

unbiased_net = LTRModel(data.num_features, width=20)
propensity = logging_policy.propensity



for features, clicks, positions in train_dl:
    print(features.shape, clicks.shape, positions.shape)
    output = biased_net(features)
    print(output.shape, clicks.shape)
    loss = unbiased_listNet_loss(output, clicks, propensity[positions.data.numpy()])
    print(loss)
    break



# ### Unbiased ListNet training (10 points)
# Now use `unbiased_listNet_loss` to train an LTR model.

# In[ ]:


from ltr.train import train_unbiased_listNet

params = Namespace(epochs=1, 
                    lr=1e-4,
                    batch_size=1,
                    propensity=logging_policy.propensity,
                    metrics={"ndcg@10", "precision@10", "recall@10"})

biased_net = LTRModel(15, width=20)
train_unbiased_listNet(biased_net, params, data)


# ### Saving the results (10 points - no implementation!)
# Similar to the biased model, here we train 10 different unbiased models and save them to inspect the average and std over them.
# 
# **IMPORTANT** Run the following cell to store your models and results. After it finishes, make sure to push the results to the git repo.
# 
# _Estimated time on Codespaces_: 5m

# In[ ]:


from ltr.utils import create_results
from ltr.train import train_unbiased_listNet

seed(42)
params = Namespace(epochs=20, 
                    lr=1e-4,
                    batch_size=1,
                    propensity=logging_policy.propensity,
                    metrics={"ndcg@10", "precision@10", "recall@10"})

for i in range(10):
    print('Training Model', i)
    unbiased_net = LTRModel(15, width=20)
    create_results(data, unbiased_net, 
                train_unbiased_listNet, 
                unbiased_net,
                f"./outputs/unbiased_listNet_{i}.json",
                params)

    torch.save(unbiased_net.state_dict(), f"./outputs/unbiased_listNet_{i}")



# ---
# 
# ## Propensity estimation (35 points)
# 
# In training our unbiased ListNet model, we assumed that we know propensity values.
# In practice, however, the propensity values have to be estimated from the clicks.
# There are several methods for estimating the propensities, such as dual learning algorithm (DLA) and regression-based EM.
# Here, we focus on DLA.
# 
# ### DLA
# 
# IPS is based on the examination hypothesis that says $P(c=1)=P(r=1)\times P(e=1)$, where $c$, $r$ and $e$ are click, relevance and examination signals, respectively.
# Initially, we are interested in $P(r=1)$, so in IPS we substitute $c$ with $\hat{r}=\frac{c}{P(e=1)}$.
# In practice, $P(e=1)$ is not given and should be estimated.
# DLA solves this by noticing that $\hat(e)=\frac{r}{P(r=1)}$ is also an unbiased estimation for the examination probability.
# This means that in DLA (as the name suggests), two models are trained at the same time:
# - Relevance prediction: A function $f$, modeled by `LTRModel` here, that estimates the relevance from the feature vectors.
# - Propensity prediction: A function $g$, modeled by `PropLTRModel` here, that estimates the propensity from the positions.
# 
# Using the `unbiased_listNet_loss` loss function with the following signature:
# $$
# \mathcal{L}_{\text{unbiased}}\big(\text{predictions}, \text{clicks}, \text{propensities}\big),
# $$
# 
# the overall loss function is as follows:
# $$
# \mathcal{L}_{\text{DLA}} = \underbrace{\mathcal{L}_{\text{unbiased}}\bigg(f(x), c, \sigma\big(g(p)\big)\bigg)}_{\text{relevance estimation}} + \underbrace{\mathcal{L}_{\text{unbiased}} \bigg(g(p), c, \sigma\big(f(x)\big)\bigg)}_{\text{propensity estimation}},
# $$
# which means that the predictions of $g$ are used as the propensities for optimizing $f$, and the predictions of $f$ are used as the propensities for optimizing $g$.
# The $\sigma()$ function is used to transform the logits to valid probability valules, as the propensities should be between 0 and 1.

# ### Logits to prob (2 points)
# First, we need a function to transform the logits to valid probability values (between 0 and 1).
# Use the sigmoid function for this transformation.

# In[ ]:


from ltr.train import logit_to_prob

logits = 10 * torch.rand(10)
probs = logit_to_prob(logits)

# Print the propensities
print('probabilities:', probs.squeeze())


# ### Propensity estimation LTR model (3 points)
# Then, we need a wrapper around the `LTRModel` that takes as input the positions (Long tensor) and outputs the logits for propensities.
# This new model uses one hot embedding as the input features.

# In[ ]:


from ltr.model import PropLTRModel

prop_net = PropLTRModel(logging_policy.topk, width=200)

logits = prop_net(torch.arange(17))
probs = logit_to_prob(logits)

# Print the propensities
print('probabilities:', probs.T)

# Print the normalized propensities
print('normalized with the first position:', probs.T/probs.squeeze()[0])        


# ### DLA training (20 points)
# Now we have all we need for the DLA implementation.

# In[ ]:


from ltr.train import train_DLA_listNet

params = Namespace(epochs=5, 
                    lr=1e-4,
                    batch_size=1,
                    prop_lr=1e-3,
                    prop_net=PropLTRModel(logging_policy.topk, width=256),
                    metrics={"ndcg@10", "precision@10", "recall@10"})

biased_net = LTRModel(15, width=256)
print('True (unknown to the model) propensities:', logging_policy.propensity.data.numpy())
train_DLA_listNet(biased_net, params, data)


# ### Saving the results (10 points - no implementation!)
# Similar to the biased model, here we train 10 different unbiased models and save them to inspect the average and std over them.
# 
# **IMPORTANT** Run the following cell to store your models and results. After it finishes, make sure to push the results to the git repo.
# 
# _Estimated time on Codespaces_: < 10m

# In[ ]:


from ltr.utils import create_results
from ltr.train import train_DLA_listNet

seed(42)
params = Namespace(epochs=20, 
                    lr=1e-4,
                    batch_size=1,
                    prop_lr=1e-3,
                    prop_net=None,
                    metrics={"ndcg@10", "precision@10", "recall@10"})

for i in range(10):
    print('Training Model', i)
    dla_net = LTRModel(15, width=256)
    params.prop_net = PropLTRModel(logging_policy.topk, width=256)
    create_results(data, dla_net, 
                train_DLA_listNet, 
                dla_net,
                f"./outputs/DLA_listNet_{i}.json",
                params)

    torch.save(dla_net.state_dict(), f"./outputs/DLA_listNet_{i}")
    torch.save(params.prop_net.state_dict(), f"./outputs/DLA_listNet_prop_{i}")


# ---
# 
# ## Comparing the models (15 points)
# 
# You have implemented three models: biased, unbiased with oracle propensity values, and unbiased with DLA-estimated propensity values.
# Given the training results and evaluation results, please elaborate on the ranking performance of these three models in `analysis.md`. See that file for further details.
# 
# Note that you need to submit the result files created in `outputs/` for full credit.

# In[ ]:


import json

def aggregate_results(model_name):
    aggregated_metrics = {}
    for i in range(10):
        with open(f"./outputs/{model_name}_{i}.json", "r") as reader:
            result = json.load(reader)
            for metric, (v, std) in result['test_metrics'].items():
                aggregated_metrics.setdefault(metric, []).append(v)
    return {metric: np.mean(vals) for metric, vals in aggregated_metrics.items()}

biased = aggregate_results('biased_listNet')
unbiased = aggregate_results('unbiased_listNet')
DLA = aggregate_results('DLA_listNet')

# save the aggregated output files
for model_avg_results, model_name in zip([biased, unbiased, DLA], ["biased_listNet", "unbiased_listNet", "DLA_listNet"]):
    json.dump(model_avg_results, open(f"outputs/{model_name}_avg.json", "wt"))

# display a handful of metrics
print_metrics = ["ndcg", "ndcg@20", "precision@05", "recall@20"]
print_biased = {metric: v for metric, v in biased.items() if metric in print_metrics}
print_unbiased = {metric: v for metric, v in unbiased.items() if metric in print_metrics}
print_DLA = {metric: v for metric, v in DLA.items() if metric in print_metrics}

import pandas as pd
pd.set_option("display.precision", 3)
df = pd.DataFrame([print_biased, print_unbiased, print_DLA], index=["biased", "unbiased", "DLA"])
print(df)

from IPython.display import display, HTML
display(df)


# In[ ]:


# remember to submit your outputs!

