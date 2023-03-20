import collections
import os
import torch
import pytest

from .public_variables import get_mock_data


def test_ClickLTRData(get_mock_data):
    from ltr.logging_policy import LoggingPolicy
    from ltr.dataset import ClickLTRData

    data = get_mock_data
    logging_policy = LoggingPolicy()

    click_data = ClickLTRData(data, logging_policy)
    assert isinstance(click_data[0][0], torch.FloatTensor)
    assert click_data[0][0].shape == torch.Size((20,15))

    ctr = 0
    for i in range(1000):
        ctr += click_data[1][1].sum()
    assert (ctr > 1800) & (ctr < 2300)

    assert torch.allclose(click_data[2][2], torch.LongTensor([18, 10,  9,  2,  6,  0,  4,  7, 11,  1,  5,  8, 12, 13, 14,  3, 15, 19,
        16, 17]))




def test_losses():
    from ltr.loss import listNet_loss
    from ltr.loss import unbiased_listNet_loss

    output = torch.tensor([[[-0.5],[-.4],[-.5],[ .002],[ 2.2],[ 7],[ 1.8],[ 6.5],
                            [ 6.8],[ 2.3],[-.3],[ .8],[-.3],[-.6],[ .005],[-.5],
                            [ .002],[ .2],[-.6],[-.07]]])
    clicks1 = torch.tensor([[1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0.]])
    clicks2 = torch.tensor([[0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    assert listNet_loss(output, clicks1) > listNet_loss(output, clicks2) 

    propensity = 1./torch.arange(1,21)
    assert unbiased_listNet_loss(output, clicks1, propensity) > listNet_loss(output, clicks1) 
    assert unbiased_listNet_loss(output, clicks2, propensity) < listNet_loss(output, clicks2) 


def test_train_biased_listNet(get_mock_data):
    from ltr.train import train_biased_listNet
    from argparse import Namespace
    from ltr.model import LTRModel
    import numpy as np
    from ltr.utils import seed
    

    data = get_mock_data

    params = Namespace(epochs=10, 
                        lr=1e-3,
                        batch_size=1,
                        metrics={"ndcg"})

    seed(42)
    biased_net = LTRModel(15, width=20)
    results = train_biased_listNet(biased_net, params, data)
    assert np.allclose(results['metrics_val'][-1]['ndcg'][0], 0.45, atol=0.1)


def test_train_unbiased_listNet(get_mock_data):
    from ltr.train import train_unbiased_listNet
    from ltr.logging_policy import LoggingPolicy
    from argparse import Namespace
    from ltr.model import LTRModel
    import numpy as np
    from ltr.utils import seed
    

    data = get_mock_data
    logging_policy = LoggingPolicy()
    params = Namespace(epochs=1, 
                        lr=1e-4,
                        batch_size=1,
                        propensity=logging_policy.propensity,
                        metrics={"ndcg"})

    seed(42)
    unbiased_net = LTRModel(15, width=20)
    results = train_unbiased_listNet(unbiased_net, params, data)
    assert np.allclose(results['metrics_val'][-1]['ndcg'][0], 0.43, atol=0.1)


def test_logit_to_prob():
    import torch

    from ltr.train import logit_to_prob

    assert torch.allclose(logit_to_prob(torch.tensor([-100, 0, 100])), torch.tensor([0,0.5,1]))

def test_PropLTRModel():
    from ltr.model import PropLTRModel
    from ltr.utils import seed

    seed(42)
    prop_net = PropLTRModel(5, width=4)
    assert prop_net(torch.arange(3)).shape == torch.Size((3, 1))


def test_train_DLA_listNet(get_mock_data):
    from ltr.train import train_DLA_listNet
    from ltr.logging_policy import LoggingPolicy
    from argparse import Namespace
    from ltr.model import PropLTRModel
    from ltr.model import LTRModel
    import numpy as np
    from ltr.utils import seed
    

    data = get_mock_data
    logging_policy = LoggingPolicy()
    params = Namespace(epochs=10, 
                        lr=1e-4,
                        batch_size=1,
                        prop_lr=1e-3,
                        prop_net=PropLTRModel(logging_policy.topk, width=256),
                        metrics={"ndcg"})


    seed(42)
    DLA_net = LTRModel(15, width=256)
    results = train_DLA_listNet(DLA_net, params, data)
    assert np.allclose(results['metrics_val'][-1]['ndcg'][0], 0.44, atol=0.1)