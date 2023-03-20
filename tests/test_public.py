import collections
import os

import pytest

minimal_data_cls = collections.namedtuple("Data", ["train"])

@pytest.fixture
def mock_data():
    return minimal_data_cls("unused")


def test_ClickLTRData_api(mock_data):
    from ltr.dataset import ClickLTRData

    ClickLTRData(mock_data, "logging policy not used in init")


def test_LTRModel_api():
    from ltr.model import LTRModel

    LTRModel(10, width=20)


def test_loss_exists():
    from ltr.loss import listNet_loss
    from ltr.loss import unbiased_listNet_loss


def test_train_exists():
    from ltr.train import train_biased_listNet
    from ltr.train import train_unbiased_listNet
    from ltr.train import train_DLA_listNet


def test_train_biased_listNet_output_exists():
    for i in ["avg"] + list(range(10)):
        assert os.path.exists(f"./outputs/biased_listNet_{i}.json")


def test_train_unbiased_listNet_output_exists():
    for i in ["avg"] + list(range(10)):
        assert os.path.exists(f"./outputs/unbiased_listNet_{i}.json")


def test_logit_to_prob():
    import torch

    from ltr.train import logit_to_prob

    logit_to_prob(10 * torch.rand(3))


def test_PropLTRModel_api():
    from ltr.model import PropLTRModel

    PropLTRModel(3, width=4)


def test_train_DLA_listNet_output_exists():
    for i in ["avg"] + list(range(10)):
        assert os.path.exists(f"./outputs/DLA_listNet_{i}.json")
