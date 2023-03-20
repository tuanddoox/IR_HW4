import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .dataset import ClickLTRData, LTRData
from .logging_policy import LoggingPolicy
from .loss import listNet_loss, unbiased_listNet_loss
from .eval import evaluate_model


# TODO: Implement this! (2 points)
def logit_to_prob(logit):
    ### BEGIN SOLUTION
    pass
    ### END SOLUTION


# TODO: Implement this! (10 points)
def train_biased_listNet(net, params, data):
    """
    This function should train the given network using the (biased) listNet loss
    
    Note: Do not change the function definition! 
    Note: You can assume params.batch_size will always be equal to 1
    
    
    net: the neural network to be trained
    
    params: params is an object which contains config used in training 
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)
        
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set. 
             You can use this to debug your models
    """

    val_metrics_epoch = []
    assert params.batch_size == 1
    logging_policy = LoggingPolicy()
    ### BEGIN SOLUTION
    ### END SOLUTION
    return {"metrics_val": val_metrics_epoch}


# TODO: Implement this! (10 points)
def train_unbiased_listNet(net, params, data):
    """
    This function should train the given network using the unbiased_listNet loss
    
    Note: Do not change the function definition! 
    Note: You can assume params.batch_size will always be equal to 1
    Note: For this function, params should also have the propensity attribute
    
    
    net: the neural network to be trained
    
    params: params is an object which contains config used in training 
            params.epochs - the number of epochs to train.
            params.lr - learning rate for Adam optimizer.
            params.batch_size - batch size (always equal to 1)
            params.propensity - the propensity values used for IPS in unbiased_listNet
        
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set. 
             You can use this to debug your models
    """

    val_metrics_epoch = []
    assert params.batch_size == 1
    assert hasattr(params, 'propensity')
    logging_policy = LoggingPolicy()
    ### BEGIN SOLUTION
    ### END SOLUTION
    return {"metrics_val": val_metrics_epoch}



# TODO: Implement this! (20 points)
def train_DLA_listNet(net, params, data):
    """
    This function should simultanously train both of the given networks 
    (i.e. net: for relevance estimation, and params.prop_net: for propensity estimation) using the unbiased_listNet loss.
    
    Note: Do not change the function definition! 
    Note: You can assume params.batch_size will always be equal to 1
    
    
    net: the neural network to be trained
    
    params: params is an object which contains config used in training 
            params.epochs - the number of epochs to train.
            params.lr - learning rate for relevance parameters.
            params.batch_size - batch size (always equal to 1)
            params.prop_net - the NN used for propensity estimation
            params.prop_lr - learning rate for propensity parameters.
        
    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and 
             "estimated_propensities" (a list of normalized propensities). 
             
             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set. 
             You can use this to debug your models
    """

    val_metrics_epoch = []
    estimated_propensities_epoch = []
    assert params.batch_size == 1
    assert hasattr(params, 'prop_net')
    assert hasattr(params, 'prop_lr')
    logging_policy = LoggingPolicy()
    ### BEGIN SOLUTION
    ### END SOLUTION
    estimated_propensity = logit_to_prob(
        prop_net(torch.arange(logging_policy.topk)).squeeze().data).numpy()
    print('estimated propensities:', estimated_propensity / estimated_propensity[0])
    return {"metrics_val": val_metrics_epoch}

