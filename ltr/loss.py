import torch
import torch.nn.functional as F

# TODO: Implement this! (20 points)
def listNet_loss(output, target):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param output: predictions from the model, shape [1, topk, 1]
    :param target: ground truth labels, shape [1, topk]
    :param eps: epsilon value, used for numerical stability
    :return: loss value, a torch.Tensor
    """
    eps = 1e-10  # epsilon value, use this for numerical stability: add it to probs before taking the logarithm!
    ### BEGIN SOLUTION
    output = output.squeeze(2)
    preds_smax = F.softmax(output.float(), dim=1)
    preds_smax = preds_smax + eps
    true_smax = F.softmax(target.float(), dim=1)
    preds_log = torch.log(preds_smax)
    return torch.mean(-torch.sum(true_smax * preds_log, dim=-1))

    # return torch.sum(true_smax * torch.log(true_smax / preds_smax), dim=-1)
    # return F.kl_div(preds_log, true_smax, reduction='batchmean')
    ### END SOLUTION

# TODO: Implement this! (10 points)
def unbiased_listNet_loss(output, target, propensity):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param output: predictions from the model, shape [1, topk, 1]
    :param target: ground truth labels, shape [1, topk]
    :param propensity: propensity, shape [1, topk] or [topk]
    :return: loss value, a torch.Tensor
    """
    eps = 1e-10 # epsilon value, use this for numerical stability: add it to probs before taking the logarithm!
    
    # The following helps the stability and lower variance
    stable_propensity = propensity.clip(0.01, 1)

    ### BEGIN SOLUTION
    output = output.squeeze(2)
    preds_smax = F.softmax(output.float(), dim=1)
    preds_smax = preds_smax + eps
    stable_propensity = torch.Tensor(stable_propensity).unsqueeze(0)
    stable_target = target / stable_propensity
    true_smax = F.softmax(stable_target.float(), dim=1)
    preds_log = torch.log(preds_smax)
    return torch.mean(-torch.sum(true_smax * preds_log, dim=-1))

    # return torch.sum(true_smax * torch.log(true_smax / preds_smax), dim=-1)
    # return F.kl_div(preds_log, true_smax, reduction='batchmean')
    ### END SOLUTION

