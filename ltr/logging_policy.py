import numpy as np
import torch
import os

class LoggingPolicy():
    def __init__(self, policy_path='./data/') -> None:
        policy = np.load(os.path.join(policy_path, 'logging_policy.npz'))
        self.dlr = policy['dlr']
        self.lv = policy['lv']
        self.positions = policy['positions']
        self.sorted_docids = policy['sorted_docids']

        self.topk = 20
        self.propensity = 1./torch.arange(1,self.topk+1, requires_grad=False)
        self.noise = 0.05

    def _query_rel_probs(self, qid):
        s_i, e_i = self.dlr[qid:qid+2]
        return self.lv[s_i:e_i]/4.

    def query_positions(self, qid) -> np.ndarray:
        s_i, e_i = self.dlr[qid:qid+2]
        return self.positions[s_i:e_i]
        
    def query_sorted_docids(self, qid) -> np.ndarray:
        s_i, e_i = self.dlr[qid:qid+2]
        return self.sorted_docids[s_i:e_i]
        
    def gather_clicks(self, qid):
        pos = self.query_positions(qid)
        rel_probs = self._query_rel_probs(qid)
        nrel_mask = (rel_probs <= 0.5) & (pos < self.topk)
        rel_mask = (rel_probs > 0.5) & (pos < self.topk)
        rel_probs[rel_mask] -= self.noise
        rel_probs[nrel_mask] += self.noise
        propensity = np.zeros(rel_probs.shape[0])
        propensity[:min(propensity.shape[0], self.propensity.shape[0])] = self.propensity[:min(propensity.shape[0], self.propensity.shape[0])]
        click_probs = propensity[pos] * rel_probs
        clicks = np.zeros(2)
        while clicks.sum() == 0:
            clicks = np.random.binomial(1, np.clip(click_probs, 0, 1))
        return clicks


