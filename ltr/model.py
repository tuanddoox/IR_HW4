from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


# TODO: Implement this! (0 points)
class LTRModel(nn.Module):
    def __init__(self, num_features, width):
        """
        Initialize LTR model
        Parameters
        ----------
        num_features: int
            number of features 
        """
        ### BEGIN SOLUTION
        super(LTRModel, self).__init__()
        self.num_features = num_features
        self.width = width

        self.layers = nn.Sequential(OrderedDict([('layer1', nn.Linear(self.num_features, self.width)),
                         ('relu1', nn.ReLU()),
                         ('out', nn.Linear(self.width, 1))]))
        ### END SOLUTION

    def forward(self, x):
        """
        Takes in an input feature matrix of size (1, N, NUM_FEATURES) and produces the output 
        Arguments
        ----------
            x: Tensor 
        Returns
        -------
            Tensor
        """
        ### BEGIN SOLUTION
        return self.layers(x)
        ### END SOLUTION

    

# TODO: Implement this! (3 points)
class PropLTRModel(LTRModel):
    def forward(self, p):
        """
        Takes in the position tensor (dtype:torch.long) of size (1, N), 
        transforms it into a one_hot embedding of size (1, N, layers[0].in_features) and produces the output
        Arguments
        ----------
            x: LongTensor 
        Returns
        -------
            FloatTensor
        """
        ### BEGIN SOLUTION
        p_one_hot = self.one_hot(p, num_classes=self.layers[0].in_features)
        out = p_one_hot.float()
        for layer in self.layers:
            out = layer(out)
        return out.squeeze()
        ### END SOLUTION
