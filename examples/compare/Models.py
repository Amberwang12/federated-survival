import argparse, json
import torch  # For building the networks
import torchtuples as tt  # Some useful functions
from pycox.models.cox_time import MLPVanillaCoxTime


class model(object):

    def __init__(self, conf=None):
        self.conf = conf

    def model_initial(self):
        # Initializing the model
        in_features = self.conf.n_features
        num_nodes = self.conf.num_nodes
        out_features = self.conf.out_features
        batch_norm = self.conf.batch_norm
        dropout = self.conf.dropout
        activation = self.conf.activation  # torch.nn.modules.activation

        if self.conf.model_type == 'CoxTime':
            net = MLPVanillaCoxTime(in_features, num_nodes, batch_norm, dropout,
                                    activation)
        else:
            net = tt.practical.MLPVanilla(in_features, num_nodes, out_features,
                                          batch_norm, dropout, activation,
                                          output_bias=False)
        return net
