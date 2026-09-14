# src/tasks/domain_classifier.py
import torch.nn as nn

class DomainClassifier(nn.Module):
    def __init__(self, latent_dim, n_domains, hidden_layers=[64], dropout=0.2):
        super().__init__()
        dims = [latent_dim] + hidden_layers
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_domains))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)