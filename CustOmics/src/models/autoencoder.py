# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the Standard Autoencoder module.
"""
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """
    Standard autoencoder on a single-source input.

    Parameters
    ----------
    encoder : nn.Module
        Encoder of the model.
    decoder : nn.Module
        Decoder of the model.
    device : torch.device
        Device used for computation.
    binary : bool, default=False
        If True, the model is trained for binary data using
        Binary Cross Entropy with logits.
        If False, the original MSE loss is used.
    """

    def __init__(self, encoder, decoder, device, binary=False):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.binary = binary

        self._relocate()

    def _relocate(self):
        self.encoder.to(self.device)
        self.decoder.to(self.device)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def loss(self, x, beta=None, sample_mask=None, x_target=None):
        """
        x_target : tenseur optionnel de même forme que x, utilisé comme cible de
            reconstruction à la place de x. Sert au mode "reconstruct" du modality
            dropout : l'entrée x peut être masquée (zéroée), mais on veut quand
            même reconstruire la valeur originale non masquée. Si None,
            comportement inchangé (cible = x).
        """
        x_hat, z = self.forward(x)
        target = x_target if x_target is not None else x

        if self.binary:
            reconstruction_loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            reconstruction_loss = nn.MSELoss(reduction='none')

        per_element = reconstruction_loss(x_hat, target)   # [batch, n_features]
        per_sample = per_element.mean(dim=1)               # [batch]

        if sample_mask is not None:
            denom = sample_mask.sum().clamp(min=1.0)
            return (per_sample * sample_mask).sum() / denom
        else:
            return per_sample.mean()






