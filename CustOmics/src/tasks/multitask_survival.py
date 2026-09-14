"""
MultiTaskSurvivalNet
---------------------
Une tête de survie (SurvivalNet / MLP style DeepSurv) indépendante par type de
cancer, toutes branchées sur la même représentation latente centrale z.

Usage :
    heads = MultiTaskSurvivalNet(
        latent_dim=32,
        cancer_types=["BRCA", "COAD", ...],   # 18 types
        surv_dims=[64, 32],
        dropout=0.2,
        device=device,
    )
    hazard = heads(z, task_ids)          # routage batché (entrainement)
    hazard_c = heads.predict(z, "COAD")  # une seule tête (evaluation par cancer)
"""
import torch
import torch.nn as nn

from src.tasks.survival import SurvivalNet


class MultiTaskSurvivalNet(nn.Module):
    def __init__(self, latent_dim, cancer_types, surv_dims, dropout,
                 norm=True, activation='SELU', l2_reg=1e-2, device='cpu'):
        super().__init__()
        self.cancer_types = list(cancer_types)
        self.device = device

        self.heads = nn.ModuleDict()
        for c in self.cancer_types:
            surv_param = {
                'drop': dropout,
                'norm': norm,
                'dims': [latent_dim] + list(surv_dims) + [1],
                'activation': activation,
                'l2_reg': l2_reg,
                'device': device,
            }
            self.heads[str(c)] = SurvivalNet(surv_param)

    def forward(self, z, task_ids):
        """
        z        : (B, latent_dim) tensor, représentation centrale partagée
        task_ids : liste/array/tensor de longueur B, un id de cancer (str ou int
                   convertible en str) par échantillon, aligné avec le batch.

        Retourne un tensor (B, 1) de hazards, dans le même ordre que le batch
        (chaque ligne est passée par la tête du cancer correspondant).
        """
        if torch.is_tensor(task_ids):
            task_ids_list = [t.item() if torch.is_tensor(t) else t for t in task_ids]
        else:
            task_ids_list = list(task_ids)

        hazard = torch.zeros(z.shape[0], 1, device=z.device)
        task_ids_arr = torch.as_tensor(
            [hash(str(t)) for t in task_ids_list]
        ) if False else None  # (non utilisé, gardé pour clarté du masquage ci-dessous)

        for c in self.cancer_types:
            mask = torch.tensor(
                [str(t) == str(c) for t in task_ids_list],
                dtype=torch.bool, device=z.device
            )
            if mask.sum() == 0:
                continue
            hazard[mask] = self.heads[str(c)](z[mask])
        return hazard

    def predict(self, z, cancer_type):
        """Prédiction directe via la tête d'un seul cancer."""
        return self.heads[str(cancer_type)](z)