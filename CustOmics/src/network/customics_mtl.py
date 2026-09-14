"""
CustOMICSMultiTask
-------------------
Variante multitâche de CustOMICS :
  - encodeurs par omique + VAE central : PARTAGES entre tous les cancers
  - tête de survie : UNE tête MLP (style DeepSurv) PAR cancer (18), via
    MultiTaskSurvivalNet, routée par l'id de cancer de chaque échantillon.

Pas de classifieur, pas d'adversarial de domaine : uniquement le multitask
survie demandé. La perte de survie est calculée séparément par sous-groupe
(cancer) DANS chaque batch, puis moyennée sur les cancers actifs du batch,
avant d'être ajoutée à la perte de reconstruction du VAE central (phase 2).

Prédiction : hazard = tête_du_cancer(z) directement (pas de CoxPH ensuite).
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.loss.survival_loss import CoxLoss
from src.datasets.multi_omics_dataset import MultiOmicsDataset
from src.models.autoencoder import AutoEncoder
from src.encoders.encoder import Encoder
from src.decoders.decoder import Decoder
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.models.vae import VAE
from src.tools.utils import get_common_samples

from src.tasks.multitask_survival import MultiTaskSurvivalNet
import sys
sys.path.append('..')
from mtl_deepsurv_utils import TaskBalancedBatchSampler

MIN_SAMPLES_PER_TASK_IN_BATCH = 2  # en dessous, la perte de Cox du cancer est ignorée pour ce batch


class CustOMICSMultiTask(nn.Module):
    def __init__(self, source_params, central_params, surv_params, train_params,
                 device, cancer_types):
        super().__init__()
        self.n_source = len(source_params)
        self.device = device
        self.cancer_types = list(cancer_types)

        self.lt_encoders = [
            Encoder(input_dim=source_params[s]['input_dim'], hidden_dim=source_params[s]['hidden_dim'],
                    latent_dim=source_params[s]['latent_dim'], norm_layer=source_params[s]['norm'],
                    dropout=source_params[s]['dropout'])
            for s in source_params
        ]
        self.lt_decoders = [
            Decoder(latent_dim=source_params[s]['latent_dim'], hidden_dim=source_params[s]['hidden_dim'],
                    output_dim=source_params[s]['input_dim'], norm_layer=source_params[s]['norm'],
                    dropout=source_params[s]['dropout'])
            for s in source_params
        ]
        self.rep_dim = sum(source_params[s]['latent_dim'] for s in source_params)

        self.central_encoder = ProbabilisticEncoder(
            input_dim=self.rep_dim, hidden_dim=central_params['hidden_dim'],
            latent_dim=central_params['latent_dim'], norm_layer=central_params['norm'],
            dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(
            latent_dim=central_params['latent_dim'], hidden_dim=central_params['hidden_dim'],
            output_dim=self.rep_dim, norm_layer=central_params['norm'],
            dropout=central_params['dropout'])
        self.beta = central_params['beta']
        self.lambda_central = central_params.get('lambda_central', 1)

        self.survival_predictor = MultiTaskSurvivalNet(
            latent_dim=central_params['latent_dim'],
            cancer_types=self.cancer_types,
            surv_dims=surv_params['dims'],
            dropout=surv_params['dropout'],
            norm=surv_params['norm'],
            activation=surv_params['activation'],
            l2_reg=surv_params['l2_reg'],
            device=device,
        ).to(device)
        self.lambda_survival = surv_params['lambda']

        self.phase = 1
        self.switch_epoch = train_params['switch']
        self.lr = train_params['lr']

        self.autoencoders = [AutoEncoder(self.lt_encoders[i], self.lt_decoders[i], device)
                              for i in range(self.n_source)]
        self.central_layer = VAE(self.central_encoder, self.central_decoder, device)
        for ae in self.autoencoders:
            ae.to(device)
        self.central_layer.to(device)

        self.history = []
        self.final_epoch = None
        self.optimizer = self._get_optimizer(self.lr)

    def _get_optimizer(self, lr):
        params = []
        for ae in self.autoencoders:
            params += list(ae.parameters())
        params += list(self.central_layer.parameters())
        params += list(self.survival_predictor.parameters())
        return Adam(params, lr=lr)

    def update_optimizer(self, lr):
        self.optimizer = self._get_optimizer(lr)

    # ----- forward helpers ------------------------------------------------

    def get_per_source_representation(self, x):
        return [self.autoencoders[i](x[i])[1] for i in range(self.n_source)]

    def _switch_phase(self, epoch):
        self.phase = 1 if epoch < self.switch_epoch else 2

    def _compute_loss(self, x, loss_eval=False):
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = sum(ae.loss(src, self.beta) for src, ae in zip(x, self.autoencoders))
            return lt_rep, loss
        else:
            lt_rep = self.get_per_source_representation(x)
            loss = sum(ae.loss(src, self.beta) for src, ae in zip(x, self.autoencoders))
            central_concat = torch.cat(lt_rep, dim=1)
            beta = 0 if loss_eval else self.beta
            loss += self.lambda_central * self.central_layer.loss(central_concat, beta)
            mean, logvar = self.central_encoder(central_concat)
            return mean, loss

    def _multitask_survival_loss(self, z, os_time, os_event, task_ids):
        """Moyenne, sur les cancers presents dans le batch, de la CoxLoss
        (calculee separement par cancer avec sa propre tete)."""
        task_ids_list = [t.item() if torch.is_tensor(t) else t for t in task_ids]
        total_loss = 0.0
        n_active = 0
        for c in self.cancer_types:
            mask = torch.tensor([str(t) == str(c) for t in task_ids_list],
                                 dtype=torch.bool, device=z.device)
            n = int(mask.sum().item())
            if n < MIN_SAMPLES_PER_TASK_IN_BATCH:
                continue
            if os_event[mask].sum() == 0:
                continue  # pas d'evenement -> Cox loss non informative pour ce sous-groupe
            hazard_c = self.survival_predictor.heads[str(c)](z[mask])
            loss_c = CoxLoss(survtime=os_time[mask], censor=os_event[mask],
                              hazard_pred=hazard_c, device=self.device)
            total_loss = total_loss + loss_c
            n_active += 1
        if n_active == 0:
            return torch.tensor(0.0, device=z.device, requires_grad=True)
        return total_loss / n_active

    def _train_loop(self, x, os_time, os_event, task_ids):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)
        self.optimizer.zero_grad()
        if self.phase == 1:
            _, loss = self._compute_loss(x)
        else:
            z, loss = self._compute_loss(x)
            surv_loss = self._multitask_survival_loss(z, os_time, os_event, task_ids)
            loss = loss + self.lambda_survival * surv_loss
        return loss

    # ----- training ---------------------------------------------------

    def fit(self, omics_train, clinical_df, cancer_id_col, event, surv_time,
            omics_val=None, batch_size=32, n_epochs=30, verbose=False,
            patience=None, min_delta=1e-3, early_stopping_on="val",
            n_per_task_train=None, n_per_task_val=None, sampler_seed=0):
        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}
 
        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])
        dataset_train = MultiOmicsDataset(
            omics_df=omics_train, clinical_df=clinical_df, lt_samples=lt_samples_train,
            label=cancer_id_col, event=event, surv_time=surv_time, domain_col=cancer_id_col,
        )
        if n_per_task_train is not None:
            cancer_ids_train = clinical_df.loc[lt_samples_train, cancer_id_col].values
            batch_sampler_train = TaskBalancedBatchSampler(
                cancer_ids_train, n_per_task=n_per_task_train, seed=sampler_seed)
            train_loader = DataLoader(dataset_train, batch_sampler=batch_sampler_train, **kwargs)
        else:
            train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, **kwargs)
 
        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            dataset_val = MultiOmicsDataset(
                omics_df=omics_val, clinical_df=clinical_df, lt_samples=lt_samples_val,
                label=cancer_id_col, event=event, surv_time=surv_time, domain_col=cancer_id_col,
            )
            if n_per_task_val is not None:
                cancer_ids_val = clinical_df.loc[lt_samples_val, cancer_id_col].values
                batch_sampler_val = TaskBalancedBatchSampler(
                    cancer_ids_val, n_per_task=n_per_task_val, seed=sampler_seed)
                val_loader = DataLoader(dataset_val, batch_sampler=batch_sampler_val, **kwargs)
            else:
                val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)
 
        if patience is not None:
            if early_stopping_on == "val" and omics_val is None:
                raise ValueError("early_stopping_on='val' necessite omics_val.")
 
        best_loss = float('inf')
        best_weights = None
        epochs_without_improvement = 0
        self.history = []
        self.final_epoch = n_epochs
 
        for epoch in range(n_epochs):
            t0 = time.time()
            overall_loss = 0.0
            if patience is None:
                self._switch_phase(epoch)
 
            for batch_idx, (x, _labels, os_time, os_event, task_ids) in enumerate(train_loader):
                if x[0].shape[0] == 1:
                    break
                self.train_all()
                loss_train = self._train_loop(x, os_time, os_event, task_ids)
                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / (batch_idx + 1)
 
            average_loss_val = None
            if omics_val:
                overall_loss = 0.0
                for batch_idx, (x, _labels, os_time, os_event, task_ids) in enumerate(val_loader):
                    if x[0].shape[0] == 1:
                        break
                    self.eval_all()
                    with torch.no_grad():
                        loss_val = self._train_loop(x, os_time, os_event, task_ids)
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / (batch_idx + 1)
                self.history.append((average_loss_train, average_loss_val))
                if verbose:
                    print(f"\tEpoch {epoch+1} | train={average_loss_train:.4f} "
                          f"val={average_loss_val:.4f} | {time.time()-t0:.1f}s")
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print(f"\tEpoch {epoch+1} | train={average_loss_train:.4f}")
 
            if patience is not None:
                monitored = average_loss_train if early_stopping_on == "train" else average_loss_val
                if monitored < best_loss * (1 - min_delta):
                    best_loss = monitored
                    best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        if self.phase == 1:
                            if verbose:
                                print(f"\tEarly stop phase1 @ {epoch+1}, passage phase 2.")
                            self.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
                            self.switch_epoch = epoch
                            best_loss, best_weights, epochs_without_improvement = float('inf'), None, 0
                            self.phase = 2
                        else:
                            if verbose:
                                print(f"\tEarly stop phase2 @ {epoch+1}.")
                            self.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
                            self.final_epoch = epoch
                            break

    # ----- inference ----------------------------------------------------

    def get_latent_representation(self, omics_df):
        self.eval_all()
        self.phase = 2
        x = [torch.Tensor(omics_df[source].values).to(self.device) for source in omics_df.keys()]
        with torch.no_grad():
            z, _ = self._compute_loss(x)
        return z.cpu().detach().numpy()

    def predict_risk(self, omics_df, cancer_type):
        """Hazard predit directement par la tete du cancer donne (style DeepSurv, pas de CoxPH)."""
        z = self.get_latent_representation(omics_df)
        z = torch.Tensor(z).to(self.device)
        self.survival_predictor.eval()
        with torch.no_grad():
            hazard = self.survival_predictor.predict(z, cancer_type)
        return hazard.cpu().detach().numpy().ravel()

    # ----- misc -----------------------------------------------------------

    def train_all(self):
        for ae in self.autoencoders:
            ae.train()
        self.central_layer.train()
        self.survival_predictor.train()

    def eval_all(self):
        for ae in self.autoencoders:
            ae.eval()
        self.central_layer.eval()
        self.survival_predictor.eval()