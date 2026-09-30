# -*- coding: utf-8 -*-
"""
Created on Wed 01 Sept 2021

@author: Hakim Benkirane

    CentraleSupelec
    MICS laboratory
    9 rue Juliot Curie, Gif-Sur-Yvette, 91190 France

Build the CustOMICS module.
"""
import os
import numpy as np

from src.loss.survival_loss import CoxLoss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from torch.optim import Adam, AdamW

import shap

from src.datasets.multi_omics_dataset import MultiOmicsDataset
from src.models.autoencoder import AutoEncoder
from src.encoders.encoder import Encoder
from src.decoders.decoder import Decoder
from src.encoders.probabilistic_encoder import ProbabilisticEncoder
from src.decoders.probabilistic_decoder import ProbabilisticDecoder
from src.tasks.classification import MultiClassifier
from src.tasks.survival import SurvivalNet
from src.models.vae import VAE
from src.loss.classification_loss import classification_loss
from src.loss.consensus_loss import consensus_loss
from src.metrics.classification import multi_classification_evaluation, plot_roc_multiclass
from src.metrics.survival import CIndex_lifeline, cox_log_rank
from src.tools.utils import save_plot_score
from src.tools.utils import get_common_samples
from src.ex_vae.shap_vae import processPhenotypeDataForSamples, randomTrainingSample, splitExprandSample, ModelWrapper, addToTensor
from lifelines import KaplanMeierFitter

from src.loss.grl import grad_reverse
from src.tasks.domain_classifier import DomainClassifier

import matplotlib.pyplot as plt
import time

plt.rcParams.update({'font.size': 22})


class CustOMICS(nn.Module):
    """
    The main CustOMICS object that represents the main network for dealing with multi-source integration and multi-task learning
    """
    def __init__(self, source_params, central_params, classif_params, surv_params, train_params, device, unsupervised, domain_params=None, linear_central_decoder=False,
        optimizer="adam",
        weight_decay=0.0,
    ):
        """
        Construct the whole architecture with intermediate autoencoders, central layer and eventually downstream predictors
        Parameters:
            source_params (dict)      -- parameters related to the different sources to integrate
            central_params (dict)     -- parameters of the central autoencoder
            classif_params (dict)     -- classifier parameters
            surv_params (dict)        -- parameters of the survival network
            train_params (dict)       -- training hyperparameters
            device (pytorch)          -- the device in which the computation is done
        """
        super(CustOMICS, self).__init__()
        self.n_source = len(list(source_params.keys()))
        self.source_names = list(source_params.keys())
        self.source_params = source_params
        self.device = device
        self.lt_encoders = [Encoder(input_dim=source_params[source]['input_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             latent_dim=source_params[source]['latent_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.lt_decoders = [Decoder(latent_dim=source_params[source]['latent_dim'], hidden_dim=source_params[source]['hidden_dim'],
                             output_dim=source_params[source]['input_dim'], norm_layer=source_params[source]['norm'], 
                             dropout=source_params[source]['dropout']) for source in source_params.keys()]
        self.rep_dim = sum([source_params[source]['latent_dim'] for source in source_params])
        self.central_encoder = ProbabilisticEncoder(input_dim=self.rep_dim, hidden_dim=central_params['hidden_dim'], 
                                                    latent_dim=central_params['latent_dim'], norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'])
        self.central_decoder = ProbabilisticDecoder(latent_dim=central_params['latent_dim'], hidden_dim=central_params['hidden_dim'], 
                                                    output_dim=self.rep_dim, norm_layer=central_params['norm'],
                                                    dropout=central_params['dropout'], activation = not linear_central_decoder)
        self.beta = central_params['beta']
        self.num_classes = classif_params['n_class']
        self.lambda_classif = classif_params['lambda']
        self.classifier =  MultiClassifier(n_class=self.num_classes, latent_dim=central_params['latent_dim'], dropout=classif_params['dropout'],
            class_dim = classif_params['hidden_layers']).to(self.device)
        self.lambda_survival = surv_params['lambda']
        surv_param = {'drop': surv_params['dropout'], 'norm': surv_params['norm'], 'dims': [central_params['latent_dim']] + surv_params['dims'] + [1], 
                    'activation': surv_params['activation'], 'l2_reg': surv_params['l2_reg'], 'device': self.device}
        self.survival_predictor = SurvivalNet(surv_param)
        self.phase = 1
        self.switch_epoch = train_params['switch']
        self.lr = train_params['lr']
        self.autoencoders = []
        self.central_layer = None
        self._set_autoencoders()
        self._set_central_layer()
        self._relocate()
        self.vae_history = []
        self.survival_history = []
        self.label_encoder = None
        self.one_hot_encoder = None
        self.lambda_central = central_params.get("lambda_central", 1)
        self.unsupervised = unsupervised
        self.modality_dropout_p = train_params.get('modality_dropout_p', 0.0) 
        self.modality_dropout_mode = train_params.get('modality_dropout_mode', 'exclude')
        if self.modality_dropout_mode not in ('exclude', 'reconstruct'):
            raise ValueError("modality_dropout_mode doit être 'exclude' ou 'reconstruct'.")

        self.use_domain_adv = domain_params is not None
        if self.use_domain_adv:
            self.n_domains = domain_params['n_domains']
            self.lambda_domain = domain_params['lambda']
            self.domain_classifier = DomainClassifier(
                latent_dim=central_params['latent_dim'],
                n_domains=self.n_domains,
                hidden_layers=domain_params['hidden_layers'],
                dropout=domain_params['dropout'],
            ).to(self.device)

        self.optimizer = self._get_optimizer(
            self.lr,
            optimizer=optimizer,
            weight_decay=weight_decay
        )
    
    def _get_optimizer(self, lr, optimizer="adam", weight_decay=0.0):
        """
        Initializes the optimizer.

        Parameters:
            lr (float): learning rate
            optimizer (str): "adam" or "adamw"
            weight_decay (float): weight decay coefficient
        """

        lt_params = []

        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())

        lt_params += list(self.central_layer.parameters())

        if not self.unsupervised:
            lt_params += list(self.survival_predictor.parameters())
            lt_params += list(self.classifier.parameters())

        if self.use_domain_adv:
            lt_params += list(self.domain_classifier.parameters())

        if optimizer.lower() == "adam":
            optimizer = Adam(
                lt_params,
                lr=lr,
                weight_decay=weight_decay
            )

        elif optimizer.lower() == "adamw":
            optimizer = AdamW(
                lt_params,
                lr=lr,
                weight_decay=weight_decay
            )

        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer}. "
                "Choose 'adam' or 'adamw'."
            )

        return optimizer
    def _set_autoencoders(self):
        """
        Initializes the autoencoders.
        Each source can optionally be binary.
        """
        for i, source in enumerate(self.source_names):

            binary = self.source_params[source].get("binary", False)

            self.autoencoders.append(
                AutoEncoder(
                    self.lt_encoders[i],
                    self.lt_decoders[i],
                    self.device,
                    binary=binary
                )
            )

    def _set_central_layer(self):
        """
        Initializes the central variational autoencoder
        """
        self.central_layer = VAE(self.central_encoder, self.central_decoder, self.device)

    def _relocate(self):
        """
        Relocates the network to specified device
        """
        for i in range(self.n_source):
            self.autoencoders[i].to(self.device)
        self.central_layer.to(self.device)

    def _switch_phase(self, epoch):
        """
        Switches phases during training
        Parameters:
            epoch (int)      -- epoch starting which to switch phases
        """
        if epoch < self.switch_epoch:
            self.phase = 1
        else:
            self.phase = 2

    def _compute_baseline(self, clinical_df, lt_samples, event, surv_time):
        kmf = KaplanMeierFitter()
        kmf.fit(clinical_df.loc[lt_samples, surv_time], clinical_df.loc[lt_samples, event])
        return kmf.survival_function_


    def per_source_forward(self, x):
        lt_forward = []
        for i in range(self.n_source):
            lt_forward.append(self.autoencoders[i](x[i]))
        return lt_forward

    def get_per_source_representation(self, x):
        lt_rep = []
        for i in range(self.n_source):
            lt_rep.append(self.autoencoders[i](x[i])[1])
        return lt_rep

    def decode_per_source_representation(self, lt_rep):
        lt_hat = []
        for i in range(self.n_source):
            lt_hat.append(self.autoencoders[i].decode(lt_rep[i]))
        return lt_hat



    def forward(self, x):
        lt_forward = self.per_source_forward(x)
        lt_hat = [element[0] for element in lt_forward]
        lt_rep = [element[1] for element in lt_forward]
        central_concat = torch.cat(lt_rep, dim=1)
        mean, logvar = self.central_encoder(central_concat)
        return lt_hat, lt_rep ,mean 

    def _compute_loss(self, x, loss_eval = False, modality_mask = None, x_target = None):
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for i, (source, autoencoder) in enumerate(zip(x, self.autoencoders)):
                sample_mask = modality_mask[:, i] if modality_mask is not None else None
                target = x_target[i] if x_target is not None else None

                loss += autoencoder.loss(source, self.beta, sample_mask=sample_mask, x_target=target)
            return lt_rep, loss
        elif self.phase == 2:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for i, (source, autoencoder) in enumerate(zip(x, self.autoencoders)):
                sample_mask = modality_mask[:, i] if modality_mask is not None else None
                target = x_target[i] if x_target is not None else None
                loss += autoencoder.loss(source, self.beta, sample_mask=sample_mask, x_target=target)

            central_concat = torch.cat(lt_rep, dim=1)
            beta = 0 if loss_eval else self.beta
            loss += self.lambda_central*self.central_layer.loss(central_concat, beta)
            mean, logvar = self.central_encoder(central_concat)
            z = mean
            return z, loss

    def _train_loop(self, x, labels, os_time, os_event, task, domain_labels,
                 observed_mask=None, apply_modality_dropout=False):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)

        # masque "vraie donnée manquante" (persistant, vient du dataset)
        modality_mask = observed_mask.to(self.device) if observed_mask is not None else None

        # dropout aléatoire additionnel
        x_target = None
        if apply_modality_dropout and self.phase == 2:
            x_clean = x
            x, dropout_mask = self._apply_modality_dropout_input(x)
            if self.modality_dropout_mode == "reconstruct":
                # l'entrée reste masquée, mais on reconstruit la valeur d'origine
                # et on n'exclut plus ces échantillons de la loss
                x_target = x_clean
            else:
                # comportement historique : échantillons droppés exclus de la loss
                modality_mask = dropout_mask if modality_mask is None else modality_mask * dropout_mask

        loss = 0
        self.optimizer.zero_grad()
        if self.phase == 1:
            lt_rep, loss = self._compute_loss(x, modality_mask=modality_mask, x_target=x_target) 
        elif self.phase == 2:
            z, loss = self._compute_loss(x, modality_mask=modality_mask, x_target=x_target)
            if task == 'survival' and not self.unsupervised:
                hazard_pred = self.survival_predictor(z)
                survival_loss = CoxLoss(survtime=os_time, censor=os_event, hazard_pred=hazard_pred, device=self.device)
                loss += self.lambda_survival * survival_loss
            elif task == 'classification':
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels)
                loss += self.lambda_classif * classification


            if self.use_domain_adv:
                z_reversed = grad_reverse(z, self.lambda_domain)
                domain_pred = self.domain_classifier(z_reversed)
                domain_loss = nn.functional.cross_entropy(domain_pred, domain_labels)
                loss += domain_loss
        return loss

    def fit(self, omics_train, clinical_df, label, event, surv_time, task, omics_val=None,
        batch_size=32, n_epochs=30, verbose=False, patience=None, min_delta=1e-3,
        early_stopping_on="val", track_loss_components=False,
        modality_mask_train=None, modality_mask_val=None, missing_strategy="impute",
        latent_pca_every=None, latent_pca_space="central", latent_pca_idx=0,
        latent_pca_dir="figures/latent_pca", latent_pca_lim=None):
        """
        latent_pca_every (int|None) : si renseigné, toutes les `latent_pca_every` epochs, trace pour
            chaque modalité une PCA de l'espace latent où l'individu `latent_pca_idx` du train a
            cette modalité mise à 0 en entrée (cf. plot_latent_pca_zeroed).
        latent_pca_space (str)      : "central" (moyenne du VAE central) ou "source" (latent de l'AE de la modalité).
        latent_pca_idx (int)        : indice (dans le train) de l'individu dont on annule la modalité.
        latent_pca_dir (str)        : dossier de sauvegarde des figures.
        latent_pca_lim (float|None) : si renseigné, axes fixés à [-lim, lim] en x et en y pour toutes les
            figures ; sinon limite symétrique recalculée à chaque figure. Échelle x/y toujours identique.
        """
        
        if missing_strategy not in ("impute", "drop"):
            raise ValueError("missing_strategy doit être 'impute' ou 'drop'.")
        if latent_pca_space not in ("central", "source"):
            raise ValueError("latent_pca_space doit être 'central' ou 'source'.")

        encoded_clinical_df = clinical_df.copy()
        self.label_encoder = LabelEncoder().fit(encoded_clinical_df.loc[:, label].values)
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)
        self.one_hot_encoder = OneHotEncoder(sparse_output=False).fit(encoded_clinical_df.loc[:, label].values.reshape(-1,1))

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_train.values()] + [clinical_df])

        if missing_strategy == "drop":
            if modality_mask_train is None:
                raise ValueError("missing_strategy='drop' nécessite modality_mask_train.")
            complete_mask = modality_mask_train.loc[lt_samples_train].all(axis=1)
            n_before = len(lt_samples_train)
            lt_samples_train = [s for s in lt_samples_train if complete_mask.loc[s]]
            if verbose:
                print(f"\t[missing_strategy=drop] {n_before - len(lt_samples_train)} patients retirés du train "
                    f"({len(lt_samples_train)} restants)")
            modality_mask_train = None  # plus besoin, tous les patients restants sont complets


        self.baseline = self._compute_baseline(clinical_df, lt_samples_train, event, surv_time)
        dataset_train = MultiOmicsDataset(
            omics_df=omics_train, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
            label=label, event=event, surv_time=surv_time,
            domain_col="domain_label" if self.use_domain_adv else None,
            modality_mask_df=modality_mask_train,
        )
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, **kwargs)

        # Données complètes du train (un seul batch, même prétraitement que l'entraînement)
        # pour le suivi PCA de l'espace latent
        x_pca = None
        if latent_pca_every:
            x_pca = next(iter(DataLoader(dataset_train, batch_size=len(dataset_train), shuffle=False)))[0]
            if verbose:
                print(f"\t[latent PCA] individu suivi : {lt_samples_train[latent_pca_idx]} "
                      f"(indice {latent_pca_idx}), tous les {latent_pca_every} epochs -> {latent_pca_dir}")

        if omics_val:
            lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])
            if missing_strategy == "drop":
                if modality_mask_val is None:
                    raise ValueError("missing_strategy='drop' nécessite modality_mask_val quand omics_val est fourni.")
                complete_mask_val = modality_mask_val.loc[lt_samples_val].all(axis=1)
                n_before_val = len(lt_samples_val)
                lt_samples_val = [s for s in lt_samples_val if complete_mask_val.loc[s]]
                if verbose:
                    print(f"\t[missing_strategy=drop] {n_before_val - len(lt_samples_val)} patients retirés du val "
                        f"({len(lt_samples_val)} restants)")
                modality_mask_val = None

            dataset_val = MultiOmicsDataset(
                omics_df=omics_val, clinical_df=encoded_clinical_df, lt_samples=lt_samples_val,
                label=label, event=event, surv_time=surv_time,
                domain_col="domain_label" if self.use_domain_adv else None,
                modality_mask_df=modality_mask_val,
            )
            val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, **kwargs)
            
        # Validation du paramètre early_stopping_on
        if patience is not None:
            if early_stopping_on == "val" and omics_val is None:
                raise ValueError("early_stopping_on='val' nécessite de fournir omics_val.")
            if early_stopping_on not in ("val", "train"):
                raise ValueError("early_stopping_on doit être 'val' ou 'train'.")

        # Early stopping state
        best_loss = float('inf')
        best_weights = None
        epochs_without_improvement = 0

        self.history = []
        self.final_epoch = n_epochs
        if track_loss_components:
            self.detailed_history = {'train': [], 'val': []}

        for epoch in range(n_epochs):
            start_time_epoch = time.time()
            overall_loss = 0
            if patience is None:
                self._switch_phase(epoch)

            epoch_components_train = {}
            for batch_idx, (x, labels, os_time, os_event, domain_labels, observed_mask) in enumerate(train_loader):
                if(x[0].shape[0] == 1):
                    break
                self.train_all()

                apply_modality_dropout = self.modality_dropout_p != 0
                if track_loss_components:
                    loss_train, comp_train = self._train_loop_with_components(
                        x, labels, os_time, os_event, task, domain_labels, observed_mask, apply_modality_dropout)
                    for k, v in comp_train.items():
                        epoch_components_train.setdefault(k, []).append(v)
                else:
                    loss_train = self._train_loop(x, labels, os_time, os_event, task, domain_labels, observed_mask, apply_modality_dropout)

                overall_loss += loss_train.item()
                loss_train.backward()
                self.optimizer.step()
            average_loss_train = overall_loss / ((batch_idx+1)*batch_size)
            overall_loss = 0
            end_time_epoch = time.time()

            if track_loss_components:
                self.detailed_history['train'].append(
                    {k: np.mean(v) for k, v in epoch_components_train.items()}
                )

            if omics_val != None:
                for batch_idx, (x, labels, os_time, os_event, domain_labels, observed_mask) in enumerate(val_loader):
                    if(x[0].shape[0] == 1):
                        break
                    self.eval_all()
                    loss_val = self._train_loop(x, labels, os_time, os_event, task, domain_labels, observed_mask)
                    overall_loss += loss_val.item()
                average_loss_val = overall_loss / ((batch_idx+1)*batch_size)
                self.history.append((average_loss_train, average_loss_val))
                
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train, "\tAverage Loss Val : ", average_loss_val, "\t Time :", end_time_epoch-start_time_epoch)

            # Suivi PCA de l'espace latent (avant l'early stopping pour ne pas être sauté par le break)
            if latent_pca_every and (epoch + 1) % latent_pca_every == 0:
                self.plot_latent_pca_zeroed(x_pca, epoch + 1, space=latent_pca_space,
                                            idx=latent_pca_idx, save_dir=latent_pca_dir,
                                            axis_lim=latent_pca_lim)
                
            if patience is not None:
                monitored_loss = average_loss_train if early_stopping_on == "train" else average_loss_val

                if monitored_loss < best_loss * (1 - min_delta):
                    best_loss = monitored_loss
                    best_weights = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if verbose:
                        print(f"\t  >> Pas d'amélioration sur {early_stopping_on}_loss ({epochs_without_improvement}/{patience})")
                    if epochs_without_improvement >= patience:
                        if self.phase == 1 :
                            if verbose:
                                print(f"\tEarly stopping à l'epoch {epoch + 1}. Restauration des meilleurs poids et passage a la phase 2.")
                            self.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
                            self.switch_epoch = epoch
                            best_loss = float('inf')
                            best_weights = None
                            epochs_without_improvement = 0
                            self.phase = 2
                        else :
                            if verbose:
                                print(f"\tEarly stopping à l'epoch {epoch + 1}. Restauration des meilleurs poids.")
                            self.load_state_dict({k: v.to(self.device) for k, v in best_weights.items()})
                            self.phase = 2
                            self.final_epoch = epoch
                            break
                        
            else:
                self.history.append(average_loss_train)
                if verbose:
                    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss Train : ", average_loss_train)

    def get_switch_epoch(self):
        return self.switch_epoch

    def get_final_epoch(self):
        return self.final_epoch

    # def get_loss_eval(self, omics_val):
    #     x_val = [
    #         torch.Tensor(omics_val[src].values).to(self.device)
    #         for src in omics_val.keys()
    #     ]
    #     self.eval_all()
    #     self.phase = 2
    #     z, loss = self._compute_loss(x_val, loss_eval = True)
    #     if self.use_domain_adv:
    #         z_reversed = grad_reverse(z, self.lambda_domain)
    #         domain_pred = self.domain_classifier(z_reversed)
    #         domain_loss = nn.functional.cross_entropy(domain_pred, domain_labels)
    #         loss += domain_loss 
    #     return loss

    def get_loss_eval(self, omics_val, clinical_df, label, event, surv_time, modality_mask_val=None):
        lt_samples_val = get_common_samples([df for df in omics_val.values()] + [clinical_df])

        encoded_clinical_df = clinical_df.copy()
        if label:
            encoded_clinical_df.loc[:, label] = self.label_encoder.transform(
                encoded_clinical_df.loc[:, label].values
            )

        dataset_val = MultiOmicsDataset(
            omics_df=omics_val,
            clinical_df=encoded_clinical_df,
            lt_samples=lt_samples_val,
            label=label,
            event=event,
            surv_time=surv_time,
            domain_col="domain_label" if self.use_domain_adv else None,
            modality_mask_df=modality_mask_val,
        )
        val_loader = DataLoader(dataset_val, batch_size=len(lt_samples_val), shuffle=False)

        self.eval_all()
        self.phase = 2

        x, labels, os_time, os_event, domain_labels, observed_mask = next(iter(val_loader))
        for i in range(len(x)):
            x[i] = x[i].to(self.device)

        modality_mask = observed_mask.to(self.device) if observed_mask is not None else None

        with torch.no_grad():
            z, loss = self._compute_loss(x, loss_eval=True, modality_mask=modality_mask)

            if self.use_domain_adv:
                domain_labels = domain_labels.to(self.device)
                z_reversed = grad_reverse(z, self.lambda_domain)
                domain_pred = self.domain_classifier(z_reversed)
                domain_loss = nn.functional.cross_entropy(domain_pred, domain_labels)
                loss += domain_loss

        return loss

    def get_latent_representation(self, omics_df, tensor=False):
        self.eval_all()
        self.phase= 2
        if tensor == False:
            x = [torch.Tensor(omics_df[source].values) for source in omics_df.keys()]
        else:
            x = [omics for omics in omics_df]
        with torch.no_grad():
            for i in range(len(x)):
                x[i] = x[i].to(self.device)
            z, loss = self._compute_loss(x)
        return z.cpu().detach().numpy()

    def reconstruct(self, x):
        x = torch.Tensor(x)
        z = self.autoencoders[0](x)[1]
        return self.autoencoders[0].decode(z).cpu().detach().numpy()


    def plot_representation(self, omics_df, clinical_df, labels, filename, title, show=True):
        labels_df = clinical_df.loc[:, labels]
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df=omics_df)
        save_plot_score(filename, z, labels_df[lt_samples].values, title, show=True)


    # ============================================================
    # PCA de l'espace latent avec une modalité mise à 0 pour un individu
    # ============================================================
    def plot_latent_pca_zeroed(self, x_ref, epoch, space="central", idx=0,
                               save_dir="figures/latent_pca", show=False, axis_lim=None):
        """
        Pour chaque modalité m : on met à 0 l'entrée de la modalité m pour l'individu `idx`
        (les autres individus et les autres modalités de idx sont inchangés), on encode,
        puis on projette par PCA. La PCA est ajustée sur la population non modifiée
        (individu idx exclu), puis appliquée à l'individu idx original et modifié.

        Parameters:
            x_ref (list[Tensor]) -- un tenseur par modalité, tous les individus du train
            epoch (int)          -- numéro d'epoch (pour le titre et le nom de fichier)
            space (str)          -- "central" : moyenne du VAE central ;
                                    "source"  : latent de l'autoencodeur de la modalité m
            idx (int)            -- indice de l'individu dont on annule la modalité
            save_dir (str)       -- dossier de sauvegarde
            show (bool)          -- afficher la figure
            axis_lim (float|None)-- si renseigné, axes fixés à [-axis_lim, axis_lim] ; sinon limite
                                    symétrique automatique. Dans les deux cas, même échelle en x et y.
        """
        os.makedirs(save_dir, exist_ok=True)
        was_training = self.central_layer.training
        self.eval_all()

        fig, axes = plt.subplots(1, self.n_source, figsize=(6 * self.n_source, 5), squeeze=False)
        with torch.no_grad():
            x_ref = [xi.to(self.device) for xi in x_ref]
            _, lt_rep_ref, z_ref = self.forward(x_ref)

            for m, name in enumerate(self.source_names):
                x_mod = [xi.clone() for xi in x_ref]
                x_mod[m][idx] = 0.0
                _, lt_rep_mod, z_mod = self.forward(x_mod)

                if space == "central":
                    lat_ref, lat_mod = z_ref, z_mod
                else:
                    lat_ref, lat_mod = lt_rep_ref[m], lt_rep_mod[m]
                lat_ref = lat_ref.cpu().numpy()
                lat_mod = lat_mod.cpu().numpy()

                others = np.delete(np.arange(lat_ref.shape[0]), idx)
                pca = PCA(n_components=2).fit(lat_ref[others])
                p_others = pca.transform(lat_ref[others])
                p_orig = pca.transform(lat_ref[idx:idx + 1])
                p_zero = pca.transform(lat_mod[idx:idx + 1])

                ax = axes[0, m]
                ax.scatter(p_others[:, 0], p_others[:, 1], s=8, c="lightgrey", label="autres individus")
                ax.scatter(p_orig[:, 0], p_orig[:, 1], s=120, facecolors="none",
                           edgecolors="tab:blue", linewidths=2, label=f"ind. {idx} (original)")
                ax.scatter(p_zero[:, 0], p_zero[:, 1], s=120, c="tab:red", marker="X",
                           label=f"ind. {idx} ({name} = 0)")

                # Même échelle en x et y, limites symétriques (la PCA centre la population en 0)
                if axis_lim is None:
                    pts = np.vstack([p_others, p_orig, p_zero])
                    lim = 1.05 * np.abs(pts).max()
                else:
                    lim = axis_lim
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_aspect("equal", adjustable="box")

                ev = pca.explained_variance_ratio_
                ax.set_title(f"{name} mis à 0 — latent {space}", fontsize=12)
                ax.set_xlabel(f"PC1 ({ev[0]:.1%})", fontsize=10)
                ax.set_ylabel(f"PC2 ({ev[1]:.1%})", fontsize=10)
                ax.tick_params(labelsize=9)
                ax.legend(fontsize=9)

        fig.suptitle(f"Epoch {epoch} — phase {self.phase}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"latent_pca_{space}_epoch{epoch:04d}.png"), bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        if was_training:
            self.train_all()


    def source_predict(self, expr_df, source):
        #tensor_expr = torch.Tensor(expr_df.values)
        tensor_expr = expr_df
        if source == 'CNV' or source == 'protein':
            z = self.lt_encoders[0](tensor_expr)
        elif source == 'RNAseq' or source == 'gene_exp':
            z = self.lt_encoders[1](tensor_expr)
        elif source == 'methyl':
            z = self.lt_encoders[2](tensor_expr)
        y_pred_proba = self.classifier(z)
        return y_pred_proba

    def predict_risk(self, omics_df):
        z = self.get_latent_representation(omics_df)
        return self.survival_predictor(torch.Tensor(z))

    def predict_survival(self, omics_df, t=None):
        lt_samples = get_common_samples([df for df in omics_df.values()])
        dt_surv = {}
        risk_score = self.predict_risk(omics_df).cpu().detach().numpy()
        for sample, risk in zip(lt_samples, risk_score):
            dt_surv[sample] = self.baseline*np.exp(risk[0])
        return dt_surv


    def evaluate(self, omics_test, clinical_df, label, event, surv_time, task, batch_size=32,
             plot_roc=False, modality_mask_test=None):

        encoded_clinical_df = clinical_df.copy()
        encoded_clinical_df.loc[:, label] = self.label_encoder.transform(encoded_clinical_df.loc[:, label].values)

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.device.type == "cuda" else {}

        lt_samples_train = get_common_samples([df for df in omics_test.values()] + [clinical_df])
        dataset_test = MultiOmicsDataset(
            omics_df=omics_test, clinical_df=encoded_clinical_df, lt_samples=lt_samples_train,
            label=label, event=event, surv_time=surv_time,
            domain_col="domain_label" if self.use_domain_adv else None,
            modality_mask_df=modality_mask_test,
        )
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, **kwargs)

        self.eval_all()
        classif_metrics = []
        c_index = []
        with torch.no_grad():
            for batch_idx, (x, labels, os_time, os_event, domain_labels, observed_mask) in enumerate(test_loader):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device)
                modality_mask = observed_mask.to(self.device) if observed_mask is not None else None
                z, loss = self._compute_loss(x, modality_mask=modality_mask)
                if task == 'survival':
                    predicted_survival_hazard = self.survival_predictor(z)
                    predicted_survival_hazard = predicted_survival_hazard.cpu().detach().numpy().reshape(-1, 1)
                    os_time = os_time.cpu().detach().numpy()
                    os_event = os_event.cpu().detach().numpy()
                    c_index.append(CIndex_lifeline(predicted_survival_hazard, os_event, os_time))
                    return np.mean(c_index)
                elif task == 'classification':
                    y_pred_proba = self.classifier(z)
                    y_pred = torch.argmax(y_pred_proba, dim=1).cpu().detach().numpy()
                    y_pred_proba = y_pred_proba.cpu().detach().numpy()
                    y_true = labels.cpu().detach().numpy()
                    classif_metrics.append(multi_classification_evaluation(y_true, y_pred, y_pred_proba, ohe=self.one_hot_encoder))
                    if plot_roc:
                        plot_roc_multiclass(y_test=y_true, y_pred_proba=y_pred_proba, filename='test', n_classes=self.num_classes,
                                            var_names=np.unique(clinical_df.loc[:, label].values.tolist()))
                    return classif_metrics



    def stratify(self, omics_df, clinical_df, event, surv_time, treshold=0.5, 
                    save_plot=False, plot_title="", filename=''):
        lt_samples = get_common_samples([df for df in omics_df.values()] + [clinical_df])
        z = self.get_latent_representation(omics_df)
        hazard_pred = self.survival_predictor(torch.Tensor(z)).cpu().detach().numpy()
        dt_strat = {'high': [], 'low': []}
        for i in range(len(lt_samples)):
            if hazard_pred[i] <= np.mean(hazard_pred):
                dt_strat['low'].append(lt_samples[i])
            else:
                dt_strat['high'].append(lt_samples[i])
        kmf_low = KaplanMeierFitter(label='low risk')
        kmf_high = KaplanMeierFitter(label='high risk')
        kmf_low.fit(clinical_df.loc[dt_strat['low'], surv_time], clinical_df.loc[dt_strat['low'], event])
        kmf_high.fit(clinical_df.loc[dt_strat['high'], surv_time], clinical_df.loc[dt_strat['high'], event])
        p_value = cox_log_rank(hazard_pred.reshape(1,-1)[0], np.array(clinical_df.loc[lt_samples, event].values, dtype=float), np.array(clinical_df.loc[lt_samples, surv_time].values, dtype=float))

        kmf_low.plot()
        kmf_high.plot()
        plt.title(plot_title + " (p-value = {:.3g})".format(p_value))
        plt.xlim((0,2500))
        if save_plot:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()


    def explain(self, sample_id, omics_df, clinical_df, source, subtype,label='PAM50', device='cpu', show=False):
        """
        :param sample_id: List of samplesid to consider for explanation.
        :param vae_model: The CustOmics model to explain, the output should be a 1xn_class tensor.
        :param expr_df: DataFrame of data to explain, the input has to match the source selected for the explanation.
        :param clinical_df: DataFrame containing the clinical data.
        :param source: Omic source to explain.
        :param subtype: Subtype that needs explanation, shap values will be computed for this subtypes against the others.
        :param le: LabelEncoder to revert back from numerical encoding to original names.
        :return:None, plots of the top 10 genes and their shap values
        """
        #This class has combined the different analysis' of the Deep SHAP values we conducted.
        #SHAP reference: Lundberg et al., 2017: http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

        expr_df = omics_df[source]
        sample_id = list(set(sample_id).intersection(set(expr_df.index)))
        phenotype = processPhenotypeDataForSamples(clinical_df, sample_id, self.label_encoder)

        conditionaltumour=phenotype.loc[:, label] == subtype

        
        expr_df = expr_df.loc[sample_id,:]
        normal_expr = randomTrainingSample(expr_df, 10)
        tumour_expr = splitExprandSample(condition=conditionaltumour, sampleSize=10, expr=expr_df)
        # put on device as correct datatype
        background = addToTensor(expr_selection=normal_expr, device=device)
        male_expr_tensor = addToTensor(expr_selection=tumour_expr, device=device)


        e = shap.DeepExplainer(ModelWrapper(self, source=source), background)
        shap_values_female = e.shap_values(male_expr_tensor, ranked_outputs=None)

        shap.summary_plot(shap_values_female[0],features=tumour_expr,feature_names=list(tumour_expr.columns), show=False, plot_type="violin", max_display=10, plot_size=[4,6])
        plt.savefig('shap_{}_{}.png'.format(source, subtype), bbox_inches='tight')
        if show:
            plt.show()
        plt.clf()

    def plot_loss(self):
        n_epochs = len(self.history)
        plt.title('Evolution of the loss function with respect to the epochs')
        plt.vlines(x=self.switch_epoch, ymin=0.1, ymax=0.7, colors='purple', ls='--', lw=2, label='phase 2 switch')
        plt.plot(range(0, n_epochs), [loss[0] for loss in self.history], label = 'train loss')
        plt.plot(range(0, n_epochs), [loss[1] for loss in self.history], label = 'val loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.show()
    
    def save_figure_loss(self, step):
        n_epochs = len(self.history)
        plt.title('Evolution of the loss function with respect to the epochs')
        plt.vlines(x=self.switch_epoch, ymin=0, ymax=2.5, colors='purple', ls='--', lw=2, label='phase 2 switch')
        if len(self.history)==2:
            plt.plot(range(0, n_epochs), [loss[0] for loss in self.history], label = 'train loss')
            plt.plot(range(0, n_epochs), [loss[1] for loss in self.history], label = 'val loss')
        else:
            plt.plot(range(0, n_epochs), [loss for loss in self.history], label = 'train loss')

        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f"figures/fig_{step}.png")

    def print_parameters(self):
        lt_params = []
        lt_names = []
        for autoencoder in self.autoencoders:
            for name, param in autoencoder.named_parameters():
                lt_params.append(param.data)
                lt_names.append(name)
        for name, param in self.central_layer.named_parameters():
            lt_params.append(param.data)
            lt_names.append(name)
        print(len(lt_params))

    def get_number_parameters(self):
        sum_params = 0
        for autoencoder in self.autoencoders:
            sum_params += sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        sum_params += sum(p.numel() for p in self.central_layer.parameters() if p.requires_grad)
        return sum_params


    def train_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.train()
            decoder.train()
        for autoencoder in self.autoencoders:
            autoencoder.train()
        self.central_layer.train()
        if self.survival_predictor:
            self.survival_predictor.train()
        if self.classifier:
            self.classifier.train()

    def eval_all(self):
        for encoder, decoder in zip(self.lt_encoders, self.lt_decoders):
            encoder.eval()
            decoder.eval()
        for autoencoder in self.autoencoders:
            autoencoder.eval()
        self.central_layer.eval()
        if self.survival_predictor:
            self.survival_predictor.eval()
        if self.classifier:
            self.classifier.eval()

    ### Transfer learning methods =======================================================================================
    
    def freeze_autoencoders(self):
        for ae in self.autoencoders:
            for p in ae.parameters():
                p.requires_grad = False

    def unfreeze_autoencoders(self):
        for ae in self.autoencoders:
            for p in ae.parameters():
                p.requires_grad = True

    def freeze_central_vae(self):
        for p in self.central_layer.parameters():
            p.requires_grad = False

    def unfreeze_central_vae(self):
        for p in self.central_layer.parameters():
            p.requires_grad = True

    def update_optimizer(self, lr):
        """
        update the optimizer
        Parameters:
            lr (float)      -- learning rate for the CustOmics network
        """
        lt_params = []
        for autoencoder in self.autoencoders:
            lt_params += list(autoencoder.parameters())
        lt_params += list(self.central_layer.parameters())
        if not self.unsupervised:
            lt_params += list(self.survival_predictor.parameters())
            lt_params += list(self.classifier.parameters())       
        if self.use_domain_adv:
            lt_params += list(self.domain_classifier.parameters())     
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad,
                lt_params),
            lr=lr
        ) 
    

    # ============================================================
    # Modality Dropout
    # ============================================================
    def _apply_modality_dropout_input(self, x):
        if self.modality_dropout_p <= 0:
            return x, None

        batch_size = x[0].shape[0]
        n_sources = len(x)
        device = x[0].device

        keep_mask = (torch.rand(batch_size, n_sources, device=device) > self.modality_dropout_p).float()

        all_dropped = keep_mask.sum(dim=1) == 0
        if all_dropped.any():
            idx = torch.randint(0, n_sources, (int(all_dropped.sum().item()),), device=device)
            keep_mask[all_dropped, idx] = 1.0

        x_dropped = [x[s] * keep_mask[:, s].unsqueeze(1) for s in range(n_sources)]
        return x_dropped, keep_mask

    # ============================================================
    # Fonctions pour afficher chaque partie de la loss séparément
    # ============================================================
    
    def _compute_loss_with_components(self, x, loss_eval=False, modality_mask=None, x_target = None):
        """
        Équivalent de _compute_loss mais retourne en plus un dictionnaire
        contenant la valeur (float) de chaque terme de reconstruction.
        Les termes de supervision (survie / classification / domaine) sont
        ajoutés ensuite dans _train_loop_with_components, car ils nécessitent
        labels / os_time / os_event / domain_labels.
    
        Utilise self.source_names (liste des noms de modalités, dans le même
        ordre que self.autoencoders) pour nommer chaque composante au lieu
        d'un simple indice. Voir __init__ : self.source_names = list(source_params.keys())
        """
        components = {}
    
        if self.phase == 1:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            for i, (name, source, autoencoder) in enumerate(zip(self.source_names, x, self.autoencoders)):
                sample_mask = modality_mask[:, i] if modality_mask is not None else None
                target = x_target[i] if x_target is not None else None
                source_loss = autoencoder.loss(source, self.beta, sample_mask=sample_mask, x_target=target)
                components[f'recon_{name}'] = source_loss.item()
                loss += source_loss
            return lt_rep, loss, components
    
        elif self.phase == 2:
            lt_rep = self.get_per_source_representation(x)
            loss = 0
            
            for i, (name, source, autoencoder) in enumerate(zip(self.source_names, x, self.autoencoders)):
                sample_mask = modality_mask[:, i] if modality_mask is not None else None
                target = x_target[i] if x_target is not None else None
                source_loss = autoencoder.loss(source, self.beta, sample_mask=sample_mask, x_target=target)
                components[f'recon_{name}'] = source_loss.item()
                loss += source_loss
    
            central_concat = torch.cat(lt_rep, dim=1)
            beta = 0 if loss_eval else self.beta
            central_loss = self.central_layer.loss(central_concat, beta)
            components['recon_central'] = self.lambda_central * central_loss.item()
            loss += self.lambda_central * central_loss
    
            mean, logvar = self.central_encoder(central_concat)
            z = mean
            return z, loss, components
    
    

    
    def _train_loop_with_components(self, x, labels, os_time, os_event, task, domain_labels, 
            observed_mask=None, apply_modality_dropout=False):
        for i in range(len(x)):
            x[i] = x[i].to(self.device)

        modality_mask = observed_mask.to(self.device) if observed_mask is not None else None

        x_target = None
        if apply_modality_dropout and self.phase == 2:
            x_clean = x
            x, dropout_mask = self._apply_modality_dropout_input(x)
            if self.modality_dropout_mode == "reconstruct":
                # l'entrée reste masquée, mais on reconstruit la valeur d'origine
                # et on n'exclut plus ces échantillons de la loss
                x_target = x_clean
            else:
                # comportement historique : échantillons droppés exclus de la loss
                modality_mask = dropout_mask if modality_mask is None else modality_mask * dropout_mask
            
        self.optimizer.zero_grad()
        components = {}
    
        if self.phase == 1:
            lt_rep, loss, components = self._compute_loss_with_components(x, modality_mask=modality_mask, x_target=x_target)
    
        elif self.phase == 2:
            z, loss, components = self._compute_loss_with_components(x, modality_mask=modality_mask, x_target=x_target)
    
            if task == 'survival' and not self.unsupervised:
                hazard_pred = self.survival_predictor(z)
                survival_loss = CoxLoss(survtime=os_time, censor=os_event,
                                        hazard_pred=hazard_pred, device=self.device)
                components['survival_raw'] = survival_loss.item()
                components['survival_weighted'] = (self.lambda_survival * survival_loss).item()
                loss += self.lambda_survival * survival_loss
    
            elif task == 'classification':
                y_pred_proba = self.classifier(z)
                classification = classification_loss('CE', y_pred_proba, labels)
                components['classification_raw'] = classification.item()
                components['classification_weighted'] = (self.lambda_classif * classification).item()
                loss += self.lambda_classif * classification
    
            if self.use_domain_adv:
                z_reversed = grad_reverse(z, self.lambda_domain)
                domain_pred = self.domain_classifier(z_reversed)
                domain_loss = torch.nn.functional.cross_entropy(domain_pred, domain_labels)
                components['domain'] = domain_loss.item()
                loss += domain_loss
    
        return loss, components
    

    

    
    def plot_loss_detailed(self, show=True, save_path=None, log_scale=False):
        """
        Plot séparant :
            - la reconstruction de chaque omique
            - la reconstruction centrale (VAE central)
            - la supervision par la survie (pondérée par lambda_survival,
            c'est-à-dire le terme tel qu'il contribue réellement à la loss totale)
    
        Nécessite d'avoir entraîné avec fit(..., track_loss_components=True).
    
        log_scale : si True, affiche l'axe des y en échelle logarithmique.
        """
        if not hasattr(self, 'detailed_history') or len(self.detailed_history['train']) == 0:
            raise ValueError("Aucun historique détaillé trouvé. "
                            "Relancer fit(..., track_loss_components=True).")
    
        history = self.detailed_history['train']
        n_epochs = len(history)
        epochs = range(n_epochs)
    
        # Récupère l'ensemble des clés rencontrées sur tout l'entraînement
        all_keys = set()
        for epoch_dict in history:
            all_keys.update(epoch_dict.keys())
    
        plt.figure(figsize=(10, 6))
        plt.title("Décomposition de la loss par composante (train)")
    
        # ordonne les modalités selon self.source_names pour un affichage stable
        def sort_key(k):
            if k.startswith('recon_') and k != 'recon_central':
                name = k[len('recon_'):]
                return (0, self.source_names.index(name)) if name in self.source_names else (0, 999)
            return (1, 0)
    
        for key in sorted(all_keys, key=sort_key):
            # certaines clés (recon_central, survival_*) n'existent qu'à partir
            # de la phase 2 : on remplit avec NaN avant, matplotlib laissera un trou
            values = [epoch_dict.get(key, np.nan) for epoch_dict in history]
    
            if key.startswith('recon_') and key != 'recon_central':
                label = f"reconstruction {key[len('recon_'):]}"
            elif key == 'recon_central':
                label = "reconstruction centrale"
            elif key == 'survival_weighted':
                label = "supervision survie (pondérée)"
            elif key == 'survival_raw':
                continue  # affichée dans l'autre plot
            elif key.startswith('classification'):
                label = key.replace('_', ' ')
            elif key == 'domain':
                label = "dann"
            else:
                label = key
    
            plt.plot(epochs, values, label=label)
    
        plt.axvline(x=self.switch_epoch, color='purple', ls='--', lw=2, label='passage phase 2')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        if log_scale:
            plt.yscale('log')
        plt.legend(fontsize=10)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    
    
    def plot_loss_recon_surv(self, show=True, save_path=None, log_scale=False):
        """
        Version simplifiée : uniquement reconstruction centrale et supervision
        par la survie, avec la loss de survie brute (CoxLoss telle quelle,
        SANS multiplication par lambda_survival).
    
        log_scale : si True, affiche l'axe des y en échelle logarithmique.
        """
        if not hasattr(self, 'detailed_history') or len(self.detailed_history['train']) == 0:
            raise ValueError("Aucun historique détaillé trouvé. "
                            "Relancer fit(..., track_loss_components=True).")
    
        history = self.detailed_history['train']
        n_epochs = len(history)
        epochs = range(n_epochs)
    
        recon_central = [epoch_dict.get('recon_central', np.nan) for epoch_dict in history]
        survival_raw = [epoch_dict.get('survival_raw', np.nan) for epoch_dict in history]
    
        plt.figure(figsize=(10, 6))
        plt.title("Reconstruction centrale vs supervision survie (sans lambda_survival)")
        plt.plot(epochs, recon_central, label="reconstruction centrale")
        plt.plot(epochs, survival_raw, label="survie (brute, sans λ)")
        plt.axvline(x=self.switch_epoch, color='purple', ls='--', lw=2, label='passage phase 2')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        if log_scale:
            plt.yscale('log')
        plt.legend(fontsize=10)
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    

    
    def plot_loss_detailed_stacked(self, show=True, save_path=None, log_scale=False):
        """
        Même contenu que plot_loss_detailed, mais en aires empilées : chaque
        composante s'ajoute par-dessus les précédentes, le sommet de la pile
        correspond à la loss totale (hors lambda_central appliqué au terme
        central, cf. remarque ci-dessous).
    
        Avant la phase 2, seules les reconstructions par omique existent : les
        composantes 'recon_central' / 'survival_weighted' valent alors 0
        (et non NaN, un stackplot ne tolère pas les trous).
    
        log_scale : si True, affiche l'axe des y en échelle logarithmique.
            Comme log(0) est indéfini, les valeurs à 0 (phase 1) sont alors
            remplacées par un epsilon (1e-8) pour rester affichables.
        """
        if not hasattr(self, 'detailed_history') or len(self.detailed_history['train']) == 0:
            raise ValueError("Aucun historique détaillé trouvé. "
                            "Relancer fit(..., track_loss_components=True).")
    
        history = self.detailed_history['train']
        n_epochs = len(history)
        epochs = np.arange(n_epochs)
    
        all_keys = set()
        for epoch_dict in history:
            all_keys.update(epoch_dict.keys())
    
        # on ignore survival_raw ici : on veut le terme pondéré, qui est celui
        # qui contribue réellement à la loss totale empilée
        all_keys.discard('survival_raw')
    
        def sort_key(k):
            # ordre d'empilement : modalités d'abord (dans l'ordre de self.source_names),
            # puis central, puis survie/classif/domaine
            if k.startswith('recon_') and k != 'recon_central':
                name = k[len('recon_'):]
                idx = self.source_names.index(name) if name in self.source_names else 999
                return (0, idx)
            if k == 'recon_central':
                return (1, 0)
            if k.startswith('classification'):
                return (2, 0)
            if k.startswith('survival'):
                return (3, 0)
            if k == 'domain':
                return (4, 0)
            return (5, 0)
    
        ordered_keys = sorted(all_keys, key=sort_key)
    
        label_map = {
            'recon_central': 'reconstruction centrale',
            'survival_weighted': 'supervision survie (pondérée)',
            'classification_weighted': 'classification (pondérée)',
            'domain': 'adversarial de domaine',
        }
    
        series = []
        labels = []
        for key in ordered_keys:
            values = [epoch_dict.get(key, 0.0) for epoch_dict in history]
            if log_scale:
                values = [v if v > 0 else 1e-8 for v in values]
            series.append(values)
            if key.startswith('recon_') and key != 'recon_central':
                labels.append(f"reconstruction {key[len('recon_'):]}")
            else:
                labels.append(label_map.get(key, key))
    
        plt.figure(figsize=(10, 6))
        plt.title("Décomposition cumulée de la loss (train)")
        plt.stackplot(epochs, *series, labels=labels)
        plt.axvline(x=self.switch_epoch, color='purple', ls='--', lw=2, label='passage phase 2')
        plt.xlabel('epochs')
        plt.ylabel('loss cumulée')
        if log_scale:
            plt.yscale('log')
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    
    def plot_loss_recon_surv_stacked(self, show=True, save_path=None, log_scale=False):
        """
        Version empilée du duo reconstruction centrale / survie brute
        (sans lambda_survival).
    
        log_scale : si True, affiche l'axe des y en échelle logarithmique
            (les valeurs à 0 sont remplacées par un epsilon 1e-8).
        """
        if not hasattr(self, 'detailed_history') or len(self.detailed_history['train']) == 0:
            raise ValueError("Aucun historique détaillé trouvé. "
                            "Relancer fit(..., track_loss_components=True).")
    
        history = self.detailed_history['train']
        n_epochs = len(history)
        epochs = np.arange(n_epochs)
    
        recon_central = [epoch_dict.get('recon_central', 0.0) for epoch_dict in history]
        survival_raw = [epoch_dict.get('survival_raw', 0.0) for epoch_dict in history]
        if log_scale:
            recon_central = [v if v > 0 else 1e-8 for v in recon_central]
            survival_raw = [v if v > 0 else 1e-8 for v in survival_raw]
    
        plt.figure(figsize=(10, 6))
        plt.title("Reconstruction centrale + survie brute (cumulé, sans lambda_survival)")
        plt.stackplot(epochs, recon_central, survival_raw,
                    labels=["reconstruction centrale", "survie (brute, sans λ)"])
        plt.axvline(x=self.switch_epoch, color='purple', ls='--', lw=2, label='passage phase 2')
        plt.xlabel('epochs')
        plt.ylabel('loss cumulée')
        if log_scale:
            plt.yscale('log')
        plt.legend(fontsize=10, loc='upper right')
        plt.tight_layout()
    
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()