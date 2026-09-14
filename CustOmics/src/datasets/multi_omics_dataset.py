import torch
from torch.utils.data import Dataset
import numpy as np


class MultiOmicsDataset(Dataset):
    """
    Load multi-omics data.
    modality_mask_df : DataFrame optionnel, index=sample_id, colonnes=noms de sources
                        (même ordre que omics_df.keys()), valeurs 1=observé / 0=manquant.
                        Si None, tous les patients sont considérés comme complets
                        (comportement rétrocompatible).
    """

    def __init__(self, omics_df, clinical_df, lt_samples, label, event, surv_time,
                 domain_col=None, modality_mask_df=None):
        self.omics_df = omics_df
        self.clinical_df = clinical_df
        self.lt_samples = lt_samples
        self.label = label
        self.event = event
        self.surv_time = surv_time
        self.domain_col = domain_col
        self.modality_mask_df = modality_mask_df
        self.source_names = list(omics_df.keys())

        if self.modality_mask_df is not None:
            missing_sources = set(self.source_names) - set(self.modality_mask_df.columns)
            if missing_sources:
                raise ValueError(
                    f"modality_mask_df ne contient pas de colonne pour {missing_sources}"
                )
            missing_samples = set(lt_samples) - set(self.modality_mask_df.index)
            if missing_samples:
                raise ValueError(
                    f"modality_mask_df n'a pas d'entrée pour {len(missing_samples)} "
                    f"échantillons (ex: {list(missing_samples)[:5]})"
                )

    def __len__(self):
        return len(self.lt_samples)

    def __getitem__(self, index):
        sample = self.lt_samples[index]
        omics_data = []
        for source, omic_df in zip(self.omics_df.keys(), self.omics_df.values()):
            omic_line = omic_df.loc[sample, :].values
            omic_line = omic_line.astype(np.float32)
            omic_line_tensor = torch.Tensor(omic_line)
            omics_data.append(omic_line_tensor)
        if self.label:
            label = self.clinical_df.loc[sample, self.label]
        else:
            label = 0
        os_time = int(self.clinical_df.loc[sample, self.surv_time])
        os_event = int(self.clinical_df.loc[sample, self.event])

        if self.domain_col:
            domain_label = int(self.clinical_df.loc[sample, self.domain_col])
        else:
            domain_label = 0

        if self.modality_mask_df is not None:
            observed_mask = torch.Tensor(
                self.modality_mask_df.loc[sample, self.source_names].values.astype(np.float32)
            )
        else:
            observed_mask = torch.ones(len(self.source_names), dtype=torch.float32)

        return omics_data, label, os_time, os_event, domain_label, observed_mask

    def return_samples(self):
        return self.lt_samples


class PANCANDataset(Dataset):
    """
    Load multi-omics data
    """

    def __init__(self, omics_df, labels, cohort):
        self.omics_df = omics_df
        self.labels = labels
        self.cohort = cohort

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        omics_data = []
        for source, omic_df in zip(self.omics_df.keys(), self.omics_df.values()):
            omic_line = omic_df.iloc[index, :].values
            omic_line = omic_line.astype(np.float32)
            omic_line_tensor = torch.Tensor(omic_line)
            omics_data.append(omic_line_tensor)
        label = self.labels[index]
        return omics_data, label