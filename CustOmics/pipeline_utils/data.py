"""Chargement des données pan-cancer / par cancer, factorisé.

Les scripts d'origine construisaient chacun leur dict `omics_df` à la main, à
partir des mêmes 4 clés brutes ('_rna', 'mirna', 'cnv', 'mutation') mais avec
deux conventions de nommage différentes selon le script :

- "clinical_labels" -> {'protein', 'gene_exp', 'methyl', 'mutation'}
  (utilisé par les scripts pan-cancer : ex search/eval_source_pretrain)
- "raw"             -> {'_rna', 'mirna', 'cnv', 'mutation'}
  (utilisé par les scripts par-cancer : ex search/eval_ncv, search/eval_target_finetune)

`build_omics_dict` couvre les deux avec un seul paramètre plutôt que de
recopier le dict literal à chaque script.
"""
import pickle

import numpy as np

_NAMING_SCHEMES = {
    "clinical_labels": {
        "protein": "_rna",
        "gene_exp": "mirna",
        "methyl": "cnv",
        "mutation": "mutation",
    },
    "raw": {
        "_rna": "_rna",
        "mirna": "mirna",
        "cnv": "cnv",
        "mutation": "mutation",
    },
}


def load_pancancer(path="../data/dict_pancancer_union_mutation.pickle"):
    """Charge le dict pan-cancer complet (tous cancers confondus)."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_cancer_data(cancer_name, per_cancer_dir="../data/union_separated_cancer",
                      prefix="dict_pancancer"):
    """Charge le dict pré-découpé pour un seul cancer (fichiers `union_separated_cancer`)."""
    path = f"{per_cancer_dir}/{prefix}_{cancer_name}.pickle"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_clinical_test(cancer_name, clinical_dir="../data/clinical"):
    """Charge le pickle clinical_test (features cliniques brutes hors
    time/status) utilisé uniquement en mode --add_clinical.
    """
    path = f"{clinical_dir}/{cancer_name}_clinical.pickle"
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df[list(set(df.columns) - {"time", "bcr_patient_barcode", "status"})]


def get_cancer_data(pancancer, cancer_name):
    """Filtre un dict pan-cancer complet sur un seul type de cancer."""
    return {
        name: df[pancancer["clinical"]["cancer_type"] == cancer_name]
        for name, df in pancancer.items()
    }


def get_source_data(pancancer, target_cancer, n_samples=-1, random_state=42):
    """Sous-dict pan-cancer EXCLUANT `target_cancer`, avec sous-échantillonnage
    optionnel. Utilisé par le pré-entraînement source (transfer learning).
    """
    clinical_all = pancancer["clinical"]
    source_mask = clinical_all["cancer_type"] != target_cancer
    candidates = clinical_all[source_mask]

    if n_samples == -1:
        source_indices = candidates.index
    else:
        source_indices = candidates.sample(n_samples, random_state=random_state).index

    return {name: df.loc[source_indices] for name, df in pancancer.items()}


def build_omics_dict(data, naming="clinical_labels", cast_float32=False):
    """Construit le dict {source_name: DataFrame} attendu par CustOMICS à
    partir d'un dict brut, selon la convention de nommage `naming`
    ("clinical_labels" ou "raw", voir docstring du module).
    """
    scheme = _NAMING_SCHEMES[naming]
    omics = {out_key: data[in_key] for out_key, in_key in scheme.items()}
    if cast_float32:
        omics = {k: v.astype(np.float32) for k, v in omics.items()}
    return omics