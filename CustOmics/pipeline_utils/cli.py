"""Blocs d'arguments argparse partagés entre les scripts du pipeline.
"""
import argparse


def base_parser(description):
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


def add_cancer_arg(parser, default="COAD"):
    parser.add_argument(
        "--cancer",
        type=str,
        default=default,
        help="Cancer type name (e.g. COAD, BRCA, LUAD).",
    )
    return parser


def add_outer_cv_args(parser, outer_fold_required=False, default_splits=5):
    parser.add_argument(
        "--outer_splits",
        type=int,
        default=default_splits,
        help="Number of outer CV folds.",
    )
    parser.add_argument(
        "--outer_fold",
        type=int,
        required=outer_fold_required,
        default=None,
        help="Index of the outer fold to process.",
    )
    parser.add_argument(
        "--use_saved_folds",
        action="store_true",
        default=False,
        help="Use pre-saved outer fold splits from splits.json instead of generating new ones.",
    )
    return parser


def add_inner_cv_args(parser, default=3):
    parser.add_argument(
        "--inner_splits",
        type=int,
        default=default,
        help="Number of inner CV folds (StratifiedKFold).",
    )
    return parser


def add_optuna_args(parser, n_trials_default=30):
    parser.add_argument(
        "--n_trials_per_worker",
        type=int,
        default=n_trials_default,
        help="Number of Optuna trials per outer fold (per worker when running multiple workers).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Optuna timeout in seconds.",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="String to add to the study name and journal filename.",
    )
    return parser


def add_training_args(parser):
    parser.add_argument(
        "--supervised",
        action="store_true",
        default=False,
        help="Train CustOMICS in supervised mode (survival loss included during training).",
    )
    parser.add_argument(
        "--limit_epochs",
        type=int,
        default=None,
        help="Limit the number of epochs for training (disables early stopping).",
    )
    parser.add_argument(
        "--add_clinical",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Concatenate clinical features to the latent representation before CoxNet.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw"],
        default="adam",
        help="Optimizer used for training.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="weight decay / L2 reg of the weights of the model",
    )
    return parser


def add_cox_args(parser, l1_ratio_default=0.01):
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=l1_ratio_default,
        help="L1 ratio for CoxNet regularization.",
    )
    parser.add_argument(
        "--ridge",
        action="store_true",
        default=False,
        help="Use ridge (L2) CoxPHSurvivalAnalysis instead of Elastic Net CoxnetSurvivalAnalysis.",
    )
    return parser


def add_output_dir_arg(parser, default="results"):
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default,
        help="Directory where results (metrics, predictions, logs) are written.",
    )
    return parser


def add_transfer_paths_args(parser):
    parser.add_argument(
        "--pretrain_ckpt",
        type=str,
        default="pretrained_model.pt",
        help="Filename (relative to --ckpt_dir, prefixed by cancer) of the pre-trained checkpoint.",
    )
    parser.add_argument(
        "--best_params_in",
        type=str,
        default="best_params_source.json",
        help="Filename (relative to --ckpt_dir, prefixed by cancer) of the best-hyperparameters JSON.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="tl_ckpt",
        help="Directory where the pre-trained checkpoint and its config are read from and written to.",
    )
    return parser

def add_missing_modality_args(parser):
    parser.add_argument(
        "--modality_dropout", 
        action="store_true", 
        default=False,
        help="Active le modality dropout "
             "(si activé, p est optimisé par Optuna entre 0.01 et 0.5)."
    )
    parser.add_argument(
        "--md_mode",
        type=str,
        choices=["exclude", "reconstruct"],
        default="exclude",
        help="'exclude': dropped modality excluded from the loss. "
             "'reconstruct': denoising-autoencoder style reconstruction.",
    )
    parser.add_argument(
        "--missing_rate",
        type=float,
        default=0.0,
        help="Fraction of patients with a simulated missing modality "
             "(0.0 = none, default; e.g. 0.3/0.5/0.7 for experiments).",
    )
    parser.add_argument(
        "--missing_strategy",
        type=str,
        choices=["impute", "drop"],
        default="impute",
        help="'impute': zero-imputation + mask in loss. "
             "'drop': incomplete patients removed from train/val.",
    )
    return parser

def add_parallel_args(parser):
    parser.add_argument("--multiproc", type=int, default=1,
        help="Number of parallel worker processes for Optuna (each runs "
             "n_trials_per_worker trials, sharing the same JournalStorage file). "
             "1 = sequential, no multiprocessing.")
    parser.add_argument("--affinity", action="store_true", default=False,
        help="Répartit les cœurs initialement alloués entre les workers "
             "(2 workers par cœur physique, 1 thread HT chacun). Fait pour Topaze. "
             "Avec --multiproc 1, restaure simplement l'affinité initiale.")