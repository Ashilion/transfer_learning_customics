"""Post-Optuna evaluation for SOURCE pan-cancer pre-training.

Recharge l'étude Optuna produite par `search_source_pretrain.py`, réentraîne
le modèle final sur toutes les données source, et sauvegarde :
  - le checkpoint des poids (`--output_dir/{cancer}_{pretrain_ckpt}`),
  - le sélecteur de features utilisé (`....selector.pkl`),
  - la config JSON (architecture + meilleurs hyperparamètres) consommée
    ensuite par `search_target_finetune.py` / `eval_target_finetune.py`.

Refactor de l'ancien script 4/6. Comportement inchangé, hormis l'ajout
optionnel (désactivé par défaut) du modality dropout et de la simulation de
modalités manquantes.
"""
import json
import os
import pickle
import sys

import torch
from sklearn.preprocessing import LabelEncoder

sys.path.append('..')

from custcox_utils import fit_feature_selector, apply_feature_selector, fit_scalers, apply_scalers

from pipeline_utils.checkpoints import TransferPaths
from pipeline_utils.cli import base_parser, add_cancer_arg, add_training_args, add_missing_modality_args
from pipeline_utils.data import load_pancancer, get_source_data
from pipeline_utils.hidden_dims_params import reconstruct_autoencoder_hidden_dims
from pipeline_utils.model import build_customics_model
from pipeline_utils.optuna_utils import load_best_trial
from pipeline_utils.missing_data import simulate_missing_modalities, apply_missing_modalities



def parse_args():
    parser = base_parser("Post-Optuna evaluation for SOURCE pan-cancer pre-training.")
    add_cancer_arg(parser)
    add_missing_modality_args(parser)
    parser.add_argument("--n_samples", type=int, default=-1,
        help="Number of source samples used during training (-1 = all).")
    parser.add_argument("--name_suffix", type=str, default="",
        help="Suffix used in journal/study name during training.")
    parser.add_argument("--output_dir", type=str, default="tl_ckpt",
        help="Directory to write the result CSV and model checkpoint.")
    parser.add_argument("--pretrain_ckpt", type=str, default="pretrained_model.pt",
        help="Path where the final model weights will be saved.")
    parser.add_argument("--best_params_out", type=str, default="best_params_source.json",
        help="Path where the best config JSON will be saved.")
    parser.add_argument("--n_epochs", type=int, default=1000,
        help="Max epochs for the final model (early stopping applies).")
    parser.add_argument("--supervised", action="store_true", default=False, help="Train in supervised mode.")
    parser.add_argument("--limit_epochs", type=int, default=None, help="Hard-limit epochs (disables early stopping).")
    parser.add_argument("--domain_adv", action="store_true", default=False,
        help="Must match the --domain_adv setting used during Optuna training.")
    parser.add_argument("--lambda_domain", type=float, default=1.0,
        help="Weight for the domain-adversarial loss (GRL coefficient ceiling).")
    parser.add_argument("--domain_hidden_dim", type=int, nargs="+", default=[64],
        help="Hidden layer sizes for the domain classifier head.")
    parser.add_argument("--validation", type=str, default="loss", help="Validation function (vvh, ibs or loss).")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam",
        help="Optimizer used for the final model training.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
        help="Weight decay / L2 reg of the final model's weights.")
    parser.add_argument("--missing_repetition_seed", type=int, default=0,
        help="Repetition seed for the missing-modality simulation. Must match the seed used in "
             "search_source_pretrain.py for the masks to be consistent (no outer-fold concept here).")
    return parser.parse_args()


def main():
    args = parse_args()
    unsupervised = not args.supervised
    n_epochs = args.limit_epochs if args.limit_epochs else args.n_epochs
    switch_epoch = n_epochs // 2

    print(f"\n{'='*60}")
    print(f"  [SOURCE eval]  Target cancer : {args.cancer}")
    print(f"  Study suffix   : '{args.name_suffix}'")
    print(f"  Max epochs     : {n_epochs}")
    print(f"  Domain Adversarial: {args.domain_adv}")
    print(f"  Modality dropout : {args.modality_dropout}  |  mode : {args.md_mode}")
    print(f"  Missing rate   : {args.missing_rate}  |  strategy : {args.missing_strategy}")
    print(f"{'='*60}\n")

    study_name = f"source_{args.name_suffix}{args.cancer}"
    journal_file = f"optuna_journal/journal_tl_{args.name_suffix}source_{args.cancer}.log"
    _, best_trial = load_best_trial(study_name, journal_file)

    best_params_hp = best_trial.params
    best_score = best_trial.value
    print(f"Best trial #{best_trial.number}")
    print(f"  params : {best_params_hp}")
    print(f"  score  : {best_score:.4f}")

    best_alpha = None
    if args.validation != "loss":
        best_alpha = best_trial.user_attrs["best_alpha"]
        print(f"  alpha  : {best_alpha:.6f}")

    pancancer = load_pancancer()
    source_data = get_source_data(pancancer, args.cancer, n_samples=args.n_samples)
    clinical_df = source_data["clinical"]

    omics_df = {
        "protein": source_data["_rna"],
        "gene_exp": source_data["mirna"],
        "methyl": source_data["cnv"],
        "mutation": source_data["mutation"],
    }

    lt_samples = list(clinical_df.index)
    sources = list(omics_df.keys())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 5
    classifier_dim = [128, 64]
    survival_dim = [64, 32]
    dropout = 0.2

    print(f"\nSource samples : {len(lt_samples)}")

    domain_params = None
    n_domains = None
    if args.domain_adv:
        domain_label_encoder = LabelEncoder().fit(clinical_df["cancer_type"])
        clinical_df = clinical_df.copy()
        clinical_df["domain_label"] = domain_label_encoder.transform(clinical_df["cancer_type"])
        n_domains = len(domain_label_encoder.classes_)
        print(f"  N domains (cancer types in source) : {n_domains}")
        domain_params = {
            "n_domains": n_domains, "lambda": args.lambda_domain,
            "hidden_layers": args.domain_hidden_dim, "dropout": dropout,
        }

    os.makedirs(args.output_dir, exist_ok=True)
    paths = TransferPaths(args.output_dir, args.cancer, args.pretrain_ckpt, args.best_params_out)

    selector_source = fit_feature_selector(omics_df, nbFeatures=5000)
    omics_source = apply_feature_selector(omics_df, selector_source)

    # ===== Simulation de modalités manquantes (avant le scaling) =====
    missing_assignment = None
    mask_source = None
    if args.missing_rate > 0:
        missing_assignment = simulate_missing_modalities(
            sample_ids=lt_samples,
            sources=sources,
            missing_rate=args.missing_rate,
            repetition_seed=args.missing_repetition_seed,
        )
        omics_source, mask_source = apply_missing_modalities(omics_source, missing_assignment)
        n_missing = sum(1 for v in missing_assignment.values() if v is not None)
        print(f"  -> {n_missing}/{len(lt_samples)} patients avec une modalité manquante simulée "
              f"(seed {args.missing_repetition_seed})")

    scalers = fit_scalers(omics_source)
    omics_source = apply_scalers(omics_source, scalers)
    with open(paths.selector, "wb") as f:
        pickle.dump(selector_source, f)

    autoencoder_hidden_dims = reconstruct_autoencoder_hidden_dims(best_params_hp, sources)
    best_params_full = {
        **best_params_hp,
        "autoencoder_hidden_dims": autoencoder_hidden_dims,
        "latent_dim": 32,
        "rep_dim": 32,
        "central_hidden_dims": [64],
        "patience": best_params_hp.get("patience", 3),
    }

    print("\n=== Final training on all source data ===")
    patience = None if args.limit_epochs else best_params_full["patience"]

    modality_dropout_p = best_params_full.get("modality_dropout_p")

    final_model = build_customics_model(
        omics_source, sources, best_params_full, device,
        classifier_dim, survival_dim, dropout, num_classes, unsupervised, switch_epoch,
        domain_params=domain_params,
        modality_dropout_p=modality_dropout_p, md_mode=args.md_mode,
        optimizer=args.optimizer, weight_decay=args.weight_decay,
    )
    final_model.fit(
        omics_train=omics_source, clinical_df=clinical_df,
        label="status", event="status", surv_time="time",
        omics_val=None, batch_size=32, n_epochs=n_epochs,
        verbose=True, task="survival",
        patience=patience, min_delta=best_params_full["delta_min"],
        early_stopping_on="train",
        modality_mask_train=mask_source,
        missing_strategy=args.missing_strategy,
    )

    torch.save(final_model.state_dict(), paths.pretrain_ckpt)
    print(f"Weights saved -> {paths.pretrain_ckpt}")

    best_config = {
        "best_params": best_params_full,
        "best_score": float(best_score),
        "architecture": {
            "hidden_dim": best_params_full["autoencoder_hidden_dims"],
            "central_hidden": best_params_full["central_hidden_dims"],
            "classifier_dim": classifier_dim,
            "survival_dim": survival_dim,
            "num_classes": num_classes,
            "dropout": dropout,
            "unsupervised": unsupervised,
        },
        "domain_adversarial": {
            "enabled": args.domain_adv,
            "lambda_domain": args.lambda_domain,
            "domain_hidden_dim": args.domain_hidden_dim,
            "n_domains": n_domains,
        },
        "modality_dropout": {
            "enabled": args.modality_dropout,
            "p": modality_dropout_p,
            "mode": args.md_mode,
        },
        "missing_modalities": {
            "rate": args.missing_rate,
            "strategy": args.missing_strategy,
            "repetition_seed": args.missing_repetition_seed,
        },
        "target_cancer": args.cancer,
        "nb_source_samples": len(lt_samples),
        "optuna_study_name": study_name,
        "optuna_journal": journal_file,
        "optuna_trial": best_trial.number,
    }
    if args.validation != "loss":
        best_config["best_alpha"] = float(best_alpha)

    with open(paths.best_params, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"Config saved -> {paths.best_params}")


if __name__ == "__main__":
    main()