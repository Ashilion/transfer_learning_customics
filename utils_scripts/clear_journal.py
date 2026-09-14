#!/usr/bin/env python3
"""
Supprime les fichiers journal Optuna pour un cancer et un name_suffix spécifiques.

Pattern des fichiers : CustOmics/optuna_journal/journal_{name_suffix}{cancer_name}_fold*.log

Usage :
    python clear_journal.py --cancer BRCA --dry-run
    python clear_journal.py --cancer BRCA --name-suffix clinical_
    python clear_journal.py --cancer BRCA --no-dry-run
"""

import argparse
import glob
import os


def delete_journals_for_cancer(journal_dir, name_suffix, cancer_name, dry_run=True):
    pattern = os.path.join(journal_dir, f"journal_{name_suffix}{cancer_name}_fold*.log")
    files = glob.glob(pattern)

    if not files:
        print(f"Aucun fichier trouvé pour le pattern : {pattern}")
        return

    print(f"{len(files)} fichier(s) trouvé(s) pour cancer='{cancer_name}', name_suffix='{name_suffix}':")
    for f in files:
        print(f"  - {f}")

    if dry_run:
        print("\n[DRY RUN] Aucun fichier supprimé. Ajoutez --no-dry-run pour supprimer réellement.")
    else:
        for f in files:
            os.remove(f)
        print(f"\n{len(files)} fichier(s) supprimé(s).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Supprime les journaux Optuna d'un cancer et name_suffix spécifiques."
    )
    parser.add_argument(
        "--journal-dir",
        default="CustOmics/optuna_journal",
        help="Dossier contenant les fichiers journal (défaut : CustOmics/optuna_journal)",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Préfixe/suffixe utilisé dans le nom des fichiers (défaut : '')",
    )
    parser.add_argument(
        "--cancer",
        required=True,
        help="Nom du cancer dont on veut supprimer les journaux (ex: BRCA)",
    )

    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Affiche seulement les fichiers concernés, sans les supprimer (défaut)",
    )
    dry_run_group.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Supprime réellement les fichiers",
    )
    parser.set_defaults(dry_run=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    delete_journals_for_cancer(
        journal_dir=args.journal_dir,
        name_suffix=args.name_suffix,
        cancer_name=args.cancer,
        dry_run=args.dry_run,
    )