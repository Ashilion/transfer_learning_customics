#!/usr/bin/env python3
"""
Supprime TOUS les fichiers journal Optuna d'un dossier.

Usage :
    python delete_journal_all.py --dry-run
    python delete_journal_all.py --no-dry-run
"""

import argparse
import glob
import os


def delete_all_journals(journal_dir, dry_run=True):
    pattern = os.path.join(journal_dir, "journal_*.log")
    files = glob.glob(pattern)

    if not files:
        print(f"Aucun fichier journal trouvé dans : {journal_dir}")
        return

    print(f"{len(files)} fichier(s) trouvé(s) :")
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
        description="Supprime tous les journaux Optuna d'un dossier."
    )
    parser.add_argument(
        "--journal-dir",
        default="CustOmics/optuna_journal",
        help="Dossier contenant les fichiers journal (défaut : optuna_journal)",
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
    delete_all_journals(
        journal_dir=args.journal_dir,
        dry_run=args.dry_run,
    )