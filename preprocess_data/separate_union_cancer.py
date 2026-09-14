import argparse
import os
import pickle
import gc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split le pickle pancancer en un fichier par type de cancer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../data/dict_pancancer_union_mutation.pickle",
        help="Chemin vers le pickle pancancer complet.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/union_separated_cancer",
        help="Dossier où écrire un fichier pickle par cancer.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="dict_pancancer",
        help="Préfixe du nom de fichier de sortie (ex: dict_pancancer_COAD.pickle).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Réécrit les fichiers déjà existants.",
    )
    return parser.parse_args()


def load_pancancer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_cancer_data(pancancer, cancer_name):
    return {
        name: df[pancancer["clinical"]["cancer_type"] == cancer_name]
        for name, df in pancancer.items()
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Chargement de {args.input} ...")
    pancancer = load_pancancer(args.input)

    cancer_types = sorted(pancancer["clinical"]["cancer_type"].unique())
    print(f"{len(cancer_types)} types de cancer trouvés : {cancer_types}")

    for cancer_name in cancer_types:
        out_path = os.path.join(args.output_dir, f"{args.prefix}_{cancer_name}.pickle")

        if os.path.exists(out_path) and not args.overwrite:
            print(f"[skip] {out_path} existe déjà (utilise --overwrite pour écraser).")
            continue

        data = get_cancer_data(pancancer, cancer_name)
        n_samples = len(data["clinical"])

        with open(out_path, "wb") as f:
            pickle.dump(data, f)

        print(f"[ok] {cancer_name}: {n_samples} échantillons -> {out_path}")

    print("Terminé.")


if __name__ == "__main__":
    main()