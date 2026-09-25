"""Résolution des chemins de checkpoint / config JSON / sélecteur de
features, selon la convention `{output_dir}/{cancer}_{basename(...)}`
utilisée par les scripts de pré-entraînement et de fine-tuning.
"""
import os


class TransferPaths:
    """Regroupe les 3 chemins dérivés de --output_dir/--cancer/--pretrain_ckpt
    /--best_params_in (ou --best_params_out côté pré-entraînement), qui
    étaient recalculés à la main et de façon identique dans 3 scripts.
    """

    def __init__(self, output_dir, cancer, pretrain_ckpt, best_params_file):
        ckpt_filename = f"{cancer}_{os.path.basename(pretrain_ckpt)}"
        config_filename = f"{cancer}_{os.path.basename(best_params_file)}"

        self.pretrain_ckpt = os.path.join(output_dir, ckpt_filename)
        self.best_params = os.path.join(output_dir, config_filename)
        self.selector = f"{self.pretrain_ckpt}.selector.pkl"

    def __repr__(self):
        return (f"TransferPaths(pretrain_ckpt={self.pretrain_ckpt!r}, "
                f"best_params={self.best_params!r}, selector={self.selector!r})")