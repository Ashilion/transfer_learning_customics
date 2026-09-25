# Figures

### 1ère étape
Les scripts d'évaluation écrivent dans des fichiers unique spécifique au fold -> il faut regrouper après pour avoir tous les résultats dans un même fichier

``` shell
# Usage: sbatch scripts/combine_results_outer.sh <cancer> <method> <name_suffix>

# Sans TL
sbatch scripts/combine_results_outer.sh KIRP custcox supclinridge_

# Avec TL
sbatch scripts/combine_results_outer.sh KIRP finetune supclinridge_
```

---

``` shell
module load python
cd /env/cnrgh/proj/math_stats/scratch/hlegrand/
source venv_sksurv_dev/bin/activate
```

## 1 cancer
![kirp allridge](../figures/KIRP/KIRP_allridge_boxplot.png)
``` shell
python results_analysis/cancer_specific/compare_all_1_cancer_sep.py --cancer KIRP --suffix ridge --graph figures/KIRP/KIRP_sep --boxplot
```

![kirp scaled](../figures/KIRP/KIRP_scaled_boxplot.png)
``` shell
python results_analysis/cancer_specific/compare_multiple_1_cancer.py 
--files results/outer_cv_results_vvh_ridge_KIRP.csv  \
    results/ncv_custcox_optuna_paral_ridge_KIRP.csv \
    results/ncv_custcox_optuna_paral_clinridge_KIRP.csv \ 
    results/ncv_custcox_optuna_paral_supridge_KIRP.csv \
    results/ncv_custcox_optuna_paral_supclinridge_KIRP.csv \
    results/ncv_finetune_optuna_paral_ridge_KIRP.csv \
    results/ncv_finetune_optuna_paral_clinridge_KIRP.csv \ 
    results/ncv_finetune_optuna_paral_supridge_KIRP.csv \
    results/ncv_finetune_optuna_paral_supclinridge_KIRP.csv \ 
--labels "Cox" "CustCox" "CustCox clin" "CustCox sup" "CustCox clin sup" \
"TL CustCox" "TL CustCox clin" "TL CustCox sup" "TL CustCox clin sup" \
--graph "KIRP_allridge" --boxplot

```


![kirp ttest](../figures/KIRP/ttest_KIRP.png)
``` shell
python results_analysis/cancer_specific/ttest_all_method_1_cancer.py --cancer KIRP --data-dir ./results --summary --plot --fdr
```


## Multi Cancer

![heatmap cox diff ](../figures/all_cancer/heatmap_cox_diff_c_index.png)
``` shell
python results_analysis/heatmap_cox_diff.py --results-dir results/ --output-dir ./figures/all_cancer/
```


![heatmap tl diff ](../figures/all_cancer/heatmap_tl_diff_c_index.png)
``` shell
python results_analysis/heatmap_tl_diff.py --results-dir results/ --output-dir ./figures/all_cancer/
```



## Missing data

<img src="../figures/KIRP/missing_data/missing_results_boxplot_cindex.png" width="500">

``` shell
python results_analysis/heatmap_tl_diff.py --results-dir results/ --output-dir ./figures/all_cancer/
```

<img src="../figures/KIRP/missing_data/missing_finetune_results_test_grouped_boxplot_cindex.png" width="700">

``` shell
python results_analysis/heatmap_tl_diff.py --results-dir results/ --output-dir ./figures/all_cancer/
```


## Hyperparamètres

<img src="../CustOmics/figures/best_params_comparison_sans_tl.png" width="400">

<img src="../CustOmics/figures/param_importances_sans_tl.png" width="400">

``` shell
sbatch scripts/optuna_importance_all_methods.sh
```
Pour un seul cancer (nécéssite avoir lancé le script précédent)
``` shell
cd CustOmics
python analysis/analyse_optuna_all_methods_1_cancer.py
```
## Espace latent

---

Se placer dans le dossier CustOmics pour les prochaines commandes

![espace latent coloration](../CustOmics/latent_space_analysis/figures_latent_pancancer/latent_cancer_type_coloring.png)

![espace latent coloration dann](../CustOmics/latent_space_analysis/figures_latent_pancancer_dann/latent_cancer_type_coloring_dann.png)

``` shell
cd CustOmics/latent_space_analysis/
python tl_source_analysis.py
```
