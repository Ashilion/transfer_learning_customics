dir.create("r_libs", showWarnings = FALSE)

.libPaths("r_libs")
# Définir les dépôts
options(repos = c(
  mlrorg   = "https://mlr-org.r-universe.dev",
  raphaels1 = "https://raphaels1.r-universe.dev",
  CRAN     = "https://cloud.r-project.org"
))

# Installer les packages
install.packages("mlr3verse")
install.packages("mlr3extralearners")
install.packages("mlr3batchmark")
install.packages("mlr3proba")