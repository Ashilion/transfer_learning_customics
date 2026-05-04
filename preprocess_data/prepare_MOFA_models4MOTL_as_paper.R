rm(list = ls())

args = commandArgs(trailingOnly=TRUE)

nbFeatures          = as.integer(args[1])
cancerToRmv         = args[2]
Prior_K             = as.integer(args[3]) # how many factors to start with - 100 for LEARNING, 
# 60 for single-project REFERENCE, 100 for multi-project REFERENCE

# Load Librairies 
library(matrixStats)
library(rhdf5)
library(reticulate)
library(tictoc)
library(data.table)

wkd_path = "TODEFINIE"
setwd(wkd_path)

pancancer_file  = "pancancer_union_mutation.RDS"

pancancer       = readRDS(file = paste0(data_path, pancancer_file))
cancer_idx      = which(pancancer$clinical$cancer_type == cancerToRmv)
clinical_data   = pancancer$clinical
pancancer       = lapply(pancancer, function(x){
  rownames(x) = pancancer$clinical[["bcr_patient_barcode"]]
  return(t(x))})
pancancer       = pancancer[-1]
source          = lapply(pancancer, function(x) x[, -cancer_idx])
target          = lapply(pancancer, function(x) x[, cancer_idx])
clinical_source = clinical_data
clinical_target = clinical_data

#############################################################################
#                                                                           #
#   For both mRNA and miRNA we removed genes if they had a count of zero    #
#        in ≥ 90% of samples, or had zero variance across samples           #
#                                                                           #
#############################################################################

# Impossible to get zero count genes has it already had been transformed with
# log-CPM with Voom (log2(((x+0.5)/(sum(x)+1))*(10^6)))

list_null_sd = sapply(source, function(x) which(apply(x, 1, sd) == 0))
source       = sapply(names(source), function(x){if(length(list_null_sd[[x]]) != 0){return(source[[x]][-list_null_sd[[x]], ])}else{return(source[[x]])}})

print("Null variance samples had been removed in the source")
print(list_null_sd)

#############################################################################
#                                                                           #
#   We normalized mRNA and miRNA counts with the DESeq2 (v.1.38.0)          #
#      R package (Love et al., 2014), and log2(x + 1) transformed           #
#                        the normalized counts.                             #
#                                                                           #
#############################################################################

print("Already Log2-CPM with Voom...")

#############################################################################
#                                                                           #
#   We included SNV records whose variant classification was either         #
#     Frame Shift Del, Frame Shift Ins, In Frame Del, In Frame Ins,         #
#       Missense Mutation, Nonsense Mutation, Nonstop Mutation,             #
#             Splice Site or Translation Start Site                         #
#                                                                           #
#############################################################################

print("For now we do not have the info (Frame Shift Del, Frame Shift Ins, 
      In Frame Del, In Frame Ins...) but it could be nice to retrieve it !")

###################################################################################
#                                                                                 #
# We removed genes from SNV matrices if the mutation rate across samples was ≤ 1% #
#                                                                                 #
###################################################################################

mutation_to_rm = which(rowSums(source$mutation)/ncol(source$mutation) <= 0.01)
if(length(mutation_to_rm) != 0){
  source$mutation = source$mutation[-mutation_to_rm, ]
}

print("mutation rate below 1% were removed")
print(mutation_to_rm)

##################################################################################
#                                                                                #
#   We filtered all omics to include only the 5,000 most variable features.      #
#     We did not perform any batch effect correction on L datasets in order      #
#                to preserve biological signal (Lee et al., 2020).               #
#                                                                                #
##################################################################################

# Define list of omics
source_filtered = list()
for (bl in names(source)){
  # If number of features is higher than nbFeatures --> filter
  if (dim(source[[bl]])[1] > nbFeatures){
    # Order variable by their standard deviation
    features_var = order(apply(source[[bl]], 1, sd), decreasing = T)
    # Keep only the top nbFeatures features
    source_filtered[[bl]] = source[[bl]][features_var[1:nbFeatures], ]
  } else { # ELSE, do not filter
    source_filtered[[bl]] = source[[bl]]
  }
  # Display progression
  print(bl)
}

print("We filtered all omics to include only the 5,000 most variable features")
print(sapply(source_filtered, dim))