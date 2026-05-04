rm(list = ls())

library(data.table)
library(survival)
library(ggplot2)
library(ggfortify)

# Path of the TLOMICS project
wkd_path = "/env/cnrgh/proj/math_stats/scratch/hlegrand/"

# Set working directory
setwd(wkd_path)

# Only takes CSV from the "data" directory --> ATTENTION si écrit un nouveau CSV !
data_path        = paste0(wkd_path, "data")
list_cancers_csv = list.files(path = data_path, pattern = ".RData", full.names = T)

######################################################
#                                                    #
# Extract data per cancer and create blocks of omics #
#                                                    #
######################################################

pancancer = list()
tmp_env   = new.env()
for (file in list_cancers_csv){
  tmp     = load(file = file, envir = tmp_env)
  dat     = as.data.table(tmp_env[[tmp]])
  
  # Extract a block per omic modality
  blocks = c("clinical|time|status|bcr_patient_barcode", "mutation", "rna", "mirna", "cnv")
  block_list = lapply(
    blocks,
    function(bl) {
      as.matrix(dat[, .SD, .SDcols=names(dat) %like% paste0("_", bl, "$")])
    }
  )
  names(block_list) <- blocks
  
  # Extract the cancer type
  cancer_type = gsub(pattern     = ".RData", 
                     replacement = "", 
                     x           = basename(file))
  # Create CLINICAL list where all variables in "_clinical" are stored along with
  # time and status
  block_list$clinical = data.frame(block_list$`clinical|time|status|bcr_patient_barcode`, 
                                   cancer_type = cancer_type)
  # Remove previous list containing clinical variables (the name was used to
  # extract those variables).
  block_list$`clinical|time|status|bcr_patient_barcode` = NULL
  
  # Store list of block in a list fof cancers
  pancancer[[cancer_type]] = block_list
  
  # Display progression
  print(cancer_type)
}
rm(list = 'tmp_env')

# save the list of cancers each containing a list of 5 omics (including clinical)
saveRDS(object = pancancer, file = paste0(data_path, "pancancer.RDS"))

#############################################################
#                                                           #
# Create pancancer data-set as a list of block, on per omic #
#                                                           #
#############################################################

# Define list of omics
blocks              = c("clinical", "mutation", "rna", "mirna", "cnv")
pancancer_intersect = list()
for (bl in blocks){
  # Per omic block, extract common features accross all cancers/
  common_features = Reduce("intersect", lapply(pancancer, function(x) colnames(x[[bl]])))
  # Concatenate each type of cancer based on common variables.
  pancancer_intersect[[bl]] = Reduce("rbind", lapply(pancancer, function(x) x[[bl]][, common_features]))
  # Display Progression
  print(bl)
  # If Mutation Block, take Union
  if(bl == "mutation"){
    union_features            = Reduce("union", lapply(pancancer, function(x) colnames(x[[bl]])))
    pancancer_intersect[[bl]] = Reduce("rbind", lapply(pancancer, function(x){
      tmp                      = matrix(0, ncol = length(union_features), nrow = nrow(x[[bl]]))
      colnames(tmp)            = union_features
      rownames(tmp)            = rownames(x[[bl]])
      tmp[, colnames(x[[bl]])] = x[[bl]]
      return(tmp)
    })) 
  }
}

# Convert some clinical variables into numerics
class(pancancer_intersect$clinical$time) = class(pancancer_intersect$clinical$status) = class(pancancer_intersect$clinical$age_clinical) = "numeric"

# Save pan-cancer data-set
saveRDS(object = pancancer_intersect, file = paste0(data_path, "pancancer_union_mutation.RDS"))
