## This program is performing concordance correlation test. Input to this program is taken from BRM_AudioFeatureExtraction_withoutOverallStats.py module multimediaFeatureCalculation()
## After this program run BRM_AudioFeatureExtraction_withoutOverallStats.py module multimediaFeatureCalculation() again to get the pictures.

library(psych)
library(irrNA)
library(vegan)
library(dendextend)
library(VIM)
Sys.setenv(RETICULATE_PYTHON = "/home/sudhakar/.local/share/r-miniconda/envs/r-reticulate/bin")
require("reticulate")
use_condaenv("r-reticulate") ##https://rdrr.io/cran/reticulate/f/vignettes/python_packages.Rmd
pd <- import("pandas")
pickle_data <-pd$read_pickle("/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/BlockInformationFor244Stimuli.pkl")

videoPrefix = 'WithThirtyVideos'  # WithAllVideos , WithThirtyVideos, With69Videos, WithThirtyVideosEGI
sourceDir = file.path('/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/NewTarget', paste(videoPrefix, '_', sep=''))

allfiles = list.files(sourceDir, pattern=".+_IRRDF_.+csv", all.files = TRUE)
totalBlocks = length(allfiles)/5

for (i in 3 : totalBlocks){
#for (i in 7 : 7){
  suffix = paste('b-', as.character(i), sep = '') ## It should be dynamically change according to videoPrefix. All the video prefix should be checked for the suffix with value and with any value(i.e. suffix='').
  targetDir = file.path(sourceDir, suffix)
  dir.create(targetDir)
  ####################################################################################################### Valence 
  fileName = Sys.glob(file.path(sourceDir, pattern=paste(videoPrefix, suffix, paste('IRRDF_valence_', '*.csv', sep = ''), sep='_')))
  x = read.csv(fileName, row.names=1)
  colNames = colnames(x)
  x = kNN(t(x), k = 3, imp_var=FALSE)
  x = t(x)  
  colnames(x)=colNames

  x = round(x, 0)
  res = kendall.global(x, nperm = 999, mult = "holm")
  #res = kendallNA(x)
  resF = data.frame(matrix(ncol = 6, nrow = 1))
  colnames(resF) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
  resF[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res$Concordance_analysis['W',],3), round(res$Concordance_analysis['F',],3), round(res$Concordance_analysis['Prob.F',],3), round(res$Concordance_analysis['Chi2',],3), round(res$Concordance_analysis['Prob.perm',],3), toString(dim(x)))
  write.csv(resF, file.path(targetDir, paste(videoPrefix, 'Valence', paste('CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  #res = kendall.global(x, nperm = 999, mult = "holm")
  respost = kendall.post(x, nperm = 999, mult = "holm") ### A posterior test for correction
  write.csv(round(t(respost$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Valence', paste('POST_CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  ###########After Removing Columns
  if (sum(respost$A_posteriori_tests["Spearman.mean",]>0) > 1) {
    new = x[, respost$A_posteriori_tests["Spearman.mean",]>0]
    respost_xColsToKeep = kendall.post(new, nperm = 999, mult = "holm") ### A posterior test for correction
    
    library(Ckmeans.1d.dp)
    result = Ckmeans.1d.dp(respost_xColsToKeep$A_posteriori_tests["Spearman.mean",], 2)
    
    colsToKeep = (result$cluster==2)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Valence', 'Cluster-2', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep2 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep2$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Valence', 'Cluster-2', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))      
    }

    colsToKeep = (result$cluster==1)
    if (sum(colsToKeep) > 2){    
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Valence', 'Cluster-1', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep1 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep1$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Valence', 'Cluster-1', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
  }
  ############################ Arousal ################################
  fileName = Sys.glob(file.path(sourceDir, pattern=paste(videoPrefix, suffix, paste('IRRDF_arousal_', '*.csv', sep = ''), sep='_')))
  x = read.csv(fileName, row.names=1)
  colNames = colnames(x)
  x = kNN(t(x), k = 3, imp_var=FALSE)
  x = t(x)  
  colnames(x)=colNames
  # if (videoPrefix == 'WithThirtyVideos'){
  #   x = hf[c(1:5,8:16,18,20:30), ]
  # }else{
  #   x=hf
  # }
  x = round(x, 0)
  res = kendall.global(x, nperm = 999, mult = "holm")
  #res = kendallNA(x)
  
  resF = data.frame(matrix(ncol = 6, nrow = 1))
  colnames(resF) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
  resF[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res$Concordance_analysis['W',],3), round(res$Concordance_analysis['F',],3), round(res$Concordance_analysis['Prob.F',],3), round(res$Concordance_analysis['Chi2',],3), round(res$Concordance_analysis['Prob.perm',],3), toString(dim(x)))
  write.csv(resF, file.path(targetDir, paste(videoPrefix, 'Arousal', paste('CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  #res = kendall.global(x, nperm = 999, mult = "holm")
  respost = kendall.post(x, nperm = 999, mult = "holm") ### A posterior test for correction
  write.csv(round(t(respost$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Arousal', paste('POST_CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  ###########After Removing Columns
  if (sum(respost$A_posteriori_tests["Spearman.mean",]>0) > 1) {
    new = x[, respost$A_posteriori_tests["Spearman.mean",]>0]
    respost_xColsToKeep = kendall.post(new, nperm = 999, mult = "holm") ### A posterior test for correction
    
    library(Ckmeans.1d.dp)
    result = Ckmeans.1d.dp(respost_xColsToKeep$A_posteriori_tests["Spearman.mean",], 2)
    
    colsToKeep = (result$cluster==2)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Arousal', 'Cluster-2', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep2 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep2$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Arousal', 'Cluster-2', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
    
    colsToKeep = (result$cluster==1)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Arousal', 'Cluster-1', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep1 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep1$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Arousal', 'Cluster-1', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
  }
  #################################### Dominance ######################################
  fileName = Sys.glob(file.path(sourceDir, pattern=paste(videoPrefix, suffix, paste('IRRDF_dominan_', '*.csv', sep = ''), sep='_')))
  x = read.csv(fileName, row.names=1)
  colNames = colnames(x)
  x = kNN(t(x), k = 3, imp_var=FALSE)
  x = t(x)  
  colnames(x)=colNames
  # if (videoPrefix == 'WithThirtyVideos'){
  #   x = hf[c(1:5,8:16,18,20:30), ]
  # }else{
  #   x=hf
  # }
  x = round(x, 0)
  res = kendall.global(x, nperm = 999, mult = "holm")
  
  resF = data.frame(matrix(ncol = 6, nrow = 1))
  colnames(resF) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
  resF[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res$Concordance_analysis['W',],3), round(res$Concordance_analysis['F',],3), round(res$Concordance_analysis['Prob.F',],3), round(res$Concordance_analysis['Chi2',],3), round(res$Concordance_analysis['Prob.perm',],3), toString(dim(x)))
  write.csv(resF, file.path(targetDir, paste(videoPrefix, 'Dominance', paste('CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  #res = kendall.global(x, nperm = 999, mult = "holm")
  respost = kendall.post(x, nperm = 999, mult = "holm") ### A posterior test for correction
  write.csv(round(t(respost$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Dominance', paste('POST_CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  ###########After Removing Columns
  if (sum(respost$A_posteriori_tests["Spearman.mean",]>0) > 1) {
    new = x[, respost$A_posteriori_tests["Spearman.mean",]>0]
    respost_xColsToKeep = kendall.post(new, nperm = 999, mult = "holm") ### A posterior test for correction
    
    library(Ckmeans.1d.dp)
    result = Ckmeans.1d.dp(respost_xColsToKeep$A_posteriori_tests["Spearman.mean",], 2)
    
    colsToKeep = (result$cluster==2)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Dominance', 'Cluster-2', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep2 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep2$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Dominance', 'Cluster-2', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
    colsToKeep = (result$cluster==1)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Dominance', 'Cluster-1', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep1 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep1$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Dominance', 'Cluster-1', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))    
    }
  }
  
  #################################### Liking #################################
  fileName = Sys.glob(file.path(sourceDir, pattern=paste(videoPrefix, suffix, paste('IRRDF_liking_', '*.csv', sep = ''), sep='_')))
  x = read.csv(fileName, row.names=1) 
  colNames = colnames(x)
  x = kNN(t(x), k = 3, imp_var=FALSE)
  x = t(x)  
  colnames(x)=colNames
  # if (videoPrefix == 'WithThirtyVideos'){
  #   x = hf[c(1:5,8:16,18,20:30), ]}else{
  #     x=hf
  #   }
  x = round(x, 0)
  res = kendall.global(x, nperm = 999, mult = "holm")
  resF = data.frame(matrix(ncol = 6, nrow = 1))
  colnames(resF) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
  resF[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res$Concordance_analysis['W',],3), round(res$Concordance_analysis['F',],3), round(res$Concordance_analysis['Prob.F',],3), round(res$Concordance_analysis['Chi2',],3), round(res$Concordance_analysis['Prob.perm',],3), toString(dim(x)))
  write.csv(resF, file.path(targetDir, paste(videoPrefix, 'Liking', paste('CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  #res = kendall.global(x, nperm = 999, mult = "holm")
  respost = kendall.post(x, nperm = 999, mult = "holm") ### A posterior test for correction
  write.csv(round(t(respost$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Liking', paste('POST_CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  ###########After Removing Columns
  if (sum(respost$A_posteriori_tests["Spearman.mean",]>0) > 1) {
    new = x[, respost$A_posteriori_tests["Spearman.mean",]>0]
    respost_xColsToKeep = kendall.post(new, nperm = 999, mult = "holm") ### A posterior test for correction
    
    library(Ckmeans.1d.dp)
    result = Ckmeans.1d.dp(respost_xColsToKeep$A_posteriori_tests["Spearman.mean",], 2)
    
    colsToKeep = (result$cluster==2)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Liking', 'Cluster-2', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep2 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep2$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Liking', 'Cluster-2', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
    colsToKeep = (result$cluster==1)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Liking', 'Cluster-1', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep1 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep1$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Liking', 'Cluster-1', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
  }
  #################################### Familiarity #################################
  fileName = Sys.glob(file.path(sourceDir, pattern=paste(videoPrefix, suffix, paste('IRRDF_familiarity_', '*.csv', sep = ''), sep='_')))
  x = read.csv(fileName, row.names=1)
  colNames = colnames(x)
  x = kNN(t(x), k = 3, imp_var=FALSE)
  x = t(x)  
  colnames(x)=colNames
  # if (videoPrefix == 'WithThirtyVideos'){
  #   x = hf[c(1:5,8:16,18,20:30), ]}else{
  #     x=hf
  #   }
  x = round(x, 0)
  res = kendall.global(x, nperm = 999, mult = "holm")
  resF = data.frame(matrix(ncol = 6, nrow = 1))
  colnames(resF) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
  resF[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res$Concordance_analysis['W',],3), round(res$Concordance_analysis['F',],3), round(res$Concordance_analysis['Prob.F',],3), round(res$Concordance_analysis['Chi2',],3), round(res$Concordance_analysis['Prob.perm',],3), toString(dim(x)))
  write.csv(resF, file.path(targetDir, paste(videoPrefix, 'Familiarity', paste('CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  #res = kendall.global(x, nperm = 999, mult = "holm")
  rm(respost)
  respost = kendall.post(x, nperm = 999, mult = "holm") ### A posterior test for correction
  write.csv(round(t(respost$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Familiarity', paste('POST_CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  ###########After Removing Columns
  if (sum(respost$A_posteriori_tests["Spearman.mean",]>0) > 1) {
    new = x[, respost$A_posteriori_tests["Spearman.mean",]>0]
    respost_xColsToKeep = kendall.post(new, nperm = 999, mult = "holm") ### A posterior test for correction
    
    library(Ckmeans.1d.dp)
    result = Ckmeans.1d.dp(respost_xColsToKeep$A_posteriori_tests["Spearman.mean",], 2)
    
    colsToKeep = (result$cluster==2)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Familiarity', 'Cluster-2', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep2 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep2$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Familiarity', 'Cluster-2', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
    
    colsToKeep = (result$cluster==1)
    if (sum(colsToKeep) > 2){
      xColsToKeep = new[, colsToKeep]
      res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
      resF_xColsToKeep = data.frame(matrix(ncol = 6, nrow = 1))
      colnames(resF_xColsToKeep) = c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')
      resF_xColsToKeep[1, c("W", 'F', "Prob.F", 'Chi2', 'Prob.perm', 'Dimension')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)))
      write.csv(resF_xColsToKeep, file.path(targetDir, paste(videoPrefix, 'Familiarity', 'Cluster-1', paste('CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      respost_xColsToKeep1 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
      write.csv(round(t(respost_xColsToKeep1$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Familiarity', 'Cluster-1', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
    }
  }
}

#}
##ICC(hf)[["results"]]["Single_random_raters", ]
#ICC(hf)[["results"]]["Average_random_raters", ]

#library(tidyverse)
# Convert to long format
#hf_long <- hf %>% 
#  tibble::rowid_to_column("participant") %>% 
#  tidyr::gather(rater, score, -participant)
#m <- lme4::lmer(
#  score ~ 1 + (1 | rater) + (1 | participant),
#  hf_long
#)
#m

#variance_total <- sum(c(0.5, 1.71, 1.76) ^ 2)
#variance_participant <- 1.71 ^ 2
#variance_rater <- 0.50 ^ 2
#variance_participant / variance_total
#> [1] 0.1693387

#res = icc(hf, model='oneway', type='agreement', unit='single')#, effect='random')  #[["results"]]["Single_random_raters", ]
#print(res)
#return(list(res$p.value, res$parameter, res$conf.int, res$statistic, effectSize))
#}
