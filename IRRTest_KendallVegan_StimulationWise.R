## This program is performing concordance correlation test. Input to this program is taken from BRM_AudioFeatureExtraction.py

library(psych)
library(irrNA)
library(vegan)
library(dendextend)
library(VIM)
library(irr)
library(stringr)

videoPrefix = 'WithThirtyVideos_'  # WithAllVideos_ , WithThirtyVideos_, With69Videos_, WithThirtyVideosEGI
if (videoPrefix == 'WithAllVideos_'){
  date = 'Oct_10-Oct_20'  
}else if ((videoPrefix == 'WithThirtyVideos_') | (videoPrefix == 'With69Videos_')){
  date = 'Oct_10-Nov_15'  
  if (videoPrefix == 'WithThirtyVideos_'){
    clipDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/block_For_30_Stimuli/Videos'
    clipsAre = list.files(clipDir, pattern=".", all.files = TRUE)      
  }else if (videoPrefix == 'With69Videos_'){
    clipDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Emotion_Name_Rating/Videos'
    clipsAre = list.files(clipDir, pattern=".", all.files = TRUE)          
  }
## No need to use clipsAre as, I will take care of the emotion specific results while filling the concordance related details in python program: 
## /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/BRM_AudioFeatureExtraction_withoutOverallStats.py
}

sourceDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/NewTarget'
fileName = paste('My_Experiment_Ratings_after_cleaning2018', paste(date, '.csv', sep=''), sep = '_')
targetFN = paste('RProgram_MeanAbsoluteDifference', strsplit(fileName, 'Ratings_')[[1]][2], sep = '_')
sourceFile = file.path(sourceDir, fileName)
DataFrame = read.csv(sourceFile, row.names=1) 
#DataFrame = DataFrame[clipsAre,]
emotions = unique(DataFrame[,'Experiment_id'])
#allResults = data.frame(matrix(ncol = 7, nrow = 1))
allResults = data.frame(matrix(ncol = 7, nrow = length(emotions)))
rownames(allResults) = emotions
colnames(allResults) = c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')

Cluster_1_Res<-data.frame(allResults)
tracemem(Cluster_1_Res)==tracemem(allResults)
Cluster_2_Res<-data.frame(allResults)
tracemem(Cluster_2_Res)==tracemem(allResults)

#kappaResults = data.frame(matrix(ncol = 5, nrow = 1))
kappaResults = data.frame(matrix(ncol = 5, nrow = length(emotions)))
rownames(kappaResults) = emotions
colnames(kappaResults) = c("kappa", 'stats', "p_val", 'subjects', 'raters')
Cluster_1_kappa<-data.frame(kappaResults)
tracemem(Cluster_1_kappa)==tracemem(kappaResults)
Cluster_2_kappa<-data.frame(kappaResults)
tracemem(Cluster_2_kappa)==tracemem(kappaResults)

#DataFrame[,'Valence'] = round(DataFrame[,'Valence'], 0)
#DataFrame[,'Arousal'] = round(DataFrame[,'Arousal'], 0)
#DataFrame[,'Dominance'] = round(DataFrame[,'Dominance'], 0)
DataFrame[,'Familiarity'] = (((DataFrame[,'Familiarity'] - 1) * (9 - 1)) / (5 - 1)) + 1
DataFrame[,'Liking'] = (((DataFrame[,'Liking'] - 1) * (9 - 1)) / (5 - 1)) + 1

Quality <- function(Wval) {
  if (round(Wval,3) > 0.8){
    AgreeQuality = 'Very Good'
  }else if ((round(Wval,3) <= 0.8) & (round(Wval,3) > 0.6)){
    AgreeQuality = 'Good'
  }else if ((round(Wval,3) <= 0.6) & (round(Wval,3) > 0.4)){
    AgreeQuality = 'Moderate'
  }else if ((round(Wval,3) <= 0.4) & (round(Wval,3) > 0.2)){
    AgreeQuality = 'Fair'
  }else if (round(Wval,3) < 0.2){
    AgreeQuality = 'Poor'
  }
  return(AgreeQuality)
}

noDomin = 0  ## This flag will trigger the option of using dominance scale or not. 0: means no use of dominance and 1: means using dominance.

for (emt in emotions){#[c(1:6, 8:13, 15:207)]){
  
  #if (str_detect(emt, 'Barbie')){
    print(emt)
    indexes = which(DataFrame['Experiment_id'] == emt)
  if (length(indexes)>5){
    if (noDomin != 1){
      consideredColmns = c('Valence','Arousal','Dominance','Liking','Familiarity')
    }else if (noDomin == 1){
      consideredColmns = c('Valence','Arousal','Liking','Familiarity')
    }
    
    if (noDomin != 1){
      emotionsWiseDF = data.frame(matrix(ncol = length(indexes), nrow = 5))
      rownames(emotionsWiseDF) = consideredColmns
      emotionsWiseDF['Valence',] = DataFrame[indexes,'Valence']
      emotionsWiseDF['Arousal',] =  DataFrame[indexes,'Arousal']      
      emotionsWiseDF['Dominance',] = DataFrame[indexes,'Dominance']
    }else if (noDomin == 1){
      emotionsWiseDF = data.frame(matrix(ncol = length(indexes), nrow = 4))
      rownames(emotionsWiseDF) = consideredColmns
      emotionsWiseDF['Valence',] = DataFrame[indexes,'Valence']
      emotionsWiseDF['Arousal',] =  DataFrame[indexes,'Arousal']            
    }
    emotionsWiseDF['Liking',] = DataFrame[indexes,'Liking']
    emotionsWiseDF['Familiarity',] = DataFrame[indexes,'Familiarity']
    
    meanVal = mapply(mean,emotionsWiseDF)
    resSum = 0
    
    for (clmn in consideredColmns){
      #print(round(emotionsWiseDF,0)[clmn, ])
      resSum = resSum + abs(round(meanVal,0)-round(emotionsWiseDF,0)[clmn, ])
    }
    emotionsWiseDF = emotionsWiseDF[, resSum!=0]
    
    kappaCoef = kappam.fleiss(emotionsWiseDF, detail=TRUE)
    kappaResults[emt, c("kappa", 'stats', "p_val", 'subjects', 'raters')] = c(round(kappaCoef$value,2), round(kappaCoef$statistic,2), round(kappaCoef$p.value,3), round(kappaCoef$subjects,2), round(kappaCoef$raters,2))

    res = kendall.global(emotionsWiseDF, nperm = 999, mult = "holm")
    AgreeQuality = Quality(res$Concordance_analysis['W',])
    resF = data.frame(matrix(ncol = 7, nrow = 1))
    colnames(resF) = c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')
    resF[1, c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')] = c(round(res$Concordance_analysis['W',],3), round(res$Concordance_analysis['F',],3), round(res$Concordance_analysis['Prob.F',],3), round(res$Concordance_analysis['Chi2',],3), round(res$Concordance_analysis['Prob.perm',],3), toString(dim(emotionsWiseDF)), AgreeQuality)
    allResults[emt, ] = resF    
    
    respost = kendall.post(emotionsWiseDF, nperm = 999, mult = "holm") ### A posterior test for correction
    if (sum(respost$A_posteriori_tests["Spearman.mean",]>0) > 1) {
      new = emotionsWiseDF[, respost$A_posteriori_tests["Spearman.mean",]>0]
      respost_xColsToKeep = kendall.post(new, nperm = 999, mult = "holm") ### A posterior test for correction
      
      library(Ckmeans.1d.dp)
      result = Ckmeans.1d.dp(respost_xColsToKeep$A_posteriori_tests["Spearman.mean",], 2)
      
      colsToKeep = (result$cluster==2)
      if (sum(colsToKeep) > 2){
        print('============================== Cluster-2 ===========================')
        xColsToKeep = new[, colsToKeep]
        kappaCoef = kappam.fleiss(xColsToKeep, detail=TRUE)
        Cluster_2_kappa[emt, c("kappa", 'stats', "p_val", 'subjects', 'raters')] = c(round(kappaCoef$value,2), round(kappaCoef$statistic,2), round(kappaCoef$p.value,3), round(kappaCoef$subjects,2), round(kappaCoef$raters,2))
        
        res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
        AgreeQuality = Quality(res_xColsToKeep$Concordance_analysis['W',])
        #print(res_xColsToKeep)
        resF_xColsToKeep = data.frame(matrix(ncol = 7, nrow = 1))
        colnames(resF_xColsToKeep) = c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')
        resF_xColsToKeep[1, c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)), AgreeQuality)
        Cluster_2_Res[emt, ] = resF_xColsToKeep
        respost_xColsToKeep2 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
        #print(round(t(respost_xColsToKeep2$A_posteriori_tests), 3))
        #write.csv(round(t(respost_xColsToKeep2$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Valence', 'Cluster-2', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))      
      }
      
      colsToKeep = (result$cluster==1)
      if (sum(colsToKeep) > 2){    
        print('============================== Cluster-1 ===========================')
        xColsToKeep = new[, colsToKeep]
        kappaCoef = kappam.fleiss(xColsToKeep, detail=TRUE)
        Cluster_1_kappa[emt, c("kappa", 'stats', "p_val", 'subjects', 'raters')] = c(round(kappaCoef$value,2), round(kappaCoef$statistic,2), round(kappaCoef$p.value,3), round(kappaCoef$subjects,2), round(kappaCoef$raters,2))    
        
        res_xColsToKeep = kendall.global(xColsToKeep, nperm = 999, mult = "holm")
        AgreeQuality = Quality(res_xColsToKeep$Concordance_analysis['W',])
        #print(res_xColsToKeep)
        resF_xColsToKeep = data.frame(matrix(ncol = 7, nrow = 1))
        colnames(resF_xColsToKeep) = c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')
        resF_xColsToKeep[1, c("Concord_W", 'Concord_F', "Concord_Prob.F", 'Concord_Chi2', 'Concord_Prob.perm', 'Concord_Dimension','ConcordCateg')] = c(round(res_xColsToKeep$Concordance_analysis['W',],3), round(res_xColsToKeep$Concordance_analysis['F',],3), round(res_xColsToKeep$Concordance_analysis['Prob.F',],3), round(res_xColsToKeep$Concordance_analysis['Chi2',],3), round(res_xColsToKeep$Concordance_analysis['Prob.perm',],3), toString(dim(xColsToKeep)), AgreeQuality)
        Cluster_1_Res[emt, ] = resF_xColsToKeep
        respost_xColsToKeep1 = kendall.post(xColsToKeep, nperm = 999, mult = "holm") ### A posterior test for correction
        #print(round(t(respost_xColsToKeep1$A_posteriori_tests), 3))
        #write.csv(round(t(respost_xColsToKeep1$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Valence', 'Cluster-1', paste('POST_CCC_Test_Result_AfterRemovingNon-SignificantCols_', suffix, '.csv', sep = ''), sep='_')))
      }    
    }
    #write.csv(round(t(respost$A_posteriori_tests), 3), file.path(targetDir, paste(videoPrefix, 'Valence', paste('POST_CCC_Test_Result_', suffix, '.csv', sep = ''), sep='_')))
  }
  #break
  #}
}  

if (noDomin != 1){
  write.csv(Cluster_1_kappa, file.path(sourceDir, videoPrefix, paste('Cluster-1', paste('Kappa_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(Cluster_2_kappa, file.path(sourceDir, videoPrefix, paste('Cluster-2', paste('Kappa_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(kappaResults, file.path(sourceDir, videoPrefix, paste('AllStimuli', paste('Kappa_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(Cluster_1_Res, file.path(sourceDir, videoPrefix, paste('Cluster-1', paste('CCC_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(Cluster_2_Res, file.path(sourceDir, videoPrefix, paste('Cluster-2', paste('CCC_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(allResults, file.path(sourceDir, videoPrefix, paste('AllStimuli', paste('CCC_Test_Result_', date, '.csv', sep = ''), sep='_')))
}else if (noDomin == 1){
  write.csv(Cluster_1_kappa, file.path(sourceDir, videoPrefix, paste('NoDominance_Cluster-1', paste('Kappa_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(Cluster_2_kappa, file.path(sourceDir, videoPrefix, paste('NoDominance_Cluster-2', paste('Kappa_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(kappaResults, file.path(sourceDir, videoPrefix, paste('NoDominance_AllStimuli', paste('Kappa_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(Cluster_1_Res, file.path(sourceDir, videoPrefix, paste('NoDominance_Cluster-1', paste('CCC_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(Cluster_2_Res, file.path(sourceDir, videoPrefix, paste('NoDominance_Cluster-2', paste('CCC_Test_Result_', date, '.csv', sep = ''), sep='_')))
  write.csv(allResults, file.path(sourceDir, videoPrefix, paste('NoDominance_AllStimuli', paste('CCC_Test_Result_', date, '.csv', sep = ''), sep='_')))
}



# emt2 = paste(strsplit(emt, ' ')[[1]], collapse = '_')
# emt2 = paste(strsplit(emt2, "'")[[1]], collapse = '_')
# #emt2 = paste(strsplit(emt2, "(")[[1]], collapse = '_')
# #emt2 = paste(strsplit(emt2, ')')[[1]], collapse = '_')
# emt2 = paste(strsplit(emt2, '&')[[1]], collapse = '_')
# if (grepl( '.w', emt2, fixed = TRUE)){
#   emt2 = strsplit(emt2, ".w")[[1]]
# }else if (grepl( '.m', emt2, fixed = TRUE)){
#   emt2 = strsplit(emt2, ".m")[[1]]
# }
# 
# if (sum(emt == clipsAre) | sum(emt2 == clipsAre)) {
#   SelectionMarker = 'S'
# }else{
#   SelectionMarker = 'NS'
# }