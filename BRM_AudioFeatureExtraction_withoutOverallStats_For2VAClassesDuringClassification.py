import glob
import re
import subprocess
#import emot
import os
from pyAudioAnalysis import ShortTermFeatures as sT
from visual_calling import visual_valence_arousal_vector
#import wave
from scipy.io import wavfile
## Lakshya Aur Abhyas

import numpy as np
import pickle
import matplotlib.pyplot as plt
#from joblib import Parallel, delayed
import pdb
import pandas as pd
from sklearn.decomposition import PCA     
from ANNClassification import svmClass               
import random
import scipy.stats as sts  
from knowingAboutBlocks import gettingBlockInformation
from matplotlib import gridspec

#import rpy2.robjects as ro
#from rpy2.robjects.packages import importr
#from rpy2.robjects.conversion import localconverter

# import R's "base" package
#base = importr('base')
# import R's "utils" package
#utils = importr('utils')
#import rpy2.robjects as robjects
#from rpy2.robjects import pandas2ri

############## Using R functionality in python
# Defining the R script and loading the instance in Python
#r = robjects.r

## IRRTest.R is use to calculate IRR. But this has to be called separately using the input being created by module multimediaFeatureCalculation()
#r['source']('IRRTest.R') 
# Loading the function we have defined in R.
#RStatsCalc_function_r = robjects.globalenv['IRRTest']
#OneSample_RStatsCalc_function_r = robjects.globalenv['RStatsCalcOneSample']

'''strToComp = ['Happy', 'Delighted', 'Excited', 'Aroused', 'Triumphant', 'Amused','Passionate', 'Joyous','Lust', 'Adventorous',
            'Alarmed', 'Tense', 'Angry', 'Afraid', 'Distress', 'Frustrated', 'Disgust', 'Distrustful', 'Hate', 'Startled',
            'Miserable', 'Melancholic', 'Sad', 'Depressed', 'Dissatisfied', 'Despondent', 'Taken Aback',
            'Pensive', 'Love', 'Pleased', 'Contented', 'Calm', 'Peaceful', 'Relaxed', 'Hopeful', 'Compassionate'] '''

strToComp = ['Adventorous', 'Afraid', 'Alarmed', 'Amused', 'Angry', 'Annoyed', 'Aroused', 'Ashamed', 'Astonished', 'Attraction', 'Brutality', 'Calm', 'Cheerful', 
'Compassionate', 'Contemplative', 'Contented', 'Convinced', 'Curious', 'DISTURBING', 'Delighted', 'Depressed', 'Despondent', 'Disgust', 'Dissatisfied', 'Distress', 'Distrustful', 'Droopy', 
'Enthusiastic', 'Excited', 'Frustrated', 'Funny', 'Gloomy', 'Happy', 'Hate', 'Hopeful', 'Impatient', 'Indignant', 'Insomnia', 'Joyous', 'Love', 'Lust', 'Melancholic', 'Miserable', 
'Passionate', 'Peaceful', 'Pensive', 'Pleased', 'Relaxed', 'Sad', 'Startled', 'Taken Aback', 'Tense', 'Tired', 'Triumphant']

shortName = {'Happy':'HPY', 'Delighted':'DLTD', 'Excited':'EXTD', 'Aroused':'ARSD', 'Triumphant':'TRFNT', 'Amused':'AMSD','Passionate':'PSNT', 'Joyous':'JYS','Lust':'LST', 'Adventorous':'ADVTS',
            'Alarmed':'ALRMD', 'Tense':'TNSE', 'Angry':'ANG', 'Afraid':'AFRD', 'Distress':'DSTRS', 'Frustrated':'FRSTD', 'Disgust':'DSGST', 'Distrustful':'DSTFL', 'Hate':'HAT', 'Startled':'STRTLD',
            'Miserable':'MSRBLE', 'Melancholic':'MLNKLC', 'Sad':'SAD', 'Depressed':'DPRSD', 'Dissatisfied':'DSTSFD', 'Despondent':'DSPNDT', 'Taken Aback':'TB',
            'Pensive':'PNSV', 'Love':'LOV', 'Pleased':'PLSD', 'Contented':'CNTD', 'Calm':'CLM', 'Peaceful':'PCFL', 'Relaxed':'RLXD', 'Hopeful':'HPFL', 'Compassionate':'CMPSN'}


videoPrefix = 'WithAllVideos_' ## WithAllVideos_, WithThirtyVideos_, With69Videos_, WithThirtyVideosEGI_, WithThirtyVideosEGIFinal_, WithClips_(this option is for clipwise classification)

if videoPrefix == 'WithAllVideos_':
    VideosFromDirectory = 1
    overallStatsFlag = 0 # It will created overall stats including mean, std and median for all scales, calculation of joint Standard deviation, marker to stimuli further selected and corresponding emotions.
else:
    VideosFromDirectory = 0
    overallStatsFlag = 0 # It will created overall stats including mean, std and median for all scales, calculation of joint Standard deviation, marker to stimuli further selected and corresponding emotions.    

if videoPrefix == "WithThirtyVideosEGIFinal_":  ### Rating Source Directory
    subDirs = '/mnt/7CBFA0EC210FC340/Processed_Emotions'
else:
    subDirs = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/' 

sourceDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey'
targetDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/NewTarget'

## Do analysis with only these set of videos.
if (videoPrefix == 'WithThirtyVideos_') or (videoPrefix == "WithThirtyVideosEGIFinal_") or (videoPrefix == "WithThirtyVideosEGI_"):
    clipDire = os.path.join('/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/block_For_30_Stimuli', 'Videos')
    targDire = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/target_clips_for_30_Stimuli'
    TargetclipFeat = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/target_clips_features_for_30_Stimuli'
    if not os.path.isdir(targDire):
        os.makedirs(targDire)
    if not os.path.isdir(TargetclipFeat):
        os.makedirs(TargetclipFeat)

elif videoPrefix == 'WithAllVideos_':
    clipDire = os.path.join(sourceDir, 'all_clips', 'Videos')
elif videoPrefix == 'With69Videos_':
    clipDire = os.path.join(sourceDir, 'Emotion_Name_Rating', 'Videos')


from moviepy.editor import *

def makingInputDataForClassificaiton(audioVideoFeatPCA, videoFeatureDf): #, valArl):

    global videoPrefix
    audioVideoFeatPCA.dropna(inplace=True)
    videoFeatureDf.dropna(inplace=True)
    #valArl.dropna(inplace=True)

    audioVideoCombined = pd.concat((videoFeatureDf, audioVideoFeatPCA), axis=1)

    if videoPrefix == 'WithClips_':
        Emotion_Name = [i.split('_')[2] for i in audioVideoFeatPCA['EmotionName']]
        audioVideoCombined['Emotion_Name'] = Emotion_Name
        audioVideoCombined.drop('EmotionName', axis=1, inplace=True)

    audioVideoCombined.drop('valence', axis=1, inplace=True)
    audioVideoCombined.drop('arousal', axis=1,  inplace=True)    
    print(len(audioVideoCombined))
    
    #classType = '4Quadrants'
    classType = '2Classes'

    if classType == '4Quadrants':

        '''duplicateIndexes = np.where(audioVideoCombined.duplicated().values)[0]
        audioVideoCombined.drop(duplicateIndexes, axis=0, inplace=True)'''
        audioVideoCombined.reset_index(drop=True, inplace=True)
        #audioVideoCombined.set_index('Emotion_Name', drop=True, inplace=True)
        target = audioVideoCombined.loc[:, 'target'].values.reshape(-1, 1)
        ############################################    
        #### If you want to remove 4 class (class with very less samples) selection remove4Class = 1; no_classes = 3, else remove4Class = 0; no_classes = 4
        remove4Class = 0
        no_classes = 4
        print(len(audioVideoCombined))
        ####### Removing HVLA Class. Since, it has less number of samples ############
        if remove4Class == 1:
            minSamplesIndex = np.argmin([sum(target==0), sum(target==1), sum(target==2), sum(target==3)])
            print('================= Classes to Remove = %s ====================' %str(minSamplesIndex))
            indexToRemove = audioVideoCombined.index.values[np.where(target==minSamplesIndex)[0]]
            audioVideoCombined.drop(indexToRemove, axis=0, inplace=True)
            audioVideoCombined.reset_index(drop=True, inplace=True)
            classList = np.arange(0, minSamplesIndex).tolist()
            classList.extend(np.arange(minSamplesIndex+1, 4).tolist())
        else:
            classList = [0, 1, 2, 3]
        ##################################

        print(len(audioVideoCombined))
        ValArlDim = audioVideoCombined.iloc[:, 0:-1].values
        target = audioVideoCombined.loc[:, 'target'].values.reshape(-1, 1)

        # Data split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(ValArlDim, target, test_size = 0.2, random_state = 0)

################################### Creating data for classification with minimum number of samples (which is present in any one class)
        print(sum(y_train==classList[0]))
        print(sum(y_train==classList[1]))
        print(sum(y_train==classList[2]))

        if remove4Class == 0:
            print(sum(y_train==classList[3]))
            minSamples = np.min([sum(y_train==classList[0]), sum(y_train==classList[1]), sum(y_train==classList[2]), sum(y_train==classList[3])])
            class1Samples = np.where(y_train==0)[0]
            class2Samples = np.where(y_train==1)[0]
            class3Samples = np.where(y_train==2)[0]
            class4Samples = np.where(y_train==3)[0]

            print('================== Testing =======================')
            print(sum(y_test==classList[0]))
            print(sum(y_test==classList[1]))
            print(sum(y_test==classList[2]))
            print(sum(y_test==classList[3]))
        else:
            minSamples = np.min([sum(y_train==classList[0]), sum(y_train==classList[1]), sum(y_train==classList[2])])
            class1Samples = np.where(y_train==classList[0])[0]
            class2Samples = np.where(y_train==classList[1])[0]
            class3Samples = np.where(y_train==classList[2])[0]
            print('================== Testing =======================')
            print(sum(y_test==classList[0]))
            print(sum(y_test==classList[1]))
            print(sum(y_test==classList[2]))

        maxRes = [0]
        svmAcc = 0
        import random

        flag = 0
        for repeat in np.arange(50):

            class1Idx = random.sample(class1Samples.tolist(), minSamples)
            class2Idx = random.sample(class2Samples.tolist(), minSamples)
            class3Idx = random.sample(class3Samples.tolist(), minSamples)            

            trainingData = X_train[class1Idx,:].copy()
            trainingData = np.concatenate((trainingData, X_train[class2Idx,:].copy()), axis=0)
            trainingData = np.concatenate((trainingData, X_train[class3Idx,:].copy()), axis=0)

            if remove4Class == 0:   
                class4Idx = random.sample(class4Samples.tolist(), minSamples)
                trainingData = np.concatenate((trainingData, X_train[class4Idx,:].copy()), axis=0)

            allTargets = y_train[class1Idx].copy()
            allTargets = np.concatenate((allTargets, y_train[class2Idx].copy()), axis=0)
            allTargets = np.concatenate((allTargets, y_train[class3Idx].copy()), axis=0)
            if remove4Class == 0:   
                allTargets = np.concatenate((allTargets, y_train[class4Idx].copy()), axis=0)

            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            trainingData = sc.fit_transform(trainingData)
            X_test = sc.fit_transform(X_test)

            svmDict = {0:'rbf', 1:'rbf_with_C-0.5', 2:'rbf_with_C-0.75'}
        #################### SVM Classification #############
            #print(sum(y_test==0))
            #print(sum(y_test==1))
            #print(sum(y_test==2))
            
            res = svmClass(trainingData, allTargets, X_test, y_test, classList)

            if flag == 0:            
                newOverallAcc = [res[2]]
                newF1Score = np.reshape(res[0], (1, no_classes))            #
                newCMatrix = np.reshape(res[1], (1,no_classes,no_classes))          #
                flag = 1            
            else:
                newOverallAcc.extend([res[2]])
                newF1Score = np.concatenate((newF1Score, np.reshape(res[0], (1, no_classes))), axis=0)
                newCMatrix = np.concatenate((newCMatrix, np.reshape(res[1], (1,no_classes,no_classes))), axis=0)


            '''if res[2] > svmAcc:     
                np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_f1_score.npy', res[0])
                np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_conf_mat.npy', res[1])
                np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_over_acc.npy', res[2])
                svmAcc = res[2]'''

        print('saving the results here')
        np.save('Mean_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_f1_score.npy', np.mean(newF1Score, axis=0))
        np.save('Mean_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_conf_mat.npy', np.mean(newCMatrix, axis=0))
        np.save('Mean_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_over_acc.npy', np.mean(newOverallAcc))

        np.save('Std_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_f1_score.npy', np.std(newF1Score, axis=0))
        np.save('Std_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_over_acc.npy', np.std(newOverallAcc))

############################################## Classification in valence and arousal only ########################################################

    if classType == '2Classes':
        DF = audioVideoCombined.copy()
        DF.reset_index(drop=True, inplace=True)
        
        '''duplicateIndexes = np.where(DF.duplicated().values)[0]        
        DF.drop(DF.index.values[duplicateIndexes], axis=0, inplace=True)
        DF.reset_index(drop=True, inplace=True)
        print(DF)'''

        #audioVideoCombined.set_index('Emotion_Name', drop=True, inplace=True)
        target = DF.loc[:, 'target'].values.reshape(-1, 1)
        ############################################    
        #### If you want to remove 4 class (class with very less samples) selection remove4Class = 1; no_classes = 3, else remove4Class = 0; no_classes = 4
        valenceFlag = 1 # if 1: valence calculation, else Arousal Calculation
        no_classes = 2
        print(len(audioVideoCombined))

        ################ For valence 
        valenceFeatColumns = ['max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3']
        valenceFeatColumns.extend(['val-'+str(i) for i in np.arange(15)])
        valFeatFrame = DF.loc[:, valenceFeatColumns].copy()

        valFeatFrame['target'] = DF['target'] 
        valFeatFrame["target"].replace({0:0, 3:0, 1:1, 2:1}, inplace=True)
        valtarget = valFeatFrame.loc[:, 'target'].values.reshape(-1, 1)
        valFeatDim = valFeatFrame.drop('target', axis=1).values

        ################ For arousal
        arousalFeatColumns = ['max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3']
        arousalFeatColumns.extend(['arl-'+str(i) for i in np.arange(15)])
        arlFeatFrame = DF.loc[:, arousalFeatColumns].copy()
        arlFeatFrame['target'] = DF['target'] 
        arlFeatFrame["target"].replace({0:0, 1:0, 2:1, 3:1}, inplace=True)
        arltarget = arlFeatFrame.loc[:, 'target'].values.reshape(-1, 1)
        arlFeatDim = arlFeatFrame.drop('target', axis=1).values
        ############################################

        if valenceFlag == 1:
            featValues = valFeatDim.copy()
            targetVal = valtarget.copy()
            prefix = 'valence'
        else:
            featValues = arlFeatDim.copy()
            targetVal = arltarget.copy()
            prefix = 'arousal'

        # Data split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(featValues, targetVal, test_size = 0.2, random_state = 0)

        print(sum(y_train==0))
        print(sum(y_train==1))

        print('================== Testing =======================')
        print(sum(y_test==0))
        print(sum(y_test==1))

        minSamples = np.min([sum(y_train==0), sum(y_train==1)])
        class1Samples = np.where(y_train==0)[0]
        class2Samples = np.where(y_train==1)[0]

        import random
        svmAcc = 0
        flag = 0
        for repeat in np.arange(50):

            class1Idx = random.sample(class1Samples.tolist(), minSamples)
            class2Idx = random.sample(class2Samples.tolist(), minSamples)

            trainingData = X_train[class1Idx,:].copy()
            trainingData = np.concatenate((trainingData, X_train[class2Idx,:].copy()), axis=0)

            allTargets = y_train[class1Idx].copy()
            allTargets = np.concatenate((allTargets, y_train[class2Idx].copy()), axis=0)

            # Feature scaling
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler()
            trainingData = sc.fit_transform(trainingData)
            X_test = sc.fit_transform(X_test)

            svmDict = {0:'rbf', 1:'rbf_with_C-0.5', 2:'rbf_with_C-0.75'}
        #################### SVM Classification #############
            #res = svmClass(trainingData, allTargets, X_test, y_test)
            res = svmClass(trainingData, allTargets, X_test, y_test, [0, 1])
            if flag == 0:            
                newOverallAcc = [res[2]]
                newF1Score = np.reshape(res[0], (1, no_classes))            #
                newCMatrix = np.reshape(res[1], (1,no_classes,no_classes))          #
                flag = 1            
            else:
                newOverallAcc.extend([res[2]])
                newF1Score = np.concatenate((newF1Score, np.reshape(res[0], (1, no_classes))), axis=0)
                newCMatrix = np.concatenate((newCMatrix, np.reshape(res[1], (1,no_classes,no_classes))), axis=0)


            '''if res[2] > svmAcc:     
                np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_f1_score.npy', res[0])
                np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_conf_mat.npy', res[1])
                np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_over_acc.npy', res[2])
                svmAcc = res[2]'''

        print('saving the results here')
        np.save('Mean_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_f1_score.npy', np.mean(newF1Score, axis=0))
        np.save('Mean_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_conf_mat.npy', np.mean(newCMatrix, axis=0))
        np.save('Mean_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_over_acc.npy', np.mean(newOverallAcc))

        np.save('Std_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_f1_score.npy', np.std(newF1Score, axis=0))
        np.save('Std_'+videoPrefix+'no_classes-%s_noSamples-%s' %(str(no_classes), str(res[3]))+'_svm_over_acc.npy', np.std(newOverallAcc))


def cutMovieClips():
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    videoFeatureDf = pd.DataFrame([], index=np.arange(1500), columns=['EmotionName', 'max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate_', 'min_shotRate_', 'mean_shotRate_', 'std_shotRate_', '_shotRate_percent_1', '_shotRate_percent_2', '_shotRate_percent_3', 'max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3'])

    if not os.path.isfile(os.path.join(subDirs, 'ClipWise_valArl.csv')):
        emotionWise = 0
        Ncomp = 15
        valCols = ['val-'+str(i) for i in np.arange(Ncomp)]
        arlCols = ['arl-'+str(i) for i in np.arange(Ncomp)]
        allCols = valCols.copy()
        allCols.extend(arlCols)
        allCols.extend(['valence', 'arousal', 'target', 'EmotionName'])  
        targetDict = {'HVHA':0,'LVHA':1,'LVLA':2,'HVLA':3}

        audioVideoFeatPCA = pd.DataFrame([], index=np.arange(1500), columns=allCols)
        valArl = pd.DataFrame([], index=np.arange(1500), columns=['CalcVal', 'CalcArl', 'RatdVal', 'RatdArl', 'EmotionName'])

        count = 0

        clickWiseRatings = pd.read_csv(os.path.join(subDirs, videoPrefix+'clickWiseRatings.csv'), index_col = 0)        
        for vid, sub, emt, clickN, clikT, val, arl in zip(clickWiseRatings['videoName'],clickWiseRatings['subjectName'],clickWiseRatings['EmotionName'],clickWiseRatings['ClickNo'],clickWiseRatings['ClickTime'],clickWiseRatings['Valence'],clickWiseRatings['Arousal']):
            
            if 'Taken Aback' in emt:
                emt = '_'.join(emt.split(' '))

            if 'neutral' in vid:
                continue

            if '[360p]' in vid:
                vid = vid.split('[360p]')[0]
            print(vid)
            try:
                vidFileName = glob.glob(os.path.join(clipDire, vid+'*'))[0]
            except:
                pdb.set_trace()

            start_time = clikT-7

            if start_time < 0:
                start_time = 0
                end_time = start_time + 7
            else:
                end_time = clikT

            targetFileName = os.path.join(targDire, sub+'_'+'click-%s' %str(clickN)+'_'+emt+'_'+vidFileName.split('/')[-1])
            fileNameEntry = targetFileName.split('/')[-1].split('.')[0]   
            new_audio_file = targetFileName.split('/')[-1].split('.')[0] + '.wav'

            if not os.path.isfile(targetFileName):
                # loading video dsa gfg intro video
                clip = VideoFileClip(vidFileName)              
                # getting subclip as video is large
                clip = clip.subclip(start_time, end_time)
                  
                # saving the clip
                print(targetFileName)
                clip.write_videofile(targetFileName)
                #ffmpeg_extract_subclip(vidFileName, start_time, end_time, targetname=targetFileName)

                command = "ffmpeg -y -i %s -ab 160k -ac 2 -ar 44100 -vn %s" %(targetFileName, os.path.join(targDire, new_audio_file))
                subprocess.call(command, shell=True)

    ############################ Calculation of Audio Features ###########################
                
            '''file_avlb_orig = glob.glob(os.path.join(targDire, new_audio_file))
            #print("Line 64: %s" %len(file_avlb_orig))        
            flag_track_file_name = 0 # this is tracking what king of modification is done in the file name
            
            #if 'Haasil_Movie_Dialogues_and_Scenes_Collection_' in new_name:
            #    pdb.set_trace()

            if not len(file_avlb_orig): # If no such audio file exists 
                try:
                    pdb.set_trace()

                except Exception as e:
                    pdb.set_trace()
                    print(new_name)
                    return 1

                fileName = os.path.join(targDire, new_audio_file)
                flag_track_file_name = 1

            else: # If audioFile exists but features are not calculated

                fileName = file_avlb_orig[0]'''

            fileName = os.path.join(targDire, new_audio_file)

            if 'The_Champ_1979_Death_Final_Ending_Scene_VERY_SAD' in fileName.split('/')[-1]:
                continue

            if not os.path.isfile(fileName):
                command = "ffmpeg -y -i %s -ab 160k -ac 2 -ar 44100 -vn %s" %(targetFileName, os.path.join(targDire, new_audio_file))
                subprocess.call(command, shell=True)            

            audio_file = wavfile.read(fileName)
            fs = audio_file[0]
            win = fs
            step = 0.2*fs
            signal = audio_file[1]

            if not os.path.isfile(os.path.join(TargetclipFeat, videoPrefix+new_audio_file.split('.wav')[0]+'.pkl')):
                #print('=========== Calculating for file = %s' %os.path.join(TargetclipFeat, videoPrefix+new_audio_file.split('.wav')[0]+'.pkl'))
                stFeatures = sT.feature_extraction(signal[:,1], sampling_rate=fs, window=win, step=step)               
                pickle.dump(stFeatures, open(os.path.join(TargetclipFeat, videoPrefix+new_audio_file.split('.wav')[0]+'.pkl'), 'wb'))
            else:
                stFeatures = pickle.load(open(os.path.join(TargetclipFeat, videoPrefix+new_audio_file.split('.wav')[0]+'.pkl'), 'rb'))

            try:
                rescaledFeat = 1+np.divide(((stFeatures[0].transpose()-np.min(stFeatures[0],1))*(9-1)), (np.max(stFeatures[0],1)-np.min(stFeatures[0],1)))
                rescaledFeat = rescaledFeat.transpose()
            except:
                pdb.set_trace()

            AudFeatures = rescaledFeat.copy()
            no_columns = AudFeatures.shape[1]
            no_rows = AudFeatures.shape[0]

            window_sec = no_columns/(signal.shape[0]/fs)
            window_minute = window_sec * 60
            distance_origin = []

            arousalIdx = [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
            arousalIdx.extend((np.array(arousalIdx)+34).tolist())

            valenceIdx = [3,4,8,9,10,11,12,13,14,15,16,17,18,19,20]
            valenceIdx.extend((np.array(valenceIdx)+34).tolist())                
            
            valence_group = AudFeatures[valenceIdx,:]                                                
            arousal_group = AudFeatures[arousalIdx,:]

            from sklearn.preprocessing import StandardScaler
            valence_group = np.transpose(valence_group)
            arousal_group = np.transpose(arousal_group)

            pcaV = PCA(n_components=Ncomp)
            ##print('======================================================================================')
            try:
                principalComponents = pcaV.fit(valence_group)
            except:
                pdb.set_trace()

            ##print(sum(principalComponents.explained_variance_ratio_))
            principalComponentsV = pcaV.fit_transform(valence_group)
            ##### Rescaling PCA transformed Data
            VrescaledPC = 1+np.divide(((principalComponentsV-np.min(principalComponentsV, 0))*(9-1)), (np.max(principalComponentsV, 0)-np.min(principalComponentsV, 0)))
            valence = np.sqrt(np.sum(np.power(np.mean(VrescaledPC,0),2)))

            pcaA = PCA(n_components=Ncomp)
            principalComponents = pcaA.fit(arousal_group)
            ##print(sum(principalComponents.explained_variance_ratio_))
            principalComponentsA = pcaA.fit_transform(arousal_group)
            ##### Rescaling PCA transformed Data
            ArescaledPC = 1+np.divide(((principalComponentsA-np.min(principalComponentsA, 0))*(9-1)), (np.max(principalComponentsA, 0)-np.min(principalComponentsA, 0)))
            arousal = np.sqrt(np.sum(np.power(np.mean(ArescaledPC,0),2)))

    ######################### This data-frame is being constructed for ANN Based prediction of rated valence and arousal from the calculated one.
            
            audioVideoFeatPCA.loc[count, 'EmotionName'] = fileNameEntry
            audioVideoFeatPCA.loc[count, valCols] = np.mean(VrescaledPC,0).tolist()
            audioVideoFeatPCA.loc[count, arlCols] = np.mean(ArescaledPC,0)

            ### If answer doesn't come rechange participantRating to participantRatingGB
            if emotionWise == 0:
                audioVideoFeatPCA.loc[count, 'valence'] = val
                audioVideoFeatPCA.loc[count, 'arousal'] = arl
            else:
                audioVideoFeatPCA.loc[count, 'valence'] = participantRatingGB.loc[count, ['Valence']].values[0]
                audioVideoFeatPCA.loc[count, 'arousal'] = participantRatingGB.loc[count, ['Arousal']].values[0]  

            val_ = audioVideoFeatPCA.loc[count, 'valence']
            arl_ = audioVideoFeatPCA.loc[count, 'arousal']
            if (val_ > 5.0) and (arl_ > 5.0):
                audioVideoFeatPCA.loc[count, 'target'] = targetDict['HVHA']
            if (val_ <= 5.0) and (arl_ > 5.0):
                audioVideoFeatPCA.loc[count, 'target'] = targetDict['LVHA']
            if (val_ <= 5.0) and (arl_ <= 5.0):
                audioVideoFeatPCA.loc[count, 'target'] = targetDict['LVLA']
            if (val_ > 5.0) and (arl_ <= 5.0):
                audioVideoFeatPCA.loc[count, 'target'] = targetDict['HVLA']
    #################################################################################################################

            valArl.loc[count, 'CalcVal'] = valence 
            valArl.loc[count, 'CalcArl'] = arousal 
            valArl.loc[count, 'RatdVal'] = val
            valArl.loc[count, 'RatdArl'] = arl
            valArl.loc[count, 'EmotionName'] = fileNameEntry

        ############################ Calculation of Video Features ###########################            
            
            max_motion_comp, min_motion_comp, mean_motion_comp, std_motion_comp, motion_percent_1, motion_percent_2, motion_percent_3, max_shotRate_, min_shotRate_, mean_shotRate_, std_shotRate_, _shotRate_percent_1, _shotRate_percent_2, _shotRate_percent_3, max_rhythm_comp, min_rhythm_comp, mean_rhythm_comp, std_rhythm_comp, rhythm_comp_percent_1, rhythm_comp_percent_2, rhythm_comp_percent_3, max_bright_array, min_bright_array, mean_bright_array, std_bright_array, bright_array_percent_1, bright_array_percent_2, bright_array_percent_3, max_shotRate, min_shotRate, mean_shotRate, std_shotRate, shotRate_percent_1, shotRate_percent_2, shotRate_percent_3 = visual_valence_arousal_vector(TargetclipFeat, targetFileName)   
            videoFeatureDf.loc[count, ['EmotionName', 'max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate_', 'min_shotRate_', 'mean_shotRate_', 'std_shotRate_', '_shotRate_percent_1', '_shotRate_percent_2', '_shotRate_percent_3', 'max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3']] = [fileNameEntry, max_motion_comp, min_motion_comp, mean_motion_comp, std_motion_comp, motion_percent_1, motion_percent_2, motion_percent_3, max_shotRate_, min_shotRate_, mean_shotRate_, std_shotRate_, _shotRate_percent_1, _shotRate_percent_2, _shotRate_percent_3, max_rhythm_comp, min_rhythm_comp, mean_rhythm_comp, std_rhythm_comp, rhythm_comp_percent_1, rhythm_comp_percent_2, rhythm_comp_percent_3, max_bright_array, min_bright_array, mean_bright_array, std_bright_array, bright_array_percent_1, bright_array_percent_2, bright_array_percent_3, max_shotRate, min_shotRate, mean_shotRate, std_shotRate, shotRate_percent_1, shotRate_percent_2, shotRate_percent_3]

            count = count + 1


        valArl.to_csv(os.path.join(subDirs, 'ClipWise_valArl.csv'))
        audioVideoFeatPCA.to_csv(os.path.join(subDirs, 'ClipWise_audioVideoFeatPCA.csv'))
        videoFeatureDf.to_csv(os.path.join(subDirs, 'ClipWise_videoFeatureDf.csv'))

    else:
        valArl = pd.read_csv(os.path.join(subDirs, 'ClipWise_valArl.csv'), index_col = 0)
        audioVideoFeatPCA = pd.read_csv(os.path.join(subDirs, 'ClipWise_audioVideoFeatPCA.csv'), index_col = 0)
        videoFeatureDf = pd.read_csv(os.path.join(subDirs, 'ClipWise_videoFeatureDf.csv'), index_col = 0)

#################################################### Doing it for the Classification ############################################################        
        makingInputDataForClassificaiton(audioVideoFeatPCA, videoFeatureDf, valArl)
###################################################################################

        valArl.dropna(axis=0, inplace=True)
        audioVideoFeatPCA.dropna(axis=0, inplace=True)
        valArl['Emotion_Name'] = [i.split('_')[2] for i in valArl['EmotionName']]
        audioVideoFeatPCA['Emotion_Name'] = [i.split('_')[2] for i in audioVideoFeatPCA['EmotionName']]

        valArl.set_index('Emotion_Name', drop=True, inplace=True)
        emotionWiseCorrelationV = pd.DataFrame([], index=np.arange(36), columns=['emtName', 'Test-Type', 'Stats', 'p-value', 'no_samples'])
        emotionWiseCorrelationA = pd.DataFrame([], index=np.arange(36), columns=['emtName', 'Test-Type', 'Stats', 'p-value', 'no_samples'])

        for idxx, i in enumerate(np.unique(valArl.index.values)):
            emotionwise = []
            emotionwise = valArl.loc[i, :].copy()

            if len(emotionwise)>20:
                emotionWiseCorrelationV.loc[idxx, 'emtName'] = i
                emotionWiseCorrelationA.loc[idxx, 'emtName'] = i
                emotionWiseCorrelationV.loc[idxx, 'no_samples'] = len(emotionwise)
                emotionWiseCorrelationA.loc[idxx, 'no_samples'] = len(emotionwise)                

                print('=============================+++++++++++++++++++++++=================================== Emotion = %s' %i)
                emotionwise['CalcVal'] = 1+np.divide((emotionwise['CalcVal']-np.min(emotionwise['CalcVal']))*(9-1), (np.max(emotionwise['CalcVal'])-np.min(emotionwise['CalcVal'])))
                emotionwise['CalcArl'] = 1+np.divide((emotionwise['CalcArl']-np.min(emotionwise['CalcArl']))*(9-1), (np.max(emotionwise['CalcArl'])-np.min(emotionwise['CalcArl'])))
                               
                print('================= With Audio Feature =================================') 
                #### Test of normality: This function tests the null hypothesis that a sample comes from a normal distribution.
                print('Calculated Valence = %s' %str(sts.normaltest(emotionwise['CalcVal'])))
                print('Calculated Arousal = %s' %str(sts.normaltest(emotionwise['CalcArl'])))
                print('Rated Valence = %s' %str(sts.normaltest(emotionwise['RatdVal'])))
                print('Rated Arousal = %s' %str(sts.normaltest(emotionwise['RatdArl'])))

                if (sts.normaltest(emotionwise['CalcVal'])[1] > 0.05) and (sts.normaltest(emotionwise['RatdVal'])[1] > 0.05): ## Normal Distribution
                    res = sts.pearsonr(emotionwise['CalcVal'], emotionwise['RatdVal'])
                    emotionWiseCorrelationV.loc[idxx, 'Test-Type'] = 'pearsonr'
                    emotionWiseCorrelationV.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationV.loc[idxx, 'p-value'] = res[1]
                    
                else:
                    res = sts.spearmanr(emotionwise['CalcVal'], emotionwise['RatdVal'])
                    emotionWiseCorrelationV.loc[idxx, 'Test-Type'] = 'spearmanr'
                    emotionWiseCorrelationV.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationV.loc[idxx, 'p-value'] = res[1]

                if (sts.normaltest(emotionwise['CalcArl'])[1] > 0.05) and (sts.normaltest(emotionwise['RatdArl'])[1] > 0.05): ## Normal Distribution
                    res = sts.pearsonr(emotionwise['CalcArl'], emotionwise['RatdArl'])
                    emotionWiseCorrelationA.loc[idxx, 'Test-Type'] = 'pearsonr'
                    emotionWiseCorrelationA.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationA.loc[idxx, 'p-value'] = res[1]                                        
                else:                
                    res = sts.spearmanr(emotionwise['CalcArl'], emotionwise['RatdArl'])
                    emotionWiseCorrelationA.loc[idxx, 'Test-Type'] = 'spearmanr'
                    emotionWiseCorrelationA.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationA.loc[idxx, 'p-value'] = res[1]      

                print('Number of videos = %s' %str(len(emotionwise)))

        emotionWiseCorrelationV.dropna(axis=0, inplace=True)
        emotionWiseCorrelationA.dropna(axis=0, inplace=True)
        emotionWiseCorrelationV.to_csv(os.path.join(subDirs, 'Clips_ForAudioFeatures_emotionWiseCorrelationV.csv'))
        emotionWiseCorrelationA.to_csv(os.path.join(subDirs, 'Clips_ForAudioFeatures_emotionWiseCorrelationA.csv'))

############################################################################## With Video

        videoFeatureDf.dropna(axis=0, inplace=True)
        
        valenceVideoFrame = videoFeatureDf.loc[:, ['EmotionName', 'max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3']]
        arousalVideoFrame = videoFeatureDf.loc[:, ['EmotionName', 'max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3']]

        valenceVideoFrame.set_index('EmotionName', drop=True, inplace=True)
        arousalVideoFrame.set_index('EmotionName', drop=True, inplace=True)

        columnsToDrop = valenceVideoFrame.columns.values[sum(valenceVideoFrame.values-np.mean(valenceVideoFrame.values,0))==0.0]
        columnsToDrop = columnsToDrop.tolist()
        #columnsToDrop.extend(['std_shotRate_', 'mean_shotRate_'])
        valenceVideoFrame.drop(columnsToDrop, axis=1, inplace=True)

        columnsToDrop = arousalVideoFrame.columns.values[sum(arousalVideoFrame.values-np.mean(arousalVideoFrame.values,0))==0]
        arousalVideoFrame.drop(columnsToDrop.tolist(), axis=1, inplace=True)

        Vrescaled = 1+np.divide(((valenceVideoFrame-np.min(valenceVideoFrame, 0))*(9-1)), (np.max(valenceVideoFrame, 0)-np.min(valenceVideoFrame, 0)))
        Arescaled = 1+np.divide(((arousalVideoFrame-np.min(arousalVideoFrame, 0))*(9-1)), (np.max(arousalVideoFrame, 0)-np.min(arousalVideoFrame, 0)))
        Vrescaled.dropna(axis=1, inplace=True)
        Arescaled.dropna(axis=1, inplace=True)
        ## DO range conversion and then PCA calculation

    #################################### Valence
        pcaV = PCA(n_components=5)
        try:
            principalComponents = pcaV.fit(Vrescaled.values)
        except:
            pdb.set_trace()

        print(sum(principalComponents.explained_variance_ratio_))
        principalComponentsV = pcaV.fit_transform(Vrescaled.values)
        ##### Rescaling PCA transformed Data
        VrescaledPC = 1+np.divide(((principalComponentsV-np.min(principalComponentsV, 0))*(9-1)), (np.max(principalComponentsV, 0)-np.min(principalComponentsV, 0)))
        
        valence = np.sqrt(np.sum(np.power(VrescaledPC, 2), 1))        
        #valence = 1+np.divide(((valence-np.min(valence, 0))*(9-1)), (np.max(valence, 0)-np.min(valence, 0)))
        
    #################################### Arousal        
        pcaV = PCA(n_components=5)
        principalComponents = pcaV.fit(Arescaled.values)
        print(sum(principalComponents.explained_variance_ratio_))
        principalComponentsA = pcaV.fit_transform(Arescaled.values)
        ##### Rescaling PCA transformed Data
        ArescaledPC = 1+np.divide(((principalComponentsA-np.min(principalComponentsA, 0))*(9-1)), (np.max(principalComponentsA, 0)-np.min(principalComponentsA, 0)))
        arousal = np.sqrt(np.sum(np.power(ArescaledPC, 2), 1)) 
        #arousal = 1+np.divide(((arousal-np.min(arousal, 0))*(9-1)), (np.max(arousal, 0)-np.min(arousal, 0)))

        audioVideoFeat = valArl
        Emotion_Name = audioVideoFeat.index.values 
        audioVideoFeat.set_index('EmotionName', drop=True, inplace=True)
        audioVideoFeat['Emotion_Name'] = Emotion_Name
        audioVideoFeat['valenceVid'] = 0
        audioVideoFeat['arousalVid'] = 0

        for emtVid in audioVideoFeat.index.values:
            audioVideoFeat.loc[emtVid, 'valenceVid'] = valence[np.where(emtVid==Vrescaled.index.values)[0][0]]
            audioVideoFeat.loc[emtVid, 'arousalVid'] = arousal[np.where(emtVid==Arescaled.index.values)[0][0]]

        emotionWiseCorrelationV = pd.DataFrame([], index=np.arange(36), columns=['emtName', 'Test-Type', 'Stats', 'p-value', 'no_samples'])
        emotionWiseCorrelationA = pd.DataFrame([], index=np.arange(36), columns=['emtName', 'Test-Type', 'Stats', 'p-value', 'no_samples'])
       
        print('================= With Video Feature =================================')
        audioVideoFeat.set_index('Emotion_Name', drop=True, inplace=True)

        for idxx, i in enumerate(np.unique(audioVideoFeat.index.values)):
            emotionwise = []
            emotionwise = audioVideoFeat.loc[i, :].copy()

            if len(emotionwise)>20:
                emotionWiseCorrelationV.loc[idxx, 'emtName'] = i
                emotionWiseCorrelationA.loc[idxx, 'emtName'] = i
                emotionWiseCorrelationV.loc[idxx, 'no_samples'] = len(emotionwise)
                emotionWiseCorrelationA.loc[idxx, 'no_samples'] = len(emotionwise)                                

                print('=============================+++++++++++++++++++++++=================================== Emotion = %s' %i)
                emotionwise['valenceVid'] = 1+np.divide((emotionwise['valenceVid']-np.min(emotionwise['valenceVid']))*(9-1), (np.max(emotionwise['valenceVid'])-np.min(emotionwise['valenceVid'])))
                emotionwise['arousalVid'] = 1+np.divide((emotionwise['arousalVid']-np.min(emotionwise['arousalVid']))*(9-1), (np.max(emotionwise['arousalVid'])-np.min(emotionwise['arousalVid'])))               
                
                #### Test of normality: This function tests the null hypothesis that a sample comes from a normal distribution.
                print('Calculated Valence = %s' %str(sts.normaltest(emotionwise['valenceVid'])))
                print('Calculated Arousal = %s' %str(sts.normaltest(emotionwise['arousalVid'])))
                print('Rated Valence = %s' %str(sts.normaltest(emotionwise['RatdVal'])))
                print('Rated Arousal = %s' %str(sts.normaltest(emotionwise['RatdArl'])))
                
                if (sts.normaltest(emotionwise['valenceVid'])[1] > 0.05) and (sts.normaltest(emotionwise['RatdVal'])[1] > 0.05): ## Normal Distribution
                    res = sts.pearsonr(emotionwise['valenceVid'], emotionwise['RatdVal'])
                    emotionWiseCorrelationV.loc[idxx, 'Test-Type'] = 'pearsonr'
                    emotionWiseCorrelationV.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationV.loc[idxx, 'p-value'] = res[1]                    
                else:
                    res = sts.spearmanr(emotionwise['valenceVid'], emotionwise['RatdVal'])
                    emotionWiseCorrelationV.loc[idxx, 'Test-Type'] = 'spearmanr'
                    emotionWiseCorrelationV.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationV.loc[idxx, 'p-value'] = res[1]                    

                if (sts.normaltest(emotionwise['arousalVid'])[1] > 0.05) and (sts.normaltest(emotionwise['RatdArl'])[1] > 0.05): ## Normal Distribution
                    res = sts.pearsonr(emotionwise['arousalVid'], emotionwise['RatdArl'])
                    emotionWiseCorrelationA.loc[idxx, 'Test-Type'] = 'pearsonr'
                    emotionWiseCorrelationA.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationA.loc[idxx, 'p-value'] = res[1]                    
                else:                
                    res = sts.spearmanr(emotionwise['arousalVid'], emotionwise['RatdArl'])
                    emotionWiseCorrelationA.loc[idxx, 'Test-Type'] = 'spearmanr'
                    emotionWiseCorrelationA.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationA.loc[idxx, 'p-value'] = res[1]                    

                print('Number of videos = %s' %str(len(emotionwise)))

        emotionWiseCorrelationV.dropna(axis=0, inplace=True)
        emotionWiseCorrelationV.to_csv(os.path.join(subDirs, 'Clips_ForVideoFeatures_emotionWiseCorrelationV.csv'))
        emotionWiseCorrelationA.dropna(axis=0, inplace=True)
        emotionWiseCorrelationA.to_csv(os.path.join(subDirs, 'Clips_ForVideoFeatures_emotionWiseCorrelationA.csv'))

    ###################### After combining the valence and arousal features #######################

        #weight = [0.88,0.12]        
        weight = [0.5,0.5]        
        audioVideoFeat['audioVisualAvgVal'] = ((weight[0]*audioVideoFeat['CalcVal']) + (weight[1]*audioVideoFeat['valenceVid']))
        audioVideoFeat['audioVisualAvgArl'] = ((weight[0]*audioVideoFeat['CalcArl']) + (weight[1]*audioVideoFeat['arousalVid']))

        emotionWiseCorrelationV = pd.DataFrame([], index=np.arange(36), columns=['emtName', 'Test-Type', 'Stats', 'p-value', 'no_samples'])
        emotionWiseCorrelationA = pd.DataFrame([], index=np.arange(36), columns=['emtName', 'Test-Type', 'Stats', 'p-value', 'no_samples'])

        print('================= For combined audio and visual features =================================')
        print(weight)

        for idxx, i in enumerate(np.unique(audioVideoFeat.index.values)):            
            emotionwise = []
            emotionwise = audioVideoFeat.loc[i, :].copy()

            if len(emotionwise)>20:
                emotionWiseCorrelationV.loc[idxx, 'emtName'] = i
                emotionWiseCorrelationA.loc[idxx, 'emtName'] = i
                emotionWiseCorrelationV.loc[idxx, 'no_samples'] = len(emotionwise)
                emotionWiseCorrelationA.loc[idxx, 'no_samples'] = len(emotionwise)                

                print('=============================+++++++++++++++++++++++=================================== Emotion = %s' %i)
                emotionwise['audioVisualAvgVal'] = 1+np.divide((emotionwise['audioVisualAvgVal']-np.min(emotionwise['audioVisualAvgVal']))*(9-1), (np.max(emotionwise['audioVisualAvgVal'])-np.min(emotionwise['audioVisualAvgVal'])))
                emotionwise['audioVisualAvgArl'] = 1+np.divide((emotionwise['audioVisualAvgArl']-np.min(emotionwise['audioVisualAvgArl']))*(9-1), (np.max(emotionwise['audioVisualAvgArl'])-np.min(emotionwise['audioVisualAvgArl'])))  

                #### Test of normality: This function tests the null hypothesis that a sample comes from a normal distribution.
                print('Calculated Valence = %s' %str(sts.normaltest(emotionwise['audioVisualAvgVal'])))
                print('Calculated Arousal = %s' %str(sts.normaltest(emotionwise['audioVisualAvgArl'])))

                if (sts.normaltest(emotionwise['audioVisualAvgVal'])[1] > 0.05) and (sts.normaltest(emotionwise['RatdVal'])[1] > 0.05): ## Normal Distribution
                    res = sts.pearsonr(emotionwise['audioVisualAvgVal'], emotionwise['RatdVal'])
                    emotionWiseCorrelationV.loc[idxx, 'Test-Type'] = 'pearsonr'
                    emotionWiseCorrelationV.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationV.loc[idxx, 'p-value'] = res[1]                      
                else:
                    res = sts.spearmanr(emotionwise['audioVisualAvgVal'], emotionwise['RatdVal'])
                    emotionWiseCorrelationV.loc[idxx, 'Test-Type'] = 'spearmanr'
                    emotionWiseCorrelationV.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationV.loc[idxx, 'p-value'] = res[1]                      

                if (sts.normaltest(emotionwise['audioVisualAvgArl'])[1] > 0.05) and (sts.normaltest(emotionwise['RatdArl'])[1] > 0.05): ## Normal Distribution
                    res = sts.pearsonr(emotionwise['audioVisualAvgArl'], emotionwise['RatdArl'])
                    emotionWiseCorrelationA.loc[idxx, 'Test-Type'] = 'pearsonr'
                    emotionWiseCorrelationA.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationA.loc[idxx, 'p-value'] = res[1]                      
                else:                
                    res = sts.spearmanr(emotionwise['audioVisualAvgArl'], emotionwise['RatdArl'])
                    emotionWiseCorrelationA.loc[idxx, 'Test-Type'] = 'spearmanr'
                    emotionWiseCorrelationA.loc[idxx, 'Stats'] = res[0]
                    emotionWiseCorrelationA.loc[idxx, 'p-value'] = res[1]                      

                print('Number of videos = %s' %str(len(emotionwise)))

        emotionWiseCorrelationV.dropna(axis=0, inplace=True)
        emotionWiseCorrelationV.to_csv(os.path.join(subDirs, 'Clips_ForCombinedAudioVideoFeatures_emotionWiseCorrelationV.csv'))
        emotionWiseCorrelationA.dropna(axis=0, inplace=True)
        emotionWiseCorrelationA.to_csv(os.path.join(subDirs, 'Clips_ForCombinedAudioVideoFeatures_emotionWiseCorrelationA.csv'))

def eegRatings():

    VideosWithEmotions = {}
    EmotionWiseStimuliRatingsWithSubjects = {}
    VideosWithEmotionsRatings = {} 
    EmotionWiseStimuliRatingsWithFamiliarity = pd.DataFrame([], index=np.arange(2000), columns=['EmotionName', 'HFLF', 'Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity'])   
    clickWiseRatings = pd.DataFrame([], index=np.arange(2000), columns=['videoName', 'subjectName', 'EmotionName', 'ClickNo', 'ClickTime', 'Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity'])
    clickWiseCount = 0

    global subDirs

    if videoPrefix == "WithThirtyVideosEGIFinal_":        
        subNames = [i.split('/')[-1] for i in glob.glob(os.path.join(subDirs, 'Done/mit*'))]
        feedbackFiles = 'csvFiles'
    else:
        feedbackFiles = 'data_For_30_Stimuli'       
        subNames = [i.split('/')[-1] for i in glob.glob(os.path.join(subDirs, feedbackFiles, '*'))]

    allRatings = pd.DataFrame([], columns=['Experiment_id','Valence','Arousal','Dominance','Liking','Familiarity'])

    ### These video names are as per the validation. Here I am mapping video names used in EEG experiment to used in validation.
    VideoDict = {'1':'Anacondas_The_Hunt_for_the_Blood_Orchid_clip', '3':'Arshad_Warsi_Bollywood_Comedy_[360p]', '4':'Best_Horror_Kills_Ghost_Ship_Opening_scene', 
    '2':'Anger_LBS', '5':'Brothers_9_10_Movie_CLIP_Sam_Loses_It_2009_HD', '6':'Catwoman_Basketball_Scene', '8':'excitingMohabbat_1', '9':'Fast_Furious_7_Plane_Scene', 
    '10':'Final_Race_of_Milkha_Singh_Career_[360p]', '12':'friendly', '13':'FunAlaBarfi_1', '15':'hateful1', '16':'hate_lbs', '17':'horror', '23':'MASOOM', 
    '24':'Milkha_Visits_His_Village_in_Pakistan_[360p]', '27':'The_Champ_1979_Death_Final_Ending_Scene_VERY_SAD',
    '7':'disgust','11':'First_Day_Of_My_Life','14':'funTamma_1','18':'How_To_Fight_Loneliness','19':'I_m_Yours',
    '20':'joyKolaveri_1','21':'loveNashe_1','22':'Madari_movie_of_best_scene_[360p]','25':'peaceful2','26':'SaddaHaq_1',
    '28':'The_Weight_Of_My_Words','29':'Titanic_2012_Sinking_Scene_HD_720p','30':'Titanic_Movie_Crash_Scene',
    'Arshad Warsi Bollywood Comedy [360p]':'Arshad_Warsi_Bollywood_Comedy_[360p]', 'Final Race of Milkha Singh Career [360p]':'Final_Race_of_Milkha_Singh_Career_[360p]',
    'First Day Of My Life':'First_Day_Of_My_Life','How To Fight Loneliness':'How_To_Fight_Loneliness', "I'm Yours":'I_m_Yours', 'Madari movie of best scene [360p]':'Madari_movie_of_best_scene_[360p]',
    'Milkha Visits His Village in Pakistan [360p]':'Milkha_Visits_His_Village_in_Pakistan_[360p]','The Weight Of My Words':'The_Weight_Of_My_Words'}

    for sub_ in subNames:
        subDir = os.path.join(subDirs, feedbackFiles, sub_)
        os.chdir(subDir)        

        try:        
            fileIndex = np.max(np.array([int(i.split('_')[0]) for i in glob.glob('*final_exp*.csv')]))
        except:
            pdb.set_trace()

        noStimShown = np.array([int(i.split('_')[-1].split('.csv')[0]) for i in glob.glob(str(fileIndex)+'_final_exp_*.csv')])

        maxStimShown = 0
        for i in noStimShown:
            if (i > maxStimShown) and (i < 50):
                maxStimShown = i

        ratings = pd.read_csv(os.path.join(subDir, glob.glob(str(fileIndex)+'_final_exp_*%s.csv' %str(maxStimShown))[0]))
        if 'MouseClickEvents' not in ratings.columns.values:
            continue    

        if len(ratings) < 10:
            pdb.set_trace()
        
        #print(subDir)
        #print(ratings)

        try:
            columnsAre = []
            for cll_ in ratings.columns.values:
                if 'filepath' in cll_:
                    columnsAre.extend([cll_])
                if ('valence' in cll_) or ('Valence' in cll_):
                    columnsAre.extend([cll_])
                if ('arousal' in cll_) or ('Arousal' in cll_):
                    columnsAre.extend([cll_])
                if ('dominance' in cll_) or ('Dominance' in cll_):
                    columnsAre.extend([cll_])
                if ('liking' in cll_) or ('Liking' in cll_):
                    columnsAre.extend([cll_])
                if ('familiarity' in cll_) or ('Familiarity' in cll_):
                    columnsAre.extend([cll_])

            #columnsAre.extend(['Emotion_Name'])          
            columnsAre.extend(['Emotion_Name', 'MouseClickEvents'])          
            ratings = ratings.loc[:, columnsAre]
        except:
            pdb.set_trace()

        ratings.drop(0, inplace=True)
        videoNames = [i.split('/')[-1].split('.')[0] for i in ratings['filepath']]
        
        newVideoNames = []

        for vid in videoNames:
            if vid in VideoDict.keys():
                newVideoNames.extend([VideoDict[vid]])
            else:
                newVideoNames.extend([vid])
        ratings['filepath'] = newVideoNames        

#################### Collecting emotion names for all the video stimuli ##################   
        stimuliCount = 1
        for fileN, emt_, _val, _arl, _dom, _lik, _fam, clicks in zip(ratings['filepath'], ratings['Emotion_Name'], ratings[columnsAre[1]], ratings[columnsAre[2]],
             ratings[columnsAre[3]], ratings[columnsAre[4]], ratings[columnsAre[5]], ratings['MouseClickEvents']):

            print(emt_)
            #if isinstance(emt_, str):
            if isinstance(emt_, str) and len(emt_) > 2:
                tmp = []

                ### Creating click array from string 
                clickArr = clicks.split(',')
                clickA = []
                print(len(clickArr))
                print(clicks)
                if len(clickArr) == 1:
                    clickA = [float(clickArr[0].split('[')[1].split(']')[0])]
                else:
                    clickA = [float(clickArr[0].split('[')[1])]  ### The first click
                    for clk in np.arange(1, len(clickArr)-1):
                        clickA.extend([float(clickArr[clk])])    ## In between clicks
                    clickA.extend([float(clickArr[-1].split(']')[0])])  ### The last click

                
                #if : ## Checking if emt_ is empty
                for _str in strToComp: ## Counting on pre-defined set of emotions               
                    if _str in emt_: ## Checking if emt_ has pre-defined emotions.
                        tmp.extend([_str]) ## Collecting all the emotions clicked by participants for each emotion stimuli

                        ###### Collecting the click number corresponding to the emotion
                        emtIndex = emt_.find(_str, 0, len(emt_))
                        for backIdx in np.arange(emtIndex, 0, -1):
                            if emt_[backIdx].isnumeric():
                                clickNumber = int(emt_[backIdx])
                                clickWiseRatings.loc[clickWiseCount, ['videoName', 'subjectName', 'EmotionName', 'ClickNo', 'ClickTime', 'Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity']] = [fileN, sub_, _str, clickNumber, clickA[clickNumber], _val, _arl, _dom, _lik, _fam]
                                #clickWiseCount = clickWiseCount + 1
                                break                        

                        if _fam < 2.5:
                            famKey = 'LF'
                        else:
                            famKey = 'HF'

############################## Emotion-Familiarity wise Ratings ##########################   
                        EmotionWiseStimuliRatingsWithFamiliarity.loc[clickWiseCount, ['EmotionName', 'HFLF', 'Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity']] = [_str, famKey, _val, _arl, _dom, _lik, _fam]

############################## Emotion-Subject wise Ratings ##########################
                        if _str not in EmotionWiseStimuliRatingsWithSubjects.keys():                            
                            EmotionWiseStimuliRatingsWithSubjects[_str] = {}
                            EmotionWiseStimuliRatingsWithSubjects[_str][sub_] = np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))
                        else:
                            if sub_ not in EmotionWiseStimuliRatingsWithSubjects[_str].keys():
                                EmotionWiseStimuliRatingsWithSubjects[_str][sub_] = np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))
                            else:
                                EmotionWiseStimuliRatingsWithSubjects[_str][sub_] = np.concatenate((EmotionWiseStimuliRatingsWithSubjects[_str][sub_], np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))), axis=0)

############################# Emotion wise Ratings ###################
                        if _str not in VideosWithEmotionsRatings.keys():
                            VideosWithEmotionsRatings[_str] = np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))
                        else:
                            VideosWithEmotionsRatings[_str] = np.concatenate((VideosWithEmotionsRatings[_str], np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))), axis=0) 

                        clickWiseCount = clickWiseCount + 1               

                if fileN not in VideosWithEmotions.keys():
                    VideosWithEmotions[fileN] = tmp            
                else:
                    VideosWithEmotions[fileN].extend(tmp)

                #pdb.set_trace()
                #ratings.loc[stimuliCount, 'Emotion_Name'] = tmp
                
###########################################################################################

        #ratings.drop('Emotion_Name', axis=1, inplace=True)
        ratings = ratings.rename({'filepath':'Experiment_id',columnsAre[1]:'Valence',columnsAre[2]:'Arousal',columnsAre[3]:'Dominance',columnsAre[4]:'Liking',columnsAre[5]:'Familiarity'}, axis=1) 
        #print(sub_)
        #print(ratings['Valence'].values)        
        pdb.set_trace()
        ratings['participant'] = sub_
        allRatings = pd.concat((allRatings, ratings), axis=0, ignore_index=True)

    ########### Familiarity based inter-rater reliability test. It will tell me which emotion needs the consideration of familiarity for reliable and consistent ratings.

    pdb.set_trace()
    clickWiseRatings.dropna(axis=0, inplace=True)
    clickWiseRatings.to_csv(os.path.join(subDirs, videoPrefix+'clickWiseRatings.csv'))

    EmotionWiseStimuliRatingsWithFamiliarity.dropna(axis=0, inplace=True)
    EmotionWiseStimuliRatingsWithFamiliarity.to_csv(os.path.join(subDirs, videoPrefix+'EmotionWiseStimuliRatingsWithFamiliarity.csv'))

    groupedEmtFamWise = EmotionWiseStimuliRatingsWithFamiliarity.groupby(by=['EmotionName', 'HFLF']).count()
    allEmts = groupedEmtFamWise.index.get_level_values('EmotionName')

    groupedEmtFam = pd.DataFrame([], index=np.unique(allEmts), columns=['HF', 'LF'])
    for _emt in allEmts:
        try:
            higF = groupedEmtFamWise.loc[[(_emt, 'HF')], ['Valence']].values[0][0]
            LowF = groupedEmtFamWise.loc[[(_emt, 'LF')], ['Valence']].values[0][0]
            if (LowF+higF)>20:
                groupedEmtFam.loc[_emt, 'HF'] = higF
                groupedEmtFam.loc[_emt, 'LF'] = LowF
        except:
            continue

    from matplotlib import gridspec
    plt.figure(figsize=(20,10))
    plt.rcParams['font.size']=35
    nRows = 2
    nCols = 1
    gs = gridspec.GridSpec(nRows, nCols, height_ratios=[10,1], width_ratios=np.ones(nCols), hspace=0.1, wspace=0.08, left=0.05, right=0.99, bottom=0.25, top=0.94)
    row = 0
    col = 0

    groupedEmtFam.dropna(inplace=True)
    ax=plt.subplot(gs[row, col])
    groupedEmtFam.plot.bar(ax=ax)
    plt.savefig(os.path.join(subDirs, videoPrefix+'FamiliarityWiseEmotionPlot.png'))
    plt.savefig(os.path.join(subDirs, videoPrefix+'FamiliarityWiseEmotionPlot.pdf'))
    plt.savefig(os.path.join(subDirs, videoPrefix+'FamiliarityWiseEmotionPlot.eps'))
    
    pdb.set_trace()

    emotionWiseDF = pd.DataFrame([], columns=['Valence','Arousal','Dominance','Liking','Familiarity'])
    for key_ in VideosWithEmotionsRatings.keys():
        
        df = pd.DataFrame(VideosWithEmotionsRatings[key_], index=[key_]*len(VideosWithEmotionsRatings[key_]), columns=['Valence','Arousal','Dominance','Liking','Familiarity'])
        emotionWiseDF = pd.concat((emotionWiseDF, df), axis=0)

    emotionWiseDF.to_csv(os.path.join(subDirs, videoPrefix+'EmotionWiseRatings.csv')) ## This contains scale ratings emotion-wise.

    toDel = np.where(np.array(allRatings['Experiment_id'].values)=='neutral')[0]
    allRatings.drop(toDel, inplace=True)
    pickle.dump(VideosWithEmotions, open(os.path.join(subDirs, videoPrefix+'VideosWithEmotionsEGI.pkl'), 'wb'))
    allRatings.to_csv(os.path.join(subDirs, videoPrefix+'allParticipantsRatingsEGI.csv'))
    #allRatings['count'] = 1
    #allRatings.groupby(by=['Experiment_id']).count()
    #allRatingsGB = allRatings.groupby(by=['Experiment_id']).mean()
    #pdb.set_trace()

directory = {'b-1':'block-1','b-2':'block-2','b-3':'block-3','b-4':'block-4','b-5':'block-5','b-6':'block-6','b-7':'block-7','b-8':'block-8','b-9':'block-9','b-10':'block-10',
'b-11':'block-11','b-12':'block-12','b-13':'block-13','b-14':'block-14','b-15':'block-15','b-16':'block-16','b-17':'block-17','b-18':'block-18','b-19':'block-19','b-20':'block-20','b-21':'block-21',
'b-22':'block-22','b-23':'block-23','b-24':'block-24'}

def multimediaFeatureCalculation():

    global videoPrefix    
    blockWiseDirs = glob.glob(os.path.join(os.getcwd(), 'Validation_Emotion', 'NewTarget', videoPrefix, 'b-*'))

    newBlockDir = []
    for blockId in np.arange(len(blockWiseDirs)):
        newBlockDir.extend([os.path.join('/'.join(blockWiseDirs[blockId].split('/')[0:-1]), 'b-%s' %str(blockId+1))])
    blockWiseDirs = newBlockDir.copy()
    del newBlockDir

    if not len(blockWiseDirs): #'/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/NewTarget/' WithAllVideos_/b-1'
        ## This module is basically creating Block wise information such as blockwise information about participants and stats. Then the output will be used to /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/IRRTest_KendallVegan_BlockWise.R        
        if videoPrefix == 'WithAllVideos_':
            date = 'Oct_10-Oct_20'
        else:
            date = 'Oct_10-Nov_15'

        ## date from Oct_10-Oct_20 For 200 videos
        ## Date from Oct_21-Nov_15 69 videos are rated. However, while calculating for 69 stimuli give the date from Oct_10-Nov_15.
        ## 30 stimuli are selected after completion of the validation study. So, for this also select date from Oct_10-Nov_15. 
        ## Basically while doing calculation for any stimuli we are taking data from whole period. In the code below we are explicitly considering 69 or 30 stimuli by using the variable "videosToConsider"

        if 'WithThirtyVideosEGI' in videoPrefix:
            emotionWiseDF_EGI = pd.read_csv(os.path.join(subDirs, videoPrefix+'EmotionWiseRatings.csv'), index_col=0)
            EGIRatings = pd.read_csv(os.path.join(subDirs, videoPrefix+'allParticipantsRatingsEGI.csv'), index_col=0)    
            VideosWithEmotions = pickle.load(open(os.path.join(subDirs, videoPrefix+'VideosWithEmotionsEGI.pkl'), 'rb')) ## Loading the video names given by participants during EEG experiment
        else:
            emotionWiseDF_EGI = pd.read_csv(os.path.join(subDirs, 'WithThirtyVideos_EmotionWiseRatings.csv'), index_col=0)    
            EGIRatings = pd.read_csv(os.path.join(subDirs, 'WithThirtyVideos_allParticipantsRatingsEGI.csv'), index_col=0)
            #VideosWithEmotions = pickle.load(open(os.path.join(subDirs, 'WithThirtyVideos_VideosWithEmotionsEGI.pkl'), 'rb')) ## Loading the video names given by participants during EEG experiment
            VideosWithEmotions = {}

        #EGIRatings['Valence'] = [float64(i) for i in EGIRatings['Valence'].values]
        #participantRating = pd.read_csv(os.path.join(sourceDir, 'Validation_Emotion/ToPublish/My_Experiment_Ratings_after_cleaning2018_Apr_24-Oct-31.csv'), index_col=0)
        #VideosWithEmotions = pickle.load(open(os.path.join(subDirs, videoPrefix+'VideosWithEmotionsEGI.pkl'), 'rb')) ## Loading the video names given by participants during EEG experiment

        ## The following file is created using program : /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/validation_data_analysis_bash.py
        if (videoPrefix == 'WithThirtyVideosEGIFinal_') or (videoPrefix == 'WithThirtyVideosEGI_'):
            participantRating = EGIRatings.copy()
        else:
            ## This file is created from /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/validation_data_analysis_bash.py
            participantRating = pd.read_csv(os.path.join(targetDir, 'My_Experiment_Ratings_after_cleaning2018_%s.csv' %date), index_col=0)
        #### The below file is create using eegRatings() module in this python file

        ### making all the videos without extension so that extension mismatch should not be a problem.
        withoutExt = participantRating['Experiment_id'].values    
        ################# Replacing all special characters with '_'
        withoutExt = ['_'.join(i.split(' ')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
        withoutExt = ['_'.join(i.split("'")) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
        withoutExt = ['_'.join(i.split('(')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
        withoutExt = ['_'.join(i.split(')')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
        withoutExt = ['_'.join(i.split('&')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file

        withoutExt = [i.split('.')[0] for i in withoutExt]
        participantRating.loc[:,'Experiment_id'] = withoutExt

        print(EGIRatings)
        print(participantRating)

        if (videoPrefix == 'With69Videos_') or ('WithThirtyVideos' in videoPrefix):  ## This is to select videos which had more than 35 ratings and less variations in the valence and arousal ratings.        
            videosToConsider = glob.glob(os.path.join(clipDire, '*'))
            videosToConsider = [i.split('/')[-1] for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split(' ')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split("'")) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split('(')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split(')')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split('&')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = [i.split('.')[0] for i in videosToConsider]            

            participantRating.set_index('Experiment_id', inplace=True)
            newFrame = pd.DataFrame([])
            
            print(participantRating)
            for vidStim in videosToConsider:
                newFrame = pd.concat((newFrame, participantRating.loc[vidStim, :]), axis=0)

            participantRating = newFrame.copy()
            del newFrame
            print(participantRating)
            participantRating['Experiment_id'] = participantRating.index.values
            participantRating.reset_index(drop=True, inplace=True)

        participantRating['count'] = 1
        participantRatingCount = participantRating.groupby(by=['Experiment_id']).count()    
        participantRatingMean = np.round(participantRating.groupby(by=['Experiment_id']).mean(), 2)
        participantRatingStd = np.round(participantRating.groupby(by=['Experiment_id']).std(), 2)
        participantRatingMedn = np.round(participantRating.groupby(by=['Experiment_id']).median(), 2)

    #########################################################################################################                    
        '''if not overallStatsFlag:
            if ('WithThirtyVideos' in videoPrefix) or (videoPrefix=='With69Videos_'):
                videosToConsider = [i.split('/')[-1] for i in glob.glob(os.path.join(clipDire, "*"))]

            allIndexes = []
            videoNameWithIndex = {}
            for _stm_ in videosToConsider:
                _stm_ = _stm_.split('.')[0]
                _stm_ = '_'.join(_stm_.split(' '))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split("'"))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split('('))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split(')'))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split('&'))  ### Renaming the experiment Ids taken from csv file
                allIndexes.extend(np.where(participantRating['Experiment_id'].values == _stm_)[0].tolist())
                videoNameWithIndex[_stm_] = np.where(participantRating['Experiment_id'].values == _stm_)[0].tolist()

            new = participantRating.iloc[np.array(allIndexes), :].copy()
            participantRating = new.copy()
            del new

            allIndexes = []
            videoNameWithIndex = {}
            for _stm_ in videosToConsider:
                _stm_ = _stm_.split('.')[0]
                _stm_ = '_'.join(_stm_.split(' '))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split("'"))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split('('))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split(')'))  ### Renaming the experiment Ids taken from csv file
                _stm_ = '_'.join(_stm_.split('&'))  ### Renaming the experiment Ids taken from csv file

                allIndexes.extend(np.where(EGIRatings['Experiment_id'].values == _stm_)[0].tolist())
                try:
                    videoNameWithIndex[_stm_] = np.where(EGIRatings['Experiment_id'].values == _stm_)[0].tolist()
                except:
                    pdb.set_trace()

            new = EGIRatings.iloc[np.array(allIndexes), :].copy()
            EGIRatings = new.copy()
            del new

        participantRating.reset_index(drop=True, inplace=True)
        print(EGIRatings)
        print(participantRating)'''

    #################### Collecting emotion names for all the video stimuli ##################   
        if overallStatsFlag: # Don't include Emotion Rated During EEG Experiment
            VideosWithEmotions = {}

        if ("WithThirtyVideosEGIFinal_" not in videoPrefix) and ("WithThirtyVideosEGI_" not in videoPrefix):
            VideosWithEmotionsRatings = {}
            #for fileN, emt_ in zip(participantRating['Experiment_id'], participantRating['Emotion_Name']):
            for fileN, emt_, _val, _arl, _dom, _lik, _fam in zip(participantRating['Experiment_id'], participantRating['Emotion_Name'], participantRating['Valence'], 
                participantRating['Arousal'], participantRating['Dominance'], participantRating['Liking'], participantRating['Familiarity']):    
                tmp = []

                if isinstance(emt_, float) or len(emt_) == 0:
                    continue

                notFoundFlag = 1
                for _str in strToComp:            
                    if _str.upper() in emt_.upper():
                        notFoundFlag = 0
                        tmp.extend([_str])
                        if _str not in VideosWithEmotionsRatings.keys():
                            VideosWithEmotionsRatings[_str] = np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))
                        else:
                            VideosWithEmotionsRatings[_str] = np.concatenate((VideosWithEmotionsRatings[_str], np.reshape(np.array([_val, _arl, _dom, _lik, _fam]), (1,5))), axis=0)                

                if notFoundFlag == 1:
                    print(emt_)
                
                if fileN not in VideosWithEmotions.keys():
                    VideosWithEmotions[fileN] = tmp
                else:
                    VideosWithEmotions[fileN].extend(tmp)

            if overallStatsFlag: # Including emotion category related information in the oveall statistics.
                sourceDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/NewTarget'
                MADFrame = pd.read_csv(os.path.join(sourceDir, 'MADFrame_'+date+'.csv'), index_col = 0)

                MADEmt = MADFrame.index.values
                MADEmt = [i.split('/')[-1] for i in MADFrame.index.values]  ### Renaming the experiment Ids taken from csv file
                MADEmt = ['_'.join(i.split(' ')) for i in MADEmt]  ### Renaming the experiment Ids taken from csv file
                MADEmt = ['_'.join(i.split("'")) for i in MADEmt]  ### Renaming the experiment Ids taken from csv file
                MADEmt = ['_'.join(i.split('(')) for i in MADEmt]  ### Renaming the experiment Ids taken from csv file
                MADEmt = ['_'.join(i.split(')')) for i in MADEmt]  ### Renaming the experiment Ids taken from csv file
                MADEmt = ['_'.join(i.split('&')) for i in MADEmt]  ### Renaming the experiment Ids taken from csv file
                MADEmt = [i.split('.')[0] for i in MADEmt]
                MADFrame['Experiment_id'] = MADEmt
                MADFrame.set_index('Experiment_id', drop=True, inplace=True)

                for vidStim in VideosWithEmotions.keys():
                    emts = np.unique(VideosWithEmotions[vidStim])
                    tmp = 0
                    
                    for _emt in emts:
                        factor = np.sum(np.array(VideosWithEmotions[vidStim])==_emt)/len(VideosWithEmotions[vidStim])
                        if tmp < factor:
                            tmp = factor
                            overallStats.loc[vidStim, 'emtFactor'] = round(factor,2)
                            overallStats.loc[vidStim, 'MostRated'] = _emt
                   
                    overallStats.loc[vidStim, 'NRatedEmotions'] = len(VideosWithEmotions[vidStim])
                    if len(MADFrame.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']]) == 1:
                        overallStats.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']] = MADFrame.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']]
                    else:
                        overallStats.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']] = MADFrame.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']].values[0]

                    overallStats.loc[vidStim, 'RatedEmotions'] = ', '.join(emts)

    ################################### Marking 69 videos here whiich were selected .

                if videoPrefix == 'WithAllVideos_':
                    sixtyNineVideosDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Emotion_Name_Rating/Videos'
                    sixyNineVideos = glob.glob(os.path.join(sixtyNineVideosDir, '*'))

                    sixyNineVideos = [i.split('/')[-1] for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
                    sixyNineVideos = ['_'.join(i.split(' ')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
                    sixyNineVideos = ['_'.join(i.split("'")) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
                    sixyNineVideos = ['_'.join(i.split('(')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
                    sixyNineVideos = ['_'.join(i.split(')')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
                    sixyNineVideos = ['_'.join(i.split('&')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
                    sixyNineVideos = [i.split('.')[0] for i in sixyNineVideos] 

                    for emot in sixyNineVideos:
                        overallStats.loc[emot, '69Videos'] = '#'

                    overallStats['Experiment_id'] = [' '.join(i.split('_')) for i in overallStats.index.values]    
                    overallStats.set_index('Experiment_id', drop=True, inplace=True)

                if VideosFromDirectory:
                    overallStats.to_csv('OverallStats_Stimuli_FromDirectory_%s.csv' %date)
                else:
                    overallStats.to_csv('OverallStats_Stimuli_%s.csv' %date)

            emotionWiseDF = pd.DataFrame([], columns=['Valence','Arousal','Dominance','Liking','Familiarity'])
            for key_ in VideosWithEmotionsRatings.keys():            
                df = pd.DataFrame(VideosWithEmotionsRatings[key_], index=[key_]*len(VideosWithEmotionsRatings[key_]), columns=['Valence','Arousal','Dominance','Liking','Familiarity'])
                emotionWiseDF = pd.concat((emotionWiseDF, df), axis=0)

        ###################### Plotting Emotion Category for each Video Stimuli ###############

            if not (date == 'Oct_10-Oct_20'):
                emotionWiseDF_All = pd.concat((emotionWiseDF_EGI, emotionWiseDF), axis=0)
                videoWithCategoryDF = pd.DataFrame(0, index=VideosWithEmotions.keys(), columns=strToComp)

                for key_ in VideosWithEmotions.keys():
                    if len(VideosWithEmotions[key_]) >= 10:  ## Taking only those video stimuli which are rated at least 10 time collectively in emotion categories
                        emtss = np.unique(VideosWithEmotions[key_])
                        for _emt in emtss:
                            videoWithCategoryDF.loc[key_, _emt] = sum(np.array(VideosWithEmotions[key_])==_emt)

                videoWithCategoryDF.to_csv(videoPrefix+'videoWithCategoryDF_WithAllVideoStimuli_%s.csv' %date)
                
                KeysToRemove = videoWithCategoryDF.index.values[np.sum(videoWithCategoryDF,1)==0]
                videoWithCategoryDF.drop(KeysToRemove, axis=0, inplace=True)
                videoWithCategoryDF.to_csv(videoPrefix+'videoWithCategoryDF_WithVideoStimuliWithAtleastTenCategories_%s.csv' %date)
                
                from matplotlib import gridspec
                plt.figure(figsize=(20,10))
                nRows = 3
                nCols = 3
                gs = gridspec.GridSpec(nRows, nCols, height_ratios=np.ones(nRows), width_ratios=np.ones(nCols), hspace=1, wspace=0.08, left=0.03, right=0.99, bottom=0.15, top=0.94)
                row = 0
                col = 0
                count = 0
                count2 = 1

                plt.rcParams['font.size']=20
                videoWithCategoryDF.rename(shortName, axis=1, inplace=True)

                for key_ in videoWithCategoryDF.index:
                    if 'neutral' not in key_:            
                        EmtRated = videoWithCategoryDF.columns[videoWithCategoryDF.loc[key_]>0]   

                        ax=plt.subplot(gs[row, col])
                        videoWithCategoryDF.loc[key_, EmtRated].plot.bar(ax=ax)
                        key_ = '_'.join(key_.split(' '))
                        ax.set_title('_'.join(key_.split('_')[:2]), loc='left')

                        count = count + 1
                        col = col + 1
                        if count % 3 == 0:
                            col = 0
                            row = row + 1

                        if count == 9:
                            plt.savefig(videoPrefix+'EmotionCategoriesPerStimulus-%s_%s.png' %(date, str(count2)))
                            plt.savefig(videoPrefix+'EmotionCategoriesPerStimulus-%s_%s.pdf' %(date, str(count2)))
                            plt.close()
                            plt.clf()

                            count2 = count2 + 1

                            plt.figure(figsize=(20,10))
                            nRows = 3
                            nCols = 3
                            gs = gridspec.GridSpec(nRows, nCols, height_ratios=np.ones(nRows), width_ratios=np.ones(nCols), hspace=1, wspace=0.08, left=0.03, right=0.99, bottom=0.15, top=0.94)
                            row = 0
                            col = 0
                            count = 0
                            plt.rcParams['font.size']=20

    ############################### Doing IRR Calculation, Block-wise grouping of videos and block wise statistics of thise groupings  ###################################
        
        #if videoPrefix == 'WithAllVideos_': 
        ## This file is created from /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/knowingAboutBlocks.py

        if videoPrefix == 'WithAllVideos_':
            dateForBlocks = 'Oct_10-Oct_20'
        else:
            dateForBlocks = 'Oct_10-Nov_15'

        ## For thirty videos, I need to consider    

        #if os.path.isfile(os.path.join('/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey', videoPrefix+'_BlockInformationForStimuli_%s.pkl' %dateForBlocks)):
        #    blockInformation = pickle.load(open(os.path.join('/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey', videoPrefix+'_BlockInformationForStimuli_%s.pkl' %dateForBlocks), 'rb'))
        #else:        
        blockInformation = gettingBlockInformation(videoPrefix, dateForBlocks)

        videoStim = np.unique(blockInformation[1])
        blockDF = pd.DataFrame([], index=videoStim, columns=['BlockInform'])

        blockToDel = ['block-'+str(j) for j in np.where(np.array([len(i) for i in blockInformation[0].values()])<5)[0]] ## Finding out blocks which has number of stimuli less than 5
        for idx_ in np.arange(len(blockToDel)):
            del blockInformation[0][blockToDel[idx_]]
            del blockInformation[2][blockToDel[idx_]]

        newBlock = {}
        participantBlock = {}
        #pdb.set_trace()
        for i, idx_ in enumerate(blockInformation[0].keys()): ### Creating dict with new blocks which has number of stimuli always greater than 5
            newBlock['block-'+str(i)] = blockInformation[0][idx_]
            participantBlock['block-'+str(i)] = blockInformation[2][idx_]
        
        #pdb.set_trace()
        ### Finding out 
        similarityFrame = pd.DataFrame(0, index=newBlock.keys(), columns=newBlock.keys())
        for key_1 in newBlock.keys():
            for key_2 in newBlock.keys():
                similarity = 0
                for vidName in newBlock[key_1]:
                    if vidName in newBlock[key_2] :
                        similarity = similarity + 1

                similarityFrame.loc[key_1, key_2] = similarity

        FramesToBeCombined = similarityFrame > 5
        ############# Now combining the blocks which has overlapping video stimuli
        alreadyProcessed = []
        if (np.sum(FramesToBeCombined) > 1).any():
            print(np.sum(FramesToBeCombined))
            FinalBlocks = {}
            participantsCombine = {}

            for index_ in FramesToBeCombined.index.values:
                #if int(index_.split('-')[1]) in alreadyProcessed:
                #    continue

                similarIdx = np.where(FramesToBeCombined.loc[index_] == True)[0]
                tmpp = []
                tmppParticipant = []

                alreadyProcessed.extend(similarIdx)
                print(similarIdx)
                for idx_ in similarIdx:
                    tmpp.extend(newBlock['block-%s' %str(idx_)])
                    tmppParticipant.extend(participantBlock['block-%s' %str(idx_)])                
                
                tmpp = np.unique(tmpp) 
                tmppParticipant = np.unique(tmppParticipant) 
                print(tmpp)
                print(tmppParticipant)

                #if (index_ == 'block-2') or (index_ == 'block-7'):
                #    pdb.set_trace()
                if len(FinalBlocks) > 0:
                    ## We want to know the if there is any key in FinalBlocks which has exactly same stimuli. If so no need to create a new block.
                    for i in FinalBlocks.keys():  ## Checking all keys of the disctionary for the same values
                        print(i)
                        #pdb.set_trace()
                        stimCount = 0  # Doing stimulation count for all keys.
                        for stimulus in tmpp:    
                            if (stimulus == np.array(FinalBlocks[i])).any():
                                stimCount = stimCount + 1                            
                            else:                                            
                                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Not found in block %s' %i) 
                                break

                        if stimCount == len(tmpp): # Already Available. If all the stimuli is already available in some block then no need to create a new block. However, participant of this block should be merged with the block having all the stimuli of this block.
                            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr Found in block %s' %i)
                            participantsCombine[str(i)].extend(tmppParticipant)
                            participantsCombine[str(i)] = np.unique(participantsCombine[str(i)]).tolist()
                            break

                    if (i == [j for j in FinalBlocks.keys()][-1]) and (stimCount != len(tmpp)): # If above loop has already reached to the last key and still stimcount is not equal to length of the stimuli then create a new block
                        FinalBlocks['block-'+str(len(FinalBlocks))] = tmpp
                        participantsCombine['block-'+str(len(participantsCombine))] = tmppParticipant.tolist()

                else:
                    FinalBlocks['block-'+str(len(FinalBlocks))] = tmpp
                    participantsCombine['block-'+str(len(participantsCombine))] = tmppParticipant.tolist()

            similarityFrame = pd.DataFrame(0, index=FinalBlocks.keys(), columns=FinalBlocks.keys())
            for key_1 in FinalBlocks.keys():
                for key_2 in FinalBlocks.keys():
                    similarity = 0
                    for vidName in FinalBlocks[key_1]:
                        if vidName in FinalBlocks[key_2] :
                            similarity = similarity + 1

                    similarityFrame.loc[key_1, key_2] = similarity

            FramesToBeCombined = similarityFrame > 5
       
        else:
            FinalBlocks = blockInformation[0].copy()
            participantsCombine = blockInformation[2].copy()
    #### This is finding the correspnding block 

        countThrs = 5
        if (videoPrefix == 'With69Videos_') or ('WithThirtyVideos' in videoPrefix):
            videosToConsider = ['_'.join(i.split(' ')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split("'")) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split('(')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split(')')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = ['_'.join(i.split('&')) for i in videosToConsider]  ### Renaming the experiment Ids taken from csv file
            videosToConsider = [i.split('.')[0] for i in videosToConsider]        

            considerTheBlock = []
            SixtyNineEmotions = []
            for key_ in FinalBlocks.keys():
                #if (key_ in ['block-1', 'block-8', 'block-17']): These are the blocks I foiund for thirty videos
                count = 0
                for vid in FinalBlocks[key_]:
                    if vid in videosToConsider:
                        count = count + 1

                #print('******************************************************')
                #print(count)
                #print(len(FinalBlocks[key_]))
                #print(abs(count - len(FinalBlocks[key_])))
                if (videoPrefix == 'With69Videos_'):
                    if count == len(FinalBlocks[key_]):
                        print('========= Found The %s' %key_)
                        considerTheBlock.extend([key_])
                        SixtyNineEmotions.extend(FinalBlocks[key_])

                if ('WithThirtyVideos_' in videoPrefix):
                    #if abs(count - len(FinalBlocks[key_]))<=5:
                    if count >= countThrs:
                        print('========= Found The %s' %key_)
                        considerTheBlock.extend([key_])
                        SixtyNineEmotions.extend(FinalBlocks[key_])

        else:
            considerTheBlock = FinalBlocks.keys()

        participantRating.reset_index(drop=True, inplace=True)
        howManyStimuliCovered = []
        numberOfFrames = 0

        VstatsResults = pd.DataFrame([])
        AstatsResults = pd.DataFrame([])
        DstatsResults = pd.DataFrame([])
        LstatsResults = pd.DataFrame([])
        FstatsResults = pd.DataFrame([])

        '''if ('WithThirtyVideos_' in videoPrefix):
            allVids = np.unique(participantRating['Experiment_id']).copy()            
            allPart = np.unique(participantRating['Participant_id']).copy()
            ratingFrame = pd.DataFrame(0, index=allVids, columns=allPart)

            for expId, partId in zip(participantRating['Experiment_id'].values, participantRating['Participant_id'].values):
                ratingFrame.loc[expId, partId] = 1

            partToDel = allPart[np.where(np.sum(ratingFrame.values, axis=0)<=3)[0]]
            ratingFrame.drop(partToDel, axis=1, inplace=True)
            partToConsider = ratingFrame.columns.values

            perplexities = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            plt.figure(figsize=(20,10))
            plt.rcParams['font.size']=35
            #perplexities = [20]#, 35, 40, 45]
            nRows = 1
            nCols = len(perplexities)
            gs = gridspec.GridSpec(nRows, nCols, width_ratios=np.ones(nCols), hspace=0.1, wspace=0.08, left=0.01, right=0.99, bottom=0.01, top=0.99)            

            from sklearn import manifold
            import sklearn.cluster

            n_clusters = 5
            for i, perplexity in enumerate(perplexities):
                tsne = manifold.TSNE(n_components=2, init='random',
                                     random_state=0, perplexity=20)
                Y = tsne.fit_transform(ratingFrame)
                
                clusterObject = sklearn.cluster.KMeans(n_clusters=n_clusters)
                clusters = clusterObject.fit(Y)

                print('======== With Perplexity = %s =============' %str(perplexity))

                for clustInd in np.arange(n_clusters):
                    cluster = ratingFrame.index.values[np.where(clusters.labels_==clustInd)[0]]                    
                    newDf = ratingFrame.loc[cluster, :]
                    print(len(cluster))
                    participantsAre = newDf.columns.values[np.where(np.sum(newDf.values, axis=0)>=(len(cluster)*0.6))[0]]
                    print(newDf.loc[cluster, participantsAre])

            blockToConsider = {}
            for i in np.arange(int(len(allVids)/10)):
                j = 1
                print('******************************************************')
                minMisses = 100000
                while j < 1000:
                    allVidsC = allVids.copy()
                    random.shuffle(allVidsC)
                    length = random.randint(8,15)
                    lower = i*length
                    upper = lower+length

                    newBlock = pd.DataFrame([], columns = participantRating.columns.values)
                    for vid_ in allVidsC[lower:upper]:
                        newBlock = pd.concat((newBlock, participantRating.loc[np.where(participantRating['Experiment_id'] == vid_)[0],:]), axis=0)     

                    ratingFrame = pd.DataFrame(1, index=np.unique(newBlock['Experiment_id'].values), columns=partToConsider)

                    for expId, partId in zip(newBlock['Experiment_id'].values, newBlock['Participant_id'].values):
                        if partId not in partToConsider:
                            continue
                        ratingFrame.loc[expId, partId] = 0

                    missingVals = sum(sum(ratingFrame.values))
                    diff_ = missingVals - (ratingFrame.shape[0]*ratingFrame.shape[1]) ## Notice this difference should be increasing because not rated is marked by one during the initialization of the ratingFrame.
                    print(missingVals, (ratingFrame.shape[0]*ratingFrame.shape[1]))
                    j = j + 1

                    if missingVals < minMisses:
                        minMisses = missingVals
                        blockToConsider[i] = allVidsC[lower:upper]

                pdb.set_trace()

        pdb.set_trace()'''

        for key_ in considerTheBlock:
    ################################## Selecting block wise stimuli
            newBlock = pd.DataFrame([], columns = participantRating.columns.values)
            for videoStim in FinalBlocks[key_]:
                '''if ('WithThirtyVideos_' in videoPrefix):
                    if videoStim not in videosToConsider:
                        continue'''
                newBlock = pd.concat((newBlock, participantRating.loc[np.where(participantRating['Experiment_id'] == videoStim)[0],:]), axis=0)
            
            '''if ('WithThirtyVideos_' in videoPrefix):
                if len(np.unique(newBlock['Experiment_id'])) < countThrs:
                    pdb.set_trace()'''

            newBlock.reset_index(drop=True, inplace=True)
    ################################## Selecting block wise participants
            partIndexes = []
            for partName in participantsCombine[key_]:
                partIndexes.extend(np.where(newBlock['Participant_id'] == partName)[0])

            newBlock = newBlock.loc[partIndexes, :].copy()
            newBlock.reset_index(drop=True, inplace=True)

            allRaters = np.unique(newBlock['Participant_id'])
            allStimul = np.unique(newBlock['Experiment_id'])
            howManyStimuliCovered.extend(allStimul)

            if len(allRaters) < 5:
                continue

            numberOfFrames = numberOfFrames + 1
            suffix = 'b-'+str(numberOfFrames)

            IRRDF_valence = pd.DataFrame([], index=allStimul, columns=allRaters)
            IRRDF_arousal = pd.DataFrame([], index=allStimul, columns=allRaters)
            IRRDF_dominan = pd.DataFrame([], index=allStimul, columns=allRaters)
            IRRDF_liking = pd.DataFrame([], index=allStimul, columns=allRaters)
            IRRDF_famili = pd.DataFrame([], index=allStimul, columns=allRaters)
            
            for stm_ in allStimul:
                indexes = np.where(newBlock['Experiment_id'].values==stm_)[0]
                try:
                    colss = newBlock.loc[indexes, 'Participant_id'].values
                    for idxx, col_ in enumerate(colss):
                        IRRDF_valence.loc[stm_, col_] = newBlock.loc[indexes, 'Valence'].values[idxx]
                        IRRDF_arousal.loc[stm_, col_] = newBlock.loc[indexes, 'Arousal'].values[idxx]
                        IRRDF_dominan.loc[stm_, col_] = newBlock.loc[indexes, 'Dominance'].values[idxx]
                        IRRDF_liking.loc[stm_, col_] = newBlock.loc[indexes, 'Liking'].values[idxx]
                        IRRDF_famili.loc[stm_, col_] = newBlock.loc[indexes, 'Familiarity'].values[idxx]
                except:
                    pdb.set_trace()
            
            if not os.path.exists(os.path.join(targetDir, videoPrefix)):
                os.makedirs(os.path.join(targetDir, videoPrefix))

            blank = pd.DataFrame([], index=[directory[suffix]], columns=['Nratings', 'mean', 'std', 'median', 'IQR'])
            statsVal = pd.DataFrame(0, index=IRRDF_valence.index.values, columns=['Nratings', 'mean', 'std', 'median', 'IQR'])      
            statsVal.loc[:, ['mean']] = np.round(np.nanmean(IRRDF_valence.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['std']] = np.round(np.sqrt(np.nanvar(IRRDF_valence.astype(np.float), axis=1).tolist()), 2)
            statsVal.loc[:, ['median']] = np.round(np.nanmedian(IRRDF_valence.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['Nratings']] = np.round(np.sum((IRRDF_valence.astype(np.float).fillna(0)!=0), axis=1).tolist(), 2)
            statsVal.loc[:, ['IQR']] = np.round(np.nanquantile(IRRDF_valence.astype(np.float), 0.75, axis=1)-np.nanquantile(IRRDF_valence.astype(np.float), 0.25, axis=1).tolist(), 2)
            statsVal.loc[:, 'Block'] = directory[suffix]
            #statsVal.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_STATS_valence_%s_%s.csv' %(suffix, key_, date)))        
            #print(statsVal)        
            #VstatsResults=pd.concat((VstatsResults, blank), axis=0)
            VstatsResults=pd.concat((VstatsResults, statsVal), axis=0)

            statsVal = pd.DataFrame(0, index=IRRDF_arousal.index.values, columns=['Nratings', 'mean', 'std', 'median'])        
            statsVal.loc[:, ['mean']] = np.round(np.nanmean(IRRDF_arousal.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['std']] = np.round(np.sqrt(np.nanvar(IRRDF_arousal.astype(np.float), axis=1).tolist()), 2)
            statsVal.loc[:, ['median']] = np.round(np.nanmedian(IRRDF_arousal.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['Nratings']] = np.round(np.sum((IRRDF_arousal.astype(np.float).fillna(0)!=0), axis=1).tolist(), 2)
            statsVal.loc[:, ['IQR']] = np.round(np.nanquantile(IRRDF_arousal.astype(np.float), 0.75, axis=1)-np.nanquantile(IRRDF_arousal.astype(np.float), 0.25, axis=1).tolist(), 2)
            statsVal.loc[:, 'Block'] = directory[suffix]
            #IRRDF_arousal.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_STATS_arousal_%s_%s.csv' %(suffix, key_, date)))

            #AstatsResults=pd.concat((AstatsResults, blank), axis=0)
            AstatsResults=pd.concat((AstatsResults, statsVal), axis=0)        
            #print(statsVal)

            statsVal = pd.DataFrame(0, index=IRRDF_dominan.index.values, columns=['Nratings', 'mean', 'std', 'median'])        
            statsVal.loc[:, ['mean']] = np.round(np.nanmean(IRRDF_dominan.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['std']] = np.round(np.sqrt(np.nanvar(IRRDF_dominan.astype(np.float), axis=1).tolist()), 2)
            statsVal.loc[:, ['median']] = np.round(np.nanmedian(IRRDF_dominan.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['Nratings']] = np.round(np.sum((IRRDF_dominan.astype(np.float).fillna(0)!=0), axis=1).tolist(), 2)
            statsVal.loc[:, ['IQR']] = np.round(np.nanquantile(IRRDF_dominan.astype(np.float), 0.75, axis=1)-np.nanquantile(IRRDF_dominan.astype(np.float), 0.25, axis=1).tolist(), 2)
            statsVal.loc[:, 'Block'] = directory[suffix]
            #IRRDF_dominan.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_STATS_dominan_%s_%s.csv' %(suffix, key_, date)))
            #print(statsVal)
            #DstatsResults=pd.concat((DstatsResults, blank), axis=0)
            DstatsResults=pd.concat((DstatsResults, statsVal), axis=0)        

            statsVal = pd.DataFrame(0, index=IRRDF_liking.index.values, columns=['Nratings', 'mean', 'std', 'median'])        
            statsVal.loc[:, ['mean']] = np.round(np.nanmean(IRRDF_liking.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['std']] = np.round(np.sqrt(np.nanvar(IRRDF_liking.astype(np.float), axis=1).tolist()), 2)
            statsVal.loc[:, ['median']] = np.round(np.nanmedian(IRRDF_liking.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['Nratings']] = np.round(np.sum((IRRDF_liking.astype(np.float).fillna(0)!=0), axis=1).tolist(), 2)
            statsVal.loc[:, ['IQR']] = np.round(np.nanquantile(IRRDF_liking.astype(np.float), 0.75, axis=1)-np.nanquantile(IRRDF_liking.astype(np.float), 0.25, axis=1).tolist(), 2)
            statsVal.loc[:, 'Block'] = directory[suffix]
            #IRRDF_liking.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_STATS_liking_%s_%s.csv' %(suffix, key_, date)))
            #print(statsVal)
            #LstatsResults=pd.concat((LstatsResults, blank), axis=0)
            LstatsResults=pd.concat((LstatsResults, statsVal), axis=0)        

            statsVal = pd.DataFrame(0, index=IRRDF_famili.index.values, columns=['Nratings', 'mean', 'std', 'median'])        
            statsVal.loc[:, ['mean']] = np.round(np.nanmean(IRRDF_famili.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['std']] = np.round(np.sqrt(np.nanvar(IRRDF_famili.astype(np.float), axis=1).tolist()), 2)
            statsVal.loc[:, ['median']] = np.round(np.nanmedian(IRRDF_famili.astype(np.float), axis=1).tolist(), 2)
            statsVal.loc[:, ['Nratings']] = np.round(np.sum((IRRDF_famili.astype(np.float).fillna(0)!=0), axis=1).tolist(), 2)
            statsVal.loc[:, ['IQR']] = np.round(np.nanquantile(IRRDF_famili.astype(np.float), 0.75, axis=1)-np.nanquantile(IRRDF_famili.astype(np.float), 0.25, axis=1).tolist(), 2)
            statsVal.loc[:, 'Block'] = directory[suffix]
            #IRRDF_famili.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_STATS_familiarity_%s_%s.csv' %(suffix, key_, date)))
            #print(statsVal)
            #FstatsResults=pd.concat((FstatsResults, blank), axis=0)
            FstatsResults=pd.concat((FstatsResults, statsVal), axis=0)        

            IRRDF_valence.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_IRRDF_valence_%s_%s.csv' %(suffix, key_, date)))
            IRRDF_arousal.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_IRRDF_arousal_%s_%s.csv' %(suffix, key_, date)))
            IRRDF_dominan.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_IRRDF_dominan_%s_%s.csv' %(suffix, key_, date)))
            IRRDF_liking.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_IRRDF_liking_%s_%s.csv' %(suffix, key_, date)))
            IRRDF_famili.to_csv(os.path.join(targetDir, videoPrefix, videoPrefix+'%s_IRRDF_familiarity_%s_%s.csv' %(suffix, key_, date)))

        ## Read these fines in R Program: IRRTest_KendallVegan.R and do the statistical analysis. Then read the results and re-arrange them using python file: .
        
        VstatsResults.to_csv(os.path.join(targetDir, videoPrefix, 'AllBlockStats', videoPrefix+'AllBlockStats_valence_%s.csv' %date))
        AstatsResults.to_csv(os.path.join(targetDir, videoPrefix, 'AllBlockStats', videoPrefix+'AllBlockStats_arousal_%s.csv' %date))
        DstatsResults.to_csv(os.path.join(targetDir, videoPrefix, 'AllBlockStats', videoPrefix+'AllBlockStats_dominance_%s.csv' %date))
        LstatsResults.to_csv(os.path.join(targetDir, videoPrefix, 'AllBlockStats', videoPrefix+'AllBlockStats_liking_%s.csv' %date))
        FstatsResults.to_csv(os.path.join(targetDir, videoPrefix, 'AllBlockStats', videoPrefix+'AllBlockStats_familiarity_%s.csv' %date))

        VstatsResults.rename({'mean':'Vmean', 'std':'Vstd', 'median':'Vmedian', 'IQR':'VIQR'}, inplace=True, axis=1)
        AstatsResults.rename({'mean':'Amean', 'std':'Astd', 'median':'Amedian', 'IQR':'AIQR'}, inplace=True, axis=1)
        DstatsResults.rename({'mean':'Dmean', 'std':'Dstd', 'median':'Dmedian', 'IQR':'DIQR'}, inplace=True, axis=1)
        LstatsResults.rename({'mean':'Lmean', 'std':'Lstd', 'median':'Lmedian', 'IQR':'LIQR'}, inplace=True, axis=1)
        FstatsResults.rename({'mean':'Fmean', 'std':'Fstd', 'median':'Fmedian', 'IQR':'FIQR'}, inplace=True, axis=1)

        allResults = VstatsResults    
        allResults = pd.concat((allResults, AstatsResults.loc[:, ['Amean', 'Astd', 'Amedian', 'AIQR']]), axis=1)
        allResults = pd.concat((allResults, DstatsResults.loc[:, ['Dmean', 'Dstd', 'Dmedian', 'DIQR']]), axis=1)
        allResults = pd.concat((allResults, LstatsResults.loc[:, ['Lmean', 'Lstd', 'Lmedian', 'LIQR']]), axis=1)
        allResults = pd.concat((allResults, FstatsResults.loc[:, ['Fmean', 'Fstd', 'Fmedian', 'FIQR']]), axis=1)
        allResults.to_csv(os.path.join(targetDir, videoPrefix, 'AllBlockStats', videoPrefix+'AllBlockStats_AllScales_%s.csv' %date))
        
        from AssigningIDs import assignVideoIds
        blockIdDict = assignVideoIds(videoPrefix)
        print(blockIdDict)        
        pickle.dump(blockIdDict, open(os.path.join(targetDir, videoPrefix, videoPrefix+'VideoIds_%s.pkl' %date), 'wb'))
        ##then call R program for IRR Test (Actually ICC test) IRRTest_KendallVegan_BlockWise.R

    else: 

        blockArr = []
        for blockId in np.arange(len(blockWiseDirs)):
            blockArr.extend([blockWiseDirs[blockId].split('/')[-1]])

        coeffDict = {}
        probDict = {}
        dimentDict = {}
        objectDict = {}
        clusterType = '_' #'_', '_Cluster-1_', '_Cluster-2_'

        for scl_ in ['Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity']:                        
            for blockId in np.arange(len(blockWiseDirs)):
                #if blockId == 6:
                #    pdb.set_trace()

                try:
                    #print(glob.glob(os.path.join(blockWiseDirs[blockId], videoPrefix+scl_+clusterType+'CCC*.csv'))[0])
                    cccRes = pd.read_csv(glob.glob(os.path.join(blockWiseDirs[blockId], videoPrefix+scl_+clusterType+'CCC*.csv'))[0], index_col=0)
                    print(blockWiseDirs[blockId].split('/')[-1])
                    print(cccRes)
                    dimm = cccRes['Dimension'].values[0].split(', ')[1]
                    objt = cccRes['Dimension'].values[0].split(', ')[0]

                    if cccRes['Prob.perm'].values[0] <= 0.01:
                        probMarker = '*'
                    elif (cccRes['Prob.perm'].values[0] > 0.01) and (cccRes['Prob.perm'].values[0] <= 0.05):
                        probMarker = '#'
                    else:
                        probMarker = ''

                    if 'Valence' in scl_:                    
                        coeffDict[blockWiseDirs[blockId].split('/')[-1]] = [cccRes['W'].values[0]]
                        probDict[blockWiseDirs[blockId].split('/')[-1]] = [probMarker]
                        dimentDict[blockWiseDirs[blockId].split('/')[-1]] = [dimm]
                        objectDict[blockWiseDirs[blockId].split('/')[-1]] = objt
                    else:
                        coeffDict[blockWiseDirs[blockId].split('/')[-1]].extend([cccRes['W'].values[0]])
                        probDict[blockWiseDirs[blockId].split('/')[-1]].extend([probMarker])
                        dimentDict[blockWiseDirs[blockId].split('/')[-1]].extend([dimm])
                        objectDict[blockWiseDirs[blockId].split('/')[-1]] = objt

                except:                    
                    if not len(glob.glob(os.path.join(blockWiseDirs[blockId], videoPrefix+scl_+clusterType+'CCC*.csv'))):
                        if 'Valence' in scl_:                    
                            coeffDict[blockWiseDirs[blockId].split('/')[-1]] = [0]
                            probDict[blockWiseDirs[blockId].split('/')[-1]] = ['']
                            dimentDict[blockWiseDirs[blockId].split('/')[-1]] = ['0']
                            if blockWiseDirs[blockId].split('/')[-1] not in objectDict.keys():
                                objectDict[blockWiseDirs[blockId].split('/')[-1]] = '0'
                        else:
                            coeffDict[blockWiseDirs[blockId].split('/')[-1]].extend([0])
                            probDict[blockWiseDirs[blockId].split('/')[-1]].extend([''])
                            dimentDict[blockWiseDirs[blockId].split('/')[-1]].extend(['0'])
                            if blockWiseDirs[blockId].split('/')[-1] not in objectDict.keys():
                                objectDict[blockWiseDirs[blockId].split('/')[-1]] = '0'
                    else:
                        pdb.set_trace()

        coefDf = pd.DataFrame.from_dict(coeffDict, orient='index', columns=['Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity'])
        meanConcord = np.mean(coefDf,axis=0)
        P_dataFrame = pd.DataFrame.from_dict(probDict, orient='index', columns=['Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity'])
        dimntDf = pd.DataFrame.from_dict(dimentDict, orient='index', columns=['Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity'])
        objectDf = pd.DataFrame.from_dict(objectDict, orient='index') #, columns=['Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity'])
        
        from matplotlib import gridspec
        fig = plt.figure(figsize=(20, 10))
        plt.rcParams['font.size'] = 35
        gs = gridspec.GridSpec(2, 1, height_ratios=[20,1], hspace=0.15, left=0.1, right=0.99, bottom=0.07, top=0.94) 
        ax = plt.subplot(gs[0, 0])   

        avgTxt = ';'.join(['Val-'+str(np.round(meanConcord['Valence'],2)), 'Arl-'+str(np.round(meanConcord['Arousal'],2)), 'Dom-'+str(np.round(meanConcord['Dominance'],2)), 
        'Lik-'+str(np.round(meanConcord['Liking'],2)), 'Fam-'+str(np.round(meanConcord['Familiarity'],2))])
        ax.set_title(avgTxt, fontsize=30)            

        coefDf.fillna(0, inplace=True)
        coefDf.plot.bar(ax=ax, width=0.8)
        #if 'WithAllVideos_' in videoPrefix:
        ax.set_yticks(np.arange(0, 1.3, 0.15))
        ax.set_xlabel('Stimuli Blocks')
        ax.set_ylabel('Kendall W Coefficient')   
        plt.legend(ncol=5, fontsize=25, fancybox=True, framealpha=0.1, bbox_to_anchor=(0.82, 0.83, 0.2, 0.2), columnspacing=0.5, markerscale=0.2)

        #pdb.set_trace()

        _row = -1
        _col = -1
        for pIdx, p_ in enumerate(ax.patches):
            if (pIdx%len(P_dataFrame)) == 0:                
                _row = -1
                _col = _col + 1
            _row = _row + 1    

            if _row == 7:
                print(objectDf.values[_row][0], dimntDf.values[_row][_col])        

            ax.annotate(format(P_dataFrame.values[_row][_col]), 
                           (p_.get_x() + p_.get_width() / 2., p_.get_height()+0.04), ha = 'center', va = 'center', 
                           size=20, xytext = (0, -8),  rotation='vertical', textcoords = 'offset points')        

            if 'Cluster' in clusterType:
                if 'WithAllVideos_' in videoPrefix:
                    if _col%2 == 0:                
                        ax.annotate(format(dimntDf.values[_row][_col]), 
                                           (p_.get_x() + p_.get_width()/2, 1.05), ha = 'center', va = 'center', size=20,xytext = (0, -12), textcoords = 'offset points')        
                    else:
                        ax.annotate(format(dimntDf.values[_row][_col]), 
                                           (p_.get_x() + p_.get_width()/2, 1.0), ha = 'center', va = 'center', size=20,xytext = (0, -12), textcoords = 'offset points')                        
                else:
                    if _col%2 == 0:                
                        ax.annotate(format(dimntDf.values[_row][_col]), 
                                           (p_.get_x() + p_.get_width()/2, 1.05), ha = 'center', va = 'center', size=25,xytext = (0, -12), textcoords = 'offset points')        
                    else:
                        ax.annotate(format(dimntDf.values[_row][_col]), 
                                           (p_.get_x() + p_.get_width()/2, 1.0), ha = 'center', va = 'center', size=25,xytext = (0, -12), textcoords = 'offset points')                                        
            else:
                if 'WithAllVideos_' in videoPrefix:
                    if _col == 0:                
                        ax.annotate(format(dimntDf.values[_row][_col]), 
                                           (p_.get_x() + 0.4, 1.05), ha = 'center', va = 'center', size=25,xytext = (0, -12), textcoords = 'offset points')        
                else:
                    if _col == 0:                
                        ax.annotate(format(dimntDf.values[_row][_col]), 
                                           (p_.get_x() + 0.4, 1.05), ha = 'center', va = 'center', size=25,xytext = (0, -12), textcoords = 'offset points')        

            if _col == 0:                
                if 'WithAllVideos_' in videoPrefix:                
                    ax.annotate(format(objectDf.values[_row][0]), 
                                       (p_.get_x() + 0.4, 1.1), ha = 'center', va = 'center', size=25,xytext = (0, -12), textcoords = 'offset points')        
                else:
                    ax.annotate(format(objectDf.values[_row][0]), 
                                       (p_.get_x() + 0.4, 1.1), ha = 'center', va = 'center', size=25,xytext = (0, -12), textcoords = 'offset points')                            
        
        plt.savefig(os.path.join(os.getcwd(), 'Validation_Emotion', 'NewTarget', videoPrefix, clusterType+videoPrefix+'BlockWise_InterRater_Agreement.png'))#, layout='tight')
        plt.savefig(os.path.join(os.getcwd(), 'Validation_Emotion', 'NewTarget', videoPrefix, clusterType+videoPrefix+'BlockWise_InterRater_Agreement.pdf'))#, layout='tight')     

        from AssigningIDs import assignVideoIds
        blockIdDict = assignVideoIds(videoPrefix)
        print(blockIdDict)        
        pickle.dump(blockIdDict, open(os.path.join(targetDir, videoPrefix, videoPrefix+'VideoIds_%s.pkl' %date), 'wb'))
        ##then call R program for IRR Test (Actually ICC test) IRRTest_KendallVegan_BlockWise.R
##############################################################################################################################################################

def corrBetCalculatedAndMeasureValenceArousal():

    ## This module is basically creating Block wise information such as blockwise information about participants and stats. Then the output will be used to /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/IRRTest_KendallVegan_BlockWise.R

    global videoPrefix    
    #videoPrefix == 'WithAllVideos_'

    if videoPrefix == 'WithAllVideos_':
        date = 'Oct_10-Oct_20'
    else:
        date = 'Oct_10-Nov_15'

    ## date from Oct_10-Oct_20 For 200 videos
    ## Date from Oct_21-Nov_15 69 videos are rated. However, while calculating for 69 stimuli give the date from Oct_10-Nov_15.
    ## 30 stimuli are selected after completion of the validation study. So, for this also select date from Oct_10-Nov_15. 
    ## Basically while doing calculation for any stimuli we are taking data from whole period. In the code below we are explicitly considering 69 or 30 stimuli by using the variable "videosToConsider"


    if 'WithThirtyVideosEGI' in videoPrefix:
        emotionWiseDF_EGI = pd.read_csv(os.path.join(subDirs, videoPrefix+'EmotionWiseRatings.csv'), index_col=0)
        EGIRatings = pd.read_csv(os.path.join(subDirs, videoPrefix+'allParticipantsRatingsEGI.csv'), index_col=0)    
        VideosWithEmotions = pickle.load(open(os.path.join(subDirs, videoPrefix+'VideosWithEmotionsEGI.pkl'), 'rb')) ## Loading the video names given by participants during EEG experiment
    else:
        emotionWiseDF_EGI = pd.read_csv(os.path.join(subDirs, 'WithThirtyVideos_EmotionWiseRatings.csv'), index_col=0)    
        EGIRatings = pd.read_csv(os.path.join(subDirs, 'WithThirtyVideos_allParticipantsRatingsEGI.csv'), index_col=0)
        #VideosWithEmotions = pickle.load(open(os.path.join(subDirs, 'WithThirtyVideos_VideosWithEmotionsEGI.pkl'), 'rb')) ## Loading the video names given by participants during EEG experiment
        VideosWithEmotions = {}

    #EGIRatings['Valence'] = [float64(i) for i in EGIRatings['Valence'].values]
    #participantRating = pd.read_csv(os.path.join(sourceDir, 'Validation_Emotion/ToPublish/My_Experiment_Ratings_after_cleaning2018_Apr_24-Oct-31.csv'), index_col=0)
    #VideosWithEmotions = pickle.load(open(os.path.join(subDirs, videoPrefix+'VideosWithEmotionsEGI.pkl'), 'rb')) ## Loading the video names given by participants during EEG experiment

    ## The following file is created using program : /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/validation_data_analysis_bash.py
    if (videoPrefix == 'WithThirtyVideosEGIFinal_') or (videoPrefix == 'WithThirtyVideosEGI_'):
        participantRating = EGIRatings.copy()
    else:
        ## This file is created from /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/validation_data_analysis_bash.py
        participantRating = pd.read_csv(os.path.join(targetDir, 'My_Experiment_Ratings_after_cleaning2018_%s.csv' %date), index_col=0)

#########################################################################################################
    
    participantRating = participantRating.loc[:, ['Experiment_id','Valence','Arousal','Dominance','Liking','Familiarity']]

    ### No consideration of EGI For Now.
    #participantRating = pd.concat((EGIRatings, participantRating), axis=0, ignore_index=True)
    participantRating['count'] = 1
    ## It is to select only those videos which has rating more than threshold.
    ratingThrs = 11
    Count_participantRatingGB = participantRating.groupby(by=['Experiment_id']).count()['count']
    print(len(Count_participantRatingGB[Count_participantRatingGB>ratingThrs]))

    participantRatingGB = participantRating.groupby(by=['Experiment_id']).mean() ## Groupby mean
    participantRatingGBStd = participantRating.groupby(by=['Experiment_id']).std() ## Groupby standard deviation
    participantRatingGB['ValenceStd'] = participantRatingGBStd['Valence']
    participantRatingGB['ArousalStd'] = participantRatingGBStd['Arousal']
    participantRatingGB['DominanceStd'] = participantRatingGBStd['Dominance']
    participantRatingGB['LikingStd'] = participantRatingGBStd['Liking']
    participantRatingGB['FamiliarityStd'] = participantRatingGBStd['Familiarity']


    videoStims = ['_'.join(i.split(' ')) for i in participantRating['Experiment_id']]  ### Renaming the experiment Ids taken from csv file
    videoStims = ['_'.join(i.split("'")) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
    videoStims = ['_'.join(i.split('(')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
    videoStims = ['_'.join(i.split(')')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
    videoStims = ['_'.join(i.split('&')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
    videoStims = [i.split('.')[0] for i in videoStims]  ### Renaming the experiment Ids taken from csv file

    participantRating['Experiment_id'] = videoStims
    participantRating.set_index('Experiment_id', drop=True, inplace=True)

    del participantRatingGBStd    
    participantRatingGB['Experiment_id'] = participantRatingGB.index.values
    participantRatingGB.drop(Count_participantRatingGB[Count_participantRatingGB<ratingThrs].index.values, inplace=True)

    Ncomp = 15
    valCols = ['val-'+str(i) for i in np.arange(Ncomp)]
    arlCols = ['arl-'+str(i) for i in np.arange(Ncomp)]
    allCols = valCols.copy()
    allCols.extend(arlCols)
    allCols.extend(['valence', 'arousal', 'target'])  
    targetDict = {'HVHA':0,'LVHA':1,'LVLA':2,'HVLA':3}
    #targetDictRev = {0:'HVHA',1:'LVHA',2:'LVLA',3:'HVLA'}


    if not os.path.isfile(os.path.join(targetDir, videoPrefix+'AudioFeatureBasedValenceArousalCalculation.csv')):
    #if os.path.isfile(os.path.join(targetDir, videoPrefix+'AudioFeatureBasedValenceArousalCalculation.csv')):
        ### This is to remove any extra special character from video name in the file
        videoNames = np.unique(participantRatingGB.index.values)
        videoNames = ['_'.join(i.split(' ')) for i in videoNames]  ### Renaming the experiment Ids taken from csv file
        videoNames = ['_'.join(i.split("'")) for i in videoNames]  ### Renaming the experiment Ids taken from csv file
        videoNames = ['_'.join(i.split('(')) for i in videoNames]  ### Renaming the experiment Ids taken from csv file
        videoNames = ['_'.join(i.split(')')) for i in videoNames]  ### Renaming the experiment Ids taken from csv file
        videoNames = ['_'.join(i.split('&')) for i in videoNames]  ### Renaming the experiment Ids taken from csv file
        videoNames = [i.split('.')[0] for i in videoNames]  ### Renaming the experiment Ids taken from csv file
        
        valArl = pd.DataFrame(0, index=videoNames, columns=['CalcVal', 'CalcArl', 'RatdVal', 'RatdArl', 'RValStd', 'RArlStd']) ## For audio features
        ### For video Features.
        videoFeatureDf = pd.DataFrame([], index=videoNames, columns=['max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate_', 'min_shotRate_', 'mean_shotRate_', 'std_shotRate_', '_shotRate_percent_1', '_shotRate_percent_2', '_shotRate_percent_3', 'max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3'])

        participantRatingGB['Experiment_id'] = videoNames  ## Only names are being assigned here.
        participantRatingGB = participantRatingGB.set_index('Experiment_id', drop=True)

        emotionWise = 0 ## If emotionWise = 0: We have grouped emotion wise ratings of all the participants else we didn't group participants ratings and taking them individually/

        if emotionWise == 0:
            audioVideoFeatPCA = pd.DataFrame(0, index=participantRating.index.values, columns=allCols) # If considering participant rating wise
        else:
            audioVideoFeatPCA = pd.DataFrame(0, index=participantRatingGB.index.values, columns=allCols) # If considering stimulus videos wise
        
        TargetclipDire = os.path.join(sourceDir, videoPrefix+'targetClips')
        if not os.path.exists(TargetclipDire):
            os.makedirs(TargetclipDire)

        clipsDir = glob.glob(os.path.join(clipDire, '*'))
        #clipsDir.extend(glob.glob(os.path.join(clipDire, '*.mp4')))
       
        for clip_ in clipsDir:

            old_name = clip_.split('/')[-1]
            new_name = '_'.join(old_name.split(' ')) ### Renaming the experiment Ids
            new_name = '_'.join(new_name.split("'")) ### Renaming the experiment Ids
            new_name = '_'.join(new_name.split('(')) ### Renaming the experiment Ids
            new_name = '_'.join(new_name.split(')')) ### Renaming the experiment Ids
            new_name = '_'.join(new_name.split('&')) ### Renaming the experiment Ids            

            os.rename(os.path.join(clipDire, old_name), os.path.join(clipDire, new_name)) ## Renaming the files without special characters in them
            
            if videoPrefix == 'WithAllVideos_':
                newName2 = new_name
            else:
                newName2 = new_name.split('.')[0]

            if any(newName2 == np.array(videoNames)): ## Comparing the file without extension.       

########################## Calculating Visual Features #####################
                max_motion_comp, min_motion_comp, mean_motion_comp, std_motion_comp, motion_percent_1, motion_percent_2, motion_percent_3, max_shotRate_, min_shotRate_, mean_shotRate_, std_shotRate_, _shotRate_percent_1, _shotRate_percent_2, _shotRate_percent_3, max_rhythm_comp, min_rhythm_comp, mean_rhythm_comp, std_rhythm_comp, rhythm_comp_percent_1, rhythm_comp_percent_2, rhythm_comp_percent_3, max_bright_array, min_bright_array, mean_bright_array, std_bright_array, bright_array_percent_1, bright_array_percent_2, bright_array_percent_3, max_shotRate, min_shotRate, mean_shotRate, std_shotRate, shotRate_percent_1, shotRate_percent_2, shotRate_percent_3 = visual_valence_arousal_vector(TargetclipDire, os.path.join(clipDire, new_name))
                videoFeatureDf.loc[newName2, ['max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate_', 'min_shotRate_', 'mean_shotRate_', 'std_shotRate_', '_shotRate_percent_1', '_shotRate_percent_2', '_shotRate_percent_3', 'max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3']] = [max_motion_comp, min_motion_comp, mean_motion_comp, std_motion_comp, motion_percent_1, motion_percent_2, motion_percent_3, max_shotRate_, min_shotRate_, mean_shotRate_, std_shotRate_, _shotRate_percent_1, _shotRate_percent_2, _shotRate_percent_3, max_rhythm_comp, min_rhythm_comp, mean_rhythm_comp, std_rhythm_comp, rhythm_comp_percent_1, rhythm_comp_percent_2, rhythm_comp_percent_3, max_bright_array, min_bright_array, mean_bright_array, std_bright_array, bright_array_percent_1, bright_array_percent_2, bright_array_percent_3, max_shotRate, min_shotRate, mean_shotRate, std_shotRate, shotRate_percent_1, shotRate_percent_2, shotRate_percent_3]


################ Extracting Audio Channel ####################
                song_title = newName2
                new_audio_file = song_title + '.wav'
                        
                file_avlb_orig = glob.glob(os.path.join(TargetclipDire, new_audio_file))
                flag_track_file_name = 0 # this is tracking what king of modification is done in the file name
                
                if not len(file_avlb_orig): # If no such audio file exists 
                    try:
                        print("Line 72 VideoFile %s being converted to AudioFile %s" %(os.path.join(clipDire, new_name), os.path.join(TargetclipDire, new_audio_file)))
                        command = "ffmpeg -y -i %s -ab 160k -ac 2 -ar 44100 -vn %s" %(os.path.join(clipDire, new_name), os.path.join(TargetclipDire, new_audio_file))
                        subprocess.call(command, shell=True)
                    except Exception as e:
                        pdb.set_trace()
                        print(new_name)
                        return 1
                        '''print("new_name %s is not found for song_title %s" %(new_name, song_title))            
                        raise ValueError'''

                    fileName = os.path.join(TargetclipDire, new_audio_file)
                    flag_track_file_name = 1

                else: # If audioFile exists but features are not calculated

                    fileName = file_avlb_orig[0]
                # Calculating Audio Features here for the new entry
                try:
                    audio_file = wavfile.read(fileName)
                except:
                    print(fileName)
                    return 1
                fs = audio_file[0]
                win = fs
                step = 0.2*fs
                signal = audio_file[1]
                
                if signal.shape[0] > (55*fs): # If the media is more than one minute

########################### Extracting Audio Features ##########################
                    if not os.path.isfile(os.path.join(TargetclipDire, new_audio_file.split('.wav')[0]+'.pkl')):
                        stFeatures = sT.feature_extraction(signal[:,1], sampling_rate=fs, window=win, step=step)               
                        pickle.dump(stFeatures, open(os.path.join(TargetclipDire, new_audio_file.split('.wav')[0]+'.pkl'), 'wb'))
                    else:
                        stFeatures = pickle.load(open(os.path.join(TargetclipDire, new_audio_file.split('.wav')[0]+'.pkl'), 'rb'))

                    ### Check if this function is taking place properly.
                    #np.sum(np.sum(stFeatures[0]))
                    try:
                        rescaledFeat = 1+np.divide(((stFeatures[0].transpose()-np.min(stFeatures[0],1))*(9-1)), (np.max(stFeatures[0],1)-np.min(stFeatures[0],1)))
                        rescaledFeat = rescaledFeat.transpose()
                    except:
                        pdb.set_trace()
                    #print(np.mean(rescaledFeat,1), np.std(rescaledFeat,1))
                    #print(np.min(rescaledFeat,1), np.max(rescaledFeat,1))
                    #AudFeatures = np.divide((rescaledFeat-np.reshape(feature_mean,(rescaledFeat.shape[0],1))), np.reshape(feature_std,(rescaledFeat.shape[0],1)))
                    AudFeatures = rescaledFeat.copy()
                    no_columns = AudFeatures.shape[1]
                    no_rows = AudFeatures.shape[0]
                    #print("Line 80: Shape of the feature is column: %d and row %d" %(no_columns,no_rows))
                    '''
                    window_sec = 11945/(5269504/44100) = 100
                    window_minute = 100 * 60 = 6000'''
                    window_sec = no_columns/(signal.shape[0]/fs)
                    window_minute = window_sec * 60
                    distance_origin = []
                    '''stFeatures is a matrix of 34 rows. These rows are according to the article(pyAudioAnalysis: An open-source python library for audio signal analysis)
                    : 1-ZCR, 2-Energy, 3-Entropy of energy, 4-Spectral centroid, 5-Spectral spread, 6-Spectral Entropy, 7-Spectral flux, 8-Spectral Rolloff, 9-21: MFCC's,
                    22-33: Chroma Vector, 34-Chroma deviation'''
                    '''Grouping is being done according to the google doc(Affective Content Analysis)
                    Valence Grouping is:4,5,9-21, 
                    Arousal Grouping is:1,2,3,6,7,8,9-21,22-34'''
                    
                    #audFeat = ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy', 'spectral_flux', 'spectral_rolloff', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 'mfcc_12', 'mfcc_13', 'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12', 'chroma_std']
    #, 'delta zcr', 'delta energy', 
    #'delta energy_entropy', 'delta spectral_centroid', 'delta spectral_spread', 'delta spectral_entropy', 'delta spectral_flux', 'delta spectral_rolloff', 
    #'delta mfcc_1', 'delta mfcc_2', 'delta mfcc_3', 'delta mfcc_4', 'delta mfcc_5', 'delta mfcc_6', 'delta mfcc_7', 'delta mfcc_8', 'delta mfcc_9', 'delta mfcc_10',
    #'delta mfcc_11', 'delta mfcc_12', 'delta mfcc_13', 'delta chroma_1', 'delta chroma_2', 'delta chroma_3', 'delta chroma_4', 'delta chroma_5', 'delta chroma_6',
    #'delta chroma_7', 'delta chroma_8', 'delta chroma_9', 'delta chroma_10', 'delta chroma_11', 'delta chroma_12', 'delta chroma_std']

                    arousalIdx = [0,1,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
                    arousalIdx.extend((np.array(arousalIdx)+34).tolist())

                    valenceIdx = [3,4,8,9,10,11,12,13,14,15,16,17,18,19,20]
                    valenceIdx.extend((np.array(valenceIdx)+34).tolist())                                    

                    valence_group = AudFeatures[valenceIdx,:]                                                
                    arousal_group = AudFeatures[arousalIdx,:]

                    from sklearn.preprocessing import StandardScaler
                    valence_group = np.transpose(valence_group)
                    arousal_group = np.transpose(arousal_group)
                    #valence_group = StandardScaler().fit_transform(valence_group)
                    #arousal_group = StandardScaler().fit_transform(arousal_group)

########### Performing dimension reduction here to less number of components covering more than 95% variance #############
                    pcaV = PCA(n_components=Ncomp)
                    principalComponents = pcaV.fit(valence_group)
                    ##print(sum(principalComponents.explained_variance_ratio_))
                    principalComponentsV = pcaV.fit_transform(valence_group)
                    ##### Rescaling PCA transformed Data
                    VrescaledPC = 1+np.divide(((principalComponentsV-np.min(principalComponentsV, 0))*(9-1)), (np.max(principalComponentsV, 0)-np.min(principalComponentsV, 0)))
                    valence = np.sqrt(np.sum(np.power(np.mean(VrescaledPC,0),2)))

                    pcaA = PCA(n_components=Ncomp)
                    principalComponents = pcaA.fit(arousal_group)
                    ##print(sum(principalComponents.explained_variance_ratio_))
                    principalComponentsA = pcaA.fit_transform(arousal_group)
                    ##### Rescaling PCA transformed Data
                    ArescaledPC = 1+np.divide(((principalComponentsA-np.min(principalComponentsA, 0))*(9-1)), (np.max(principalComponentsA, 0)-np.min(principalComponentsA, 0)))
                    arousal = np.sqrt(np.sum(np.power(np.mean(ArescaledPC,0),2)))

######################### This data-frame is being constructed for ANN Based prediction of rated valence and arousal from the calculated one.
                    
                    audioVideoFeatPCA.loc[newName2, valCols] = np.mean(VrescaledPC,0).tolist()  # Taking mean across all the window segments in which audio signal is automatically divided.
                    audioVideoFeatPCA.loc[newName2, arlCols] = np.mean(ArescaledPC,0)

                    ### If answer doesn't come rechange participantRating to participantRatingGB
                    #### Storing rated values to the data frame.
                    if emotionWise == 0:
                        try:
                            audioVideoFeatPCA.loc[newName2, 'valence'] = participantRating.loc[newName2, ['Valence']].values
                        except:
                            pdb.set_trace()

                        audioVideoFeatPCA.loc[newName2, 'arousal'] = participantRating.loc[newName2, ['Arousal']].values
                    else:
                        audioVideoFeatPCA.loc[newName2, 'valence'] = participantRatingGB.loc[newName2, ['Valence']].values[0]
                        audioVideoFeatPCA.loc[newName2, 'arousal'] = participantRatingGB.loc[newName2, ['Arousal']].values[0]  

#################### Encoding the valence-arousal quadrants in 0, 1, 2, and 3 ##################
                    targetName = []
                    if emotionWise == 0:
                        for val_, arl_ in zip(audioVideoFeatPCA.loc[newName2, 'valence'], audioVideoFeatPCA.loc[newName2, 'arousal']):                        
                            if (val_ > 5.0) and (arl_ > 5.0):
                                targetName.extend([targetDict['HVHA']])
                            if (val_ <= 5.0) and (arl_ > 5.0):
                                targetName.extend([targetDict['LVHA']])
                            if (val_ <= 5.0) and (arl_ <= 5.0):
                                targetName.extend([targetDict['LVLA']])
                            if (val_ > 5.0) and (arl_ <= 5.0):
                                targetName.extend([targetDict['HVLA']])

                        audioVideoFeatPCA.loc[newName2, 'target'] = targetName

                    else:
                        val_ = audioVideoFeatPCA.loc[newName2, 'valence']
                        arl_ = audioVideoFeatPCA.loc[newName2, 'arousal']
                        if (val_ > 5.0) and (arl_ > 5.0):
                            audioVideoFeatPCA.loc[newName2, 'target'] = targetDict['HVHA']
                        if (val_ <= 5.0) and (arl_ > 5.0):
                            audioVideoFeatPCA.loc[newName2, 'target'] = targetDict['LVHA']
                        if (val_ <= 5.0) and (arl_ <= 5.0):
                            audioVideoFeatPCA.loc[newName2, 'target'] = targetDict['LVLA']
                        if (val_ > 5.0) and (arl_ <= 5.0):
                            audioVideoFeatPCA.loc[newName2, 'target'] = targetDict['HVLA']

#################################################################################################################

                    valArl.loc[newName2, 'CalcVal'] = valence 
                    valArl.loc[newName2, 'CalcArl'] = arousal 
                    try:
                        valArl.loc[newName2, 'RatdVal'] = participantRatingGB.loc[newName2, ['Valence']].values[0]
                        valArl.loc[newName2, 'RatdArl'] = participantRatingGB.loc[newName2, ['Arousal']].values[0]
                        valArl.loc[newName2, 'RValStd'] = participantRatingGB.loc[newName2, ['ValenceStd']].values[0]
                        valArl.loc[newName2, 'RArlStd'] = participantRatingGB.loc[newName2, ['ArousalStd']].values[0] * 15 ## Scaling so the size of the dots in scatter plot is increased                        
                    except:
                        if len(participantRatingGB.loc[newName2, ['Valence']].values) > 1:
                            valArl.loc[newName2, 'RatdVal'] = participantRatingGB.loc[newName2, ['Valence']].values[0][0]
                            valArl.loc[newName2, 'RatdArl'] = participantRatingGB.loc[newName2, ['Arousal']].values[0][0]
                            valArl.loc[newName2, 'RValStd'] = participantRatingGB.loc[newName2, ['ValenceStd']].values[0][0]
                            valArl.loc[newName2, 'RArlStd'] = participantRatingGB.loc[newName2, ['ArousalStd']].values[0][0] * 15 ## Scaling so the size of the dots in scatter plot is increased                        
                        else:
                            pdb.set_trace()

                            
                else:                
                    print(newName2)
                    print(signal.shape[0]/fs)
                    #pdb.set_trace()  
            else:            
                print(newName2)
                print(newName2 in videoNames)                
        
        if emotionWise == 0:
            videoFeat_class = pd.DataFrame([], columns=videoFeatureDf.columns.values)
            videoFeat_class['target'] = audioVideoFeatPCA['target']
            videoFeat_class.drop('target', axis=1, inplace=True)
            for _key in videoFeatureDf.index.values:
                try:                    
                    videoFeat_class.loc[_key] = videoFeatureDf.loc[_key,:].values
                except:
                    if videoFeatureDf.loc[_key,:].values.shape[0] > 1:
                       videoFeat_class.loc[_key] = videoFeatureDf.loc[_key,:].values[0] 
                    else:
                        pdb.set_trace()

            allFeatures = pd.concat((videoFeat_class, audioVideoFeatPCA), axis=1)
            allFeatures.dropna(axis=0, inplace=True)
            allFeatures.to_csv(os.path.join(targetDir, videoPrefix+'ForClassificaion_AllMultimediaCalculatedPCADimensionsValenceArousal.csv'))  ## This file will be used for ANN classification
        else:
            allFeatures = pd.concat((videoFeatureDf, audioVideoFeatPCA), axis=1)
            allFeatures.dropna(axis=0, inplace=True)
            allFeatures.to_csv(os.path.join(targetDir, videoPrefix+'ForClassificaion_AllMultimediaCalculatedPCADimensionsValenceArousalGB.csv'))  ## This file will be used for ANN classification            

        valenceVideoFrame = videoFeatureDf.loc[:, ['max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3']]
        arousalVideoFrame = videoFeatureDf.loc[:, ['max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3']]
        #valenceVideoFrame.dropna(axis=0, inplace=True)
        #arousalVideoFrame.dropna(axis=0, inplace=True)

        valArl.drop(valArl.index.values[np.sum(valArl.values,1)==0], inplace=True)
        valArl.to_csv(os.path.join(targetDir, videoPrefix+'AudioFeatureBasedValenceArousalCalculation.csv'))

        #### For video: take different from visual and audio features.
        valenceVideoFrame.to_csv(os.path.join(targetDir, videoPrefix+'VideoFeatureBasedValenceCalculation.csv'))
        arousalVideoFrame.to_csv(os.path.join(targetDir, videoPrefix+'VideoFeatureBasedArousalCalculation.csv'))

    else:
        print('=========== VideoPrefix = %s ============' %videoPrefix)
        #valArl_WithThirtyVideos = pd.read_csv(os.path.join(clipDire, 'WithThirtyVideos_AudioFeatureBasedValenceArousalCalculation.csv'), index_col=0)
        valArl = pd.read_csv(os.path.join(targetDir, videoPrefix+'AudioFeatureBasedValenceArousalCalculation.csv'), index_col=0)
        #valArl = pd.read_csv(os.path.join(clipDire, 'AudioFeatureBasedValenceArousalCalculation (copy).csv'), index_col=0)
        valArl['Experiment_id'] = [i.split('.')[0] for i in  valArl.index.values]
        valArl.set_index('Experiment_id', drop=True, inplace=True)
        valArlCopy = valArl.copy()

        #valArl_WithThirtyVideos.drop(['disgust', 'loveNashe_1'], axis=0, inplace=True)
        #valArlCopy = valArl.loc[valArl_WithThirtyVideos.index.values, :].copy()
        #valArlCopy.drop_duplicates(inplace=True)

        valArlCopy['CalcVal'] = 1+np.divide((valArlCopy['CalcVal']-np.min(valArlCopy['CalcVal']))*(9-1), (np.max(valArlCopy['CalcVal'])-np.min(valArlCopy['CalcVal'])))
        valArlCopy['CalcArl'] = 1+np.divide((valArlCopy['CalcArl']-np.min(valArlCopy['CalcArl']))*(9-1), (np.max(valArlCopy['CalcArl'])-np.min(valArlCopy['CalcArl'])))
               
        from matplotlib import gridspec
        plt.figure(figsize=(20,10))
        plt.rcParams['font.size']=35
        nRows = 2
        nCols = 1
        gs = gridspec.GridSpec(nRows, nCols, height_ratios=[20,1], width_ratios=np.ones(nCols), hspace=0.1, wspace=0.08, left=0.05, right=0.99, bottom=0.12, top=0.98)
        ax=plt.subplot(gs[:, 0])
        
        ax.scatter(valArlCopy['CalcVal'], valArlCopy['CalcArl'], label='Calculated', s=60)
        ax.scatter(valArlCopy['RatdVal'], valArlCopy['RatdArl'], label='Rated', s=60)
        ax.legend(['Calculated', 'Rated'])
        ax.set_ylabel('Arousal')
        ax.set_xlabel('Valence')
        plt.savefig(os.path.join(targetDir, videoPrefix+'Valence-Arousal_ScatterPlot.png'))
        plt.savefig(os.path.join(targetDir, videoPrefix+'Valence-Arousal_ScatterPlot.pdf'))

        print('================= With Audio Feature =================================') 
        print('This function tests the null hypothesis that a sample comes from a normal distribution')
        print('Calculated Valence = %s' %str(sts.normaltest(valArlCopy['CalcVal'])))
        print('Calculated Arousal = %s' %str(sts.normaltest(valArlCopy['CalcArl'])))
        print('Rated Valence = %s' %str(sts.normaltest(valArlCopy['RatdVal'])))
        print('Rated Arousal = %s' %str(sts.normaltest(valArlCopy['RatdArl'])))

        '''pdb.set_trace()
        duplicateIndexes = np.where(valArlCopy.duplicated().values)[0]
        audioVideoCombined.drop(duplicateIndexes, axis=0, inplace=True)
        audioVideoCombined.reset_index(drop=True, inplace=True)'''

        print('Valence = %s' %str(sts.spearmanr(valArlCopy['CalcVal'], valArlCopy['RatdVal'])))
        print('Arousal = %s' %str(sts.spearmanr(valArlCopy['CalcArl'], valArlCopy['RatdArl'])))
        print('Number of videos = %s' %str(len(valArlCopy)))
        print('minimum number of ratings = %s' %str(np.min(Count_participantRatingGB[Count_participantRatingGB>ratingThrs])))
        print('maximum number of ratings = %s' %str(np.max(Count_participantRatingGB[Count_participantRatingGB>ratingThrs])))

        print('======================================================================')
        valArlCopy.plot.scatter(x='RatdVal', y='RatdArl', c='RValStd', s='RArlStd')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel('Rated Valence', fontsize=30)
        plt.ylabel('Rated Arosal', fontsize=30)
        #plt.show()

        valArlCopy.plot.scatter(x='CalcVal', y='CalcArl')
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel('Calculated Valence', fontsize=30)
        plt.ylabel('Calculated Arosal', fontsize=30)
        #plt.show()

###################################### Video Feature Calculation #########################################
        valenceVideoFrame = pd.read_csv(os.path.join(targetDir, videoPrefix+'VideoFeatureBasedValenceCalculation.csv'), index_col=0)
        valenceVideoFrame['Experiment_id'] = [i.split('.')[0] for i in  valenceVideoFrame.index.values]
        valenceVideoFrame.set_index('Experiment_id', drop=True, inplace=True)
        valenceVideoFrame.drop(['max_rhythm_comp'], axis=1, inplace=True)

        arousalVideoFrame = pd.read_csv(os.path.join(targetDir, videoPrefix+'VideoFeatureBasedArousalCalculation.csv'), index_col=0)
        arousalVideoFrame['Experiment_id'] = [i.split('.')[0] for i in  arousalVideoFrame.index.values]
        arousalVideoFrame.set_index('Experiment_id', drop=True, inplace=True)

        columnsToDrop = valenceVideoFrame.columns.values[sum(valenceVideoFrame.values-np.mean(valenceVideoFrame.values,0))==0.0]
        columnsToDrop = columnsToDrop.tolist()
        #columnsToDrop.extend(['std_shotRate_', 'mean_shotRate_'])
        valenceVideoFrame.drop(columnsToDrop, axis=1, inplace=True)

        columnsToDrop = arousalVideoFrame.columns.values[sum(arousalVideoFrame.values-np.mean(arousalVideoFrame.values,0))==0]
        arousalVideoFrame.drop(columnsToDrop.tolist(), axis=1, inplace=True)

        Vrescaled = 1+np.divide(((valenceVideoFrame-np.min(valenceVideoFrame, 0))*(9-1)), (np.max(valenceVideoFrame, 0)-np.min(valenceVideoFrame, 0)))
        Arescaled = 1+np.divide(((arousalVideoFrame-np.min(arousalVideoFrame, 0))*(9-1)), (np.max(arousalVideoFrame, 0)-np.min(arousalVideoFrame, 0)))
        Vrescaled.dropna(axis=0, inplace=True)

        if sum(np.isnan(Arescaled['max_motion_comp'])) > len(Arescaled)/2:
            Arescaled.drop('max_motion_comp', axis=1,  inplace=True)    
        Arescaled.dropna(axis=0,  inplace=True)
        ## DO range conversion and then PCA calculation

#################################### Valence
        pcaV = PCA(n_components=5)
        principalComponents = pcaV.fit(Vrescaled.values)
        print(sum(principalComponents.explained_variance_ratio_))
        principalComponentsV = pcaV.fit_transform(Vrescaled.values)
        ##### Rescaling PCA transformed Data
        VrescaledPC = 1+np.divide(((principalComponentsV-np.min(principalComponentsV, 0))*(9-1)), (np.max(principalComponentsV, 0)-np.min(principalComponentsV, 0)))
        
        valence = np.sqrt(np.sum(np.power(VrescaledPC, 2), 1))        
        valence = 1+np.divide(((valence-np.min(valence, 0))*(9-1)), (np.max(valence, 0)-np.min(valence, 0)))
        
#################################### Arousal        
        pcaV = PCA(n_components=5)
        principalComponents = pcaV.fit(Arescaled.values)
        print(sum(principalComponents.explained_variance_ratio_))
        principalComponentsA = pcaV.fit_transform(Arescaled.values)
        ##### Rescaling PCA transformed Data
        ArescaledPC = 1+np.divide(((principalComponentsA-np.min(principalComponentsA, 0))*(9-1)), (np.max(principalComponentsA, 0)-np.min(principalComponentsA, 0)))
        arousal = np.sqrt(np.sum(np.power(ArescaledPC, 2), 1)) 
        arousal = 1+np.divide(((arousal-np.min(arousal, 0))*(9-1)), (np.max(arousal, 0)-np.min(arousal, 0)))

        audioVideoFeat = valArlCopy
        audioVideoFeat['valenceVid'] = 0
        audioVideoFeat['arousalVid'] = 0

        for emtVid in audioVideoFeat.index.values:
            #print(emtVid)
            #print(Vrescaled.index.values[np.where(emtVid==Vrescaled.index.values)[0][0]])
            try:
                audioVideoFeat.loc[emtVid, 'valenceVid'] = valence[np.where(emtVid==Vrescaled.index.values)[0][0]]
            except:
                pdb.set_trace()

            audioVideoFeat.loc[emtVid, 'arousalVid'] = arousal[np.where(emtVid==Arescaled.index.values)[0][0]]

        print('================= With Video Feature =================================')
        print('Calculated Valence = %s' %str(sts.normaltest(audioVideoFeat['valenceVid'])))
        print('Calculated Arousal = %s' %str(sts.normaltest(audioVideoFeat['arousalVid'])))
        print('Rated Valence = %s' %str(sts.normaltest(audioVideoFeat['RatdVal'])))
        print('Rated Arousal = %s' %str(sts.normaltest(audioVideoFeat['RatdArl'])))

        print('Valence = %s' %str(sts.spearmanr(audioVideoFeat['valenceVid'], audioVideoFeat['RatdVal'])))
        print('Arousal = %s' %str(sts.spearmanr(audioVideoFeat['arousalVid'], audioVideoFeat['RatdArl'])))

###################### After combining the valence and arousal features #######################

        #weight = [0.88,0.12]        
        weight = [0.5,0.5]        
        audioVideoFeat['audioVisualAvgVal'] = ((weight[0]*audioVideoFeat['CalcVal']) + (weight[1]*audioVideoFeat['valenceVid']))
        audioVideoFeat['audioVisualAvgArl'] = ((weight[0]*audioVideoFeat['CalcArl']) + (weight[1]*audioVideoFeat['arousalVid']))

        print('================= For combined audio and visual features =================================')
        print(weight)
        print('Calculated Valence = %s' %str(sts.normaltest(audioVideoFeat['audioVisualAvgVal'])))
        print('Calculated Arousal = %s' %str(sts.normaltest(audioVideoFeat['audioVisualAvgArl'])))

        print('Valence = %s' %str(sts.spearmanr(audioVideoFeat['audioVisualAvgVal'], audioVideoFeat['RatdVal'])))
        print('Arousal = %s' %str(sts.spearmanr(audioVideoFeat['audioVisualAvgArl'], audioVideoFeat['RatdArl'])))

        print("Put condition when to chech correlation with pearson and when with spearmanr")




        print('Classification')
        allFeatures = pd.read_csv(os.path.join(targetDir, videoPrefix+'ForClassificaion_AllMultimediaCalculatedPCADimensionsValenceArousal.csv'), index_col = 0) 
        videoFeatureDf = allFeatures.iloc[:, 0:35]
        audioVideoFeatPCA = allFeatures.iloc[:, 35:]
        #makingInputDataForClassificaiton(audioVideoFeatPCA, videoFeatureDf)  #, valArl):