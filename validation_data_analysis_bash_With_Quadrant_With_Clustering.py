'''This module is to test validation of the study'''

import numpy as np
import pandas as pd
import os
import glob
from adjustText import adjust_text
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import shutil
import re
import pdb

'''Important files are:
    1. validation_analysis: 
        errorbars for valence, arousal, dominance, liking, familiarity, relevance. 
        WithVIdeoId-VideoEmotionProfile, 
        VideoEmotionProfile; 
        summary_data_frame_, 
        Greater_Then_50_, My_Experiment_Ratings_
    2 VAD_Plotting: 
        'all_emotions_and_mean_'+range_vid_key+'_'+cleaning_flag+date+'.png'; 'all_emotions_and_mean_greater_10_VA_Norm_'+cleaning_flag+date+'.png'; all_emotions_and_mean_arousal_domiannce0_10_before_cleaning2018_Oct_14-Nov_15
        'all_emotions_and_mean_greater_10_VA_Mean_'+cleaning_flag+date+'.png': all_emotions_and_mean_greater_10_VA_Norm_before_cleaning2018_Apr_24-Oct-31.png
        'all_emotions_and_mean_greater_10_obs_'+cleaning_flag+date+'.png'
'''
#   valence_rat_5_wale = ['chankya','harsha','upasna','prachi','pankaj','AananditaDhawan','rakhi','rohit','swarnima','Lovely','sarthak']
#def validation_analysis(interval_record,recalculate_rating_file, no_blocks, cleaning_flag = None):
#sourceDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion'

'''strToComp = ['Adventorous', 'Afraid', 'Alarmed', 'Amused', 'Angry', 'Annoyed', 'Aroused', 'Ashamed', 'Astonished', 'Attraction', 'Brutality', 'Calm', 'Cheerful', 
'Compassionate', 'Contemplative', 'Contented', 'Convinced', 'Curious', 'DISTURBING', 'Delighted', 'Depressed', 'Despondent', 'Disgust', 'Dissatisfied', 'Distress', 'Distrustful', 'Droopy', 
'Enthusiastic', 'Excited', 'Frustrated', 'Funny', 'Gloomy', 'Happy', 'Hate', 'Hopeful', 'Impatient', 'Indignant', 'Insomnia', 'Joyous', 'Love', 'Lust', 'Melancholic', 'Miserable', 
'Passionate', 'Peaceful', 'Pensive', 'Pleased', 'Relaxed', 'Sad', 'Startled', 'Taken Aback', 'Tense', 'Tired', 'Triumphant']'''


############ Note: Always use : after the name of the emotion. Because definition of emotions also have emotion words.
strToComp = ['Adventorous:', 'Afraid:', 'Alarmed:', 'Amused:', 'Angry:', 'Annoyed:', 'Aroused:', 'Ashamed:', 'Astonished:', 'Attraction:', 'Brutality:', 'Calm:', 'Cheerful:', 
'Compassionate:', 'Contemplative:', 'Contented:', 'Convinced:', 'Curious:', 'DISTURBING:', 'Delighted:', 'Depressed:', 'Despondent:', 'Disgust:', 'Dissatisfied:', 'Distress:', 'Distrustful:', 'Droopy:', 
'Enthusiastic:', 'Excited:', 'Frustrated:', 'Funny:', 'Gloomy:', 'Happy:', 'Hate:', 'Hopeful:', 'Impatient:', 'Indignant:', 'Joyous:', 'Love:', 'Lust:', 'Melancholic:', 'Miserable:', 
'Passionate:', 'Peaceful:', 'Pensive:', 'Pleased:', 'Relaxed:', 'Sad:', 'Startled:', 'Taken Aback:', 'Tense:', 'Tired:', 'Triumphant:']

#strToComp = ['Taken Aback:']

NotHappy = ['Best_Of_Amitabh_Bachchan_Scenes', 'Brothers_9_10_Movie_CLIP_Sam_Loses', 'Butterfly', 'Carving_A_Giant', 'Darkest_Things', 'Corporate_Cannibal', 'Loneliness', 
            'Hobbeling_scene', 'My_Funeral_', 'Procrastination', 'Beautiful', 'The_One_I_Once_Was', 'The_Champ_1979', 'Titanic_2012_Sinking', 'hate_lbs', 'hopeful2', 'pursuit_of_happyness',
            'Madari_movie_of_best_scene', 'Milkha_Visits_His_Village', 'The_Weight_Of_My_Words', 'Anacondas_The_Hunt_for']

'''strToCompQuad = {'Afraid':'LVHA', 'Distress':'LVHA', 'Amused':'HVHA', 'Disgust':'LVHA', 'Miserable':'LVLA', 
                 'Enthusiastic':'HVHA', 'Aroused':'HVHA', 'Adventurous':'HVHA', 'Triumphant':'HVHA', 'Happy':'HVHA', 
                 'Joyous':'HVHA', 'Ashamed':'LVLA', 'Angry':'LVHA', 'Droopy':'LVLA', 'Sad':'LVLA', 'Calm':'HVLA', 
                 'Love':'HVLA', 'Melancholic':'LVLA', 'Alarmed':'LVHA'}'''

_thisDir = os.path.dirname(os.path.abspath(__file__))
videoPrefix = 'With69Videos_'  # 'WithThirtyVideos_'  # WithAllVideos_ , WithThirtyVideos_, With69Videos_, WithThirtyVideosEGI

def validation_analysis(interval_record=1, recalculate_rating_file = 1, no_blocks=1, cleaning_flag = 0, date='', no_months=1,start_date=14, end_date=15, partcipant_details=''):

    # Dates:
    # For all videos: Oct_10-Oct_20
    # For 69 and 30 videos: Oct_10-Nov_15

    if 'Oct_10-Nov_15' in date:
        videoPrefix = 'With69Videos_' ##  With69Videos_, WithThirtyVideos_  ## Change this option for scatter plot on V-A space
    if 'Oct_10-Oct_20' in date:
        videoPrefix = 'WithAllVideos_'

    #For without summary
    #validation_analysis(interval_record=1, recalculate_rating_file = 1, no_blocks=1, cleaning_flag = 0, date='2018_Oct_14-Nov_15', no_months=1,start_date=14, end_date=15, partcipant_details='Participant_Details_New.xlsx')
    #For summary
    #validation_analysis(interval_record=1, recalculate_rating_file = 0, no_blocks=1, cleaning_flag = 1, date='2018_Oct_10-Oct_20', no_months=1,start_date=10, end_date=20, partcipant_details='Participant_Details_New.xlsx')


    #validation_analysis(1,1,1,0,date='2018_Oct_31-Nov_2',response_file='ResponseFor69VideosOn27102018.xlsx')
    #validation_analysis(1,1,1,0,date='2018_Nov_14-Nov_15')    
    import ast        
    response_data = pd.read_excel(partcipant_details)#, encoding='utf-7')    
    enroll_ = response_data['Enrollment']
    enroll_ = [str(j) for j in enroll_]   
    enroll_ = np.char.lower(enroll_)
    enroll_ = [i.split(' ')[0] for i in enroll_]

    fid_ = open('errornuos_data.txt','w') 
    fid__ = open('right_data.txt','w') 
    

    months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    indexMonth = {'1':'Jan', '2':'Feb', '3':'Mar', '4':'Apr', '5':'May', '6':'Jun', '7':'Jul', '8':'Aug', 
        '9':'Sep', '10':'Oct', '11':'Nov', '12':'Dec'}
    LastMonthDate = {'1':'31', '2':'29', '3':'31', '4':'30', '5':'31', '6':'30', '7':'31', '8':'31', 
        '9':'30', '10':'31', '11':'30', '12':'31'}


    ### =========================Finding out month names
    monthNames = []
    monthIndex = []
    
    for month_ in months.keys():
        if month_ in date:
            monthNames = np.append(monthNames, month_)
            monthIndex = np.append(monthIndex, months[month_])

    sorted_ = monthIndex.argsort()
    sorted_month_index = monthIndex[sorted_]
    sorted_month = monthNames[sorted_]
    no_months = len(sorted_month)

    ### ======================== Finding out year names

    substring = sorted_month[0]
    start_index = 0
    year_array = []
    temp_ = []

    for char_ in date:
        if char_ in substring:
            month_name_start_index = start_index
            break

        if ('-' == char_) or ('_' == char_):            
            year_array = np.append(year_array, ''.join(temp_))
            temp_ = []
        else:
            temp_ = np.append(temp_, char_)        

        start_index = start_index + 1

    no_years = len(year_array)
    dateArr = []


    if (no_months == 1) and (no_years == 1):
 
        dates_are = np.arange(start_date, end_date+1)

        for dt_ in dates_are:
            dateArr = np.append(dateArr, year_array[0]+'_'+sorted_month[0]+'_'+str(dt_))

    elif (no_months > 1) and (no_years == 1):
        
        flag_start = 0
        flag_end = 0
        monthIndexArr = np.arange(int(sorted_month_index[0]), int(sorted_month_index[1])+1)

        for MIA_ in monthIndexArr:
            if flag_start == 0: # If start date is included
                dates_are = np.arange(start_date, int(LastMonthDate[str(MIA_)])+1)

                for dt_ in dates_are:                    
                    dateArr = np.append(dateArr, year_array[0]+'_'+indexMonth[str(MIA_)]+'_'+str(dt_))

                flag_start = 1
            else : # When start date is not included
                dates_are = np.arange(1, int(LastMonthDate[str(MIA_)])+1)
                for dt_ in dates_are:                    
                    dateArr = np.append(dateArr, year_array[0]+'_'+indexMonth[str(MIA_)]+'_'+str(dt_))                

    if cleaning_flag == None:
        raise ValueError("Please enter cleaning flag. 0 for without cleaning and 1 for after cleaning")
    elif cleaning_flag == 0:
        cleaning_flag = 'before_cleaning'
    else:
        cleaning_flag = 'after_cleaning'

    _thisDir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(_thisDir, 'data')

    # Since during the recording I have recorded the response intermittently also
    # That's why I need to select on which has highest index since that is final.
    os.chdir(path)

    count_enroll = 0
    correct_subject = 0
    all_the_subjects = []

 
    countFront = 0
    countBack = 0
    notFound = []
    allEmotions = []    

    if recalculate_rating_file == 1:
        if cleaning_flag == 'before_cleaning':
            if interval_record == 1:
                flag = 1
                
#                pdb.set_trace()
                for dirs_ in glob.glob("*"):

                    root_dirs_ = os.path.join(path,dirs_)
                    #os.chdir(root_dirs_)    
                    #print("The Node Name is = %s" %root_dirs_)                

                    for folders in glob.glob(os.path.join(root_dirs_, "*")):     

                        if folders.split('/')[-1] not in enroll_:
                            notFound.extend([folders])
                        #print('=================================== Front %s ========================================' %folders)
                        countFront = countFront + 1

                        array_csv_indexes = np.zeros((no_blocks,2)) 
                        array_csv_indexes[:,0] = range(no_blocks)                        
                        final_file = {}                        

                        if 1:
                            root_dirs_folder_ = os.path.join(root_dirs_, folders)                                       

                            if len(glob.glob(os.path.join(root_dirs_folder_, '*final_exp*.csv')))<4:
                                continue
                            try:        
                                fileIndex = np.max(np.array([int(i.split('/')[-1].split('_')[0]) for i in glob.glob(os.path.join(root_dirs_folder_, '*final_exp*.csv'))]))
                            except:
                                pdb.set_trace()

                            noStimShown = np.array([int(i.split('/')[-1].split('_')[-1].split('.csv')[0]) for i in glob.glob(os.path.join(root_dirs_folder_, '*'+str(fileIndex)+'_final_exp_*_*_*_*_*.csv'))])

                            maxStimShown = 0
                            for i in noStimShown:
                                if (i > maxStimShown) and (i < 30):
                                    maxStimShown = i

                            try:#01_final_exp_2018_Oct_30_0020_5
                                files = glob.glob(os.path.join(root_dirs_folder_, '*'+str(fileIndex)+'_final_exp_*_*_*_*_%s.csv' %str(maxStimShown)))[0]
                            except:
                                pdb.set_trace()

                            for date_ in dateArr:
                                if date_ in files.split('/')[-1]:
                                    continue_flag = 1
                                    break
                                else:
                                    continue_flag = 0

                            if continue_flag == 0:
                                #print("================ Back Dated file = %s" %files)
                                continue
                                #break                    

                            df = pd.read_csv(files)
                            df = df.rename(columns={'filepath':'Experiment_id', 'rating_valence.response':'Valence', 'Arousal.response':'Arousal','Dominance.response' : 'Dominance','liking.response' : 'Liking','familiarity.response' : 'Familiarity'})

                            try:
                                #pdb.set_trace()
                                index_gen = np.where((np.array(str(np.char.lower(folders.split('/')[-1])).split(' ')[0])==enroll_)==True)[0][0]
                            except:
                                #print('====== Participant information not availble = %s ================' %folders)    
                                continue
                            fid__.write("%s is name as directory and %s in the response data \n" %(folders, enroll_[index_gen]))
                            gender_part = str(response_data.loc[index_gen, 'Gender'])

                            negative_valence_flag = 0                                        
                            temp_ = np.where(('Emotion_Name'==df.columns.values)==True)[0]   

                            if len(temp_) > 0:
                                #if 'iec2015073' in folders.split('/')[-1]:
                                #    pdb.set_trace()

                                upto_emotion_name_ind = temp_[0]
                                upto_emotion_name = df.columns.values[0:upto_emotion_name_ind+1].tolist()
                                #upto_emotion_name = np.append(upto_emotion_name, ['participant', 'Quadrant'])   
                                upto_emotion_name = np.append(upto_emotion_name, ['participant'])   
                                df = df.loc[:, upto_emotion_name]
                                allEmotions.extend(df['Emotion_Name'])   
                                dfCopy = df.copy()

                                for count, values in enumerate(df[['Valence', 'Arousal', 'Emotion_Name']].values):
                                    if (isinstance(values[0], str) and ('None' in values[0])) or (isinstance(values[1], str) and ('None' in values[1])):
                                        dfCopy.drop(count, axis=0, inplace=True)

                                df = dfCopy.copy()
                                del dfCopy

                                df.reset_index(drop=True, inplace=True)

                                for count, values in enumerate(df[['Valence', 'Arousal', 'Emotion_Name']].values):

                                    try:
                                        val_ = float(values[0])
                                    except:
                                        val_ = values[0]
                                    try:
                                        arl_ = float(values[1])
                                    except:
                                        arl_ = values[1]

                                    emt = values[2]                                        

                                    #print("@@@@ Outside: Subject = %s and Emotion is = %s ********" %(folders.split('/')[-1], emt))
                                    try:
                                        np.array([np.isnan(val_), np.isnan(arl_)]).any() or (not isinstance(emt, str)) or (len(emt)<3) or ('[]' in emt) or ('Without Categorization' in emt) or ('No Emotion From List' in emt) or ('None' in emt) or ('default option' in emt)
                                    except:
                                        pdb.set_trace()

                                    if np.array([np.isnan(val_), np.isnan(arl_)]).any() or (not isinstance(emt, str)) or (len(emt)<3) or ('[]' in emt) or ('Without Categorization' in emt) or ('No Emotion From List' in emt) or ('None' in emt) or ('default option' in emt):
                                        #print("@@@@ Inside: Subject = %s and Emotion is = %s ********" %(folders.split('/')[-1], emt))
                                        #pdb.set_trace()
                                        df.loc[count, 'Quadrant'] = None

                                    else:                                                                                                                                    
                                        if val_ > 5 and arl_ > 5:
                                            df.loc[count, 'Quadrant'] = 'HVHA'
                                        if val_ > 5 and arl_ <= 5:
                                            df.loc[count, 'Quadrant'] = 'HVLA'
                                        if val_ <= 5 and arl_ > 5:
                                            df.loc[count, 'Quadrant'] = 'LVHA'
                                        if val_ <= 5 and arl_ <= 5:
                                            df.loc[count, 'Quadrant'] = 'LVLA'                                                                                                                                                                                                          

                            idx, clIdx = np.where(df.values=='None')                            
                            for r_, c_ in zip(idx, clIdx):
                                #print(r_, c_)
                                if df.columns.values[c_] in ['Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity', 'Emotion_Name']:
                                    if r_ not in df.index.values.astype(float):
                                        continue
                                    else:
                                        df.drop(r_, axis=0, inplace=True)                                        

                            df.reset_index(drop=True, inplace=True)
                            if len(np.where(df.values=='None')[0]) > 0:
                                pdb.set_trace()
                            
                            try:
                                df.drop(np.where(np.isnan(df['Valence'].values)==True)[0], axis=0, inplace=True)
                            except:
                                df.drop(np.where(np.isnan(df['Valence'].values.astype(float))==True)[0], axis=0, inplace=True)

                            '''if isinstance(df.index.values[1], str):
                                df.drop([str(i) for i in np.where(np.isnan(df['Valence'].values)==True)[0]], axis=0, inplace=True)
                            else:'''

                            df.reset_index(drop=True, inplace=True)
                            df['Valence'] = df['Valence'].values.astype(float)
                            df['Arousal'] = df['Arousal'].values.astype(float)
                            df['Dominance'] = df['Dominance'].values.astype(float)
                            try:
                                df['Liking'] = df['Liking'].values.astype(float)
                            except:
                                pdb.set_trace()
                            df['Familiarity'] = df['Familiarity'].values.astype(float)
                            
                            if 'Emotion_Name' in df.columns.values:
                                emt_flag = 1
                                temp = df[['Experiment_id', 'trials.thisTrialN', 'trials.thisIndex', 'Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity', 'Emotion_Name','participant', 'Quadrant']]
                                temp['Gender'] = gender_part
                            else:                                            
                                emt_flag = 0
                                temp = df[['Experiment_id', 'trials.thisTrialN', 'trials.thisIndex', 'Valence', 'Arousal', 'Dominance', 'Liking', 'Familiarity', 'participant']]
                                temp['Emotion_Name'] = 'Without Categorization'
                                temp['Quadrant'] = None
                                temp['Gender'] = gender_part

                            files_in_blocks = len(df)
                            try:
                                if (np.where((temp['Valence']<0)==True)[0]).any():
                                    negative_valence_flag = 1
                            except:
                                pdb.set_trace()

                            # Every block has 7 files corresponding to different emotions
                            no_stimulus = 0
                            frames_to_drop = []
                            entryToDel = []

                            for file_ind in range(files_in_blocks):                                
                                val_ce = df['Valence'][file_ind]
                                aro_al = df['Arousal'][file_ind]
                                dom_ce = df['Dominance'][file_ind]

                                '''if ('IEC2015016' in folders):
                                    print(file_ind)
                                    pdb.set_trace()
                                    if (file_ind==5):# and _str == 'Compassionate:
                                        pdb.set_trace()'''

                                if emt_flag == 1:
                                    try:
                                        emt_N = ast.literal_eval(df['Emotion_Name'][file_ind])
                                    except:
                                        emt_N = df['Emotion_Name'][file_ind]
                                else:
                                    emt_N = 'Without Categorization'

                                ### Removing some entries in which participants rated happy but it should not be happy.

                                searchingId = temp['Experiment_id'][file_ind].split('/')[-1]
                                searchingId = '_'.join(searchingId.split(' '))
                                searchingId = '_'.join(searchingId.split("'"))
                                searchingId = '_'.join(searchingId.split('('))
                                searchingId = '_'.join(searchingId.split(')'))
                                searchingId = '_'.join(searchingId.split('&'))
                                searchingId = searchingId.split('.')[0]

                                nhFlag = 0
                                if isinstance(emt_N, str):
                                    if 'No Emotion From List' in emt_N:
                                        nhFlag = 1
                                    else:                                   
                                        if ('Happy:' == np.array(emt_N.split())).any() or ('happy:' == np.array(emt_N.split())).any(): 
                                            for nh in NotHappy:                        
                                                if nh in searchingId:
                                                    nhFlag = 1
                                                    break

                                if isinstance(emt_N, list):
                                    for _emt in emt_N:
                                        if 'No Emotion From List' in _emt:
                                            nhFlag = 1
                                            break
                                        if len(_emt) == 0:
                                            continue
                                        if ('Happy:' == np.array(_emt.split())).any() or ('happy:' == np.array(_emt.split())).any(): 
                                            #print(searchingId)
                                            for nh in NotHappy:                        
                                                if nh in searchingId:
                                                    nhFlag = 1
                                                    break

                                if nhFlag == 1:
                                    entryToDel.extend([file_ind])
                                    continue

                                try:                                                
                                    if np.isnan(df['Experiment_id'][file_ind]): # Remove nan entries from the extracted data frame from the original file.
                                        frames_to_drop = np.append(frames_to_drop, file_ind)                                                    

                                except:                                                
                                    if emt_flag == 1:
                                        try:
                                            #if (emt_N == None) or (np.isnan(emt_N)):
                                            if (emt_N == None):
                                                frames_to_drop = np.append(frames_to_drop, file_ind)
                                                pdb.set_trace()
                                                continue
                                        except:
                                            print('Emotion is entered')

                                        try:
                                            if (not isinstance(emt_N, str)) and (not isinstance(emt_N, list)):
                                                continue

                                            #if ('IEC2015016' in folders) and (file_ind==5):# and _str == 'Compassionate:
                                            #    pdb.set_trace()

                                            tmp = []
                                            if isinstance(emt_N, list):
                                                for _emt in emt_N:                                            
                                                    if 'TAKEN ABACK' in _emt.upper():
                                                        tmp = ['Taken Aback']
                                                    else:                                                    
                                                        for _str in strToComp: ## Counting on pre-defined set of emotions      
                                                            for word in _emt.upper().split():
                                                                if _str.upper()==word:
                                                                    tmp.extend([_str.split(':')[0]])

                                                        #if re.search(r"\b" + _str.upper() + r"\b", _emt.upper()):
                                                        #if _str.upper() in _emt.upper(): ## Checking if emt_ has pre-defined emotions.
                                            if isinstance(emt_N, str):
                                                if 'TAKEN ABACK' in emt_N.upper():
                                                    tmp = ['Taken Aback']
                                                else:
                                                    for _str in strToComp: ## Counting on pre-defined set of emotions      
                                                        #if ('iec2016077' in folders) and (file_ind==3):# and _str == 'Compassionate:':                                                                
                                                        #    if 'Ashamed:' == _str:
                                                        #        pdb.set_trace()

                                                        for word in emt_N.upper().split():
                                                            if '[' in word:
                                                                start = len(word)-len(_str)
                                                                end = start+len(_str)
                                                            else:
                                                                start = 0
                                                                end = len(_str)

                                                            if _str.upper()==word[start:end]:
                                                                tmp = [_str.split(':')[0]]

                                            if len(tmp) > 1:
                                                pdb.set_trace()

                                            if len(tmp) == 1:
                                                temp['Emotion_Name'][file_ind] = tmp[0]

                                            if len(tmp) == 0: ## Emotion is not from the desired list
                                                entryToDel.extend([file_ind])
                                            
                                            #del tmp

                                            try:
                                                temp['Experiment_id'][file_ind] = df['Experiment_id'][file_ind].split("/")[-1]                                                                
                                            except:
                                                pdb.set_trace()

                                        except:
                                            if (not isinstance(emt_N, str)) and (not isinstance(emt_N, list)):
                                                continue

                                            #if ('IEC2015016' in folders) and (file_ind==5):# and _str == 'Compassionate:
                                            #    pdb.set_trace()

                                            tmp = []
                                            if isinstance(emt_N, list):                                                
                                                for _emt in emt_N:                                            
                                                    if 'TAKEN ABACK' in _emt.upper():
                                                        tmp = ['Taken Aback']
                                                    else:                 
                                                        for _str in strToComp: ## Counting on pre-defined set of emotions      
                                                            for word in _emt.upper().split():
                                                                if _str.upper()==word:
                                                                    tmp.extend([_str.split(':')[0]])

                                            if isinstance(emt_N, str):
                                                if 'TAKEN ABACK' in emt_N.upper():
                                                    tmp = ['Taken Aback']
                                                else:                                                
                                                    for _str in strToComp: ## Counting on pre-defined set of emotions      
                                                        for word in emt_N.upper().split():
                                                            if '[' in word:
                                                                start = len(word)-len(_str)
                                                                end = start+len(_str)
                                                            else:
                                                                start = 0
                                                                end = len(_str)

                                                            if _str.upper()==word[start:end]:
                                                                tmp = [_str.split(':')[0]]
                                            
                                            #if ('IEC2015016' in folders) and (file_ind==5):# and _str == 'Compassionate:
                                            #    pdb.set_trace()

                                            if len(tmp) > 1:
                                                pdb.set_trace()

                                            if len(tmp) == 1:
                                                temp['Emotion_Name'][file_ind] = tmp[0]
                                            if len(tmp) == 0: ## Emotion is not from the desired list
                                                entryToDel.extend([file_ind])

                                            #del tmp

                                            try:
                                                temp['Experiment_id'][file_ind] = df['Experiment_id'][file_ind].split("/")[-1]                                                                
                                            except:
                                                pdb.set_trace()

                                    else: # If no emotion file is there

                                        try:
                                            if (np.isnan(val_ce) and np.isnan(aro_al) and np.isnan(dom_ce)):# and np.isnan(df['Liking'][file_ind])
                                                frames_to_drop = np.append(frames_to_drop, file_ind)
                                            else:
                                                temp['Experiment_id'][file_ind] = df['Experiment_id'][file_ind].split("/")[-1]

                                        except:                                                     
                                            temp['Experiment_id'][file_ind] = df['Experiment_id'][file_ind].split("/")[-1]

                            for drop_frames in frames_to_drop:                                            
                                temp = temp.drop(df.index[[int(drop_frames)]])

                            if np.any(temp.values=='None'): # If any entry is None then drop that stimulus for the particular subject
                                
                                touple = np.where((temp.values=='None')==True)[0]
                                temp = temp.drop(temp.index[[touple]])

                            if negative_valence_flag == 1:
                                temp['Valence'] = 1+(((temp['Valence']-(-4))/8)*(9-1))  # Rescaling from 1 to 9 from -4 to 4
                            
                            if len(temp.values) != 0:
                                correct_subject = correct_subject + 1

                            temp['Experiment_id'] = [i.split('/')[-1] for i in temp['Experiment_id']]

                            if len(entryToDel) > 0:
                                temp.drop(entryToDel, axis=0, inplace=True)                                

                            temp.reset_index(drop=True, inplace=True)

                            if emt_flag == 1:
                                print(folders)
                                print(temp[['Experiment_id', 'Emotion_Name']])                            
                                print(df[['Experiment_id', 'Emotion_Name']])

                            if emt_flag == 1:
                                for exp_id, _emt in zip(temp['Experiment_id'], temp['Emotion_Name']):
                                    if 'Amitabh' in exp_id and ('Happy' == _emt or 'happy' == _emt):
                                        pdb.set_trace()

                            if flag == 1:
                                subject_data_frame = temp
                                flag = 0
                            else:
                                subject_data_frame = pd.concat([subject_data_frame,temp],axis=0,ignore_index=True)

                        countBack = countBack + 1
                        #print('=================================== Back %s ========================================' %folders)

                print("Total Participants are = %s . But the correct number is = %s" %(str(count_enroll), str(correct_subject)))            
                os.chdir('../..')

                directory_to_save = os.path.join(_thisDir, 'NewTarget','My_Experiment_Ratings_'+cleaning_flag+date+'.csv')

                subject_data_frame.to_csv(directory_to_save)    
                print("Now go for creating summary by selecting the option recalculate_rating_file = 0")    
                fid_.close()    
                fid__.close()

            else:
                
                for files in glob.glob("*.csv"):
                    splitted = files.split("_")
                    block_name = int(splitted[0])-1
                    #pdb.set_trace()
                    csv_index = int(splitted[-1].split(".csv")[0])
                    print("Block Name is %d and Index is %d" %(block_name,csv_index))

                    #pdb.set_trace()
                    if (array_csv_indexes[block_name,0]==block_name) and (array_csv_indexes[block_name,1]<csv_index) and (csv_index < 10):
                        array_csv_indexes[block_name,0] = block_name
                        array_csv_indexes[block_name,1] = csv_index
                        final_file[str(block_name)] = files

    else:

        import ast
        from textwrap import wrap, fill

        metadata_dir = os.path.join(_thisDir, 'Metadata')
        deap_data = pd.read_csv(os.path.join(metadata_dir, 'participant_ratings.csv')).values
        os.chdir('..')

        subject_data_frame = pd.read_csv(os.path.join(_thisDir, 'NewTarget', 'My_Experiment_Ratings_'+cleaning_flag+date+'.csv'))
        emotions_index = np.unique(subject_data_frame['Experiment_id'])

        no_emt = len(emotions_index)
        group_mean = subject_data_frame.groupby(['Experiment_id']).mean()
        group_std = subject_data_frame.groupby(['Experiment_id']).std()
        group_count = subject_data_frame.groupby(['Experiment_id'])['Experiment_id'].count()

        hindi_engligh_dict = {'04':'4','05':'5','6':'6','7':'7',
                              '08':'8','09':'9','10':'10','':'','SadTujheBhulaDiya':'Sad-Hindi'}

        filesNames = np.sort(np.unique(subject_data_frame['Experiment_id'].values))
        emt_dict = {}
        emt_part_1 = []
        emt_part_2 = []

        for vdNames_, index_ in zip(filesNames, range(len(filesNames))):
            if index_ < 10:                
                emt_dict[vdNames_] = '0'+str(index_)
            else:
                emt_dict[vdNames_] = str(index_)

            if index_ < (no_emt/2):
                emt_part_1 = np.append(emt_part_1, vdNames_)
            else:
                emt_part_2 = np.append(emt_part_2, vdNames_)


        sign_dict = {'04':2, '05':2, '06':2, '07':2, '08':2, '09':2, '10':2, '17':2, '18':2, '19':2, '20':2,
       '21':2, '27':2, '28':2, '29':2, '30':2, '34':2, '35':2, '36':2, '37':2, '38':2, '39':2,
       '40':2, 'FunAlaBarfi':1, 'SadTujheBhulaDiya':-1, 'SaddaHaq':-1, 'cheerful':1,
       'cheerfulRang':1, 'depressing':-1, 'depressingYahiHota':-1, 'exciting':1,
       'excitingMohabbat':1, 'fun':1, 'funTamma':1, 'happy':1, 'happy1':1, 'hate':-1,
       'joy':1, 'joyKolaveri':1, 'love':1, 'loveMeraPehlaPyar':1, 'loveMereSang':1,
       'loveNashe':1, 'loveTumHo':1, 'lovely':1, 'lovelySadma':1, 'melancholy':-1,
       'mellow':1, 'sad':-1, 'sadEmptiness':-1, 'sentimental':-1, 'sentimental1':-1,
       'sentimentalYehJo':-1, 'shock':-1, 'terrible':-1}

        deap_dict = {'04':4,'05':5,'06':6,'07':7,'08':8,'09':9,'10':10,'17':17,'18':18,'19':19,
                     '20':20,'21':21,'27':27,'28':28,'29':29,'30':30,'34':34,'35':35,'36':36,'37':37,
                     '38':38,'39':39,'40':40,'fun':1,'exciting':2,'joy':3,'happy':11,'cheerful':12,
                     'love':13,'happy':14,'lovely':15,'sentimental':16,'sentimental':22,'melancholy':23,
                     'sad':24,'depressing':25,'mellow':26,'terrible':31,'shock':32,'hate':33}
        
        plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.27, left=0.06, right=0.98, bottom=0.1) 
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(range(len(emt_part_1)), group_mean.loc[emt_part_1, 'Valence'], 
            group_std.loc[emt_part_1, 'Valence'], fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax1.set_xticks(np.arange(len(emt_part_1),step=3))
        ax1.grid(b=True, which='both', axis='x', linestyle='--')

        flip_counter = 0
        for val_, x_ in zip(group_count.values[0:len(emt_part_1)], np.arange(len(emt_part_1))):
            if (flip_counter % 2) == 0:
                ax1.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax1.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        ax2 = plt.subplot(gs[1])
        ax2.errorbar(len(emt_part_1) + np.arange(len(emt_part_2)), group_mean.loc[emt_part_2, 'Valence'], group_std.loc[emt_part_2, 'Valence'],
            fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax2.set_xticks(len(emt_part_1) + np.arange(len(emt_part_2), step=3))
        ax2.grid(b=True, which='both', axis='x', linestyle='--')
        
        flip_counter = 0
        for val_, x_ in zip(group_count.values[len(emt_part_1):len(emt_part_1)+len(emt_part_2)], len(emt_part_1) + np.arange(len(emt_part_2))):
            if (flip_counter % 2) == 0:
                ax2.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax2.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        plt.suptitle('Valence')
        plt.savefig('Valence_'+cleaning_flag+date+'.png')
        plt.clf()
        
        plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.27, left=0.06, right=0.98, bottom=0.1) 
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(range(len(emt_part_1)), group_mean.loc[emt_part_1, 'Arousal'], 
            group_std.loc[emt_part_1, 'Arousal'], fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax1.set_xticks(np.arange(len(emt_part_1),step=3))
        ax1.grid(b=True, which='both', axis='x', linestyle='--')

        flip_counter = 0
        for val_, x_ in zip(group_count.values[0:len(emt_part_1)], np.arange(len(emt_part_1))):
            if (flip_counter % 2) == 0:
                ax1.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax1.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        ax2 = plt.subplot(gs[1])
        ax2.errorbar(len(emt_part_1) + np.arange(len(emt_part_2)), group_mean.loc[emt_part_2, 'Arousal'], group_std.loc[emt_part_2, 'Arousal'],
            fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax2.set_xticks(len(emt_part_1) + np.arange(len(emt_part_2), step=3))
        ax2.grid(b=True, which='both', axis='x', linestyle='--')
        
        flip_counter = 0
        for val_, x_ in zip(group_count.values[len(emt_part_1):len(emt_part_1)+len(emt_part_2)], len(emt_part_1) + np.arange(len(emt_part_2))):
            if (flip_counter % 2) == 0:
                ax2.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax2.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        plt.suptitle('Arousal')
        plt.savefig('Arousal_'+cleaning_flag+date+'.png')
        plt.clf()

        plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.27, left=0.06, right=0.98, bottom=0.1) 
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(range(len(emt_part_1)), group_mean.loc[emt_part_1, 'Dominance'], 
            group_std.loc[emt_part_1, 'Dominance'], fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax1.set_xticks(np.arange(len(emt_part_1),step=3))
        ax1.grid(b=True, which='both', axis='x', linestyle='--')

        flip_counter = 0
        for val_, x_ in zip(group_count.values[0:len(emt_part_1)], np.arange(len(emt_part_1))):
            if (flip_counter % 2) == 0:
                ax1.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax1.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        ax2 = plt.subplot(gs[1])
        ax2.errorbar(len(emt_part_1) + np.arange(len(emt_part_2)), group_mean.loc[emt_part_2, 'Dominance'], group_std.loc[emt_part_2, 'Dominance'],
            fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax2.set_xticks(len(emt_part_1) + np.arange(len(emt_part_2), step=3))
        ax2.grid(b=True, which='both', axis='x', linestyle='--')
        
        flip_counter = 0
        for val_, x_ in zip(group_count.values[len(emt_part_1):len(emt_part_1)+len(emt_part_2)], len(emt_part_1) + np.arange(len(emt_part_2))):
            if (flip_counter % 2) == 0:
                ax2.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax2.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        plt.suptitle('Dominance')
        plt.savefig('Dominance_'+cleaning_flag+date+'.png')
        plt.clf()

        plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.27, left=0.06, right=0.98, bottom=0.1) 
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(range(len(emt_part_1)), group_mean.loc[emt_part_1, 'Liking'], 
            group_std.loc[emt_part_1, 'Liking'], fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax1.set_xticks(np.arange(len(emt_part_1),step=3))
        ax1.grid(b=True, which='both', axis='x', linestyle='--')

        flip_counter = 0
        for val_, x_ in zip(group_count.values[0:len(emt_part_1)], np.arange(len(emt_part_1))):
            if (flip_counter % 2) == 0:
                ax1.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax1.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        ax2 = plt.subplot(gs[1])
        ax2.errorbar(len(emt_part_1) + np.arange(len(emt_part_2)), group_mean.loc[emt_part_2, 'Liking'], group_std.loc[emt_part_2, 'Liking'],
            fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax2.set_xticks(len(emt_part_1) + np.arange(len(emt_part_2), step=3))
        ax2.grid(b=True, which='both', axis='x', linestyle='--')
        
        flip_counter = 0
        for val_, x_ in zip(group_count.values[len(emt_part_1):len(emt_part_1)+len(emt_part_2)], len(emt_part_1) + np.arange(len(emt_part_2))):
            if (flip_counter % 2) == 0:
                ax2.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax2.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        plt.suptitle('Liking')
        plt.savefig('Liking_'+cleaning_flag+date+'.png')
        plt.clf()

        plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.27, left=0.06, right=0.98, bottom=0.1) 
        ax1 = plt.subplot(gs[0])
        ax1.errorbar(range(len(emt_part_1)), group_mean.loc[emt_part_1, 'Familiarity'], 
            group_std.loc[emt_part_1, 'Familiarity'], fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax1.set_xticks(np.arange(len(emt_part_1),step=3))
        ax1.grid(b=True, which='both', axis='x', linestyle='--')

        flip_counter = 0
        for val_, x_ in zip(group_count.values[0:len(emt_part_1)], np.arange(len(emt_part_1))):
            if (flip_counter % 2) == 0:
                ax1.annotate(s=str(val_), xy=(x_, 9.5))
            else:
                ax1.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        ax2 = plt.subplot(gs[1])
        ax2.errorbar(len(emt_part_1) + np.arange(len(emt_part_2)), group_mean.loc[emt_part_2, 'Familiarity'], group_std.loc[emt_part_2, 'Familiarity'],
            fmt='o', elinewidth=0.8, markersize=1.0, ecolor='red')
        ax2.set_xticks(len(emt_part_1) + np.arange(len(emt_part_2), step=3))
        ax2.grid(b=True, which='both', axis='x', linestyle='--')
        
        flip_counter = 0
        for val_, x_ in zip(group_count.values[len(emt_part_1):len(emt_part_1)+len(emt_part_2)], len(emt_part_1) + np.arange(len(emt_part_2))):
            if (flip_counter % 2) == 0:
                ax2.annotate(s=str(val_), xy=(x_, 9.5))
            else: 
                ax2.annotate(s=str(val_), xy=(x_, 9.8))
            flip_counter = flip_counter + 1

        plt.suptitle('Familiarity')
        plt.savefig('Familiarity_'+cleaning_flag+date+'.png')
        plt.clf()

        if 'Emotion_Name' in subject_data_frame.columns.values:
            flag_emt = 1
        else:
            flag_emt = 0

        numpy_array = subject_data_frame.values
        vald_fig_path = os.path.join(_thisDir, 'Validation_figures_')

        if not os.path.exists(vald_fig_path):
            os.makedirs(vald_fig_path)

        os.chdir(vald_fig_path)

        summary_data_frame = pd.DataFrame(0, index=filesNames, columns=['VideoId',  
            'total_obs', 'V_mean', 'A_mean', 'D_mean', 'mean_dist_origin', 'V_med', 'A_med', 'D_med', 
            'median_dist_origin', 'V_std', 'A_std', 'VA_std','D_std', 'overall_std'])

        print(summary_data_frame)

        greater_than_50 = []
        VideoId = 0
        fig_count = 0
        with_VideoId = 0 # 0: if title of subfigure is video name; 1: if title of subfigure is videoID
        
        for key in filesNames:
            key = key.split('/')[-1]
            #print(key)
            touple = np.where((numpy_array[:,1]==key)==True)[0]
            emotion_wise = numpy_array[touple,1:]

            if (flag_emt == 1) and (not (date == '2018_Oct_10-Oct_20')):
                if (VideoId != 0) and (((VideoId%16)==0)):
                    # Save the figure
                    if with_VideoId == 1:
                        plt.savefig('WithVIdeoId-VideoEmotionProfile'+date+'_'+str(fig_count)+'.png')
                    else:
                        plt.savefig('VideoEmotionProfile'+date+'_'+str(fig_count)+'.png')

                    plt.clf()
                    fig_count = fig_count + 1
                    # And, create the new one
                    fig = plt.figure(figsize=(20, 10.5))
                    gs = gridspec.GridSpec(4, 4, hspace=0.8, left=0.05, right=0.99, bottom=0.15, top=0.95)                 
                    row = 0
                    col = 0                

                if VideoId == 0:
                    # Create the new figure
                    fig = plt.figure(figsize=(20, 10.5))
                    gs = gridspec.GridSpec(4, 4, hspace=0.8, left=0.05, right=0.99, bottom=0.15, top=0.95)                 
                    row = 0
                    col = 0

                emt_name_list = np.array([])
                QDArr = []
                for rate_ind, QD in zip(emotion_wise[:,8], emotion_wise[:,10]):                    

                    if not isinstance(rate_ind, str):
                        continue

                    #if isinstance(rate_ind, str):
                        #print("Emotion Name = %s; for video = %s" %(rate_ind, ))
                    try:
                        flag = 1
                        emt_name_part = str(ast.literal_eval(rate_ind)[0])
                    except Exception as e:
                        try:    
                            flag = 2                        
                            emt_name_part = (rate_ind.split(':')[0]).split('"')[1]
                        except:
                            flag = 3
                            try:
                                emt_name_part = (rate_ind.split(':')[0])
                            except:
                                pdb.set_trace()

                    #print(str("No Emotion From List"))
                    #print("Rate Ind is = %s and flag = %s" %(rate_ind, str(flag)))
                    try:
                        if (str("No Emotion From List") == str(emt_name_part)):
                            proposed_name = str(ast.literal_eval(rate_ind)[1])
                            if len(proposed_name) > 2:
                                emt_name_part = proposed_name
                        else:
                            emt_name_part = (emt_name_part.split(':')[0])
                    except Exception as e:
                        #print("The Error is = %s" %e)
                        pdb.set_trace()

                    nhFlag = 0
                    print(emt_name_part)
                    if emt_name_part == 'Happy' or emt_name_part == 'happy': 
                        #if 'Best_Of_Amitabh_Bachchan_Scenes' in '_'.join(data.loc[index, 'Experiment_id'].split(' ')):
                        #    pdb.set_trace()                                                
                        for nh in NotHappy:                        
                            if nh in '_'.join(key.split(' ')):
                                nhFlag = 1
                                break

                    if nhFlag == 0:                                            
                        emt_name_list = np.append(emt_name_list, emt_name_part)                        
                        if isinstance(QD, str):
                            QDArr.extend([QD])

                summary_data_frame.loc[key, 'Emotion_Name'] = str(emt_name_list.tolist())
                try:
                    summary_data_frame.loc[key, 'Quadrant'] = ','.join(QDArr)
                except:
                    pdb.set_trace()
                    #summary_data_frame.loc[key, 'Quadrant'] = ','.join(QDArr)

                df_emt = pd.DataFrame(emt_name_list.tolist(), index=emt_name_list.tolist(), columns=['Emotion_Name'])
                df_emt['count'] = 1

                if (col!=0) and ((col%4)==0):
                    row = row + 1
                    col = 0

                df_emt_group = df_emt.groupby(by='Emotion_Name').count()
                ax = plt.subplot(gs[row, col])   
                if with_VideoId == 1:
                    ax.set_title('VideoId-'+str(VideoId), fontsize=12)
                else:
                    ax.set_title(key, fontsize=12)

                try:
                    mf_ax = df_emt_group.plot.bar(y = 'count', ax=ax, rot=45, legend=False, xticks=None)
                except:
                    pdb.set_trace()
                    
                loc = np.arange(len(df_emt['Emotion_Name'].values))

                #wrapped_labels = ['\n'.join(wrap(l,20)) for l in df_emt['Emotion_Name'].values]
                #if 'Anaconda' in key:                
                wrapped_labels = [fill(l,20) for l in df_emt_group.index.values]

                #print(wrapped_labels)

                mf_ax.set_xticklabels(wrapped_labels)            
                mf_ax.set_xlabel('')


                if (VideoId==(len(filesNames))-1):
                    # Save the figure
                    if with_VideoId == 1:
                        plt.savefig('WithVIdeoId-VideoEmotionProfile'+date+'_'+str(fig_count)+'.png')
                    else:
                        plt.savefig('VideoEmotionProfile'+date+'_'+str(fig_count)+'.png')

                col = col + 1

            valence_mean = np.mean(emotion_wise[:,3])
            arousal_mean = np.mean(emotion_wise[:,4])
            dominance_mean = np.mean(emotion_wise[:,5])
            liking_mean = np.mean(emotion_wise[:,6])
            familiarity_mean = np.mean(emotion_wise[:,7])

            #mean_dist_from_origin = (((5-valence_mean)**2)+((5-arousal_mean)**2)+((5-dominance_mean)**2))**0.5
            mean_dist_from_origin = (((5-valence_mean)**2)+((5-arousal_mean)**2))**0.5

            valence_std = np.std(emotion_wise[:,3])
            arousal_std = np.std(emotion_wise[:,4])
            dominance_std = np.std(emotion_wise[:,5])
            liking_std = np.std(emotion_wise[:,6])
            familiarity_std = np.std(emotion_wise[:,7])

            dist_from_centroid = (((emotion_wise[:,3]-valence_mean)**2)+((emotion_wise[:,4]-arousal_mean)**2))**0.5
            thrs = np.percentile(dist_from_centroid,90)
            outlier_index = np.where((dist_from_centroid>thrs)==True)

            valence_medoid = np.percentile(emotion_wise[:,3],50)
            arousal_medoid = np.percentile(emotion_wise[:,4],50)
            dominance_medoid = np.percentile(emotion_wise[:,5],50)            
            liking_medoid = np.percentile(emotion_wise[:,6],50)            
            familiarity_medoid = np.percentile(emotion_wise[:,7],50)            

            median_dist_from_origin = (((5-valence_medoid)**2)+((5-arousal_medoid)**2)+((5-dominance_medoid)**2))**0.5

            dist_from_med = (((emotion_wise[:,3]-valence_medoid)**2)+((emotion_wise[:,4]-arousal_medoid)**2))**0.5
            thrs_med = np.percentile(dist_from_med,90)
            outlier_index_med = np.where((dist_from_med>thrs_med)==True)

            if len(touple) > 50:
                greater_than_50 = np.append(greater_than_50, key)

            summary_data_frame.loc[key, 'VideoId'] = VideoId                
            summary_data_frame.loc[key, 'total_obs'] = len(touple)
            summary_data_frame.loc[key, 'V_mean'] = np.round(valence_mean, 2)
            summary_data_frame.loc[key, 'A_mean'] = np.round(arousal_mean, 2)
            summary_data_frame.loc[key, 'D_mean'] = np.round(dominance_mean, 2)
            summary_data_frame.loc[key, 'L_mean'] = np.round(liking_mean, 2)
            summary_data_frame.loc[key, 'F_mean'] = np.round(familiarity_mean, 2)
            summary_data_frame.loc[key, 'mean_dist_origin'] = np.round(mean_dist_from_origin, 2)

            summary_data_frame.loc[key, 'V_med'] = np.round(valence_medoid, 2)
            summary_data_frame.loc[key, 'A_med'] = np.round(arousal_medoid, 2)
            summary_data_frame.loc[key, 'D_med'] = np.round(dominance_medoid, 2)
            summary_data_frame.loc[key, 'L_med'] = np.round(liking_medoid, 2)
            summary_data_frame.loc[key, 'F_med'] = np.round(familiarity_medoid, 2)
            summary_data_frame.loc[key, 'median_dist_origin'] = np.round(median_dist_from_origin, 2)

            summary_data_frame.loc[key, 'V_std'] = np.round(valence_std, 2)
            summary_data_frame.loc[key, 'A_std'] = np.round(arousal_std, 2)
            summary_data_frame.loc[key, 'VA_std'] = np.round(np.sqrt(valence_std**2 + arousal_std**2), 2)
            summary_data_frame.loc[key, 'D_std'] = np.round(dominance_std, 2)
            summary_data_frame.loc[key, 'L_std'] = np.round(liking_std, 2)
            summary_data_frame.loc[key, 'F_std'] = np.round(familiarity_std, 2)
            summary_data_frame.loc[key, 'overall_std'] = np.round(valence_std+arousal_std+dominance_std, 2)

            VideoId = VideoId + 1
        
        print(summary_data_frame)

        if (cleaning_flag == 'after_cleaning') and (not (date == '2018_Oct_10-Oct_20')):
            if 'Emotion_Name' in summary_data_frame.columns.values:
                summary_data_frame['Emt_Name'] = None
                emtNameDict = {}
                QDrantDict = {}

                for vidName, emtArray in zip(summary_data_frame.index.values, summary_data_frame['Emotion_Name']):
                    if not isinstance(emtArray, str):
                        continue
                    
                    tmpArr = []
                    emtNameDict[vidName] = []

                    for str_ in strToComp:                    
                        str_ = str_.split(':')[0]
                        if str_.upper() in emtArray.upper():  ############### Changed the letter case here.
                            #print(emtArray)
                            #print(str_)
                            count = emtArray.count(str_)
                            if count == 1:
                                tmpArr.extend([str_])
                                emtNameDict[vidName].extend([str_])                                
                            else:                                                     
                                tmpArr.extend([str_]*count)
                                emtNameDict[vidName].extend([str_]*count)
                    
                    if len(tmpArr):
                        summary_data_frame.loc[vidName, 'Emt_Name'] = ','.join(tmpArr)
                
                for vidStim in emtNameDict.keys():                                             
                    summary_data_frame.loc[vidStim, 'NEmtRated'] = len(emtNameDict[vidStim])
                    emts = np.unique(emtNameDict[vidStim])
                    tmp = 0
                    for _emt in emts:
                        factor = np.sum(np.array(emtNameDict[vidStim])==_emt)/len(emtNameDict[vidStim])
                        if tmp < factor:
                            tmp = factor
                            summary_data_frame.loc[vidStim, 'emtFactor'] = round(factor,2)
                            summary_data_frame.loc[vidStim, 'MostRated'] = _emt

                    tmp = 0
                    allQds = summary_data_frame.loc[vidStim, 'Quadrant'].split(',')
                    print(allQds)

                    if len(allQds) > 1:
                        summary_data_frame.loc[vidStim, 'NQDRated'] = len(allQds)
                        for quad_ in np.unique(allQds):
                            factor = np.sum(np.array(allQds)==quad_)/summary_data_frame.loc[vidStim, 'NQDRated']
                            if tmp < factor:
                                tmp = factor
                                summary_data_frame.loc[vidStim, 'QuadFactor'] = round(factor,2)
                                summary_data_frame.loc[vidStim, 'MostQuad'] = quad_                                                        
            
        print(summary_data_frame)
        #if videoPrefix == 'WithSixtyNine':
        Stimulat = [i.split('/')[-1] for i in summary_data_frame.index.values]  ### Renaming the experiment Ids taken from csv file
        Stimulat = ['_'.join(i.split(' ')) for i in Stimulat]  ### Renaming the experiment Ids taken from csv file
        Stimulat = ['_'.join(i.split("'")) for i in Stimulat]  ### Renaming the experiment Ids taken from csv file
        Stimulat = ['_'.join(i.split('(')) for i in Stimulat]  ### Renaming the experiment Ids taken from csv file
        Stimulat = ['_'.join(i.split(')')) for i in Stimulat]  ### Renaming the experiment Ids taken from csv file
        Stimulat = ['_'.join(i.split('&')) for i in Stimulat]  ### Renaming the experiment Ids taken from csv file
        Stimulat = [i.split('.')[0] for i in Stimulat] 
        summary_data_frame['Experiment_id'] = Stimulat
        summary_data_frame.set_index('Experiment_id', drop=True, inplace=True)

        if (date == '2018_Oct_10-Nov_15') or (date == '2018_Oct_10-Oct_20') or (date == '2018_Oct_10-Oct_31'):
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
                summary_data_frame.loc[emot, '69Videos'] = 'Final'

        if (date == '2018_Oct_10-Nov_15'): #videoPrefix == 'WithThirty':

            clipDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/block_For_30_Stimuli/Videos'
            videoStims = glob.glob(os.path.join(clipDir, '*'))

            videoStims = [i.split('/')[-1] for i in videoStims]  ### Renaming the experiment Ids taken from csv file
            videoStims = ['_'.join(i.split(' ')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
            videoStims = ['_'.join(i.split("'")) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
            videoStims = ['_'.join(i.split('(')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
            videoStims = ['_'.join(i.split(')')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
            videoStims = ['_'.join(i.split('&')) for i in videoStims]  ### Renaming the experiment Ids taken from csv file
            videoStims = [i.split('.')[0] for i in videoStims] 

            for emot in videoStims:
                summary_data_frame.loc[emot, '30Videos'] = 'Final'

        elif videoPrefix == 'WithForty':
            import pickle
            ### This file is created using the program /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/knowingAboutBlocks.py.
            fortyVideosDir = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey'
            fortyVidoes = pickle.load(open(os.path.join(fortyVideosDir, 'WithAllVideos__BlockInformationForStimuli_Nov_1-Nov_15.pkl'), 'rb'))[1]

            sixyNineVideos = [i.split('/')[-1] for i in summary_data_frame.index.values]  ### Renaming the experiment Ids taken from csv file
            sixyNineVideos = ['_'.join(i.split(' ')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
            sixyNineVideos = ['_'.join(i.split("'")) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
            sixyNineVideos = ['_'.join(i.split('(')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
            sixyNineVideos = ['_'.join(i.split(')')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
            sixyNineVideos = ['_'.join(i.split('&')) for i in sixyNineVideos]  ### Renaming the experiment Ids taken from csv file
            sixyNineVideos = [i.split('.')[0] for i in sixyNineVideos] 
            summary_data_frame['Experiment_id'] = sixyNineVideos
            summary_data_frame.set_index('Experiment_id', drop=True, inplace=True)

            for emot in fortyVidoes:
                summary_data_frame.loc[emot, '40Videos'] = 'Final'            

        os.chdir('..')
        summary_data_frame.to_csv(os.path.join(_thisDir, 'NewTarget', 'summary_data_frame_'+cleaning_flag+date+'.csv'))
        pd.DataFrame(greater_than_50).to_csv(os.path.join(_thisDir, 'NewTarget', 'Greater_Then_50_'+cleaning_flag+date+'.csv'))

########################## Clustering

        if cleaning_flag == 'after_cleaning':
            import sklearn.cluster
            dataToCluster = summary_data_frame[["mean_dist_origin", "VA_std"]]        
            dataToCluster.dropna(inplace=True)

            clusterObject = sklearn.cluster.KMeans(n_clusters=2)
            cluster = clusterObject.fit(dataToCluster)

            cluster_1 = dataToCluster.index.values[np.where(cluster.labels_==0)[0]]
            cluster_2 = dataToCluster.index.values[np.where(cluster.labels_==1)[0]]

            summary_data_frame.loc[cluster_1, 'Cluster_Marker'] = '*'
            summary_data_frame.loc[cluster_2, 'Cluster_Marker'] = '#'            

            if (date == '2018_Oct_10-Oct_20'): #videoPrefix == 'WithSixtyNine':
                NaNVal = summary_data_frame.index.values[np.where(np.isnan(summary_data_frame['VideoId']))[0]]
                summary_data_frame.drop(NaNVal, inplace=True, axis=0)
                selectedForScatter = summary_data_frame.index.values[np.where(summary_data_frame['69Videos']=='Final')[0]]
            elif (date == '2018_Oct_10-Nov_15') and (videoPrefix == 'With69Videos_'):
                selectedForScatter = summary_data_frame.index.values[np.where(summary_data_frame['69Videos']=='Final')[0]]
            elif (date == '2018_Oct_10-Nov_15'): #videoPrefix == 'WithThirty':
                selectedForScatter = summary_data_frame.index.values[np.where(summary_data_frame['30Videos']=='Final')[0]]

            summary_data_frame.to_csv(os.path.join(_thisDir, 'NewTarget', 'WithClustering_summary_data_frame_'+cleaning_flag+date+'.csv'))
            
            from adjustText import adjust_text
            if (videoPrefix == 'WithThirtyVideos_'):
                ax = summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean']].plot.scatter(x='V_mean', y='A_mean', s=40, figsize=(20,10), fontsize=40)
                texts = []
                for v, a, emt in summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean','MostRated']].values:                    
                    texts.append(plt.text(v, a, emt, fontsize=35))
                    #ax.annotate(emt, (v,a), fontsize=30)
              
                plt.ylabel('Arousal Mean', fontsize=30)
                plt.xlabel('Valence Mean', fontsize=30)
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=1.5))

                plt.savefig(os.path.join(_thisDir, 'NewTarget', '%s_Valence_ArousalRepresentationSelectedStimuli%s_%s.png' %(videoPrefix, date, cleaning_flag)), bbox_inches='tight')
                plt.savefig(os.path.join(_thisDir, 'NewTarget', '%s_Valence_ArousalRepresentationSelectedStimuli%s_%s.pdf' %(videoPrefix, date, cleaning_flag)), bbox_inches='tight')
            elif (videoPrefix == 'With69Videos_'):
                plt.clf()
                plt.close()
                ax = summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean']].plot.scatter(x='V_mean', y='A_mean', s=40, fontsize=40)#, figsize=(15,15), fontsize=40)
                texts = []
                
                '''### Annotating data ponts with emotion names
                for v, a, emt in summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean','MostRated']].values:
                    texts.append(plt.text(v, a, emt, fontsize=23))
                    #ax.annotate(emt, (v,a), fontsize=30)
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=1.5))'''
              
                plt.ylabel('Arousal Mean', fontsize=30)
                plt.xlabel('Valence Mean', fontsize=30)                
                
                plt.savefig(os.path.join(_thisDir, 'NewTarget', '%s_Valence_ArousalRepresentationSelectedStimuli%s_%s.png' %(videoPrefix, date, cleaning_flag)), bbox_inches='tight')
                plt.savefig(os.path.join(_thisDir, 'NewTarget', '%s_Valence_ArousalRepresentationSelectedStimuli%s_%s.pdf' %(videoPrefix, date, cleaning_flag)), bbox_inches='tight')
                plt.clf()
                plt.close()

            #elif videoPrefix == 'WithSixtyNine':            
            '''else:
                pdb.set_trace()
                ax = summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean']].plot.scatter(x='V_mean', y='A_mean', s=40, figsize=(20,10), fontsize=40)
                texts = []
                for v, a, emt in summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean','WarinnerName']].values:                    
                    texts.append(plt.text(v, a, emt, fontsize=35))
                    #ax.annotate(emt, (v,a), fontsize=30)
              
                plt.ylabel('Arousal Mean', fontsize=30)
                plt.xlabel('Valence Mean', fontsize=30)
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=1.5))

                #summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean']].plot.scatter(x='V_mean', y='A_mean', s=40, figsize=(20,10), fontsize=40)
                #plt.ylabel('Arousal Mean', fontsize=30)
                #plt.xlabel('Valence Mean', fontsize=30)

                plt.savefig(os.path.join(_thisDir, 'NewTarget', 'Valence_ArousalRepresentationSelectedStimuli%s_%s.png' %(date, cleaning_flag)), bbox_inches='tight')
                plt.savefig(os.path.join(_thisDir, 'NewTarget', 'Valence_ArousalRepresentationSelectedStimuli%s_%s.pdf' %(date, cleaning_flag)), bbox_inches='tight')'''
        print("After this please go to VAD_Plotting for summarized data in this module or go for Data cleaning")

        #################################### Incluse MAD information also.

        if cleaning_flag == 'after_cleaning':
            MADFile = 'RProgram_MeanAbsoluteDifference_%s%s.csv' %(cleaning_flag, date)
            MADFrame = pd.read_csv(os.path.join(_thisDir, 'NewTarget', MADFile), index_col = 0)

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

            for vidStim in summary_data_frame.index.values:

                try:
                    summary_data_frame.loc[vidStim, 'VMAD'] = MADFrame.loc[vidStim, 'VMAD']
                    summary_data_frame.loc[vidStim, 'AMAD'] = MADFrame.loc[vidStim, 'AMAD']
                    summary_data_frame.loc[vidStim, 'DMAD'] = MADFrame.loc[vidStim, 'DMAD']
                    summary_data_frame.loc[vidStim, 'LMAD'] = MADFrame.loc[vidStim, 'LMAD']
                    summary_data_frame.loc[vidStim, 'FMAD'] = MADFrame.loc[vidStim, 'FMAD']           
                except:
                    continue

    ####################### Concordance Results 
    ## This file is created using R Program: /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/IRRTest_KendallVegan_StimulationWise.R
            CCCFile = 'AllStimuli_CCC_Test_Result_%s.csv' %(date.split('2018_')[1])
            CCCFile_Cluster_1 = 'Cluster-1_CCC_Test_Result_%s.csv' %(date.split('2018_')[1])
            CCCFile_Cluster_2 = 'Cluster-2_CCC_Test_Result_%s.csv' %(date.split('2018_')[1])
            CCCFrame = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, CCCFile), index_col = 0)
            CCCFrame_Cluster_1 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, CCCFile_Cluster_1), index_col = 0)
            CCCFrame_Cluster_2 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, CCCFile_Cluster_2), index_col = 0)

            CCCEmt = CCCFrame.index.values
            CCCEmt = [i.split('/')[-1] for i in CCCFrame.index.values]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split(' ')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split("'")) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split('(')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split(')')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split('&')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = [i.split('.')[0] for i in CCCEmt]
            CCCFrame['Experiment_id'] = CCCEmt
            CCCFrame.set_index('Experiment_id', drop=True, inplace=True)

            CCCEmt = [i.split('/')[-1] for i in CCCFrame_Cluster_1.index.values]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split(' ')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split("'")) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split('(')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split(')')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split('&')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = [i.split('.')[0] for i in CCCEmt]
            CCCFrame_Cluster_1['Experiment_id'] = CCCEmt
            CCCFrame_Cluster_1.set_index('Experiment_id', drop=True, inplace=True)

            CCCEmt = [i.split('/')[-1] for i in CCCFrame_Cluster_2.index.values]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split(' ')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split("'")) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split('(')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split(')')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = ['_'.join(i.split('&')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
            CCCEmt = [i.split('.')[0] for i in CCCEmt]
            CCCFrame_Cluster_2['Experiment_id'] = CCCEmt
            CCCFrame_Cluster_2.set_index('Experiment_id', drop=True, inplace=True)

            for vidStim in summary_data_frame.index.values:
                try:
                    summary_data_frame.loc[vidStim, 'AllStim_W'] = CCCFrame.loc[vidStim, 'Concord_W']
                    summary_data_frame.loc[vidStim, 'AllStim_F'] = CCCFrame.loc[vidStim, 'Concord_F']
                    summary_data_frame.loc[vidStim, 'AllStim_Prob.F'] = CCCFrame.loc[vidStim, 'Concord_Prob.F']
                    summary_data_frame.loc[vidStim, 'AllStim_Chi2'] = CCCFrame.loc[vidStim, 'Concord_Chi2']
                    summary_data_frame.loc[vidStim, 'AllStim_Prob.perm'] = CCCFrame.loc[vidStim, 'Concord_Prob.perm']           
                    summary_data_frame.loc[vidStim, 'AllStim_Dimension'] = CCCFrame.loc[vidStim, 'Concord_Dimension']
                    summary_data_frame.loc[vidStim, 'AllStimCateg'] = CCCFrame.loc[vidStim, 'ConcordCateg']                

                    summary_data_frame.loc[vidStim, 'Clust_1_W'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_W']
                    summary_data_frame.loc[vidStim, 'Clust_1_F'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_F']
                    summary_data_frame.loc[vidStim, 'Clust_1_Prob.F'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Prob.F']
                    summary_data_frame.loc[vidStim, 'Clust_1_Chi2'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Chi2']
                    summary_data_frame.loc[vidStim, 'Clust_1_Prob.perm'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Prob.perm']           
                    summary_data_frame.loc[vidStim, 'Clust_1_Dimension'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Dimension']
                    summary_data_frame.loc[vidStim, 'Clust_1Categ'] = CCCFrame_Cluster_1.loc[vidStim, 'ConcordCateg']                

                    summary_data_frame.loc[vidStim, 'Clust_2_W'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_W']
                    summary_data_frame.loc[vidStim, 'Clust_2_F'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_F']
                    summary_data_frame.loc[vidStim, 'Clust_2_Prob.F'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Prob.F']
                    summary_data_frame.loc[vidStim, 'Clust_2_Chi2'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Chi2']
                    summary_data_frame.loc[vidStim, 'Clust_2_Prob.perm'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Prob.perm']           
                    summary_data_frame.loc[vidStim, 'Clust_2_Dimension'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Dimension']
                    summary_data_frame.loc[vidStim, 'Clust_2Categ'] = CCCFrame_Cluster_2.loc[vidStim, 'ConcordCateg']                
                                                    
                except:
                    continue


            #### Assigning emotions
            if 'Oct_10-Nov_15' in date:
                '''result_ = pd.read_excel(os.path.join('/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/Validation_Emotion/NewTarget', 
                    'WithCCC_WithClustering_summary_data_frame_after_cleaning2018_Oct_10-Nov_15_fullRepresentationWith69AndThirty_WithClusteringInformation.xlsx'), index_col=0)

                thirtyVid = []
                for vidId, thrt in zip(summary_data_frame.index.values, summary_data_frame['30Videos']):
                    pdb.set_trace()
                    try:
                        summary_data_frame.loc[vidId, 'Emt_Name'] = result_.loc[vidId, 'Emt_Name']
                    except:
                        pdb.set_trace()

                    summary_data_frame.loc[vidId, 'emtFactor'] = result_.loc[vidId, 'emtFactor']
                    summary_data_frame.loc[vidId, 'MostRated'] = result_.loc[vidId, 'MostRated']

                    if isinstance(thrt, str):
                        thirtyVid.extend([result_.loc[vidId, 'MostRated']])
                thirtyVid = np.unique(thirtyVid)'''

                thirtyVid = []
                for vidId, thrt in zip(summary_data_frame.index.values, summary_data_frame['30Videos']):
                    if isinstance(thrt, str):
                        thirtyVid.extend([summary_data_frame.loc[vidId, 'MostRated']])
                thirtyVid = np.unique(thirtyVid)                

            if 'Oct_10-Oct_20' in date:

                if summary_data_frame.index.duplicated().any():                   
                    dupIdx = np.where(summary_data_frame.index.duplicated())[0][0] ## Finding the duplicate index here.
                    summary_data_frame.reset_index(inplace=True)  ## Reseting the index to serial number so that dupIdx can be deleted.
                    summary_data_frame.drop(dupIdx, axis=0, inplace=True) ## Dropping the duplicate index here.
                    summary_data_frame.set_index('Experiment_id', drop=True, inplace=True) ## Setting the experiment Id again.

                emtConsid = []
                emtConsidDict = {}
                emtConsidConsist = {}
                emtConsidDimenst = {}
                emtConsidProbabl = {}
                strToCompNew = [i.lower() for i in strToComp]
                strToCompNew = [i.split(':')[0] for i in strToCompNew]

                warinnerData = pd.read_csv(os.path.join(_thisDir, 'Ratings_Warriner_et_al.csv'), index_col = 0)

                for ke_, allstim, clust1, clust2, dim, dim1, dim2, prob, prob1, prob2 in zip(summary_data_frame.index.values, summary_data_frame['AllStim_W'], summary_data_frame['Clust_1_W'], 
                    summary_data_frame['Clust_2_W'], summary_data_frame['AllStim_Dimension'], summary_data_frame['Clust_1_Dimension'], summary_data_frame['Clust_2_Dimension'], 
                    summary_data_frame['AllStim_Prob.perm'], summary_data_frame['Clust_1_Prob.perm'], summary_data_frame['Clust_2_Prob.perm']):  
                    minn = 1000
                    for ke_2, _val, _arl in zip(warinnerData['Word'].values, warinnerData['V.Mean.Sum'].values, warinnerData['A.Mean.Sum'].values):

                        if not isinstance(ke_2, str):
                            continue
                        if ke_2.lower() in strToCompNew:                            
                            EuclidDist = np.sqrt(np.power(summary_data_frame.loc[ke_, 'V_mean']-_val, 2) + (np.power(summary_data_frame.loc[ke_, 'A_mean']-_arl, 2)))

                            try:
                                if EuclidDist < minn:
                                    minn = EuclidDist
                                    summary_data_frame.loc[ke_, 'WarinnerName'] = ke_2
                                    summary_data_frame.loc[ke_, 'WarinnerDist'] = np.round(EuclidDist,2)
                                    print(ke_, ke_2)
                            except:
                                pdb.set_trace()                            

                    if isinstance(summary_data_frame.loc[ke_, '69Videos'], str):
                        if 'Final' in summary_data_frame.loc[ke_, '69Videos'] :
                            emtConsid.extend([summary_data_frame.loc[ke_, 'WarinnerName']])
                            #coeffArr = np.array([allstim, clust1, clust2]).reshape(1,3)

                            if isinstance(dim, str):
                                coeffArrD = [int(dim.split(',')[1])] #np.array([int(dim.split(',')[1]), int(dim1.split(',')[1]), int(dim2.split(',')[1])]).reshape(1,3)
                                coeffArrP = [prob]
                                coeffArrW = [allstim]
                            else:
                                coeffArrD = [0]
                                coeffArrP = [0]
                                coeffArrW = [0]
                            if isinstance(dim1, str):
                                coeffArrD.extend([int(dim1.split(',')[1])])
                                coeffArrP.extend([prob1])
                                coeffArrW.extend([clust1])
                            else:
                                coeffArrD.extend([0])
                                coeffArrP.extend([0])
                                coeffArrW.extend([0])

                            if isinstance(dim2, str):
                                coeffArrD.extend([int(dim2.split(',')[1])])
                                coeffArrP.extend([prob2])
                                coeffArrW.extend([clust2])
                            else:
                                coeffArrD.extend([0])
                                coeffArrP.extend([0])
                                coeffArrW.extend([0])

                            coeffArrD = np.array(coeffArrD).reshape(1,3)
                            coeffArrP = np.array(coeffArrP).reshape(1,3)
                            coeffArrW = np.array(coeffArrW).reshape(1,3)

                            if summary_data_frame.loc[ke_, 'WarinnerName'] in emtConsidDict.keys():
                                emtConsidDict[summary_data_frame.loc[ke_, 'WarinnerName']] += 1                            
                                emtConsidConsist[summary_data_frame.loc[ke_, 'WarinnerName']] = np.concatenate((emtConsidConsist[summary_data_frame.loc[ke_, 'WarinnerName']], coeffArrW), axis=0)
                                emtConsidDimenst[summary_data_frame.loc[ke_, 'WarinnerName']] = np.concatenate((emtConsidDimenst[summary_data_frame.loc[ke_, 'WarinnerName']], coeffArrD), axis=0)
                                emtConsidProbabl[summary_data_frame.loc[ke_, 'WarinnerName']] = np.concatenate((emtConsidProbabl[summary_data_frame.loc[ke_, 'WarinnerName']], coeffArrP), axis=0)
                            else:
                                emtConsidDict[summary_data_frame.loc[ke_, 'WarinnerName']] = 1
                                emtConsidConsist[summary_data_frame.loc[ke_, 'WarinnerName']] = coeffArrW
                                emtConsidDimenst[summary_data_frame.loc[ke_, 'WarinnerName']] = coeffArrD
                                emtConsidProbabl[summary_data_frame.loc[ke_, 'WarinnerName']] = coeffArrP

                ax = summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean']].plot.scatter(x='V_mean', y='A_mean', s=40, figsize=(20,10), fontsize=40)
                texts = []
                for v, a, emt in summary_data_frame.loc[selectedForScatter, ['V_mean', 'A_mean','WarinnerName']].values:                    
                    texts.append(plt.text(v, a, emt, fontsize=25))
              
                plt.ylabel('Arousal Mean', fontsize=30)
                plt.xlabel('Valence Mean', fontsize=30)
                adjust_text(texts, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=1.5))

                plt.savefig(os.path.join(_thisDir, 'NewTarget', 'Valence_ArousalRepresentationSelectedStimuli%s_%s.png' %(date, cleaning_flag)), bbox_inches='tight')
                plt.savefig(os.path.join(_thisDir, 'NewTarget', 'Valence_ArousalRepresentationSelectedStimuli%s_%s.pdf' %(date, cleaning_flag)), bbox_inches='tight')                                

                pdb.set_trace()
            else:
                if summary_data_frame.index.duplicated().any():                   
                    dupIdx = np.where(summary_data_frame.index.duplicated())[0][0] ## Finding the duplicate index here.
                    summary_data_frame.reset_index(inplace=True)  ## Reseting the index to serial number so that dupIdx can be deleted.
                    summary_data_frame.drop(dupIdx, axis=0, inplace=True) ## Dropping the duplicate index here.
                    summary_data_frame.set_index('Experiment_id', drop=True, inplace=True) ## Setting the experiment Id again.
                emtConsid = []
                emtConsidDict = {}
                emtConsidConsist = {}
                emtConsidDimenst = {}
                emtConsidProbabl = {}
                emtThrtyStim = {}

                for ke_, most, allstim, clust1, clust2, dim, dim1, dim2, prob, prob1, prob2 in zip(summary_data_frame.index.values, summary_data_frame['MostRated'], summary_data_frame['AllStim_W'], 
                    summary_data_frame['Clust_1_W'], summary_data_frame['Clust_2_W'], summary_data_frame['AllStim_Dimension'], summary_data_frame['Clust_1_Dimension'], 
                    summary_data_frame['Clust_2_Dimension'], summary_data_frame['AllStim_Prob.perm'], summary_data_frame['Clust_1_Prob.perm'], summary_data_frame['Clust_2_Prob.perm']):  
                    
                    if isinstance(most, str):
                        if isinstance(dim, str):
                            coeffArrD = [int(dim.split(',')[1])] #np.array([int(dim.split(',')[1]), int(dim1.split(',')[1]), int(dim2.split(',')[1])]).reshape(1,3)
                            coeffArrP = [prob]
                            coeffArrW = [allstim]
                        else:
                            coeffArrD = [0]
                            coeffArrP = [0]
                            coeffArrW = [0]
                        if isinstance(dim1, str):
                            coeffArrD.extend([int(dim1.split(',')[1])])
                            coeffArrP.extend([prob1])
                            coeffArrW.extend([clust1])
                        else:
                            coeffArrD.extend([0])
                            coeffArrP.extend([0])
                            coeffArrW.extend([0])

                        if isinstance(dim2, str):
                            coeffArrD.extend([int(dim2.split(',')[1])])
                            coeffArrP.extend([prob2])
                            coeffArrW.extend([clust2])
                        else:
                            coeffArrD.extend([0])
                            coeffArrP.extend([0])
                            coeffArrW.extend([0])

                        coeffArrD = np.array(coeffArrD).reshape(1,3)
                        coeffArrP = np.array(coeffArrP).reshape(1,3)
                        coeffArrW = np.array(coeffArrW).reshape(1,3)

                        if most in emtConsidDict.keys():                            
                            emtConsidDict[most] += 1
                            emtConsidConsist[most] = np.concatenate((emtConsidConsist[most], coeffArrW), axis=0)
                            emtConsidDimenst[most] = np.concatenate((emtConsidDimenst[most], coeffArrD), axis=0)
                            emtConsidProbabl[most] = np.concatenate((emtConsidProbabl[most], coeffArrP), axis=0)

                        else:
                            emtConsidDict[most] = 1
                            emtConsidConsist[most] = coeffArrW
                            emtConsidDimenst[most] = coeffArrD
                            emtConsidProbabl[most] = coeffArrP

                        if most in thirtyVid:
                            emtThrtyStim[most] = '$'

            emtConsidConsistOld = emtConsidConsist.copy()
            emtConsidProbablMark = {}

            for ke_ in emtConsidConsist.keys():
                emtConsidConsist[ke_] = np.round([np.mean(i[i!=0]) for i in np.transpose(emtConsidConsist[ke_])], 3)   
                emtConsidDimenst[ke_] = np.round(np.mean(emtConsidDimenst[ke_],axis=0),0).astype(np.int) #np.round([np.mean(i[i!=0]) for i in np.transpose(emtConsidDimenst[ke_])], 0).astype(np.int)
                emtConsidProbabl[ke_] = np.round([np.mean(i[i!=0]) for i in np.transpose(emtConsidProbabl[ke_])], 4)

                
                prob, prob1, prob2 = emtConsidProbabl[ke_]
                print(prob, prob1, prob2)

                if prob < 0.01:
                    coeffArrP = ['*']
                elif (prob < 0.05) and (prob > 0.01):
                    coeffArrP = ['#']
                else:
                    coeffArrP = ['']

                if prob1 < 0.01:
                    coeffArrP.extend(['*']) 
                elif (prob1 < 0.05) and (prob1 > 0.01):
                    coeffArrP.extend(['#'])
                else:
                    coeffArrP.extend([''])

                if prob2 < 0.01:
                    coeffArrP.extend(['*']) 
                elif (prob2 < 0.05) and (prob2 > 0.01):
                    coeffArrP.extend(['#'])
                else:
                    coeffArrP.extend([''])

                emtConsidProbablMark[ke_] = coeffArrP

            print(np.unique(emtConsid))
            print(emtConsidDict)

            W_dataFrame = pd.DataFrame.from_dict(emtConsidConsist, orient='index', columns=['W_AllPart', 'W_Clust1', 'W_Clust2'])
            D_dataFrame = pd.DataFrame.from_dict(emtConsidDimenst, orient='index', columns=['W_AllPart', 'W_Clust1', 'W_Clust2'])
            P_dataFrame = pd.DataFrame.from_dict(emtConsidProbablMark, orient='index', columns=['W_AllPart', 'W_Clust1', 'W_Clust2'])
            #Thrty_dataFrame = pd.DataFrame.from_dict(emtThrtyStim, orient='index')

            fig = plt.figure(figsize=(20, 10))
            plt.rcParams['font.size'] = 35
            gs = gridspec.GridSpec(2, 1, height_ratios=[20,1], hspace=0.15, left=0.1, right=0.99, bottom=0.31, top=0.94) 
            ax = plt.subplot(gs[0, 0])   
            if 'Oct_10-Oct_20' in date:
                #ax.set_title("208 Stimuli Assigned Emotion Name From Affective Norm Space by Warriner", fontsize=30)
                ax.set_title("For stimuli selected after the first stage", fontsize=30)
            else:
                ax.set_title("For stimuli selected after the second stage", fontsize=30)

            W_dataFrame.fillna(0, inplace=True)
            W_dataFrame.plot.bar(ax=ax, width=0.8)
            ax.set_yticks(np.arange(0, np.max(W_dataFrame.values)+0.2, 0.1))
            ax.set_ylabel('Kendall W Coefficient')   
            if 'Oct_10-Oct_20' in date:
                plt.legend(ncol=3, fontsize=25, fancybox=True, framealpha=0.1, bbox_to_anchor=(0.81, np.max(W_dataFrame.values)-0.03, 0.2, 0.2), columnspacing=0.5, markerscale=0.2)
                ax.set_xlabel('Emotion', labelpad=5)
            else:
                plt.legend(ncol=3, fontsize=25, fancybox=True, framealpha=0.1, bbox_to_anchor=(0.35, np.max(W_dataFrame.values)+0.08, 0.2, 0.2), columnspacing=0.5, markerscale=0.5)
                ax.set_xlabel('Emotion', labelpad=-30)

            _row = -1
            _col = -1
            for pIdx, p_ in enumerate(ax.patches):
                if (pIdx%len(D_dataFrame)) == 0:                
                    _row = -1
                    _col = _col + 1
                _row = _row + 1            
                
                #print(_row, pIdx%3, D_dataFrame.values[_row][_col], p_.xy)

                if _col == 0:
                    ax.annotate(format(D_dataFrame.values[_row][_col]), 
                                   (p_.get_x() + p_.get_width() / 2., p_.get_height()+0.09), 
                                   ha = 'center', va = 'center', 
                                   size=25,
                                   xytext = (0, -12), 
                                   textcoords = 'offset points')        
                else:
                    ax.annotate(format(D_dataFrame.values[_row][_col]), 
                                   (p_.get_x() + p_.get_width() / 2., p_.get_height()+0.05), 
                                   ha = 'center', va = 'center', 
                                   size=25,
                                   xytext = (0, -12), 
                                   textcoords = 'offset points')                        

                if _col == 0:
                    ax.annotate(format(P_dataFrame.values[_row][_col]), 
                                   (p_.get_x() + p_.get_width() / 2., p_.get_height()+0.15), 
                                   ha = 'center', va = 'center', 
                                   size=25,
                                   xytext = (0, -12), 
                                   textcoords = 'offset points')        
                else:
                    ax.annotate(format(P_dataFrame.values[_row][_col]), 
                                   (p_.get_x() + p_.get_width() / 2., p_.get_height()+0.1), 
                                   ha = 'center', va = 'center', 
                                   size=25,
                                   xytext = (0, -12), 
                                   textcoords = 'offset points')           

                #if _col == 1:
                #    print(ax.get_xticklabels()[_row].get_text())
                #    if ax.get_xticklabels()[_row].get_text() in thirtyVid:
                #        ax.get_xticklabels()[_row].set_color("red")
                    #Thrty_dataFrame[_row][_col-1]
            pdb.set_trace()
            plt.savefig(os.path.join(_thisDir, 'NewTarget', 'InterRater_Agreement_'+cleaning_flag+date+'.png'))#, layout='tight')
            plt.savefig(os.path.join(_thisDir, 'NewTarget', 'InterRater_Agreement_'+cleaning_flag+date+'.pdf'))#, layout='tight')     

            pdb.set_trace()
####################### kappam.fleiss Statistics
## This file is created using R Program: /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/IRRTest_KendallVegan_StimulationWise.R
        '''kappaFile = 'AllStimuli_Kappa_Test_Result_%s.csv' %(date.split('2018_')[1])
        kappaFile_Cluster_1 = 'Cluster-1_Kappa_Test_Result_%s.csv' %(date.split('2018_')[1])
        kappaFile_Cluster_2 = 'Cluster-2_Kappa_Test_Result_%s.csv' %(date.split('2018_')[1])
        kappaFrame = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, kappaFile), index_col = 0)
        kappaFrame_Cluster_1 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, kappaFile_Cluster_1), index_col = 0)
        kappaFrame_Cluster_2 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, kappaFile_Cluster_2), index_col = 0)

        kappaEmt = kappaFrame.index.values
        kappaEmt = [i.split('/')[-1] for i in kappaFrame.index.values]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(' ')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split("'")) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('(')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(')')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('&')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = [i.split('.')[0] for i in kappaEmt]
        kappaFrame['Experiment_id'] = kappaEmt
        kappaFrame.set_index('Experiment_id', drop=True, inplace=True)

        kappaEmt = [i.split('/')[-1] for i in kappaFrame_Cluster_1.index.values]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(' ')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split("'")) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('(')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(')')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('&')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = [i.split('.')[0] for i in kappaEmt]
        kappaFrame_Cluster_1['Experiment_id'] = kappaEmt
        kappaFrame_Cluster_1.set_index('Experiment_id', drop=True, inplace=True)

        kappaEmt = [i.split('/')[-1] for i in kappaFrame_Cluster_2.index.values]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(' ')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split("'")) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('(')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(')')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('&')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = [i.split('.')[0] for i in kappaEmt]
        kappaFrame_Cluster_2['Experiment_id'] = kappaEmt
        kappaFrame_Cluster_2.set_index('Experiment_id', drop=True, inplace=True)

        for vidStim in summary_data_frame.index.values:
            try:
                "kappa", 'stats', "p_val", 'subjects', 'raters'
                summary_data_frame.loc[vidStim, 'K_kappa'] = kappaFrame.loc[vidStim, 'kappa']
                summary_data_frame.loc[vidStim, 'K_stats'] = kappaFrame.loc[vidStim, 'stats']
                summary_data_frame.loc[vidStim, 'K_pVal'] = kappaFrame.loc[vidStim, 'p_val']
                summary_data_frame.loc[vidStim, 'K_subjects'] = kappaFrame.loc[vidStim, 'subjects']
                summary_data_frame.loc[vidStim, 'K_raters'] = kappaFrame.loc[vidStim, 'raters']           

                summary_data_frame.loc[vidStim, 'K_kappa1'] = kappaFrame_Cluster_1.loc[vidStim, 'kappa']
                summary_data_frame.loc[vidStim, 'K_stats1'] = kappaFrame_Cluster_1.loc[vidStim, 'stats']
                summary_data_frame.loc[vidStim, 'K_pVal1'] = kappaFrame_Cluster_1.loc[vidStim, 'p_val']
                summary_data_frame.loc[vidStim, 'K_subjects1'] = kappaFrame_Cluster_1.loc[vidStim, 'subjects']
                summary_data_frame.loc[vidStim, 'K_raters1'] = kappaFrame_Cluster_1.loc[vidStim, 'raters']           

                summary_data_frame.loc[vidStim, 'K_kappa2'] = kappaFrame_Cluster_2.loc[vidStim, 'kappa']
                summary_data_frame.loc[vidStim, 'K_stats2'] = kappaFrame_Cluster_2.loc[vidStim, 'stats']
                summary_data_frame.loc[vidStim, 'K_pVal2'] = kappaFrame_Cluster_2.loc[vidStim, 'p_val']
                summary_data_frame.loc[vidStim, 'K_subjects2'] = kappaFrame_Cluster_2.loc[vidStim, 'subjects']
                summary_data_frame.loc[vidStim, 'K_raters2'] = kappaFrame_Cluster_2.loc[vidStim, 'raters']           
                                                
            except:
                continue'''


####################### No Dominance: Concordance Results 
## This file is created using R Program: /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/IRRTest_KendallVegan_StimulationWise.R
        '''CCCFile = 'NoDominance_AllStimuli_CCC_Test_Result_%s.csv' %(date.split('2018_')[1])
        CCCFile_Cluster_1 = 'NoDominance_Cluster-1_CCC_Test_Result_%s.csv' %(date.split('2018_')[1])
        CCCFile_Cluster_2 = 'NoDominance_Cluster-2_CCC_Test_Result_%s.csv' %(date.split('2018_')[1])
        CCCFrame = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, CCCFile), index_col = 0)
        CCCFrame_Cluster_1 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, CCCFile_Cluster_1), index_col = 0)
        CCCFrame_Cluster_2 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, CCCFile_Cluster_2), index_col = 0)

        CCCEmt = CCCFrame.index.values
        CCCEmt = [i.split('/')[-1] for i in CCCFrame.index.values]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split(' ')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split("'")) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split('(')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split(')')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split('&')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = [i.split('.')[0] for i in CCCEmt]
        CCCFrame['Experiment_id'] = CCCEmt
        CCCFrame.set_index('Experiment_id', drop=True, inplace=True)

        CCCEmt = [i.split('/')[-1] for i in CCCFrame_Cluster_1.index.values]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split(' ')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split("'")) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split('(')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split(')')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split('&')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = [i.split('.')[0] for i in CCCEmt]
        CCCFrame_Cluster_1['Experiment_id'] = CCCEmt
        CCCFrame_Cluster_1.set_index('Experiment_id', drop=True, inplace=True)

        CCCEmt = [i.split('/')[-1] for i in CCCFrame_Cluster_2.index.values]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split(' ')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split("'")) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split('(')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split(')')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = ['_'.join(i.split('&')) for i in CCCEmt]  ### Renaming the experiment Ids taken from csv file
        CCCEmt = [i.split('.')[0] for i in CCCEmt]
        CCCFrame_Cluster_2['Experiment_id'] = CCCEmt
        CCCFrame_Cluster_2.set_index('Experiment_id', drop=True, inplace=True)

        for vidStim in summary_data_frame.index.values:
            try:
                summary_data_frame.loc[vidStim, 'NpDom_W'] = CCCFrame.loc[vidStim, 'Concord_W']
                summary_data_frame.loc[vidStim, 'NpDom_F'] = CCCFrame.loc[vidStim, 'Concord_F']
                summary_data_frame.loc[vidStim, 'NpDom_Prob.F'] = CCCFrame.loc[vidStim, 'Concord_Prob.F']
                summary_data_frame.loc[vidStim, 'NpDom_Chi2'] = CCCFrame.loc[vidStim, 'Concord_Chi2']
                summary_data_frame.loc[vidStim, 'NpDom_Prob.perm'] = CCCFrame.loc[vidStim, 'Concord_Prob.perm']           
                summary_data_frame.loc[vidStim, 'NpDom_Dimension'] = CCCFrame.loc[vidStim, 'Concord_Dimension']
                summary_data_frame.loc[vidStim, 'NpDom_Categ'] = CCCFrame.loc[vidStim, 'ConcordCateg']                

                summary_data_frame.loc[vidStim, 'NpDom_W1'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_W']
                summary_data_frame.loc[vidStim, 'NpDom_F1'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_F']
                summary_data_frame.loc[vidStim, 'NpDom_Prob.F1'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Prob.F']
                summary_data_frame.loc[vidStim, 'NpDom_Chi21'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Chi2']
                summary_data_frame.loc[vidStim, 'NpDom_Prob.perm1'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Prob.perm']           
                summary_data_frame.loc[vidStim, 'NpDom_Dimension1'] = CCCFrame_Cluster_1.loc[vidStim, 'Concord_Dimension']
                summary_data_frame.loc[vidStim, 'NpDomCateg1'] = CCCFrame_Cluster_1.loc[vidStim, 'ConcordCateg']                

                summary_data_frame.loc[vidStim, 'NpDom_W2'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_W']
                summary_data_frame.loc[vidStim, 'NpDom_F2'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_F']
                summary_data_frame.loc[vidStim, 'NpDom_Prob.F2'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Prob.F']
                summary_data_frame.loc[vidStim, 'NpDom_Chi22'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Chi2']
                summary_data_frame.loc[vidStim, 'NpDom_Prob.perm2'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Prob.perm']           
                summary_data_frame.loc[vidStim, 'NpDom_Dimension2'] = CCCFrame_Cluster_2.loc[vidStim, 'Concord_Dimension']
                summary_data_frame.loc[vidStim, 'NpDomCateg2'] = CCCFrame_Cluster_2.loc[vidStim, 'ConcordCateg']                
                                                
            except:
                continue

####################### No Dominance: kappam.fleiss Statistics
## This file is created using R Program: /mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/IRRTest_KendallVegan_StimulationWise.R
        kappaFile = 'NoDominance_AllStimuli_Kappa_Test_Result_%s.csv' %(date.split('2018_')[1])
        kappaFile_Cluster_1 = 'NoDominance_Cluster-1_Kappa_Test_Result_%s.csv' %(date.split('2018_')[1])
        kappaFile_Cluster_2 = 'NoDominance_Cluster-2_Kappa_Test_Result_%s.csv' %(date.split('2018_')[1])
        kappaFrame = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, kappaFile), index_col = 0)
        kappaFrame_Cluster_1 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, kappaFile_Cluster_1), index_col = 0)
        kappaFrame_Cluster_2 = pd.read_csv(os.path.join(_thisDir, 'NewTarget', videoPrefix, kappaFile_Cluster_2), index_col = 0)

        kappaEmt = kappaFrame.index.values
        kappaEmt = [i.split('/')[-1] for i in kappaFrame.index.values]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(' ')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split("'")) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('(')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(')')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('&')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = [i.split('.')[0] for i in kappaEmt]
        kappaFrame['Experiment_id'] = kappaEmt
        kappaFrame.set_index('Experiment_id', drop=True, inplace=True)

        kappaEmt = [i.split('/')[-1] for i in kappaFrame_Cluster_1.index.values]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(' ')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split("'")) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('(')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(')')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('&')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = [i.split('.')[0] for i in kappaEmt]
        kappaFrame_Cluster_1['Experiment_id'] = kappaEmt
        kappaFrame_Cluster_1.set_index('Experiment_id', drop=True, inplace=True)

        kappaEmt = [i.split('/')[-1] for i in kappaFrame_Cluster_2.index.values]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(' ')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split("'")) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('(')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split(')')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = ['_'.join(i.split('&')) for i in kappaEmt]  ### Renaming the experiment Ids taken from csv file
        kappaEmt = [i.split('.')[0] for i in kappaEmt]
        kappaFrame_Cluster_2['Experiment_id'] = kappaEmt
        kappaFrame_Cluster_2.set_index('Experiment_id', drop=True, inplace=True)

        for vidStim in summary_data_frame.index.values:
            try:
                #"kappa", 'stats', "p_val", 'subjects', 'raters'
                summary_data_frame.loc[vidStim, 'NpDom_kappa'] = kappaFrame.loc[vidStim, 'kappa']
                summary_data_frame.loc[vidStim, 'NpDom_stats'] = kappaFrame.loc[vidStim, 'stats']
                summary_data_frame.loc[vidStim, 'NpDom_pVal'] = kappaFrame.loc[vidStim, 'p_val']
                summary_data_frame.loc[vidStim, 'NpDom_subjects'] = kappaFrame.loc[vidStim, 'subjects']
                summary_data_frame.loc[vidStim, 'NpDom_raters'] = kappaFrame.loc[vidStim, 'raters']           

                summary_data_frame.loc[vidStim, 'NpDom_kappa1'] = kappaFrame_Cluster_1.loc[vidStim, 'kappa']
                summary_data_frame.loc[vidStim, 'NpDom_stats1'] = kappaFrame_Cluster_1.loc[vidStim, 'stats']
                summary_data_frame.loc[vidStim, 'NpDom_pVal1'] = kappaFrame_Cluster_1.loc[vidStim, 'p_val']
                summary_data_frame.loc[vidStim, 'NpDom_subjects1'] = kappaFrame_Cluster_1.loc[vidStim, 'subjects']
                summary_data_frame.loc[vidStim, 'NpDom_raters1'] = kappaFrame_Cluster_1.loc[vidStim, 'raters']           

                summary_data_frame.loc[vidStim, 'NpDom_kappa2'] = kappaFrame_Cluster_2.loc[vidStim, 'kappa']
                summary_data_frame.loc[vidStim, 'NpDom_stats2'] = kappaFrame_Cluster_2.loc[vidStim, 'stats']
                summary_data_frame.loc[vidStim, 'NpDom_pVal2'] = kappaFrame_Cluster_2.loc[vidStim, 'p_val']
                summary_data_frame.loc[vidStim, 'NpDom_subjects2'] = kappaFrame_Cluster_2.loc[vidStim, 'subjects']
                summary_data_frame.loc[vidStim, 'NpDom_raters2'] = kappaFrame_Cluster_2.loc[vidStim, 'raters']           
                                                
            except:
                continue'''

        pdb.set_trace()
        summary_data_frame.to_csv(os.path.join(_thisDir, 'NewTarget', 'WithCCC_kappa_WithClustering_summary_data_frame_'+cleaning_flag+date+'_withClusterInformation.csv'))
        
        '''for vidStim in VideosWithEmotions.keys():
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
                overallStats.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']] = MADFrame.loc[vidStim, ['VMAD', 'AMAD', 'DMAD', 'LMAD', 'FMAD']].values[0]'''


def VAD_Plotting(file_name = None, cleaning_flag = None, date=''):
# ========================>>>>>>>>  >>>>>>>>>>>>>>>>> VAD Plotting <<<<<<<<<<<<<<<<<<<<<<===================================

    # file_name = My_Experiment_Ratings_after_cleaning.csv
    if cleaning_flag == None:
        raise ValueError("Please enter cleaning flag. 0 for without cleaning and 1 for after cleaning")
    elif cleaning_flag == 0:
        cleaning_flag = 'before_cleaning'
    else:
        cleaning_flag = 'after_cleaning'

    path_for_file = os.getcwd()

    rating_data = pd.read_csv(os.path.join(_thisDir, 'NewTarget', file_name))
    data_summary = pd.read_csv(os.path.join(_thisDir, 'NewTarget', 'summary_data_frame_'+cleaning_flag+date+'.csv'), encoding='utf-7')

    if 'Emotion_Name' in rating_data.columns.values:
        emt_flag = 1
    else:
        emt_flag = 0

    pdb.set_trace()

    #Arousal, Dominance, Emotion_Name, Experiment_id, Familiarity, Gender, Liking, Valence, participant, trials.thisIndex, trials.thisTrialN

    if emt_flag == 1:
        rating_data.columns=['Serial','Experiment_id', 'trial_serial', 'trial_index', 'Valence', 'Arousal', 
        'Dominance', 'Liking', 'Familiarity', 'Emotion_Name','Participant_id']
        
        data_summary.columns = ['Experiment_id', 'VideoId', 'total_obs', 'V_mean', 'A_mean', 'D_mean', 
        'mean_dist_origin', 'V_med', 'A_med', 'D_med', 'median_dist_origin', 'V_std', 'A_std', 'VA_std',
        'D_std', 'overall_std', 'Emotion_Name', 'L_mean', 'F_mean', 'L_med', 'F_med', 'L_std', 'F_std']    

    else:
        rating_data.columns=['Serial','Experiment_id', 'trial_serial', 'trial_index', 'Valence', 'Arousal', 
        'Dominance', 'Liking', 'Familiarity','Participant_id']
        
        data_summary.columns = ['Experiment_id', 'VideoId', 'total_obs', 'V_mean', 'A_mean', 'D_mean', 
        'mean_dist_origin', 'V_med', 'A_med', 'D_med', 'median_dist_origin', 'V_std', 'A_std', 'VA_std',
        'D_std', 'overall_std', 'L_mean', 'F_mean', 'L_med', 'F_med', 'L_std', 'F_std']    
        
    emotions_to_plot = {}
    emt_counter = 0

    for vid_id in data_summary['Experiment_id']:        
        index = np.where((vid_id == data_summary['Experiment_id'].values) == True)[0][0]
        emotions_to_plot[vid_id] = data_summary.loc[index, 'VideoId']

    '''df1 = pd.DataFrame(emotions_to_plot.values(), columns=['scatter_plot_index'])
    df2 = pd.DataFrame(emotions_to_plot.keys(), columns=['videos'])    
    emotion_frame = pd.concat((df1, df2), axis=1)'''

    plt_counter = 0
    lower_limit = [45, 30, 20, 10, 0]
    upper_limit = [60, 45, 30, 20, 10]
    range_videos = {}
    dict_keys = []

    for emtId in emotions_to_plot.keys():

        print("Video Name is = %s" %emtId)
        #emotions_name = np.append(emotions_name, emotions_to_plot[emtId])
        indexx = np.where((data_summary['Experiment_id'].values == emtId)==True)[0]

        for lower, upper in zip(lower_limit, upper_limit):

            if (data_summary.loc[indexx, 'total_obs'].values >= lower) and (data_summary.loc[indexx, 'total_obs'].values < upper):
                dict_key = str(lower) + '_' + str(upper)

                if dict_key not in dict_keys:
                    range_videos[dict_key] = emtId
                    dict_keys = np.append(dict_keys, dict_key)                    
                else:
                    range_videos[dict_key] = np.append(range_videos[dict_key], emtId)


    all_greater_10_val = []
    all_greater_10_ars = []
    all_greater_10_dom = []
    all_greater_10_val_norm = []
    all_greater_10_ars_norm = []
    all_greater_10_dom_norm = []
    all_greater_10_emt = []
    all_greater_10_obs = []

    for range_vid_key in range_videos.keys():

        # This lower value is created so that all the plots which are more than 10 should be together
        lower_val = int(range_vid_key.split('_')[0])
        valence_mean = []
        arousal_mean = []
        dominance_mean = []        
        valence_std = []
        arousal_std = []
        dominance_std = []

        norm_valence = []
        norm_arousal = []
        norm_dominance = []

        all_observations = []
        obs_norm_val = []
        obs_norm_ars = []
        obs_norm_dom = []

        valence_mean_male = []
        arousal_mean_male = []
        dominance_mean_male = []
        valence_std_male = []
        arousal_std_male = []
        dominance_std_male = []            

        valence_mean_female = []
        arousal_mean_female = []
        dominance_mean_female = []
        valence_std_female = []
        arousal_std_female = []
        dominance_std_female = []

        distance_frame = pd.DataFrame(0, index=emotions_to_plot.values(), columns=emotions_to_plot.values())

        fig = plt.figure(figsize=(20, 10.5))
        gs = gridspec.GridSpec(2, 2, hspace=0.15, left=0.05, right=0.99, bottom=0.05, top=0.95) 

        all_valence = []
        all_arousal = []
        all_dominance = []
        all_emotions = []
        all_gender = []        
        emotions_name = []
        emotion_name_range_wise = {}

        if isinstance(range_videos[range_vid_key], str):
            VidNames = [range_videos[range_vid_key]]
        else:
            VidNames = range_videos[range_vid_key]

        for emtId in VidNames:

            indxx = np.where((emtId == rating_data['Experiment_id'])==True)[0]
            indxx_summ = np.where((emtId == data_summary['Experiment_id'])==True)[0]            

            # Selected 40 because I am assuming that for all the videos I will get at least 40 responses
            tot_obs_factor = data_summary['total_obs'][indxx_summ]/40

            valence_per_emotion = rating_data['Valence'][indxx]
            arousal_per_emotion = rating_data['Arousal'][indxx]
            dominance_per_emotion = rating_data['Dominance'][indxx]
            if (np.std(valence_per_emotion)==0) or (np.std(arousal_per_emotion)==0) or (np.std(dominance_per_emotion)==0):
                continue

            print("Video Name is = %s" %emtId)
            all_observations = np.append(all_observations, data_summary['total_obs'][indxx_summ])
            try:
                emotions_name = np.append(emotions_name, emotions_to_plot[emtId])
            except Exception as e:
                print(e)
                pdb.set_trace()

            all_emotions = np.append(all_emotions, [emotions_to_plot[emtId]]*len(indxx))            

    # ==========================>>>>>>>>>>>>>>>>>>>> Valence <<<<<<<<<<<<<<<<<<<<<<<===============================
            # Male and Females are mixed
            
            all_valence = np.append(all_valence, valence_per_emotion)
            valence_mean = np.append(valence_mean, np.round(np.mean(valence_per_emotion),2))
            valence_std = np.append(valence_std, np.round(np.std(valence_per_emotion),2))
            pdb.set_trace()
            norm_valence = np.append(norm_valence, (np.mean(valence_per_emotion)-5) / np.std(valence_per_emotion))
            obs_norm_val = np.append(obs_norm_val, tot_obs_factor*(np.mean(valence_per_emotion)-5) / np.std(valence_per_emotion))

    # ==========================>>>>>>>>>>>>>>>>>>>> Arousal <<<<<<<<<<<<<<<<<<<<<<<===============================
            # Male and Females are mixed
            
            all_arousal = np.append(all_arousal, arousal_per_emotion)
            arousal_mean = np.append(arousal_mean, np.round(np.mean(arousal_per_emotion),2))
            arousal_std = np.append(arousal_std, np.round(np.std(arousal_per_emotion),2))
            norm_arousal = np.append(norm_arousal, (np.mean(arousal_per_emotion)-5) / np.std(arousal_per_emotion))
            obs_norm_ars = np.append(obs_norm_ars, tot_obs_factor*(np.mean(arousal_per_emotion)-5) / np.std(arousal_per_emotion))

    # ==========================>>>>>>>>>>>>>>>>>>>> Dominance <<<<<<<<<<<<<<<<<<<<<<<===============================
            # Male and Females are mixed
            
            all_dominance = np.append(all_dominance, dominance_per_emotion)
            dominance_mean = np.append(dominance_mean, np.round(np.mean(dominance_per_emotion),2))
            dominance_std = np.append(dominance_std, np.round(np.std(dominance_per_emotion),2))                
            norm_dominance = np.append(norm_dominance, (np.mean(dominance_per_emotion)-5) / np.std(dominance_per_emotion))
            obs_norm_dom = np.append(obs_norm_dom, tot_obs_factor*(np.mean(dominance_per_emotion)-5) / np.std(dominance_per_emotion))

            if lower_val >= 0:
                all_greater_10_val_norm = np.append(all_greater_10_val_norm, norm_valence[-1])
                all_greater_10_ars_norm = np.append(all_greater_10_ars_norm, norm_arousal[-1])
                all_greater_10_dom_norm = np.append(all_greater_10_dom_norm, norm_dominance[-1])

                all_greater_10_val = np.append(all_greater_10_val, valence_mean[-1])
                all_greater_10_ars = np.append(all_greater_10_ars, arousal_mean[-1])
                all_greater_10_dom = np.append(all_greater_10_dom, dominance_mean[-1])

                all_greater_10_emt = np.append(all_greater_10_emt, emotions_name[-1])
                all_greater_10_obs = np.append(all_greater_10_obs, data_summary['total_obs'][indxx_summ])

        male_female_frame = pd.DataFrame({'Mean_v':valence_mean, 'STD_v':valence_std, 'Mean_a':arousal_mean, 
        'STD_a':arousal_std, 'Mean_d':dominance_mean, 'STD_d':dominance_std, 'Emotions': emotions_name.astype('int')})

        observation_frame = pd.DataFrame({'Mean_v':valence_mean, 'Mean_a':arousal_mean, 
            'Mean_d':dominance_mean, 'Emotions': all_observations.astype('int')})

        normalized_frame = pd.DataFrame({'norm_val': norm_valence, 'norm_arsl': norm_arousal, 
            'norm_dom': norm_dominance, 'Emotions': emotions_name.astype('int')})

        osb_norm_frame = pd.DataFrame({'obs_norm_val': obs_norm_val, 'obs_norm_arsl': obs_norm_ars, 
            'obs_norm_dom': obs_norm_dom, 'Emotions': emotions_name.astype('int')})                

        ax = plt.subplot(gs[0, 0])   
        ax.set_title("Without Normalization")
        mf_ax = male_female_frame.plot.scatter(x='Mean_v', y='Mean_a', ax=ax, xlim=(1,9), ylim=(1,9))
        mf_ax.set_xlabel('Valence', fontsize=12)
        mf_ax.set_ylabel('Arousal', fontsize=12)    
        for i, point in male_female_frame.iterrows():                            
            mf_ax.text(point['Mean_v'], point['Mean_a'], str(int(point['Emotions'])))

        ax = plt.subplot(gs[0, 1])   
        ax.set_title("Divided by SD normalized")
        max_val_f = np.max(normalized_frame['norm_val'].values) + 0.2
        max_val_s = np.max(normalized_frame['norm_arsl'].values) + 0.2
        pdb.set_trace()
        mf_ax = normalized_frame.plot.scatter(x='norm_val', y='norm_arsl', ax=ax, 
            xlim=(-max_val_f, max_val_f), ylim=(-max_val_s, max_val_s))

        mf_ax.set_xlabel('Valence', fontsize=12)
        mf_ax.set_ylabel('Arousal', fontsize=12)    
        for i, point in normalized_frame.iterrows():                            
            mf_ax.text(point['norm_val'], point['norm_arsl'], str(int(point['Emotions'])))            

        ax = plt.subplot(gs[1, 0])
        ax.set_title("Observation Factor Normalized")
        max_val_f = np.max(osb_norm_frame['obs_norm_val'].values) + 0.2
        max_val_s = np.max(osb_norm_frame['obs_norm_arsl'].values) + 0.2   
        mf_ax = osb_norm_frame.plot.scatter(x='obs_norm_val', y='obs_norm_arsl', ax=ax,
            xlim=(-max_val_f, max_val_f), ylim=(-max_val_s, max_val_s))
        mf_ax.set_xlabel('Valence', fontsize=12)
        mf_ax.set_ylabel('Arousal', fontsize=12)    
        for i, point in osb_norm_frame.iterrows():                            
            mf_ax.text(point['obs_norm_val'], point['obs_norm_arsl'], str(int(point['Emotions'])))            

        ax = plt.subplot(gs[1, 1])
        ax.set_title("Total Number of Observations")
        mf_ax = observation_frame.plot.scatter(x='Mean_v', y='Mean_a', ax=ax, xlim=(1,9), ylim=(1,9))
        mf_ax.set_xlabel('Valence', fontsize=12)
        mf_ax.set_ylabel('Arousal', fontsize=12)    
        for i, point in observation_frame.iterrows():                            
            mf_ax.text(point['Mean_v'], point['Mean_a'], str(int(point['Emotions'])))            

        '''ax = plt.subplot(gs[1, 1])   
        mf_ax = male_female_frame.plot.scatter(x='Mean_v', y='Mean_d', color='DarkBlue', ax=ax, xlim=(1,9), ylim=(1,9))
        mf_ax.set_xlabel('Valence', fontsize=12)
        mf_ax.set_ylabel('Dominance', fontsize=12)                    
        for i, point in male_female_frame.iterrows():                            
            mf_ax.text(point['Mean_v'], point['Mean_d'], point['Emotions'])'''

        '''ax = plt.subplot(gs[1, 1])   
        mf_ax = male_female_frame.plot.scatter(x='Mean_a', y='Mean_d', color='DarkBlue', ax=ax, xlim=(1,9), ylim=(1,9))
        mf_ax.set_xlabel('Arousal', fontsize=12)
        mf_ax.set_ylabel('Dominance', fontsize=12)                    
        for i, point in male_female_frame.iterrows():                            
            mf_ax.text(point['Mean_a'], point['Mean_d'], point['Emotions'])'''

        plt.savefig('all_emotions_and_mean_'+range_vid_key+'_'+cleaning_flag+date+'.png')

####### Without Normalization
    all_greater_10_frame = pd.DataFrame({'Mean_v':all_greater_10_val, 'Mean_a':all_greater_10_ars, 
    'Mean_d':all_greater_10_dom, 'Emotions': all_greater_10_emt.astype('int')})

    fig = plt.figure(figsize=(20, 10.5))
    #gs = gridspec.GridSpec(1, 2, hspace=0.15, left=0.05, right=0.99, bottom=0.05, top=0.95) 
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0, :])   
#    max_val_f = np.ceil(np.max(all_greater_10_frame['Mean_v'].values) + 0.2)
#    max_val_s = np.ceil(np.max(all_greater_10_frame['Mean_a'].values) + 0.2)      
    ax.set_title("Without Normalization")    
    mf_ax = all_greater_10_frame.plot.scatter(x='Mean_v', y='Mean_a', ax=ax, xlim=(1,9), ylim=(1,9))        
    mf_ax.grid(b=True, which='both', axis='both', linestyle='--')
    mf_ax.set_xlabel('Valence', fontsize=12)
    mf_ax.set_ylabel('Arousal', fontsize=12)    
    mf_ax.set_xticks(np.arange(1, 9, step=0.5))
    mf_ax.set_yticks(np.arange(1, 9, step=0.5))    
    for i, point in all_greater_10_frame.iterrows():                            
        mf_ax.text(point['Mean_v'], point['Mean_a'], str(int(point['Emotions'])))

    plt.savefig('all_emotions_and_mean_greater_10_VA_Mean_'+cleaning_flag+date+'.png')

###### With Normalization
    all_greater_10_frame = pd.DataFrame({'Mean_v':all_greater_10_val_norm, 'Mean_a':all_greater_10_ars_norm, 
    'Mean_d':all_greater_10_dom_norm, 'Emotions': all_greater_10_emt.astype('int')})

    fig = plt.figure(figsize=(20, 10.5))
    #gs = gridspec.GridSpec(1, 2, hspace=0.15, left=0.05, right=0.99, bottom=0.05, top=0.95) 
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0, :])   
    max_val_f = np.ceil(np.max(all_greater_10_frame['Mean_v'].values) + 0.2)
    max_val_s = np.ceil(np.max(all_greater_10_frame['Mean_a'].values) + 0.2)
    ax.set_title("Divided by SD normalized")    
    mf_ax = all_greater_10_frame.plot.scatter(x='Mean_v', y='Mean_a', ax=ax,
        xlim=(-max_val_f, max_val_f), ylim=(-max_val_s, max_val_s))
    mf_ax.grid(b=True, which='both', axis='both', linestyle='--')
    mf_ax.set_xlabel('Valence', fontsize=12)
    mf_ax.set_ylabel('Arousal', fontsize=12)    
    mf_ax.set_xticks(np.arange(-max_val_f, max_val_f, step=0.5))
    mf_ax.set_yticks(np.arange(-max_val_s, max_val_s, step=0.5))
    for i, point in all_greater_10_frame.iterrows():                            
        mf_ax.text(point['Mean_v'], point['Mean_a'], str(int(point['Emotions'])))

    plt.savefig('all_emotions_and_mean_greater_10_VA_Norm_'+cleaning_flag+date+'.png')

####### Total Observations
    all_greater_10_obs = pd.DataFrame({'Mean_v':all_greater_10_val, 'Mean_a':all_greater_10_ars, 
    'Mean_d':all_greater_10_dom, 'Emotions': all_greater_10_obs.astype('int')})
    fig = plt.figure(figsize=(20, 10.5))
    gs = gridspec.GridSpec(1, 2)

    ax = plt.subplot(gs[0, :])   
    max_val_f = np.max(all_greater_10_frame['Mean_v'].values) + 0.2
    max_val_s = np.max(all_greater_10_frame['Mean_a'].values) + 0.2           
    ax.set_title("Total_Observations")
    ax.grid(b=True, which='both', axis='both', linestyle='--')
    mf_ax = all_greater_10_obs.plot.scatter(x='Mean_v', y='Mean_a', ax=ax, xlim=(1,9), ylim=(1,9))
    #    xlim=(-max_val_f, max_val_f), ylim=(-max_val_s, max_val_s))
    mf_ax.grid(b=True, which='both', axis='x', linestyle='--')
    mf_ax.set_xlabel('Valence', fontsize=12)
    mf_ax.set_ylabel('Arousal', fontsize=12)    
    for i, point in all_greater_10_obs.iterrows():                            
        mf_ax.text(point['Mean_v'], point['Mean_a'], str(int(point['Emotions'])))

    plt.savefig('all_emotions_and_mean_greater_10_obs_'+cleaning_flag+date+'.png')

    print("After this please go to Data_Cleaning Module")

def Data_Cleaning(file_name = None, date=''):

    # This module should be run if you want to perform cleaning of the data. How by detecting outliers.
    # file_name = My_Experiment_Ratings_before_cleaning.csv

    # Data_Cleaning(date='Oct_10-Nov_15')
    rating_data = pd.read_csv(os.path.join(_thisDir, 'NewTarget', 'My_Experiment_Ratings_before_cleaning2018_'+date+'.csv'), encoding='utf-7', index_col=0)    
    data_summary = pd.read_csv(os.path.join(_thisDir, 'NewTarget', 'summary_data_frame_before_cleaning2018_'+date+'.csv'))    

    if 'Emotion_Name' in rating_data.columns.values:
        emt_flag = 1
    else:
        emt_flag = 0

    '''if emt_flag == 1:
        rating_data.columns=['Serial','Experiment_id', 'trial_serial', 'trial_index', 'Valence', 'Arousal', 
        'Dominance', 'Liking', 'Familiarity', 'Emotion_Name','Participant_id']
        
        data_summary.columns = ['Experiment_id', 'VideoId', 'total_obs', 'V_mean', 'A_mean', 'D_mean', 
        'mean_dist_origin', 'V_med', 'A_med', 'D_med', 'median_dist_origin', 'V_std', 'A_std', 'VA_std',
        'D_std', 'overall_std', 'Emotion_Name', 'L_mean', 'F_mean', 'L_med', 'F_med', 'L_std', 'F_std']    

    else:
        rating_data.columns=['Serial','Experiment_id', 'trial_serial', 'trial_index', 'Valence', 'Arousal', 
        'Dominance', 'Liking', 'Familiarity','Participant_id']
        
        data_summary.columns = ['Experiment_id', 'VideoId', 'total_obs', 'V_mean', 'A_mean', 'D_mean', 
        'mean_dist_origin', 'V_med', 'A_med', 'D_med', 'median_dist_origin', 'V_std', 'A_std', 'VA_std',
        'D_std', 'overall_std', 'L_mean', 'F_mean', 'L_med', 'F_med', 'L_std', 'F_std']    '''

    print(rating_data)
    print(rating_data.index.values)
    print(rating_data.columns.values)    
    print(data_summary.columns.values)

    if emt_flag == 1:

        renameDict = {'Experiment_id':'Experiment_id', 'trials.thisTrialN':'trial_serial', 'trials.thisIndex':'trial_index', 'Valence':'Valence', 'Arousal':'Arousal', 
        'Dominance':'Dominance', 'Liking':'Liking', 'Familiarity':'Familiarity', 'Emotion_Name':'Emotion_Name', 'participant':'Participant_id', 'Quadrant':'Quadrant', 'Gender':'Gender'}

        rating_data.rename(renameDict, axis=1, inplace=True)
        
        '''data_summary.columns = ['Experiment_id', 'VideoId', 'total_obs', 'V_mean', 'A_mean', 'D_mean', 
        'mean_dist_origin', 'V_med', 'A_med', 'D_med', 'median_dist_origin', 'V_std', 'A_std', 'VA_std',
        'D_std', 'overall_std', 'Emotion_Name', 'Quadrant', 'L_mean', 'F_mean', 'L_med', 'F_med', 'L_std', 'F_std', '69Videos', '30Videos']    '''

    else:
        renameDict = {'Experiment_id':'Experiment_id', 'trials.thisTrialN':'trial_serial', 'trials.thisIndex':'trial_index', 'Valence':'Valence', 'Arousal':'Arousal', 
        'Dominance':'Dominance', 'Liking':'Liking', 'Familiarity':'Familiarity', 'participant':'Participant_id'}
        rating_data.rename(renameDict, axis=1, inplace=True)
        
        data_summary.columns = ['Experiment_id', 'VideoId', 'total_obs', 'V_mean', 'A_mean', 'D_mean', 
        'mean_dist_origin', 'V_med', 'A_med', 'D_med', 'median_dist_origin', 'V_std', 'A_std', 'VA_std',
        'D_std', 'overall_std', 'L_mean', 'F_mean', 'L_med', 'F_med', 'L_std', 'F_std']            

    P_ids = np.unique(rating_data['Participant_id'].values)    


    P_corr = {}
    P_random_response = {}
    corr_dict = {}

    ### Reassining the experiment Id by normalizing it
    withoutExt = ['_'.join(i.split(' ')) for i in data_summary['Experiment_id']]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split("'")) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split('(')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split(')')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split('&')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = [i.split('.')[0] for i in withoutExt]    
    data_summary['Experiment_id'] = withoutExt    
    del withoutExt

    withoutExt = ['_'.join(i.split(' ')) for i in rating_data['Experiment_id']]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split("'")) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split('(')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split(')')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = ['_'.join(i.split('&')) for i in withoutExt]  ### Renaming the experiment Ids taken from csv file
    withoutExt = [i.split('.')[0] for i in withoutExt]    
    rating_data['Experiment_id'] = withoutExt    
    del withoutExt

    for ids_ in P_ids:
        P_data_index = np.where((rating_data['Participant_id'].values==ids_)==True)[0]
        P_data = rating_data.loc[P_data_index,['Experiment_id', 'Valence', 'Arousal', 'Dominance']]

        switch_flag = 0

        print('===================================================')
        for exp_id in P_data['Experiment_id']:
            exp_id = exp_id.split('.')[0]
            exp_id = '_'.join(exp_id.split(' '))
            exp_id = '_'.join(exp_id.split("'"))
            exp_id = '_'.join(exp_id.split('('))
            exp_id = '_'.join(exp_id.split(')'))
            exp_id = '_'.join(exp_id.split('&'))

            #if exp_id == 'Homage_For_Satan':
            #    pdb.set_trace()

            indxx = np.where(exp_id == data_summary['Experiment_id'])[0]            

            if len(indxx) > 1:
                temp_frame = data_summary.loc[[indxx[0]], ['Experiment_id', 'V_mean', 'A_mean', 'D_mean']]
            else:
                temp_frame = data_summary.loc[indxx, ['Experiment_id', 'V_mean', 'A_mean', 'D_mean']]

            if switch_flag == 0:
                summary_frame = temp_frame
                switch_flag = 1
            else:
                summary_frame = pd.concat((summary_frame, temp_frame))

        '''Formula for multiple variable coorelation is 
        R^2 = transpose(vector_C)*inverse(Rxx)*vector_C        
        Source: Wikipedia'''

        # For data cleaning I must use mean and sd or some range.
        try:
            cor_coef_V = np.corrcoef(x=summary_frame['V_mean'].values,y=P_data['Valence'].values)[0,1]
        except:
            pdb.set_trace()

        cor_coef_A = np.corrcoef(x=summary_frame['A_mean'].values,y=P_data['Arousal'].values)[0,1]
        cor_coef_D = np.corrcoef(x=summary_frame['D_mean'].values,y=P_data['Dominance'].values)[0,1]
        vector_C = np.reshape(np.array([cor_coef_V, cor_coef_A, cor_coef_D]), (3, 1))
        corr_dict[ids_] = [cor_coef_V, cor_coef_A, cor_coef_D]

        if (cor_coef_V > 0.25) and (cor_coef_A > 0.25):
            P_corr[ids_] = cor_coef_V
        else:
            P_random_response[ids_] = cor_coef_V
    
    df_ = pd.DataFrame(corr_dict.values(), index=corr_dict.keys(), columns=['corr_val','corr_arl','corr_dom'])
    p_ids_df = pd.DataFrame(P_random_response.keys(), columns=['Enroll'])
    p_ids_cor = pd.DataFrame(P_random_response.values(), columns=['corr'])
    pd.concat((p_ids_df, p_ids_cor), axis=1).to_csv(os.path.join(_thisDir, 'NewTarget', 'Participants_Who_Gave_Random_Responses'+date+'.csv'))

    for ids_ in P_random_response.keys():
        to_check_index = np.where((rating_data['Participant_id'].values==ids_)==True)
        print("Before =====================")
        print(to_check_index)
        rating_data = rating_data[rating_data.Participant_id.str.contains(ids_) == False]
        to_check_index = np.where((rating_data['Participant_id'].values==ids_)==True)
        print("After =====================")
        print(to_check_index)


    #del rating_data['Serial']
    rating_data.reset_index(drop=True, inplace=True)
    rating_data.to_csv(os.path.join(_thisDir, 'NewTarget', 'My_Experiment_Ratings_after_cleaning2018_'+date+'.csv'))
    print("After this module please run validation_analysis module to create summary after clearning")    

# ================>>>>>>>>>>>>>>>>>>>>>>>> Statistical Significance of VAD Values <<<<<<<<<<<<<<<<<<<<=============================
def statistical_significance(date=''):
    import random
    from joblib import Parallel, delayed           

    os.chdir(path_for_file)
    rating_data = pd.read_csv(os.path.join(_thisDir, 'NewTarget', 'participant_ratings.csv'))
    emotions_to_plot = {1:'fun', 2:'exciting', 3:'joy', 11:'happy', 12:'cheerful', 13:'love', 14:'happy1',15:'lovely',
      16:'senti', 22:'senti1', 23:'melanch', 24:'sad', 25:'depress', 26:'mellow', 31:'terrible', 32:'shock', 33:'hate',}

    male_sub = [0,4,5,6,11,15,16,17,18,19,20,22,25,26,27,28,29]
    female_sub = [1,2,3,7,8,9,10,12,13,14,21,23,24,30,31] 
    no_male = len(male_sub)
    no_female = len(female_sub)
    male_cont = int(np.ceil(no_male/2))
    female_cont = int(np.ceil(no_female/2))

    male_female_label = ['M', 'F', 'F', 'F', 'M', 'M', 'M', 'F', 'F', 'F', 'F', 'M', 'F', 'F', 'F', 'M', 
                        'M', 'M', 'M', 'M', 'M', 'F', 'M', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'F', 'F']

    fig = plt.figure(figsize=(20, 10.5))
    gs = gridspec.GridSpec(4, 5, hspace=0.15, left=0.01, right=0.99, bottom=0.05, top=0.95) 

    first_second_male = np.append(np.arange(no_male),np.arange(no_male)+no_male)
    first_second_female = np.append(np.arange(no_female),np.arange(no_female)+no_female)
    all_emotions_male = {}
    all_emotions_female = {}
    
    if vad_opt == 'va':
        file_to_check = 'all_emotions_male_va.pkl'
    if vad_opt == 'vad':
        file_to_check = 'all_emotions_male_vad.pkl'

    inner_count = int(factorial(no_male)/((factorial(male_cont))*(factorial(no_male - male_cont)))) # Inner Count for Males
    
    if not os.path.isfile(file_to_check):
        for S_emtId in emotions_to_plot.keys():
            emotions_name = np.append(emotions_name, emotions_to_plot[S_emtId])
            emt_name_1 = emotions_to_plot[S_emtId]
            male_shuffled_significance_matrix = np.zeros((len(emotions_to_plot)-1, inner_count))
            all_shuffled = {}

            inner_emotion_count = 0
            for T_emtId in emotions_to_plot.keys():                          
                emt_name_2 = emotions_to_plot[T_emtId]   
                if emt_name_1 != emt_name_2:
                    print("Now Processing Emotion: %s" %emt_name_2)
                    S_indxx = np.where((S_emtId == rating_data['Experiment_id'])==True)[0]
                    T_indxx = np.where((T_emtId == rating_data['Experiment_id'])==True)[0]

    # ==========================>>>>>>>>>>>>>>>>>>>> Valence <<<<<<<<<<<<<<<<<<<<<<<===============================
                    # Male wise
                    S_valence_per_emotion_male = rating_data['Valence'][S_indxx[male_sub]]
                    T_valence_per_emotion_male = rating_data['Valence'][T_indxx[male_sub]]
    # ==========================>>>>>>>>>>>>>>>>>>>> Arousal <<<<<<<<<<<<<<<<<<<<<<<===============================
                    # Male wise
                    S_arousal_per_emotion_male = rating_data['Arousal'][S_indxx[male_sub]]
                    T_arousal_per_emotion_male = rating_data['Arousal'][T_indxx[male_sub]]
    # ==========================>>>>>>>>>>>>>>>>>>>> Dominance <<<<<<<<<<<<<<<<<<<<<<<===============================
                    # Male wise
                    S_dominance_per_emotion_male = rating_data['Dominance'][S_indxx[male_sub]]
                    T_dominance_per_emotion_male = rating_data['Dominance'][T_indxx[male_sub]]

                    if vad_opt == 'va':
                        S_mag_va_male = ((S_valence_per_emotion_male**2) + (S_arousal_per_emotion_male**2))**0.5
                        T_mag_va_male = ((T_valence_per_emotion_male**2) + (T_arousal_per_emotion_male**2))**0.5
                        

                    if vad_opt == 'vad':
                        S_mag_va_male = ((S_valence_per_emotion_male**2) + (S_arousal_per_emotion_male**2) + (S_dominance_per_emotion_male**2))**0.5
                        T_mag_va_male = ((T_valence_per_emotion_male**2) + (T_arousal_per_emotion_male**2) + (T_dominance_per_emotion_male**2))**0.5

                    first_second_male_emotion = np.append(S_mag_va_male, T_mag_va_male)

                    orig_tstat_male_emotion = for_shuffled_data_VAD(list([S_mag_va_male, T_mag_va_male, -1, len(S_mag_va_male)]))
                    np.save('orig_tstatistics_male_emotion.npy',orig_tstat_male_emotion)

                    if orig_tstat_male_emotion == 0:
                        try:
                            shuffle_count_male_file = 'shuffle_count_male_' + emt_name_1+':'+str(inner_emotion_count)+'.npy'
                            if os.path.exists(shuffle_count_male_file):
                                count_data = np.load(shuffle_count_male_file)                  
                                if count_data == (inner_count-1):
                                    result_sum_male_emotion = np.load('result_sum_final_male_emotion.npy')[inner_emotion_count,:]    
                                    shuffle_count_male = count_data
                                    print("significant_connection Module: "+"Shuffle_count = %d" %shuffle_count_male)    
                                    print("Shape of Result Sum = %d" %result_sum_male_emotion.shape[0])
                                else:
                                    raise Exception("shuffle_count.npy file does not exist")
                            else:
                                raise Exception("shuffle_count.npy file does not exist")
                                                            
                        except Exception as e:
                            print("Doing Permutation Test to find if there is voxel wise connectivity between emotion and baseline for Males")            

                            shuffle_count_male = 0
                            tasks = []
                            for shuffle_count_male in range(1, inner_count):
                                index_to_shuffle = np.array(random.sample(range(0,no_male), male_cont))
                                
                                first_second_index = index_to_shuffle + no_male
                                second_first_index = index_to_shuffle

                                first_second_shuffle = first_second_male.copy()
                                first_second_shuffle[first_second_index] = second_first_index
                                first_second_shuffle[second_first_index] = first_second_index
                                
                                first_index = first_second_shuffle[0:no_male]
                                second_index = first_second_shuffle[no_male:]                                
                                S_va_male = first_second_male_emotion[first_index]
                                T_va_male = first_second_male_emotion[second_index]

                                '''print("First Emotion Index")
                                print(first_index)
                                print("Second Emotion index")
                                print(second_index)
                                # Extracting data as per the shuffled labelling
                                print("Assigning Shuffled Indexed Vaues...")                        '''                               
                                
                                tasks.append([S_va_male, T_va_male, orig_tstat_male_emotion, no_male])

                            res = Parallel(n_jobs=nCores)(delayed(for_shuffled_data_VAD)(t) for t in tasks)   
                            
                            all_shuffled[emt_name_2] = len(np.where((np.array(res)==0)==True)[0])/shuffle_count_male

                    else:
                        all_shuffled[emt_name_2] = 1                                
                    inner_emotion_count = inner_emotion_count + 1                                                  
                        
            all_emotions_male[emt_name_1] = all_shuffled

        if vad_opt == 'va':
            fid = open('all_emotions_male_va.pkl','wb')
            pickle.dump(all_emotions_male,fid)
        if vad_opt == 'vad':
            fid = open('all_emotions_male_vad.pkl','wb')
            pickle.dump(all_emotions_male,fid)
        fid.close()
        
        
    if vad_opt == 'va':
        file_to_check = 'all_emotions_female_va.pkl'
    if vad_opt == 'vad':
        file_to_check = 'all_emotions_female_vad.pkl'
        
    inner_count = int(factorial(no_male)/((factorial(male_cont))*(factorial(no_male - male_cont)))) # Inner Count for Females
    
    if not os.path.isfile(file_to_check):                

        for S_emtId in emotions_to_plot.keys():
            emotions_name = np.append(emotions_name, emotions_to_plot[S_emtId])
            emt_name_1 = emotions_to_plot[S_emtId]
            female_shuffled_significance_matrix = np.zeros((len(emotions_to_plot)-1, inner_count))
            all_shuffled = {}

            inner_emotion_count = 0

            for T_emtId in emotions_to_plot.keys():                          
                emt_name_2 = emotions_to_plot[T_emtId]   
                if emt_name_1 != emt_name_2:
                    print("Now Processing Emotion: %s" %emt_name_2)
                    S_indxx = np.where((S_emtId == rating_data['Experiment_id'])==True)[0]
                    T_indxx = np.where((T_emtId == rating_data['Experiment_id'])==True)[0]

    # ==========================>>>>>>>>>>>>>>>>>>>> Valence <<<<<<<<<<<<<<<<<<<<<<<===============================
                    # Female wise
                    S_valence_per_emotion_female = rating_data['Valence'][S_indxx[female_sub]]                
                    T_valence_per_emotion_female = rating_data['Valence'][T_indxx[female_sub]]                
    # ==========================>>>>>>>>>>>>>>>>>>>> Arousal <<<<<<<<<<<<<<<<<<<<<<<===============================
                    # Female wise
                    S_arousal_per_emotion_female = rating_data['Arousal'][S_indxx[female_sub]]                
                    T_arousal_per_emotion_female = rating_data['Arousal'][T_indxx[female_sub]]                
    # ==========================>>>>>>>>>>>>>>>>>>>> Dominance <<<<<<<<<<<<<<<<<<<<<<<===============================
                    # Female wise
                    S_dominance_per_emotion_female = rating_data['Dominance'][S_indxx[female_sub]]                
                    T_dominance_per_emotion_female = rating_data['Dominance'][T_indxx[female_sub]]                

                    if vad_opt == 'va':
                        S_mag_va_female = ((S_valence_per_emotion_female**2) + (S_arousal_per_emotion_female**2))**0.5
                        T_mag_va_female = ((T_valence_per_emotion_female**2) + (T_arousal_per_emotion_female**2))**0.5
                    if vad_opt == 'vad':
                        S_mag_va_female = ((S_valence_per_emotion_female**2) + (S_arousal_per_emotion_female**2) + (S_dominance_per_emotion_female**2))**0.5
                        T_mag_va_female = ((T_valence_per_emotion_female**2) + (T_arousal_per_emotion_female**2) + (T_dominance_per_emotion_female**2))**0.5                    
                    
                    first_second_female_emotion = np.append(S_mag_va_female, T_mag_va_female)

                    orig_tstat_female_emotion = for_shuffled_data_VAD(list([S_mag_va_female, T_mag_va_female, -1, len(S_mag_va_female)]))
                    np.save('orig_tstatistics_female_emotion.npy',orig_tstat_female_emotion)                    

                    if orig_tstat_female_emotion == 0:
                        try:
                            shuffle_count_female_file = 'shuffle_count_female_' + emt_name_1+':'+str(inner_emotion_count)+'.npy'
                            if os.path.exists(shuffle_count_female_file):
                                count_data = np.load(shuffle_count_female_file)                  
                                if count_data == (inner_count-1):
                                    result_sum_female_emotion = np.load('result_sum_final_female_emotion.npy')[inner_emotion_count,:]    
                                    shuffle_count_female = count_data
                                    print("significant_connection Module: "+"Shuffle_count = %d" %shuffle_count_female)    
                                    print("Shape of Result Sum = %d" %result_sum_female_emotion.shape[0])
                                else:
                                    raise Exception("shuffle_count.npy file does not exist")
                            else:
                                raise Exception("shuffle_count.npy file does not exist")
                                                            
                        except Exception as e:
                            print("Doing Permutation Test to find if there is voxel wise connectivity between emotion and baseline for females")            

                            shuffle_count_female = 0
                            tasks = []
                            for shuffle_count_female in range(1, inner_count):
                                index_to_shuffle = np.array(random.sample(range(0,no_female), female_cont))
                                
                                first_second_index = index_to_shuffle + no_female
                                second_first_index = index_to_shuffle

                                first_second_shuffle = first_second_female.copy()
                                first_second_shuffle[first_second_index] = second_first_index
                                first_second_shuffle[second_first_index] = first_second_index
                                
                                first_index = first_second_shuffle[0:no_female]
                                second_index = first_second_shuffle[no_female:]                                
                                S_va_female = first_second_female_emotion[first_index]
                                T_va_female = first_second_female_emotion[second_index]

                                '''print("First Emotion Index")
                                print(first_index)
                                print("Second Emotion index")
                                print(second_index)
                                # Extracting data as per the shuffled labelling
                                print("Assigning Shuffled Indexed Vaues...")                        '''                               
                                
                                tasks.append([S_va_female, T_va_female, orig_tstat_female_emotion, no_female])

                            res = Parallel(n_jobs=nCores)(delayed(for_shuffled_data_VAD)(t) for t in tasks)   

                            #all_shuffled[emt_name_2] = np.sum(res)/shuffle_count_female
# If my alternate hypothesis is becoming true for only one case which is the origin case than alternate hypothesis is acceptable since it is not random by chance.
# If I am getting 0 more number of times that means my alternate hypothesis is random by chance. But if it turns out to be 
                            all_shuffled[emt_name_2] = len(np.where((np.array(res)==0)==True)[0])/shuffle_count_female
                    else:
                        all_shuffled[emt_name_2] = 1     

                    inner_emotion_count = inner_emotion_count + 1                                                  
                        
            all_emotions_female[emt_name_1] = all_shuffled                                      

        if vad_opt == 'va':
            fid = open('all_emotions_female_va.pkl','wb')
            pickle.dump(all_emotions_female,fid)
        if vad_opt == 'vad':
            fid = open('all_emotions_female_vad.pkl','wb')
            pickle.dump(all_emotions_female,fid)
        fid.close()

def only_selected_stimulus(file_name=None, date=''):
    VideoId = np.array([65,204,209,93,220,41,201,90,68,52,99,69,59,105,62,154,211,216,100,
    110,67,32,16,156,123,121,145,48,188,226,24,178,84,23,37,56,40,170,114,127,173,
    175,137,7,189,38,51,185,74,187,10,29,36,49,184,83])

    summary_data = pd.read_csv(os.path.join(_thisDir, 'NewTarget', file_name), index_col=False, encoding='utf-7')
    os.chdir('..')
    not_copied = []
    counter = 0

    NewFrame = pd.DataFrame([[0, 0]] * len(VideoId), index = summary_data['vid'][VideoId].values, columns=['Label', 'VideoId'])

    for vidId in VideoId:

        indxx = np.where((vidId == summary_data['VideoId'].values)==True)[0][0] 
        VidName = summary_data['vid'][indxx]        
        try:
            shutil.copy(os.path.join('all_clips', VidName), os.path.join('Emotion_Name_Rating', VidName))
            valence_ = (summary_data['V_mean'][indxx])
            arousal_ = (summary_data['A_mean'][indxx])

            if (valence_ > 5) and (arousal_ > 5):
                label = "HVHA"
            if (valence_ < 5) and (arousal_ > 5):
                label = "LVHA"
            if (valence_ < 5) and (arousal_ < 5):
                label = "LVLA"                                
            if (valence_ > 5) and (arousal_ < 5):
                label = "HVLA"                

            if 'mp4' in VidName:
                videoID = "Video-"+str(counter)+'.mp4'
            if 'webm' in VidName:
                videoID = "Video-"+str(counter)+'.webm'
            counter = counter + 1
            
            NewFrame.loc[VidName, 'Label'] = label
            NewFrame.loc[VidName, 'VideoId'] = videoID
            shutil.copy(os.path.join('all_clips', VidName), os.path.join('EmotionWithVideoID', videoID))            

        except Exception as e:
            not_copied = np.append(not_copied, VidName)
            pdb.set_trace()

    os.chdir(os.path.join(os.getcwd(),'New_Corrected_Clips/New'))

    all_files = glob.glob("*")
    New_Frame = pd.DataFrame([[0, 0]]*len(all_files), index=all_files, columns=['Label', 'VideoId'])    

    for new_files in all_files:

        print(new_files)        

        if 'mp4' in VidName:
            videoID = "Video-"+str(counter)+'.mp4'
        if 'webm' in VidName:
            videoID = "Video-"+str(counter)+'.webm'        

        counter = counter + 1               
        New_Frame.loc[new_files, 'Label'] = ''
        New_Frame.loc[new_files, 'VideoId'] = videoID
        try:
            shutil.copy(new_files, os.path.join('/home/iiita/Experiment/EmotionWithVideoID', videoID))                    
        except Exception as e:
            pdb.set_trace()
            print(e)

        shutil.copy(new_files, os.path.join('/home/iiita/Experiment/Emotion_Name_Rating', new_files))

    os.chdir('../../')
    complete_frame = pd.concat((NewFrame, New_Frame))
    complete_frame.to_csv(os.path.join(_thisDir, 'NewTarget', 'Experiment_Videos_with_Ids'+date+'.csv'))

'''def main():

    interval_record = int(sys.argv[1])
    # Set this variable to fetch and structure rating data from individual subject folders.
    recalculate_rating_file = int(sys.argv[2])
    # With 56 stimulus we have created 8 blocks. And with 17 and 25 stimulus we have create only 1 block
    no_blocks = int(sys.argv[3])

    Argument options:
    1. With less_stimulus and more_stimulus option in stimulus_quantity no_blocks should be 1.
    But with medium_stimulus option in stimulus_quantity no_blocks should be 8
    
    validation_analysis(interval_record, recalculate_rating_file, no_blocks)


if __name__ == "__main__":
        main()'''
