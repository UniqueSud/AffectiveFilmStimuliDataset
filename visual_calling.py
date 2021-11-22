import cv2
import os
import numpy as np
from visual_content_analysis import motion_intensity,shot_detection,rhythm_components,brightness,shot_rate,face_recognition
import matplotlib.pyplot as plt
import pandas as pd
import glob
import subprocess
import sys
import multiprocessing as np
from joblib import Parallel, delayed
import pickle
import pdb

def range_conv(val, source_min, source_max, target_min,target_max):

    if len(val) == 1:
        return target_min+(((val[0]-source_min)/(source_max-source_min))*(target_max-target_min))
    else:
        converted_val = []
        for index in range(len(val)):
            converted_val.append(target_min+(((val[index]-source_min)/(source_max-source_min))*(target_max-target_min)))
    return converted_val

def visual_valence_arousal_vector(TargetclipDire, t):
    import numpy as np
    videoFile = t
    print("Inside visual_valence_arousal module")
    file_name = videoFile.split('/')[-1].split('.')[0]
    print('======')
    print(file_name)
    #path = os.path.join(cur_path, file_name)
    cur_path = '/'.join(videoFile.split('/')[:-1])
    path = TargetclipDire
  
    os.chdir(cur_path)    

    cap  = cv2.VideoCapture(videoFile)
    image_width = cap.get(3)
    image_height = cap.get(4)
    total_pixels = image_width * image_height
    fs = cap.get(5)
    no_frames = cap.get(7)
    try:
        total_time = no_frames/fs
    except:
        pdb.set_trace()
        
    fs = int(fs)
    no_frames = int(no_frames)
    valence = []
    arousal = []

    kaiser_window_length = 5*fs    

    duration_limit = 60 # In seconds
    low_index = 0
    halt_point = 60 * fs    
    window_minute = duration_limit * fs
    window_count = 0
    print("visual_calling.py module: VideoResolution:(%dx%d), Sampling: %d, Number_Frames: %d" %(image_width,image_height,fs,no_frames))
    
    #if no_frames > (duration_limit * fs):
    #try:
    '''if not os.path.exists(os.path.join(TargetclipDire, file_name+'_face_count.npy')):
        face_vector = face_recognition(path, cap, image_width, image_height, total_pixels, fs, no_frames)
        np.save(os.path.join(TargetclipDire, file_name+'_face_count.npy'), face_vector)
    else:
        face_vector = np.load(os.path.join(TargetclipDire, file_name+'_face_count.npy'))
        if face_vector.shape[0] == 0:
            face_vector = face_recognition(cap, image_width, image_height, total_pixels, fs, no_frames)
            np.save(os.path.join(TargetclipDire, file_name+'_face_count.npy'), face_vector)
    cap.release()'''

    print(fs)
    if fs > 30:
        print('Problem with the video file = %s' %file_name)
        return 1
    #pdb.set_trace()
    cap  = cv2.VideoCapture(videoFile)                    
    if not os.path.exists(os.path.join(TargetclipDire, file_name+'_motion_component.npy')):
    #if os.path.exists(os.path.join(TargetclipDire, file_name+'_motion_component.npy')):
        motion_component = motion_intensity(cap, image_width, image_height, total_pixels, fs, no_frames,
                 kaiser_window_length,path)
        try:
            if motion_component == 1:
                return 1
        except:
            np.save(os.path.join(TargetclipDire, file_name+'_motion_component.npy'),motion_component)
    else:
        motion_component = np.load(os.path.join(TargetclipDire, file_name+'_motion_component.npy'))
        if motion_component.shape[0] == 0:
            motion_component = motion_intensity(cap, image_width, image_height, total_pixels, fs, no_frames,
                     kaiser_window_length,path)            
            np.save(os.path.join(TargetclipDire, file_name+'_motion_component.npy'),motion_component)

    try:
        print(len(motion_component))
    except:
        pdb.set_trace()

    max_motion_comp = motion_component.max()
    min_motion_comp = motion_component.min()
    mean_motion_comp = motion_component.mean()
    std_motion_comp = motion_component.std()
    motion_percent_1, motion_percent_2, motion_percent_3 = np.percentile(motion_component, [25, 50, 75])

    cap.release()
    

    # Calculating Shots
    cap  = cv2.VideoCapture(videoFile)
    if not os.path.isfile(os.path.join(TargetclipDire, file_name+'_shot_boundary_frames.npy')):
        frame_diff,shot_boundary_frames,frames_timing = shot_detection(cap,fs,kaiser_window_length,total_pixels,path)
        np.save(os.path.join(TargetclipDire, file_name+'_shot_boundary_frames.npy'), shot_boundary_frames)
    else:
        shot_boundary_frames = np.load(os.path.join(TargetclipDire, file_name+'_shot_boundary_frames.npy'))
        if shot_boundary_frames.shape[0] == 0:
            frame_diff,shot_boundary_frames,frames_timing = shot_detection(cap,fs,kaiser_window_length,total_pixels,path)
    
    shotRate_ = shot_boundary_frames[1:]-shot_boundary_frames[0:-1]
    max_shotRate_ = shotRate_.max()
    min_shotRate_ = shotRate_.min()
    mean_shotRate_ = shotRate_.mean()
    std_shotRate_ = shotRate_.std()
    _shotRate_percent_1, _shotRate_percent_2, _shotRate_percent_3 = np.percentile(shotRate_, [25, 50, 75])

    cap.release()
    
    # Calculating Rhythm Component here            
    if not os.path.isfile(os.path.join(TargetclipDire, file_name+'_rhythm_component.npy')):
        rhythm_comp = rhythm_components(shot_boundary_frames,kaiser_window_length,fs,path)
        np.save(os.path.join(TargetclipDire, file_name+'_rhythm_component.npy'),rhythm_comp)
    else:            
        rhythm_comp = np.load(os.path.join(TargetclipDire, file_name+'_rhythm_component.npy'))
        if rhythm_comp.shape[0] == 0:
            rhythm_comp = rhythm_components(shot_boundary_frames,kaiser_window_length,fs,path)
            np.save(os.path.join(TargetclipDire, file_name+'_rhythm_component.npy'),rhythm_comp)
    max_rhythm_comp = rhythm_comp.max()
    min_rhythm_comp = rhythm_comp.min()
    mean_rhythm_comp = rhythm_comp.mean()
    std_rhythm_comp = rhythm_comp.std()
    rhythm_comp_percent_1, rhythm_comp_percent_2, rhythm_comp_percent_3 = np.percentile(rhythm_comp, [25, 50, 75])

    # Calculating Brightness here
    cap  = cv2.VideoCapture(videoFile)
    if not os.path.isfile(os.path.join(TargetclipDire, file_name+'_brightness_array.npy')):
        bright_array, mean_bright = brightness(cap,total_pixels,fs,path)
        np.save(os.path.join(TargetclipDire, file_name+'_brightness_array.npy'),bright_array)
    else:
        bright_array = np.load(os.path.join(TargetclipDire, file_name+'_brightness_array.npy'))
        if bright_array.shape[0] == 0:
            bright_array, mean_bright = brightness(cap,total_pixels,fs,path)
            np.save(os.path.join(TargetclipDire, file_name+'_brightness_array.npy'),bright_array)                    
    max_bright_array = bright_array.max()
    min_bright_array = bright_array.min()            
    mean_bright_array = bright_array.mean()
    std_bright_array = bright_array.std()
    bright_array_percent_1, bright_array_percent_2, bright_array_percent_3 = np.percentile(bright_array, [25, 50, 75])

    cap.release()
    
    # Shot Rate
    if not os.path.isfile(os.path.join(TargetclipDire, file_name+'_shot_rate.npy')):
        shotRate, overall_diff_rate = shot_rate(fs,no_frames,path)
        np.save(os.path.join(TargetclipDire, file_name+'_shot_rate.npy'),shotRate)
    else:
        shotRate = np.load(os.path.join(TargetclipDire, file_name+'_shot_rate.npy'))
        if shotRate.shape[0] == 0:
            shotRate, overall_diff_rate = shot_rate(fs,no_frames,path)
            np.save(os.path.join(TargetclipDire, file_name+'_shot_rate.npy'),shotRate)

    shotRate = np.array(shotRate)
    try:
        max_shotRate = shotRate.max()
    except:
        pdb.set_trace()

    min_shotRate = shotRate.min()            
    mean_shotRate = shotRate.mean()
    std_shotRate = shotRate.std()
    shotRate_percent_1, shotRate_percent_2, shotRate_percent_3 = np.percentile(shotRate, [25, 50, 75])

    valence_stack = []
    arousal_stack = []
    valence_len = [0]
    arousal_len = [0]
        
    return(list([max_motion_comp, min_motion_comp, mean_motion_comp, std_motion_comp, motion_percent_1, motion_percent_2, motion_percent_3, max_shotRate_, min_shotRate_, mean_shotRate_, std_shotRate_, _shotRate_percent_1, _shotRate_percent_2, _shotRate_percent_3, max_rhythm_comp, min_rhythm_comp, mean_rhythm_comp, std_rhythm_comp, rhythm_comp_percent_1, rhythm_comp_percent_2, rhythm_comp_percent_3, max_bright_array, min_bright_array, mean_bright_array, std_bright_array, bright_array_percent_1, bright_array_percent_2, bright_array_percent_3, max_shotRate, min_shotRate, mean_shotRate, std_shotRate, shotRate_percent_1, shotRate_percent_2, shotRate_percent_3]))
    #except Exception as e:
    #    print("*********************Line 176: Visual_calling.py: The Error is %s***********************" %e)

    pdb.set_trace()
    for frame_ind in range(no_frames - 1):
        #try:
        arousal.append(range_conv([motion_component[frame_ind]],min_motion_comp,max_motion_comp,0,1))
        valence.append(range_conv([bright_array[frame_ind]],min_bright_array,max_bright_array,0,1))

        if frame_ind == halt_point:                                 
            
            '''The following variable has been defined explicitly to have the track of "Number of samples each time being considered_samples"
            for the calculation of valence and arousal value while appending in the valence_stack and arousal_stack'''
            no_considered_samples = halt_point - low_index + 1

            '''This calculation is being done to calculate shot boundaries occured between the low_index and halt_point.
            which are end markers of one minute segment. Later on this first index of temp1 and temp2 will be used for calculating 
            shot rate.'''                  

            temp1 = np.where((low_index<shot_boundary_frames)==True)[0]
            temp2 = np.where((halt_point<shot_boundary_frames)==True)[0]                    
            
            # If temp1 is zero it means we have already encountered all the shots and now shot rate should be 0
            if (len(temp1)==0):
                arousal.append(0)
                cur_valence = valence[valence_len[-1],:]  # For every 60 seconds
                cur_arousal = arousal[valence_len[-1]:]  # For every 60 seconds                  
                valence_stack.append(np.sqrt((np.array(cur_valence)**2).sum()))
                arousal_stack.append(np.sqrt((np.array(cur_arousal)**2).sum()))

                window_count = window_count + 1
                # For the next time halting
                low_index = window_count * fs # 0, fs, 2*fs, 3*fs, 4*fs 
                halt_point=low_index+window_minute # 60*fs,61*fs,62*fs,63*fs,64*fs

                valence_len.append(low_index)
                arousal_len.append(low_index)                        
                del cur_valence
                del cur_arousal
                print("============>>>>>>>>>>> Length temp1 is Zero <<<<<<<<<<<<=============")
                continue
            else:
                first_shot_index = temp1[0]
            
            if (len(temp2)==0):
                last_shot_index = len(shot_boundary_frames)
            else:
                last_shot_index = temp2[0]

            ''' Calculating number of shots per second by subtracting last shot index from first shot index and dividing it by 
            time of one second segment.'''                    

            Shot_Rate = (last_shot_index-first_shot_index)/float(duration_limit)
            if Shot_Rate > 1.0:
                print("===========>>>>>>>>>>>>Shot Rate is more than 1<<<<<<<<=============")

            cur_valence = valence[valence_len[-1]:]  # For every 60 seconds
            cur_arousal = arousal[arousal_len[-1]:]  # For every 60 seconds                                      
            if not ((len(cur_valence) == no_considered_samples) and (len(cur_arousal) == no_considered_samples)):
                raise ValueError("Length of valence and arousal vector should match with number of frames considered")

            valence_stack.append(np.sqrt((np.array(cur_valence)**2).sum()))
            arousal_stack.append(np.sqrt((np.array(np.append(cur_arousal,Shot_Rate))**2).sum()))
                
            '''valence_vector.append({str(window_count):np.sqrt((np.array(cur_valence)**2).sum())})
            arousal_vector.append({str(window_count):np.sqrt((np.array(cur_arousal)**2).sum())})'''                
            
            print("Low Index: %d, Halt_Point: %d, len(cur_valence): %d, len(cur_arousal): %d Shot Rate: %f" %(low_index, halt_point,len(cur_valence),len(cur_arousal), Shot_Rate))
            del cur_valence
            del cur_arousal
             
            window_count = window_count + 1
            # For the next time halting
            low_index = window_count * fs # 0, fs, 2*fs, 3*fs, 4*fs 
            halt_point=low_index+window_minute # 60*fs,61*fs,62*fs,63*fs,64*fs
            
            if halt_point > no_frames:
                halt_point = no_frames-2

            # Updating the valence_len and arousal_len so that last index of these arrays could become starting point 
            # to fetch the content from valence and arousal array which are storing everything and ever growing.
            valence_len.append(low_index)
            arousal_len.append(low_index)                        
                
        #except Exception as e:
        #    print('*********************Line 259: Visual_calling.py: Inside Exception. Error is %s*******************' %e)   

    try:
        valence_stack = np.array(valence_stack)
        arousal_stack = np.array(arousal_stack)
        valence_vector = range_conv(valence_stack,valence_stack.min(),valence_stack.max(),1,9)
        arousal_vector = range_conv(arousal_stack,arousal_stack.min(),arousal_stack.max(),1,9)
        plt.plot(valence_vector)
        plt.plot(arousal_vector)
        plt.xlabel("Start Time of Clip")
        plt.xticks(np.arange(0,valence_stack.shape[0],5),rotation=45)
        plt.ylabel('Valece-Arousal Values')
        plt.title('Valence-Arousal Plot of the Movie')
        plt.legend(['Valence','Arousal'],loc=0)
        plt.text(valence_stack.shape[0]/2,valence_stack.max(),'Total Time:'+str(total_time))            
        plt.savefig(os.path.join(path,file_name+'_VA_Plot.png'),dpi=900)
        plt.clf()

        plt.plot(valence_len)
        plt.plot(arousal_len)
        plt.xlabel("Start Time of Clip")
        plt.xticks(np.arange(0,valence_stack.shape[0],5),rotation=45)
        plt.ylabel('Valece-Arousal Values')
        plt.title('Valence-Arousal Length Plot of the Movie')
        plt.legend(['Valence','Arousal'],loc=0)
        plt.text(valence_stack.shape[0]/2,valence_stack.max(),'Total Time:'+str(total_time))            
        plt.savefig(os.path.join(path,file_name+'_VA_length_Plot.png'),dpi=900)
        plt.clf()
        
    except Exception as e:
        print("Line-252: The Error is %s" %e)
        #pdb.set_trace()
            
    return list([valence_vector,arousal_vector])

def processing_in_parallel(t):
    import numpy as np
    cur_dir = os.getcwd()
    
    try:
        for file_item in t[0]: # Getting inside Emotion Directory
            try:                    
                fullPath = os.path.join(cur_dir, file_item)
                print("Emotion Name is %s" %fullPath)
                os.chdir(fullPath)
                videoFile = np.array(glob.glob("*.mp*"))                
                videoFile = np.append(videoFile,glob.glob("*.web*"))
                videoFile = videoFile.tolist()

                if len(videoFile[0]) > 0:
                    vec = {}
                    if not os.path.exists('VideoContentFeatures'+file_item+'.pkl'):
                        '''with open('VideoContentFeatures.pkl') as outfile:
                            pickle_file = pickle.load(outfile)
                        if len(pickle_file) > 0:'''

                        for fileName in videoFile: # Counting on all video files
                            print("=============>>>>>>>>>>>> Now Processing the File %s" %fileName)
                            vec_res = visual_valence_arousal_vector(fileName)

                            try:
                                if vec_res == 1:
                                    print("******************************************Vector Result is 1***************************************************")
                                    continue
                                else:
                                    vec[fileName] = vec_res
                            except:                                
                                #vec.append({fileName:vec_res})
                                vec[fileName] = vec_res

                        with open('VideoContentFeatures'+file_item+'.pkl','wb') as outfile:                
                            #pdb.set_trace()            
                            pickle.dump(vec,outfile)
                            #pdb.set_trace()
            except Exception as e:
                print("Line: 330 - visual_calling.py, The Error is %s" %e)                    
                os.chdir(cur_dir)
                continue
            os.chdir(cur_dir)

    except Exception as e:
        print("Line: 184 - visual_calling.py, The Error is %s" %e)
        os.chdir(cur_dir)        
    
def loading_files(nCores=1):
    import numpy as np
    cur_dir = os.getcwd()
    emotion_df = pd.read_csv('Wordnet_Emotions.csv').as_matrix()
    emotions = emotion_df[:,0]
    polarity = emotion_df[:,1]
    tasks = []
    
    for emt_index in range(len(emotions)):
        print("===================>>>>>>>>>>>>>>>The index counter being processed is = %d<<<<<<<<<<<<<<<<<<<<<<<==========================" %emt_index)
        emt_item = emotions[emt_index]        
        files = glob.glob(emt_item+"*")
        tasks.append([files])
    Parallel(n_jobs=nCores)(delayed(processing_in_parallel)(t) for t in tasks)
    
               
    
'''
from visual_calling import visual_valence_arousal_vector
vec = visual_valence_arousal_vector('Rage_Official_Trailer.mp4')'''
