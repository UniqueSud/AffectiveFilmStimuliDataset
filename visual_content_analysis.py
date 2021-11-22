# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.anchored_artists import AnchoredDrawingArea
from matplotlib.patches import Circle
from skvideo.io import FFmpegWriter
import glob
import random
from adjustText import adjust_text
import pdb

def face_recognition(path,cap, image_width, image_height, total_pixels, fs, no_frames):
    print("Counting number of faces and there they are")
    opencvPath = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/Survey/opencv-master/data/haarcascades'
    xml_file = os.path.join(opencvPath,'haarcascade_frontalface_default.xml')    
    face_cascade = cv2.CascadeClassifier(xml_file)

    try:
        frames_to_pick = random.sample(np.arange(0,no_frames).tolist(),20)
    except:
        pdb.set_trace()

    face_count = []
    frame_count = 0
    
    while(1):
        ret, frame = cap.read()
        if ret == 0:
            break

        if frame_count in frames_to_pick:            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                faces = face_cascade.detectMultiScale(gray, 1.3, 3)
            except:
                pdb.set_trace()

            for (x,y,w,h) in faces:
                img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
            
            no_faces = len(faces)
            print("Number of faces are %d" %no_faces)
            if no_faces > 0:
                cv2.imwrite(os.path.join(path,'Image'+str(frame_count)+'.jpeg'),img)
            else:
                cv2.imwrite(os.path.join(path,'Image'+str(frame_count)+'.jpeg'),gray)            
            
            face_count.append(no_faces)
            frame_count = frame_count + 1
        else:
            frame_count = frame_count + 1
            continue
    print("visual content analysis-face_recognition module Face Processing Done")                
    return face_count
    
def motion_intensity(cap, image_width, image_height, total_pixels, fs, no_frames,
                     kaiser_window_length,path): # For Arousal or Visual Excitement

    #pdb.set_trace()
    print('visual_content_analysis module: Calculating Motion Intensity')
    print("motion_intensity module: Video Properties are - image width=%d image_height=%d sampling rate=%d" %(cap.get(3),cap.get(4),cap.get(5)))
    fl = glob.glob(os.path.join(path,'normalized_motion_in_frame.npy'))
    if len(fl) == 1:
        normalized_motion_in_frame = np.load(os.path.join(path,'normalized_motion_in_frame.npy'))
    else:
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 100,
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = (15,15),
                          maxLevel = 2,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Take first frame and find corners in it        
        
        flag = 0
        frame_counter = 0
        while flag == 0:
            ret, old_frame = cap.read()
            if ret == 1:
                flag = 1
            if (frame_counter == no_frames):
                print("motion_intensity module: Path is %s" %os.getcwd())
                print("motion_intensity module: flag is %d , frame_counter is %d and no_frames are %d" %(flag,frame_counter,no_frames))
                return 1
            frame_counter = frame_counter + 1
        
        print("motion_intensity module: The frame counter before starting motion calculation is ")    
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        normalized_motion_in_frame = []
        mean_motion = []
        
        frame_lost_val = []
        max_motion = 0
        
        while(1):
            try:
                frame_counter = frame_counter + 1
                ret,frame = cap.read()
                if ret == 0:
                    break
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                # Select good points
                good_new = p1[st==1]
                good_old = p0[st==1]

                if not any(st==1):
                    
                    frame_lost_val.append(frame_counter)
                    normalized_motion_in_frame.append(0)
                    old_gray = frame_gray.copy()
                    old_frame = frame.copy()
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                    # Do iteration untill you are getting next set of corner points
                    try:
                        while p0 == None:
                            frame_lost_val.append(frame_counter)
                            normalized_motion_in_frame.append(0)
                            frame_counter = frame_counter + 1
                            ret,old_frame = cap.read()
                            if ret == 0:
                                break                        
                            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                    except:
                        continue
                    # Create a mask image for drawing purposes                    
                    mask = np.zeros_like(old_frame)                    
                
                else:                    
                    motion_magnitude = ((good_new[:,0]-good_old[:,0])**2) + ((good_new[:,1]-good_old[:,1])**2)                    
                    max_mag = np.max(motion_magnitude)
                    if max_mag == 0:
                        max_mag = 0.000001
                        
                    #frame_motion = ((np.sqrt(motion_magnitude).sum())/(max_mag*float(total_pixels)))*100 # 100 for percent
                    frame_motion = ((np.sqrt(motion_magnitude).sum())/np.sqrt(float(total_pixels))) # To make it uniform everywhere, I have removed max term from denominator.
                        
                    '''if max_motion < frame_motion:
                        max_motion = frame_motion
                        max_index = frame_counter'''
                        
                    mean_motion_frame = (np.sqrt(motion_magnitude).sum())/len(motion_magnitude)
                    normalized_motion_in_frame.append(frame_motion)
                    mean_motion.append(mean_motion_frame) 
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1,1,2)
                
            except Exception as e:                
                print("Frame Count = %d. The error is \"%s\"" %(frame_counter, e))
                if p0 == None:
                    frame_lost_val.append(frame_counter)
                    normalized_motion_in_frame.append(0)
                    old_gray = frame_gray.copy()
                    old_frame = frame.copy()
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                    # Do iteration untill you are getting next set of corner points
                    try:
                        while p0 == None:
                            frame_lost_val.append(frame_counter)
                            normalized_motion_in_frame.append(0)
                            frame_counter = frame_counter + 1
                            ret,old_frame = cap.read()
                            if ret == 0:
                                break                        
                            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
                            p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
                            print("Frame with No Feature is %d" %frame_counter)
                    except:
                        continue
                    # Create a mask image for drawing purposes
                    pdb.set_trace()
                    mask = np.zeros_like(old_frame)                                        
                continue
            
        # Kaiser window is an smoothing function with steady decay and rising rate
        # It is suitable for mapping arousal from motion intensity since motion intensity
        # has maximum impact on the arousal and this arousal builds up and decays smoothly.
        normalized_motion_in_frame = np.array(normalized_motion_in_frame)        
        np.save(os.path.join(path,'normalized_motion_in_frame.npy'),normalized_motion_in_frame)
        np.save(os.path.join(path,'mean_motion_frame.npy'),mean_motion)
        
    # Kaiser Window for Smoothing
    
    window = np.kaiser(kaiser_window_length,5)
    plt.plot(window)
    plt.title('Kaiser Window for Smoothing of Video Content Attributes')
    plt.savefig(os.path.join(path,'kaiser_window.png'),dpi=600)
    plt.clf()
    # Calculating motion component to map characteristic of arousal
    convolved_motion = np.convolve(window,normalized_motion_in_frame,mode='same')
    motion_component = (np.max(normalized_motion_in_frame)/np.max(convolved_motion))*convolved_motion
    plt.figure()    
    plt.plot(np.arange(0,normalized_motion_in_frame.shape[0])/fs,normalized_motion_in_frame)
    try:
        plt.plot(np.arange(0,normalized_motion_in_frame.shape[0])/fs,motion_component)    
    except:
        pdb.set_trace()

    plt.xlabel('Time Stamps (Seconds)')    
    #max_ind = (np.arange(0,normalized_motion_in_frame.shape[0])/fs).max()
    #plt.xlim([0 max_ind])
    plt.xticks((np.arange(0,normalized_motion_in_frame.shape[0],(10*fs))/fs),rotation=45)
    plt.ylabel('Motion Magnitude')
    plt.title('Motion Component')
    plt.legend(['Raw Motion','Smoothed_Kaiser_window'])    
    plt.savefig(os.path.join(path,'Motion Component.png'),dpi=900)
    plt.clf()    
    '''Resource: Tomasi, C., & Kanade, T. (1991). Detection and tracking of point features.'''
    
    '''cut_off_thrs = (1/np.sqrt(2))*np.max(motion_component)
    arousal_bandwidth = np.where((np.isclose(motion_component,cut_off_thrs))==True)
    try:
        time_window_max_change = arousal_bandwidth[0][0]/fs
        return list([motion_component, time_window_max_change])
    except:
        print("visual_content_analysis.py module: Some Problem  with the time Calculation")'''
    print("visual content analysis - motion_intensity module Motion Intensity Calculation Done")                
    return motion_component
    cap.release()

def shot_detection(cap,fs,kaiser_window_length,total_pixels,path):  # 
    
    print('visual_content_analysis module: Calculating Shot Boundries')
    if not os.path.exists(os.path.join(path,'frame_diff.npy')):    
        ret, old_frame = cap.read()    
        no_bins = 255

        try:
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            histogram_old = np.histogram(old_gray, bins=no_bins,range=(0,no_bins))
            frame_counter = 0
            frame_diff = []
        except Exception as e:
            pdb.set_trace()

        # Histogram based frame difference to detect shot boundaries
        while(1):
            try:
                frame_counter = frame_counter + 1
                ret,frame = cap.read()            
                if ret == 0:
                    break            
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                histogram_new = np.histogram(frame_gray,bins=no_bins,range=(0,no_bins))                
                df = (np.abs(histogram_new[0] - histogram_old[0]).sum())/no_bins                
                frame_diff.append(df)
                histogram_old = histogram_new
                
            except Exception as e:
                print("Frame Count = %d. The error is \"%s\"" %(frame_counter, e))
                pdb.set_trace()
                continue

        frame_diff = np.array(frame_diff)
        np.save(os.path.join(path,'frame_diff.npy'),frame_diff)
    else:
        frame_diff = np.load(os.path.join(path,'frame_diff.npy'))
        
    mean_fd = np.mean(frame_diff)
    std_fd  = np.std(frame_diff)
    regular_diff = mean_fd + std_fd
    gradual_diff = mean_fd + (2*std_fd)
    shot_cut_diff = mean_fd + (3*std_fd)
    #shot_cut_diff = mean_fd + (4*std_fd) # Working well with 4*std, why?
    shot_boundary_frames = np.where((frame_diff>shot_cut_diff)==True)[0]
    # plot figures
    
    plt.plot(np.arange(0,frame_diff.shape[0])/fs,frame_diff)
    plt.plot(shot_boundary_frames/fs,frame_diff[shot_boundary_frames],'o')
    plt.xlabel('Time Stamps (Seconds)')
    plt.xticks((np.arange(0,frame_diff.shape[0],(5*fs))/fs),rotation=45)
    plt.ylabel('interFrame Difference using Histogram')
    plt.title('Shot Boundaries in the video')
    plt.savefig(os.path.join(path,'Shot Boundaries.png'),dpi=600)
    plt.clf()    
    frames_timing = shot_boundary_frames/fs
    np.save(os.path.join(path,'shot_boundary_frames.npy'),shot_boundary_frames)
    print('visual_content_analysis module: Done with Shot Boundries')
    print("visual content analysis - shot_detection module shot detection Done")    
    return list([frame_diff,shot_boundary_frames,frames_timing])
    '''Resource: Pardo, A. (2005, November). Simple and robust hard cut detection using
    interframe differences. In Iberoamerican Congress on Pattern Recognition (pp. 409-419). Springer, Berlin, Heidelberg.'''
    
def rhythm_components(shot_boundary_frames,kaiser_window_length,fs,path): # For Arousal

    print('visual_content_analysis module: Calculating Rhythm Component')
    # Trying to calculate rhythm component as an determining dimension of arousal.
    if len(shot_boundary_frames.shape) > 1:
        shot_boundary_frames = shot_boundary_frames
    no_cuts = len(shot_boundary_frames)
    left_cut = 0
    arousal_based_rhythm = []
    for cut_index in range(no_cuts):
        right_cut = shot_boundary_frames[cut_index]        
    # Since negative exponential is inverse mapping function. So, if input is increased the output will be decreased.
        no_repeats = right_cut-left_cut        
    #repeated = np.repeat(100*np.exp(1-(right_cut-left_cut)),no_repeats) # 100 here for percent
        repeated = np.repeat(np.exp(1-(right_cut-left_cut)),no_repeats)
        arousal_based_rhythm =  np.append(arousal_based_rhythm,repeated,axis=0)
        left_cut = right_cut

    # Kaiser window is an smoothing function with steady decay and rising rate
    # It is suitable for mapping arousal from motion intensity since motion intensity
    # has maximum impact on the arousal and this arousal builds up and decays smoothly.
    window = np.kaiser(kaiser_window_length,5)    
    # Calculating motion component to map characteristic of arousal
    convolved_rhythm = np.convolve(arousal_based_rhythm, window, mode='full')
    
    rhythm_component = (np.max(arousal_based_rhythm)/np.max(convolved_rhythm))*convolved_rhythm

    plt.plot(np.arange(0,rhythm_component.shape[0])/fs,rhythm_component)
    plt.plot(np.arange(0,arousal_based_rhythm.shape[0])/fs,arousal_based_rhythm)
    plt.xlabel('Time Stamps (Seconds)')
    plt.xticks((np.arange(0,rhythm_component.shape[0],(10*fs))/fs),rotation=45)
    plt.ylabel('Value of Rhythm')
    plt.title('Rhythm of The Video')
    plt.legend(['Raw Rhythm','Smoothed_Kaiser_Rhythm'])    
    plt.savefig(os.path.join(path,'Rhythm.png'),dpi=600)
    plt.clf()
    print("visual content analysis - rhythm compoent module rhythm components Done")    
    return rhythm_component
    '''Resource:Hanjalic, A., & Xu, L. Q. (2005). Affective video content representation and modeling.
    IEEE transactions on multimedia, 7(1), 143-154.'''
    
def brightness(cap,total_pixels,fs,path): # For Valence
    print('visual_content_analysis module: Calculating Brightness')        

    bright_array = []    
    while(1):
        try:
            ret, frame = cap.read()
            if ret == 0:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            bright_array.append(hsv[:,:,2].sum()/total_pixels)
        except Exception as e:
            pdb.set_trace()
    bright_array = np.array(bright_array)
    np.save(os.path.join(path,'bright_array.npy'),bright_array)
   
    plt.plot(np.arange(0,bright_array.shape[0])/fs,bright_array)
    mean_bright = np.mean(bright_array)
    
    plt.plot([0,bright_array.shape[0]/fs],[mean_bright,mean_bright],'-')
    plt.xlabel('Time Stamps (Seconds)')
    plt.xticks((np.arange(0,bright_array.shape[0],(10*fs))/fs),rotation=45)
    plt.ylabel('Average Brightness')
    plt.title('Brightness Property')    
    plt.savefig(os.path.join(path,'Brightness.png'),dpi=600)
    plt.clf()
    print("visual content analysis - shot_detection module shot detection Done")    
    return list([bright_array, mean_bright])
    '''Resource: mohamedameen93. “mohamedameen93/Advanced-Lane-Finding-Using-OpenCV.”
GitHub, 25 Dec. 2017, github.com/mohamedameen93/Advanced-Lane-Finding-Using-OpenCV.'''

def shot_rate(fs,no_frames,path): # Arousal
    print('visual_content_analysis module: Calculating Shot Rate')
    
    shot_boundary_frames = np.load(os.path.join(path,'shot_boundary_frames.npy'))
    frames_timing = shot_boundary_frames/fs    
    no_shots = frames_timing.shape[0]
    diff_rate = []
    
    for shot_index in range(1,no_shots):
        diff_rate.append((frames_timing[shot_index]-frames_timing[shot_index-1]))
        
    overall_diff_rate = (np.array(diff_rate).sum())/no_frames
    np.save(os.path.join(path,'diff_rate.npy'),diff_rate)
    plt.plot(diff_rate)
    plt.xlabel('cut_index')
    plt.ylabel('Differentiation_val')
    plt.title('Shot_Rate')
    plt.text(no_shots/2,np.max(diff_rate)-1,'overall_shot_rate:'+str(np.round(overall_diff_rate,4)))
    plt.savefig(os.path.join(path,'Shot_Rate.png'),dpi=600)
    plt.clf()
    
    return list([diff_rate, overall_diff_rate])

'''if __int__ = "__main__":
    shot_detection()''dsfsd'''
