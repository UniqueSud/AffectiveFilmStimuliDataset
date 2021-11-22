import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras.layers import Dense, Dropout, GaussianNoise
#from tensorflow.keras import regularizers
import random
from sklearn.metrics import confusion_matrix, f1_score
import os
import pdb

videoPrefix = 'WithThirtyVideos_'
if videoPrefix == 'WithThirtyVideos_':
    clipDire = '/mnt/7CBFA0EC210FC340/ExperimentRelatedData/FromUbuntuAcerSystem/Experiment/block_For_30_Stimuli'
else:
    clipDire = os.path.join(sourceDir, 'all_clips')

#if emotionWise 
DF=pd.read_csv(os.path.join(clipDire, videoPrefix+'ForClassificaion_AllMultimediaCalculatedPCADimensionsValenceArousal.csv'), index_col=0)
#DF=pd.read_csv(os.path.join(sourceDir, 'all_clips/ForClassificaion_AllMultimediaCalculatedPCADimensionsValenceArousalGB.csv'), index_col=0)

# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold

'''from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD'''

# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model()
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('&gt; %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    return scores, histories

# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()

# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()

# run the test harness for evaluating a model
def DeepLearning(trainX,trainY,testX, testY, classList):
    # load dataset
    # prepare pixel data
    #trainX, testX = prep_pixels(trainX, testX)
    # evaluate model
    pdb.set_trace()
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)

def svmClass(X, y, X_test, y_test, classList):
    from sklearn import svm
    # kernels: kernel='rbf', kernel='linear', kernel='sigmoid'

    no_classes = len(classList)
    clf = svm.SVC()
    clf.fit(X, y)
    
    overallAccMax = 0
    flag = 0
    for iter_ in np.arange(20):
        minn = 1000
        for class_ in classList:
            if minn > sum(y_test==class_):
                minn = sum(y_test==class_)
        
        for class_ in classList:
            print('How minn[0] is working here when minn is a scalar')            
            sampIdxs = random.sample(np.where(y_test==class_)[0].tolist(), minn[0])

            if class_ == 0:
                newTestSet = X_test[sampIdxs, :]
                newTargSet = y_test[sampIdxs]
            else:
                newTestSet = np.concatenate((newTestSet, X_test[sampIdxs, :]), axis=0)
                newTargSet = np.concatenate((newTargSet, y_test[sampIdxs]), axis=0)

        print(f' Test Target 0 = {sum(newTargSet.reshape(-1)==0)}')
        print(f' Test Target 1 = {sum(newTargSet.reshape(-1)==1)}')
        print(f' Test Target 2 = {sum(newTargSet.reshape(-1)==2)}')

        pre_labels = clf.predict(newTestSet).reshape(-1,1)
        res = f1_score(newTargSet, pre_labels, average=None).reshape(1,no_classes)
        cm = confusion_matrix(newTargSet, pre_labels)
        overallAccuracy = [sum(cm[np.arange(no_classes),np.arange(no_classes)])/np.sum(cm)]
        cmArr = cm.reshape(1,no_classes,no_classes)

        clf = svm.SVC(C=0.5)
        clf.fit(X, y)
        pre_labels = clf.predict(newTestSet).reshape(-1,1)
        res = np.concatenate((res, f1_score(newTargSet, pre_labels, average=None).reshape(1,no_classes)), axis=0)
        cm = confusion_matrix(newTargSet, pre_labels)
        cmArr = np.concatenate((cmArr, cm.reshape(1,no_classes,no_classes)), axis=0)
        overallAccuracy.extend([sum(cm[np.arange(no_classes),np.arange(no_classes)])/np.sum(cm)])

        clf = svm.SVC(C=0.75)
        clf.fit(X, y)
        pre_labels = clf.predict(newTestSet).reshape(-1,1)
        res = np.concatenate((res, f1_score(newTargSet, pre_labels, average=None).reshape(1,no_classes)), axis=0)
        cm = confusion_matrix(newTargSet, pre_labels)
        cmArr = np.concatenate((cmArr, cm.reshape(1,no_classes,no_classes)), axis=0)
        overallAccuracy.extend([sum(cm[np.arange(no_classes),np.arange(no_classes)])/np.sum(cm)])


        index = np.argmax(overallAccuracy)

        if flag == 0:            
            newOverallAcc = overallAccuracy[index]
            newF1Score = np.reshape(res[index], (1, no_classes))            #
            newCMatrix = np.reshape(cmArr[index], (1,no_classes,no_classes))          #
            flag = 1            
        else:
            newOverallAcc = newOverallAcc+overallAccuracy[index]    # 
            newF1Score = np.concatenate((newF1Score, np.reshape(res[index], (1, no_classes))), axis=0)
            newCMatrix = np.concatenate((newCMatrix, np.reshape(cmArr[index], (1,no_classes,no_classes))), axis=0)

        '''if flag == 0:
            if max(overallAccuracy) > overallAccMax:
                index = np.argmax(overallAccuracy) # Finding that out of three settings in which setting the accuracy is maximum
                overallAccMax = overallAccuracy[index] #
                newOverallAcc = overallAccuracy    # 
                newF1Score = res[index]            #
                newCMatrix = cmArr[index]          #
                flag = 1

        if flag == 1:
            if max(overallAccuracy) > overallAccMax:
                index = np.argmax(overallAccuracy) # Finding that out of three settings in which setting the accuracy is maximum
                overallAccMax = overallAccuracy[index] #
                newOverallAcc = newOverallAcc+overallAccuracy    # 
                newF1Score = newF1Score + res[index]            #
                newCMatrix = newCMatrix + cmArr[index]          #'''

    print(newOverallAcc/20)
    print(np.mean(newF1Score, axis=0))
    print(np.mean(newCMatrix, axis=0))

    index = np.argmax(overallAccuracy)
    return list([np.mean(newF1Score, axis=0), np.mean(newCMatrix, axis=0), newOverallAcc/20, minn])


def with4Quadrants():

    #############################  Data Cleaning 

    noFeature = len(DF.keys())-3
    DF['index'] = np.arange(len(DF))
    Experiment_ids = DF.index.values

    DF.set_index('index', drop=True, inplace=True)      
    # Find out the duplicate data.
    duplicateIndexes = np.where(DF.duplicated().values)[0]
    DF.drop(duplicateIndexes, axis=0, inplace=True)
    DF.reset_index(drop=True, inplace=True)
    target = DF.iloc[:, noFeature+2].values.reshape(-1, 1)
    ############################################    
    #### If you want to remove 4 class (class with very less samples) selection remove4Class = 1; no_classes = 3, else remove4Class = 0; no_classes = 4
    remove4Class = 1
    no_classes = 3
    ####### Removing HVLA Class. Since, it has less number of samples ############
    if remove4Class == 1:
        indexToRemove = DF.index.values[np.where(target==3)[0]]
        DF.drop(indexToRemove, axis=0, inplace=True)
        DF.reset_index(drop=True, inplace=True)
        #DF['Experiment_id'] = Experiment_ids
        #DF.set_index('Experiment_id', drop=True, inplace=True)
    ##################################

    #### Here I am selecting the random features. It was an attempt to reduce the overfitting problem ############
    randomFeature = random.sample(np.arange(noFeature).tolist(), noFeature)
    np.save('randomFeatures.npy', randomFeature)
    #########################################################

    ValArlDim = DF.iloc[:, randomFeature].values
    target = DF.iloc[:, noFeature+2].values.reshape(-1, 1)

    # Data split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(ValArlDim, target, test_size = 0.2, random_state = 0)

    print(sum(y_train==0))
    print(sum(y_train==1))
    print(sum(y_train==2))
    if remove4Class == 0:
        print(sum(y_train==3))
        minSamples = np.min([sum(y_train==0), sum(y_train==1), sum(y_train==2), sum(y_train==3)])
        class1Samples = np.where(y_train==0)[0]
        class2Samples = np.where(y_train==1)[0]
        class3Samples = np.where(y_train==2)[0]
        class4Samples = np.where(y_train==3)[0]
    else:
        minSamples = np.min([sum(y_train==0), sum(y_train==1), sum(y_train==2)])
        class1Samples = np.where(y_train==0)[0]
        class2Samples = np.where(y_train==1)[0]
        class3Samples = np.where(y_train==2)[0]

    maxRes = [0]
    svmAcc = 0
    for repeat in np.arange(50):

        class1Idx = random.sample(class1Samples.tolist(), minSamples)
        class2Idx = random.sample(class2Samples.tolist(), minSamples)
        class3Idx = random.sample(class3Samples.tolist(), minSamples)

        trainingData = X_train[class1Idx,:]
        trainingData = np.concatenate((trainingData, X_train[class2Idx,:]), axis=0)
        trainingData = np.concatenate((trainingData, X_train[class3Idx,:]), axis=0)

        if remove4Class == 0:   
            trainingData = np.concatenate((trainingData, X_train[class4Samples,:]), axis=0)
        #### Adding Noise here
        #noise = np.random.normal(0, 1, trainingData.shape)
        #trainingData = trainingData + noise
        #trainingData = np.concatenate((trainingData, noise), axis=0) 

        allTargets = y_train[class1Idx]
        allTargets = np.concatenate((allTargets, y_train[class2Idx]), axis=0)
        allTargets = np.concatenate((allTargets, y_train[class3Idx]), axis=0)
        if remove4Class == 0:   
            allTargets = np.concatenate((allTargets, y_train[class4Samples]), axis=0)
        #allTargets = np.concatenate((allTargets, allTargets), axis=0) ### THese are the labels for noise.

        # Feature scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        trainingData = sc.fit_transform(trainingData)
        X_test = sc.fit_transform(X_test)

        transDataTrain = trainingData.copy()
        transDataTest = X_test.copy()

        svmDict = {0:'rbf', 1:'rbf_with_C-0.5', 2:'rbf_with_C-0.75'}
    #################### SVM Classification #############
        #res = svmClass(trainingData, allTargets, X_test, y_test)
        res = svmClass(transDataTrain, allTargets, transDataTest, y_test, no_classes)
        if res[2] > svmAcc:     
            np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_f1_score.npy', res[0])
            np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_conf_mat.npy', res[1])
            np.save(videoPrefix+'no_classes-%s_' %str(no_classes)+svmDict[res[3]]+'_svm_over_acc.npy', res[2])
            svmAcc = res[2]


############################################## Classification in valence and arousal only ########################################################

def with2Classes():
    #############################  Data Cleaning 
    valenceFlag = 1 # if 1: valence calculation, else Arousal Calculation
    no_classes = 2
    DF['index'] = np.arange(len(DF))
    Experiment_ids = DF.index.values
    pdb.set_trace()
    DF.set_index('index', drop=True, inplace=True)      
    # Find out the duplicate data.
    duplicateIndexes = np.where(DF.duplicated().values)[0]
    DF.drop(duplicateIndexes, axis=0, inplace=True)
    DF.reset_index(drop=True, inplace=True)

    ################ For valence 
    valenceFeatColumns = ['max_rhythm_comp', 'min_rhythm_comp', 'mean_rhythm_comp', 'std_rhythm_comp', 'rhythm_comp_percent_1', 'rhythm_comp_percent_2', 'rhythm_comp_percent_3', 'max_bright_array', 'min_bright_array', 'mean_bright_array', 'std_bright_array', 'bright_array_percent_1', 'bright_array_percent_2', 'bright_array_percent_3']
    valenceFeatColumns.extend(['val-'+str(i) for i in np.arange(15)])
    valFeatFrame = DF.loc[:, valenceFeatColumns]
    valFeatFrame['target'] = DF['target'] 
    valFeatFrame["target"].replace({0:0, 3:0, 1:1, 2:1}, inplace=True)
    valtarget = valFeatFrame.loc[:, 'target'].values.reshape(-1, 1)
    valFeatDim = valFeatFrame.drop('target', axis=1).values

    arousalFeatColumns = ['max_motion_comp', 'min_motion_comp', 'mean_motion_comp', 'std_motion_comp', 'motion_percent_1', 'motion_percent_2', 'motion_percent_3', 'max_shotRate', 'min_shotRate', 'mean_shotRate', 'std_shotRate', 'shotRate_percent_1', 'shotRate_percent_2', 'shotRate_percent_3']
    arousalFeatColumns.extend(['arl-'+str(i) for i in np.arange(15)])
    arlFeatFrame = DF.loc[:, arousalFeatColumns]
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

    minSamples = np.min([sum(y_train==0), sum(y_train==1)])
    class1Samples = np.where(y_train==0)[0]
    class2Samples = np.where(y_train==1)[0]

    svmAcc = 0
    for repeat in np.arange(50):

        class1Idx = random.sample(class1Samples.tolist(), minSamples)
        class2Idx = random.sample(class2Samples.tolist(), minSamples)

        trainingData = X_train[class1Idx,:]
        trainingData = np.concatenate((trainingData, X_train[class2Idx,:]), axis=0)

        allTargets = y_train[class1Idx]
        allTargets = np.concatenate((allTargets, y_train[class2Idx]), axis=0)

        # Feature scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        trainingData = sc.fit_transform(trainingData)
        X_test = sc.fit_transform(X_test)

        svmDict = {0:'rbf', 1:'rbf_with_C-0.5', 2:'rbf_with_C-0.75'}
    #################### SVM Classification #############
        #res = svmClass(trainingData, allTargets, X_test, y_test)
        res = svmClass(trainingData, allTargets, X_test, y_test, no_classes)
        if res[2] > svmAcc:     
            np.save(videoPrefix+prefix+'-'+svmDict[res[3]]+'_svm_f1_score.npy', res[0])
            np.save(videoPrefix+prefix+'-'+svmDict[res[3]]+'_svm_conf_mat.npy', res[1])
            np.save(videoPrefix+prefix+'-'+svmDict[res[3]]+'_svm_over_acc.npy', res[2])
            svmAcc = res[2]

    pdb.set_trace()
    '''
    from sklearn.preprocessing import OneHotEncoder
    onehotencoder = OneHotEncoder()
    allTargets = onehotencoder.fit_transform(allTargets).toarray()

    # Model building
    classifier = keras.Sequential()
    #add input layer and first hidden layer
    #classifier.add(Dropout(0.5, name='dropout1'))
    classifier.add(Dense(30, kernel_regularizer=regularizers.l2(0.001), activation = 'relu', input_dim = noFeature))
    #classifier.add(GaussianNoise(0.5))
    #classifier.add(Dropout(0.5, name='dropout2'))
    #classifier.add(Dense(30, kernel_regularizer=regularizers.l2(0.001), activation = 'relu'))
    #classifier.add(Dropout(0.1, name='dropout1'))

    #add 2nd hidden layer
    #classifier.add(Dense(output_dim = 6, init = ‘uniform’, activation = ‘relu’))
    if remove4Class == 0:   
        classifier.add(Dense(4, activation = 'softmax'))
    else:
        classifier.add(Dense(3, activation = 'softmax'))

    # Model compiling
    batchsize = 20 #int(trainingData.shape[0]/100)
    classifier.compile(optimizer = 'Adam', loss ='categorical_crossentropy', metrics = ['accuracy'])
    print(trainingData.shape)
    print(allTargets.shape)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=sourceDir, save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)

    history = classifier.fit(trainingData, allTargets, batch_size = batchsize, epochs = 1000, validation_split=0.1)

    # Model prediction
    y_pred = classifier.predict(X_test)
    y_prid_ = np.array([np.where(y_pred[i,:]==np.max(y_pred[i]))[0][0] for i in np.arange(len(y_pred))]).reshape(-1,1)

    # confusion_matrix  
    res = f1_score(y_test, y_prid_, average=None)

    if sum(res) > sum(maxRes):
        print(sum(res)) 
        max_history = history.history.copy()
        maxRes = res

pdb.set_trace()
tf.keras.utils.plot_model(classifier, to_file='AfterBalancing_model_%s.png' %str(maxRes), show_shapes=True, show_dtype=True)
print(max_history.keys())
# summarize history for accuracy
plt.plot(max_history['accuracy'])
plt.plot(max_history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('AfterBalancing_AccuracyTrace_%s.png' %str(maxRes))
plt.savefig('AfterBalancing_AccuracyTrace_%s.pdf' %str(maxRes))
# summarize history for loss
plt.plot(max_history['loss'])
plt.plot(max_history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('AfterBalancing_LossTrace_%s.png' %str(maxRes))
plt.savefig('AfterBalancing_LossTrace_%s.pdf' %str(maxRes))'''

#pdb.set_trace()
#cm = confusion_matrix(y_test, y_prid_)
#pdb.set_trace()

'''################################# Trying to reduce the overfit
1. I can try regularization technique.
2. 
'''