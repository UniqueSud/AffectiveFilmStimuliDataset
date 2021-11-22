import os
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import pdb

def tableCreation():
    data = pd.read_excel(os.path.join('NewTarget', 'WithCCC_kappa_WithClustering_summary_data_frame_after_cleaning2018_Oct_10-Nov_15_withClusterInformation.xlsx'), sheet_name='69Stim', index_col=0)
    data = data.replace(['Adventorous'],'Adventurous')    

    for id_, emtName in zip(data.index.values, data['Emt_Name']):
        print(id_)
        try:
            emtNArr = emtName.split(',')
        except:
            break

        emtons = np.unique(emtNArr)
        emtCountDf = pd.DataFrame(0, index=emtons, columns=['Count'])

        for emt_ in emtNArr:
            emtCountDf.loc[emt_, 'Count'] += 1

        data.loc[id_, 'firstEmt'] = emtCountDf.index.values[np.argmax(emtCountDf)]
        data.loc[id_, 'firstPro'] = np.round(emtCountDf.loc[data.loc[id_, 'firstEmt'], 'Count']/sum(emtCountDf.values)[0], 2)
        data.loc[id_, 'secondEmt'] = emtCountDf.index.values[np.argsort(emtCountDf.values.reshape(-1))[::-1][1]]

        if data.loc[id_, 'firstEmt'] == data.loc[id_, 'secondEmt']:
            data.loc[id_, 'secondEmt'] = emtCountDf.index.values[np.argsort(emtCountDf.values.reshape(-1))[::-1][0]]
            
        data.loc[id_, 'secondPro'] = np.round(emtCountDf.loc[data.loc[id_, 'secondEmt'], 'Count']/sum(emtCountDf.values)[0], 2)
        data.loc[id_, 'Emt:I(II)'] = '%s (%s)' %(data.loc[id_, 'firstEmt'], data.loc[id_, 'secondEmt'])
        data.loc[id_, 'Prob:I(II)'] = '%s(%s)' %(str(data.loc[id_, 'firstPro']), str(data.loc[id_, 'secondPro']))

    data.to_csv(os.path.join('NewTarget', '69Stim_after_cleaning2018_Oct_10-Nov_15_withEmtProb_Prog-creatingEmotionSet.csv'))

def vadPlot():
    data = pd.read_excel(os.path.join('NewTarget', 'WithCCC_kappa_WithClustering_summary_data_frame_after_cleaning2018_Oct_10-Nov_15_withClusterInformation.xlsx'), sheet_name='69Stim_VA', index_col=0)
    data = data.replace(['Adventorous'],'Adventurous')    
    valMean = data['V_mean']
    arlMean = data['A_mean']
    prob_ = data['Prob']
    mostR = data['MostR']

    plt.scatter(valMean, arlMean)

    texts = []
    highProbDict = {}
    for val_, arl_, pr_, mR_ in zip(valMean, arlMean, prob_, mostR):        
        
        if mR_ not in highProbDict.keys():
            highProbDict[mR_] = [val_, arl_, pr_]
        else:
            if highProbDict[mR_][2] < pr_:
                highProbDict[mR_] = [val_, arl_, pr_]

    for key_ in highProbDict.keys():
        texts.append(plt.text(highProbDict[key_][0], highProbDict[key_][1], key_, fontsize=15))
  
    plt.ylabel('Arousal Mean', fontsize=20)
    plt.xlabel('Valence Mean', fontsize=20)
    plt.tick_params(axis='both', which='both', labelsize=20)
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='r', lw=1.5))#, only_move={'points':'y', 'texts':'y'}, arrowprops=dict(arrowstyle="->", color='r', lw=1.5))
    plt.savefig('VAPlotWithEmotionAnnotation.pdf', bbox_inches='tight')
    pdb.set_trace()