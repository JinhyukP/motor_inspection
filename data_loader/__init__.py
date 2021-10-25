import numpy as np
import csv

# path to data directory
root = '/home/robotics/Public_HDD/Jinhyuk/MFCC_data/'

def get_dataset(data_cfg):
    """
    get dataset for motor inspection
    data_cfg: data configuration dictionary
    """
    name = data_cfg["dataType"]   # raw, MFCC, LPC, PLP, CWT, DWT
    division = data_cfg["division"]   # whole, steady, transient
    dataDim = data_cfg["dataDimension"]
    
    if name == 'raw':
        dataset = []
    elif name == 'MFCC':
#         filename = 'data_MFCC_only_whole_good'
        f = open(root + 'data_MFCC_only_whole_good' + '.txt', 'r', encoding='utf-8')
        raw_data = np.array(list(csv.reader(f)))
        f.close
        data_MFCC_only_good = np.zeros([np.shape(raw_data)[0], dataDim])
        for j in range(np.shape(raw_data)[0]):
            data_MFCC_only_good[j, :] = raw_data[j, 0].split(' ')
        del raw_data
        f = open(root + 'data_MFCC_only_whole_worst' + '.txt', 'r', encoding='utf-8')
        raw_data = np.array(list(csv.reader(f)))
        f.close
        data_MFCC_only_worst = np.zeros([np.shape(raw_data)[0], dataDim])
        for j in range(np.shape(raw_data)[0]):
            data_MFCC_only_worst[j, :] = raw_data[j, 0].split(' ')
        del raw_data
        data_MFCC_only_whole = np.append(data_MFCC_only_good, data_MFCC_only_worst, 0)    
        print("data_MFCC_only_whole.shape: {}".format(np.shape(data_MFCC_only_whole)))        
        dataset = data_MFCC_only_whole
    elif name == 'LPC':
        dataset = []
    elif name == 'CWT':
        dataset = []
    elif name == 'DWT':
        dataset = []
    
#     f = open(root + filename + )
    return dataset


def get_label():
    y_whole = np.append(np.zeros(100),np.ones(50))    # 0 for normal, 1 for defect
    return y_whole