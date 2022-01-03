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
    
    # ====================== special case for raw data =========================
    if name == 'raw':
        filename_front = 'data_' + name
        # 1. good (normal) motor data
        # open text file
        filename_good = filename_front + '_good.txt'
        f = open(root + filename_good, 'r', encoding='utf-8')
        raw_data = np.array(list(csv.reader(f)))
        f.close
        # convert data into the numpy array
        raw_length = np.shape(raw_data)[0]   # raw_length = 220500
        data_num = len(raw_data[0][0].split(' '))   # 100 for good
        data_raw_good = np.zeros([data_num, raw_length])
        for j in range(raw_length):
            data_raw_good[:, j] = raw_data[j, 0].split(' ')
        del raw_data
        
        # 2. worst (defective) motor data
        # open text file
        filename_worst = filename_front + '_worst.txt'
        f = open(root + filename_worst, 'r', encoding='utf-8')
        raw_data = np.array(list(csv.reader(f)))
        f.close
        # convert data into the numpy array
        raw_length = np.shape(raw_data)[0]   # raw_length = 220500
        data_num = len(raw_data[0][0].split(' '))   # 50 for worst
        data_raw_worst = np.zeros([data_num, raw_length])
        for j in range(raw_length):
            data_raw_worst[:, j] = raw_data[j, 0].split(' ')
        del raw_data
        
        # concatenate good and worst data
        raw_dataset = np.append(data_raw_good, data_raw_worst, 0)
        del data_raw_good, data_raw_worst
        
        return raw_dataset
        
    # ===========================================================================
    elif name == 'MFCC':
        filename_front = 'data_' + name + '_only_' + division
        
    elif name == 'LPC':
        pass
        
    elif name == 'CWT':
        pass
#         filename = 'data_CWT_whole_good'
        
    elif name == 'DWT':
        pass
#         filename = 'data_DWT_whole_good'
    
    # 1. good (normal) motor data
    # open text file
    filename_good = filename_front + '_good.txt'
    f = open(root + filename_good, 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    # convert data into the numpy array
    data_num = np.shape(raw_data)[0]
    data_good = np.zeros([data_num, dataDim])
    for j in range(data_num):
        data_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    
    # 2. worst (defective) motor data
    # open text file
    filename_worst = filename_front + '_worst.txt'
    f = open(root + filename_worst, 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    # convert data into the numpy array
    data_num = np.shape(raw_data)[0]
    data_worst = np.zeros([data_num, dataDim])
    for j in range(data_num):
        data_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    
    # concatenate good and worst data
    dataset = np.append(data_good, data_worst, 0)
    del data_good, data_worst
#     print("dataset.shape: {}".format(np.shape(dataset)))
    
    return dataset

def get_label():
    # 0 for normal, 1 for defect
    return np.append(np.zeros(100),np.ones(50))