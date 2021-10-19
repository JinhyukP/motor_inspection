import numpy as np
import csv

data_dim_MFCC = 13
trSampleRatio = 0.85
data_dim_CWT = 13
data_dim_DWT = 13
# root = 'C:\\Users\\JINHYUK2019\\Desktop\\motorInspection_my_codes\\MFCC_data\\'  
# root = '/home/robotics/Jinhyuk/motor_inspection/MFCC_data/'
root = '/home/robotics/Public_HDD/Jinhyuk/MFCC_data/'

def load_raw_whole():
    pass



def load_MFCC_whole():
    f = open(root + 'data_MFCC_only_whole_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_MFCC_only_good = np.zeros([np.shape(raw_data)[0], data_dim_MFCC])
    for j in range(np.shape(raw_data)[0]):
        data_MFCC_only_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_MFCC_only_whole_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_MFCC_only_worst = np.zeros([np.shape(raw_data)[0], data_dim_MFCC])
    for j in range(np.shape(raw_data)[0]):
        data_MFCC_only_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_MFCC_only_whole = np.append(data_MFCC_only_good, data_MFCC_only_worst, 0)    
    print("data_MFCC_only_whole.shape: {}".format(np.shape(data_MFCC_only_whole)))
    return data_MFCC_only_whole

def load_MFCC_steady():
    f = open(root + 'data_MFCC_only_uniform_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_MFCC_only_uniform_good = np.zeros([np.shape(raw_data)[0], data_dim_MFCC])
    for j in range(np.shape(raw_data)[0]):
        data_MFCC_only_uniform_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_MFCC_only_uniform_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_MFCC_only_uniform_worst = np.zeros([np.shape(raw_data)[0], data_dim_MFCC])
    for j in range(np.shape(raw_data)[0]):
        data_MFCC_only_uniform_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_MFCC_only_steady = np.append(data_MFCC_only_uniform_good, data_MFCC_only_uniform_worst, 0)
    print("data_MFCC_only_steady.shape: {}".format(np.shape(data_MFCC_only_steady)))
    return data_MFCC_only_steady        

def load_MFCC_transient():
    f = open(root + 'data_MFCC_only_transient_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_MFCC_only_transient_good = np.zeros([np.shape(raw_data)[0], data_dim_MFCC])
    for j in range(np.shape(raw_data)[0]):
        data_MFCC_only_transient_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_MFCC_only_uniform_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_MFCC_only_transient_worst = np.zeros([np.shape(raw_data)[0], data_dim_MFCC])
    for j in range(np.shape(raw_data)[0]):
        data_MFCC_only_transient_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_MFCC_only_transient = np.append(data_MFCC_only_transient_good, data_MFCC_only_transient_worst, 0)
    print("data_MFCC_only_transient.shape: {}".format(np.shape(data_MFCC_only_transient)))
    return data_MFCC_only_transient      

def load_CWT_whole():
    f = open(root + 'data_CWT_whole_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_CWT_whole_good = np.zeros([np.shape(raw_data)[0], data_dim_CWT])
    for j in range(np.shape(raw_data)[0]):
        data_CWT_whole_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_CWT_whole_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_CWT_whole_worst = np.zeros([np.shape(raw_data)[0], data_dim_CWT])
    for j in range(np.shape(raw_data)[0]):
        data_CWT_whole_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_CWT_whole = np.append(data_CWT_whole_good, data_CWT_whole_worst, 0)
    print("data_CWT_whole.shape: {}".format(np.shape(data_CWT_whole)))
    return data_CWT_whole

def load_CWT_steady():
    f = open(root + 'data_CWT_steady_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_CWT_steady_good = np.zeros([np.shape(raw_data)[0], data_dim_CWT])
    for j in range(np.shape(raw_data)[0]):
        data_CWT_steady_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_CWT_steady_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_CWT_steady_worst = np.zeros([np.shape(raw_data)[0], data_dim_CWT])
    for j in range(np.shape(raw_data)[0]):
        data_CWT_steady_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_CWT_steady = np.append(data_CWT_steady_good, data_CWT_steady_worst, 0)
    print("data_CWT_steady.shape: {}".format(np.shape(data_CWT_steady)))
    return data_CWT_steady
    
def load_CWT_transient():
    f = open(root + 'data_CWT_transient_good' + '.txt', 'r', encoding='utf-8')    
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_CWT_transient_good = np.zeros([np.shape(raw_data)[0], data_dim_CWT])
    for j in range(np.shape(raw_data)[0]):
        data_CWT_transient_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_CWT_transient_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_CWT_transient_worst = np.zeros([np.shape(raw_data)[0], data_dim_CWT])
    for j in range(np.shape(raw_data)[0]):
        data_CWT_transient_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_CWT_transient = np.append(data_CWT_transient_good, data_CWT_transient_worst, 0)
    print("data_CWT_transient.shape: {}".format(np.shape(data_CWT_transient)))
    return data_CWT_transient    
    
def load_DWT_whole():
    f = open(root + 'data_DWT_whole_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_DWT_whole_good = np.zeros([np.shape(raw_data)[0], data_dim_DWT])
    for j in range(np.shape(raw_data)[0]):
        data_DWT_whole_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_DWT_whole_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_DWT_whole_worst = np.zeros([np.shape(raw_data)[0], data_dim_DWT])
    for j in range(np.shape(raw_data)[0]):
        data_DWT_whole_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_DWT_whole = np.append(data_DWT_whole_good, data_DWT_whole_worst, 0)
    print("data_DWT_whole.shape: {}".format(np.shape(data_DWT_whole)))
    return data_DWT_whole

def load_DWT_steady():
    f = open(root + 'data_DWT_steady_good' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_DWT_steady_good = np.zeros([np.shape(raw_data)[0], data_dim_DWT])
    for j in range(np.shape(raw_data)[0]):
        data_DWT_steady_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_DWT_steady_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_DWT_steady_worst = np.zeros([np.shape(raw_data)[0], data_dim_DWT])
    for j in range(np.shape(raw_data)[0]):
        data_DWT_steady_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_DWT_steady = np.append(data_DWT_steady_good, data_DWT_steady_worst, 0)
    print("data_DWT_steady.shape: {}".format(np.shape(data_DWT_steady)))
    return data_DWT_steady    

def load_DWT_transient():
    f = open(root + 'data_DWT_transient_good' + '.txt', 'r', encoding='utf-8')    
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_DWT_transient_good = np.zeros([np.shape(raw_data)[0], data_dim_DWT])
    for j in range(np.shape(raw_data)[0]):
        data_DWT_transient_good[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    f = open(root + 'data_DWT_transient_worst' + '.txt', 'r', encoding='utf-8')
    raw_data = np.array(list(csv.reader(f)))
    f.close
    data_DWT_transient_worst = np.zeros([np.shape(raw_data)[0], data_dim_DWT])
    for j in range(np.shape(raw_data)[0]):
        data_DWT_transient_worst[j, :] = raw_data[j, 0].split(' ')
    del raw_data
    data_DWT_transient = np.append(data_DWT_transient_good, data_DWT_transient_worst, 0)
    print("data_DWT_transient.shape: {}".format(np.shape(data_DWT_transient)))
    return data_DWT_transient   


def load_label():
    y_whole = np.append(np.zeros(100),np.ones(50))    # 0 for normal, 1 for defect
    return y_whole

def resampleData(data_whole, y_whole):
    num_data = np.shape(data_whole)[0]
    num_good = 100
    num_worst = 50
    # random sampling from [0, 100]
    trIdx_good = np.sort(np.random.permutation(num_good)[:int(num_good * trSampleRatio)])
    # random sampling from [100,150]
    trIdx_worst = np.sort(np.random.permutation(num_worst)[:int(num_worst * trSampleRatio)]) + num_good
    
    # exclude trIdx_good from [0,100]
    tstIdx_good = np.array(range(num_good))
    tstIdx_good = np.delete(tstIdx_good, trIdx_good)
    # exclude trIdx_worst from [100,150]
    tstIdx_worst = np.array(range(num_worst)) + num_good
    tstIdx_worst = np.delete(tstIdx_worst, trIdx_worst - num_good)

    trIdx = np.append(trIdx_good, trIdx_worst)
    tstIdx = np.append(tstIdx_good, tstIdx_worst)
    
#     trIdx = np.sort(np.random.permutation(num_data)[:int(num_data * trSampleRatio)])
#     tstIdx = np.array(range(num_data))
#     tstIdx = np.delete(tstIdx, trIdx)

    train_data = data_whole[trIdx]
    train_label = y_whole[trIdx]
    test_data = data_whole[tstIdx]
    test_label = y_whole[tstIdx]
    
    return train_data, train_label, test_data, test_label

def fitClassifier_y_pred(model, train_data, train_label, test_data, test_label, epoch_size=200, batch_size=150):
    model.fit(np.transpose(np.reshape(train_data, [-1, model.input.shape[2], model.input.shape[1]]), [0, 2, 1]), 
              train_label, 
              batch_size=batch_size, 
              epochs=epoch_size,
              verbose=0
             )    
    
    y_pred = model.predict(np.transpose(np.reshape(test_data, [-1, model.input.shape[2], model.input.shape[1]]), [0, 2, 1]))

#     tmp = y_pred > 0.5
#     tmp.astype(int)
#     tmp = np.reshape(tmp, -1)

#     compare = tmp == test_label

#     tp = np.sum(compare[test_label==0])
#     tn = np.sum(compare[test_label==1])
#     fp = np.sum(test_label==0) - tp
#     fn = np.sum(test_label==1) - tn
#     accuracy = np.mean(compare)
#     return tp, tn, fp, fn
    return y_pred, test_label

def fitClassifier_y_pred_MLP(model, train_data, train_label, test_data, test_label, epoch_size=200, batch_size=150):
    model.fit(np.reshape(train_data, [-1, model.input.shape[1]]),
              train_label, 
              batch_size=batch_size, 
              epochs=epoch_size,
              verbose=0
             )    
    
    y_pred = model.predict(np.reshape(test_data, [-1, model.input.shape[1]]))
    return y_pred, test_label

