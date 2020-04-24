from sklearn.preprocessing import StandardScaler
import numpy as np

def data_processing(train_data,train_label,test_data,test_label,peak_rate):
    Y = np.array(train_data.T,dtype = 'float32')
    filter_peak = np.sum(Y >= 1, axis=1) >= round(peak_rate*Y.shape[1])
    Y = Y[filter_peak,:]
    train_data = Y.T
    test_data = test_data[:,filter_peak]
    train_data = train_data[:,np.sum(test_data,axis=0)>0]
    test_data = test_data[:,np.sum(test_data,axis=0)>0]
    k = 0
    train = np.sum(train_data,axis = 1)
    for i in range(train.shape[0]):
        if train[i]==0:
            print(i)
            train_data = np.delete(train_data,i+k,axis = 0)
            train_label = np.delete(train_label,i+k,axis = 0)
            k = k-1
    Y = train_data.T
    nfreqs = 1.0 * Y / np.tile(np.sum(Y,axis=0), (Y.shape[0],1))
    Y_mat = nfreqs * np.tile(np.log(1 + 1.0 * Y.shape[1] / np.sum(Y,axis=1)).reshape(-1,1), (1,Y.shape[1]))
    train_data = Y_mat.T
    test = np.sum(test_data,axis = 1)
    k = 0
    for i in range(test.shape[0]):
        if test[i]==0:
            print(i)
            test_data = np.delete(test_data,i+k,axis = 0)
            test_label = np.delete(test_label,i+k,axis = 0)
            k = k-1
    Y = test_data.T
    nfreqs = 1.0 * Y / np.tile(np.sum(Y,axis=0), (Y.shape[0],1))
    Y_mat = nfreqs * np.tile(np.log(1 + 1.0 * Y.shape[1] / np.sum(Y,axis=1)).reshape(-1,1), (1,Y.shape[1]))
    test_data = Y_mat.T
    ss = StandardScaler()
    train_data = ss.fit_transform(train_data)
    test_data = ss.transform(test_data)
    return train_data,train_label,test_data,test_label
def data_selection(train_data,train_label,test_data,test_label,peak_rate):
    Y = np.array(train_data.T,dtype = 'float32')
    filter_peak = np.sum(Y >= 1, axis=1) >= round(peak_rate*Y.shape[1])
    Y = Y[filter_peak,:]
    train_data = Y.T
    test_data = test_data[:,filter_peak]
    train_data = train_data[:,np.sum(test_data,axis=0)>0]
    test_data = test_data[:,np.sum(test_data,axis=0)>0]
    k = 0
    train = np.sum(train_data,axis = 1)
    for i in range(train.shape[0]):
        if train[i]==0:
            print(i)
            train_data = np.delete(train_data,i+k,axis = 0)
            train_label = np.delete(train_label,i+k,axis = 0)
            k = k-1
    test = np.sum(test_data,axis = 1)
    k = 0
    for i in range(test.shape[0]):
        if test[i]==0:
            print(i)
            test_data = np.delete(test_data,i+k,axis = 0)
            test_label = np.delete(test_label,i+k,axis = 0)
            k = k-1
    return train_data,train_label,test_data,test_label
    