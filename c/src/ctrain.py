import numpy as np
import csv
import keras
import random
import utm
import math
from scipy import stats
import matplotlib.pyplot as plt
import keras.optimizers as op
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras.optimizers import SGD
from keras.utils import plot_model
from keras import losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

#读训练集
directory = '../../predict/'
train_file = 'train_2g.csv'
test_file = "test_2g.csv"
sta_file = "gongcan.csv"
stations = []

with open(directory+sta_file) as f:
    reader = csv.reader(f)
    for row in reader:
        if(reader.line_num != 1):
            sta = [0,0,0,0]
            for i in range(4):
                sta[i] = float(row[i])
            stations.append(sta)

sta_num = len(stations) 
#find station
def findSta(RID,CID):
    for i in range(sta_num):
        if(stations[i][0] == RID):
            if(stations[i][1] == CID):
                return i
    return -1

def standardlize(feature):
    feature = np.array(feature)
    shape = feature.shape
    transfer = []
    for i in range(feature.shape[1]):
        top = np.max(feature[:,i])
        bot = np.min(feature[:,i])
        for j in range(feature.shape[0]):
            if(top != bot):
                feature[j][i] = (feature[j][i] - bot)/(top - bot)
            else:
                feature[j][i] = 1
    return feature

def normalize(array):
    min1 = 328770
    min2 = 3462224
    range1 = 330420 - 328770 
    range2 = 3463487 - 3462224
    after1 = (array[0]-min1)/range1
    after2 = (array[1]-min2)/range2
    return [after1,after2]

def denormalize(array):
    min1 = 328770
    min2 = 3462224
    range1 = 330420 - 328770 
    range2 = 3463487 - 3462224
    after1 = range1*array[0]+min1
    after2 = range2*array[1]+min2
    return [after1,after2]

def shuffle_train_test(x, y, size):
    random.seed(1)
    random_list = random.sample(range(size), k=int(0.1*size))
    X_Train = []
    Y_Train = []
    X_Test = []
    Y_Test = []
    for i in range(size):
        if i in random_list:
            X_Test.append(x[i])
            Y_Test.append(y[i])
        else:
            X_Train.append(x[i])
            Y_Train.append(y[i])
    return np.array(X_Train), np.array(Y_Train), np.array(X_Test), np.array(Y_Test)


ismis = []
ismi_group = []

with open(directory+train_file) as f:
    reader = csv.reader(f)
    for row in reader:
        if(reader.line_num > 1):
            ismis.append(row)

ismis = np.array(ismis,dtype="float")

def removeNan(ismi_set):
    for i in range(len(ismi_set)):
        for j in range(len(ismi_set[i])):
            if(ismi_set[i][j] == -999 or ismi_set[i][j] == -1):
                ismi_set[i][j] = ismi_set[i-1][j]
    return ismi_set

def sliptSet(array,steps):
    array = np.array(array)
    width = array.shape[1]
    result = []
    count = 0
    one_set = []
    for feature in array:
        one_set.append(feature)
        count += 1
        if count == steps:
            result.append(one_set)
            one_set = []
            count = 0
    if count != 0:
        makeup = []
        for j in range(steps - count):
            makeup.append(result[-1][count+j])
        for one in one_set:
            makeup.append(one)
        result.append(makeup)
    return result

def calMeters(point1,point2):
    c = math.sin(point1[1])*math.sin(point2[1])*math.cos(point1[0]-point2[0])+math.cos(point1[1])*math.cos(point2[1])
    distance = 6731*1000*math.acos(c)*3.14/180
    return distance


def calError(pred,true,steps):
    error = []
    for i in range(len(pred)):
        pred_point = pred[i]
        true_point = true[i]
        pred_u = utm.to_latlon(pred_point[0],pred_point[1],51,'R')
        true_u = utm.to_latlon(true_point[0],true_point[1],51,'R')
        pred_l = [pred_u[1],pred_u[0]]
        true_l = [true_u[1],true_u[0]]
        error.append(calMeters(pred_l,true_l))
    error = np.sort(error)
    print(error)
    mean_error = np.mean(error)
    print("mean_error",mean_error)
    print("median",np.median(error))
    print("90%error",error[int(len(error)*0.9)])
    val, cnt = np.unique(error, return_counts=True)
    pmf = cnt / (len(error))
    fs_rv_dist2 = stats.rv_discrete(name='fs_rv_dist2', values=(val, pmf))
    plt.plot(val, fs_rv_dist2.cdf(val))
    plt.title("CDF")
    plt.show()
# def sliptSet(array,steps):
#     array = np.array(array)
#     width = array.shape[1]
#     result = []
#     count = 0
#     one_set = []
#     for feature in array:
#         one_set.append(feature)
#         count += 1
#         if count == steps:
#             result.append(one_set)
#             one_set = []
#             count = 0
#     if count != 0:
#         for j in range(steps - count):
#             one_set.append(np.zeros(width))
#         result.append(one_set)
#     return result


def validateModel(ismi_set,steps):
    model = keras.models.load_model("./c 39.LSTM")
    #处理缺省
    ismi_set = removeNan(ismi_set)

    #获得原生特征矩阵
    all_feature = []
    all_y = []
    for row in ismi_set:
        sample = np.zeros((32),dtype=float)
        sample[0] = row[3]#timestamp
        sample[1] = row[0]#traceId
        for k in range(6):
            RID = float(row[4+k*5])
            CID = float(row[5+k*5])
            no = findSta(RID,CID)
            sta = stations[no]
            sample[5*k+2] = sta[2]
            sample[5*k+3] = sta[3]#站点经纬度
            for p in range(3):#信号值
                sample[4+p+5*k] = float(row[6+k*5+p])
        all_feature.append(sample)
        u = utm.from_latlon(float(row[35]),float(row[34]))
        all_y.append([u[0],u[1]])
    all_feature = np.array(all_feature)
    all_y = np.array(all_y)

    #归一化
    scalerX = preprocessing.StandardScaler()
    scalerY = preprocessing.StandardScaler()

    all_feature = scalerX.fit_transform(all_feature)
    all_y = scalerY.fit_transform(all_y)

    track_first = int(ismi_set[0][0])
    track_last = int(ismi_set[-1][0])

    #轨迹个数
    track_no = track_last - track_first + 1
    test_no = 1
    train_no = track_no - test_no

    #每个轨迹的点数量和其开始位置
    every_track_no = np.zeros(track_no,dtype=int)
    track_start = np.zeros(track_no,dtype=int)
    start = 0
    for one in ismi_set:
        every_track_no[int(one[0])-track_first] += 1
    for i in range(track_no):
        track_start[i] = start
        start += every_track_no[i]

    test_set = []
    test_y= []
    all_track_test = []

    #根据轨迹划分
    feature_set = []
    y_set = []
    for i in range(track_no):
        track_feature = []
        track_y = []
        #每个轨迹id分为一组
        for j in range(every_track_no[i]):
            index = j + track_start[i]
            row = all_feature[index]
            track_feature.append(row)
            track_y.append(all_y[index])
        #切割
        split_track_feature = sliptSet(track_feature,steps)
        split_track_test = sliptSet(track_y,steps)
        for j in range(len(split_track_feature)):
            feature_set.append(split_track_feature[j])
            y_set.append(split_track_test[j])  
    
    train_set = np.array(feature_set)
    y_set = np.array(y_set)
    y_set = y_set.reshape(y_set.shape[0],steps*2)

    #划分训练集和测试集
    x_train, y_train, x_test, y_test = shuffle_train_test(feature_set, y_set,round(len(all_feature)/steps))

    y_pred = model.predict(x_test)

    y_test = y_test.reshape(y_test.shape[0]*steps,2)
    y_pred = y_pred.reshape(y_pred.shape[0]*steps,2)

    y_test = scalerY.inverse_transform(y_test)
    y_pred = scalerY.inverse_transform(y_pred)

    calError(y_test,y_pred,steps)



validateModel(ismis,6)
