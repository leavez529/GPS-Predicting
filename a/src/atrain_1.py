import numpy as np
import csv
import keras
import utm
import math
import matplotlib.pyplot as plt
from scipy import stats
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
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

#参数
feature_no = 5
feature_row = 6

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

#随机划分训练街和测试集
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

#读取原始数据
ismis = []
with open(directory+train_file) as f:
    reader = csv.reader(f)
    for row in reader:
        if(reader.line_num > 1):
            ismis.append(row)

ismis = np.array(ismis,dtype=float)

#缺省值处理
def removeNan(ismi_set):
    for i in range(len(ismi_set)):
        for j in range(len(ismi_set[i])):
            if(ismi_set[i][j] == -999 or ismi_set[i][j] == -1):
                ismi_set[i][j] = ismi_set[i-1][j]
    return ismi_set

def calMeters(point1,point2):
    c = math.sin(point1[1])*math.sin(point2[1])*math.cos(point1[0]-point2[0])+math.cos(point1[1])*math.cos(point2[1])
    distance = 6731*1000*math.acos(c)*3.14/180
    return distance

def calError(y_pred_utm,y_test_utm):
    y_pred_true = []
    y_test_true = []
    for i in range(len(y_pred_utm)):
        u1 = utm.to_latlon(y_pred_utm[i][0],y_pred_utm[i][1],51,'R')
        u2 = utm.to_latlon(y_test_utm[i][0],y_test_utm[i][1],51,'R')
        y_pred_true.append([u1[1],u1[0]])
        y_test_true.append([u2[1],u2[0]])

    error = []
    for i in range(len(y_pred_true)):
        error.append(calMeters(y_pred_true[i],y_test_true[i]))

    error = np.array(error)
    error = np.sort(error)

    print(error)
    mean_error = np.mean(error)

    print("mean_error",mean_error)
    print("median",np.median(error))
    print("90%error",error[int(len(error)*0.9)])
    val, cnt = np.unique(error, return_counts=True)
    #print(val,cnt)
    pmf = cnt / (len(error))

    fs_rv_dist2 = stats.rv_discrete(name='fs_rv_dist2', values=(val, pmf))
    plt.plot(val, fs_rv_dist2.cdf(val))
    plt.title("CDF")
    plt.show()

train_set = []
train_y = []

def trainModel(ismi_set):
    #处理缺省
    ismi_set = removeNan(ismi_set)

    #获得原生特征矩阵
    all_feature = []
    all_y = []
    for row in ismi_set:
        feature = []
        for i in range(feature_row):
            sample = np.zeros((feature_no),dtype=float)
            RID = float(row[4+i*5])
            CID = float(row[5+i*5])
            no = findSta(RID,CID)
            sta = stations[no]
            sample[0] = sta[2]
            sample[1] = sta[3]
            for j in range(3):
                sample[2+j] = float(row[6+i*5+j])
            feature.append(sample)
        feature = np.array(feature)
        all_feature.append(feature)
        u = utm.from_latlon(float(row[35]),float(row[34]))
        all_y.append([u[0],u[1]])
    all_feature = np.array(all_feature)
    all_y = np.array(all_y)

    #归一化
    scalerX = preprocessing.StandardScaler()
    scalerY = preprocessing.StandardScaler()

    total = len(all_feature)
    all_feature = all_feature.reshape(total*feature_row,feature_no)
    all_feature = scalerX.fit_transform(all_feature)
    all_y = scalerY.fit_transform(all_y)

    all_feature = all_feature.reshape(total,feature_row,feature_no,1)
    #划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(all_feature,all_y,test_size=0.10,random_state=0)

    model = keras.models.load_model("./a 13.CNN")
    y_pred = model.predict(x_test)
    y_pred = scalerY.inverse_transform(y_pred)
    y_test = scalerY.inverse_transform(y_test)

    calError(y_pred,y_test)

trainModel(ismis)
