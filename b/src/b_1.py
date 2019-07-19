import numpy as np
import csv
import keras
import utm
import math

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
classes_no = 3600
wid = 60
hid = 60

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

def standardlize(feature):
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

#计算鸽子id
def calId(array):
    len1 = (330420 - 328770)/wid
    len2 = (3463487 - 3462224)/hid
    to1 = array[0] - 328770
    to2 = array[1] - 3462224
    id1 = int(to1/len1)
    id2 = int(to2/len2)
    if id1 >= wid:
        id1 = wid-1
    if id2 >= hid:
        id2 = hid-1
    return int(id1+id2*wid)

#计算鸽子中心
def calCenter(no):
    len1 = (330420 - 328770)/wid
    len2 = (3463487 - 3462224)/hid
    id1 = int(no%wid)
    id2 = int(no/hid)
    to1 = id1*len1+len1/2
    to2 = id2*len2+len2/2
    return [328770+to1,3462224+to2]


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
        all_y.append([float(row[34]),float(row[35])])
    all_feature = np.array(all_feature)
    all_y = np.array(all_y)

    #归一化
    scalerX = preprocessing.StandardScaler()

    total = len(all_feature)
    all_feature = all_feature.reshape(total*feature_row,feature_no)
    all_feature = scalerX.fit_transform(all_feature)
    all_feature = all_feature.reshape(total,feature_row,feature_no,1)
    
    #划分训练集和测试集
    x_train,x_test,y_train,y_test = train_test_split(all_feature,all_y,test_size=0.10,random_state=0)

    #转化成classes
    y_classes = []
    for point in y_train:
        u = utm.from_latlon(point[1],point[0])
        p_id = calId([u[0],u[1]])
        classes = np.zeros(classes_no,dtype=int)
        classes[p_id] = 1
        y_classes.append(classes)
    
    y_classes = np.array(y_classes)

    #训练模型
    model = Sequential()
    model.add(Conv2D(filters = 25,kernel_size=(2,2),activation="relu",input_shape=(feature_row,feature_no,1)))
    model.add(Conv2D(filters = 15,kernel_size=(2,2),activation="relu"))
    model.add(Flatten())
    model.add(Dense(output_dim=5000,activation="relu"))
    model.add(Dense(output_dim=3600,activation="softmax"))
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    model.fit(x_train,y_classes,batch_size=32,epochs=50)
    model.save('./b_1.CNN')





trainModel(ismis)
