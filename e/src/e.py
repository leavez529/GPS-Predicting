import numpy as np
import csv
import keras
import random
import utm
import math
import keras.optimizers as op
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, LSTM
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
classes_no = 3600
wid = 60
hid = 60
rate = 64
feature_row = 6
feature_col = 5
feature_no = 32
steps = 6

#读取基站信息
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

#读取原始数据
ismis = []
with open(directory+train_file) as f:
    reader = csv.reader(f)
    for row in reader:
        if(reader.line_num > 1):
            ismis.append(row)

ismis = np.array(ismis,dtype="float")

#随机划分训练集和验证集
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

#处理缺省值
def removeNan(ismi_set):
    for i in range(len(ismi_set)):
        for j in range(len(ismi_set[i])):
            if(ismi_set[i][j] == -999 or ismi_set[i][j] == -1):
                ismi_set[i][j] = ismi_set[i-1][j]
    return ismi_set

#轨迹划分
def sliptSet(array,steps):
    array = np.array(array)
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

def trainModel(ismi_set):
    ismi_set = removeNan(ismi_set)

    #获得原生特征矩阵
    all_feature = []
    all_y = []
    for row in ismi_set:
        feature = []
        for i in range(feature_row):
            sample = np.zeros((feature_col),dtype=float)
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
    all_feature = all_feature.reshape(total*feature_row,feature_col)
    all_feature = scalerX.fit_transform(all_feature)
    all_feature = all_feature.reshape(total,feature_row,feature_col,1)

    #CNN多分类模型
    modelCNN = keras.models.load_model("./b_1.CNN")
    all_prob = modelCNN.predict(all_feature)

    #轨迹个数
    track_first = int(ismi_set[0][0])
    track_last = int(ismi_set[-1][0])
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

    #根据轨迹划分
    feature_set = []
    y_set = []
    for i in range(track_no):
        track_feature = []
        track_y = []
        #每个轨迹id分为一组
        for j in range(every_track_no[i]):
            index = j + track_start[i]
            row = all_prob[index]
            track_feature.append(row)
            track_y.append(all_y[index])
        #切割
        split_track_feature = sliptSet(track_feature,steps)
        split_track_y = sliptSet(track_y,steps)
        for j in range(len(split_track_feature)):
            feature_set.append(split_track_feature[j])
            y_set.append(split_track_y[j])

    feature_set = np.array(feature_set)
    y_set = np.array(y_set)
    
    #划分训练集和测试集
    x_train, y_train, x_test, y_test = shuffle_train_test(feature_set, y_set,round(len(all_feature)/steps))


    #转换成classes
    y_classes = []
    for track in y_train:
        track_classes = []
        for point in track:
            classes = np.zeros(classes_no,dtype=int)
            u = utm.from_latlon(point[1],point[0])
            p_id = calId([u[0],u[1]])
            classes[p_id] = 1
            track_classes.append(classes)
        y_classes.append(track_classes)

    y_classes = np.array(y_classes)
    print(y_classes.shape)

    #训练模型
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

    model = Sequential()
    model.add(LSTM(64*steps, input_shape = (steps,x_train.shape[2])))

    print(model.get_layer(index=0).output_shape)
    model.add(Reshape((steps,64)))
    print(model.get_layer(index=1).output_shape)

    model.add(Dense(output_dim=classes_no, activation="softmax"))
    model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x_train,y_classes,epochs=150, batch_size=64)
    model.save('./e.LSTM')
    
trainModel(ismis)
