# Report

## 代码说明

主目录下a,b,c,d,e,f分别表示六个实验，其中pred.csv为该实验下test_2g测试集的预测数据。src文件夹里有着相应的源码，其中名字中没有train的源码为模型构造和训练，名字中有train的是对自行划分的验证集进行预测的结果。data内为训练集和测试集数据。

## 实验目的

该实验通过手机关于六个基站的信息数据来预测当时手机所在的经纬度位置。其中需要划分训练集和测试集，运用CNN, LSTM的模型进行训练，对训练集进行预测检查其误差(米)。最后用训练得到的模型预测测试数据，得到测试数据位置，并将其投影至地图上。



## 实验方法

在实验中，为了得到理想的机器学习模型，主要需要考虑到以下几点关键思想：

### 数据预处理

数据预处理是实验结果效果好坏的关键，合适的数据预处理方法能够大幅度提升训练效果，从实验实际过程来看，数据预处理的重要性是最高的。

在数据预处理环节中主要需要考虑以下几个问题：

#### 特征选择

特征是训练的主体，因此选择有用的特征才能使模型学习到有用的信息。在本实验中，主体特征主要就是六个基站的信号数据，包括这六个基站本身的经纬度位置以及三个信号数值（每个基站对应五个特征）。而在训练LSTM时，由于时间序列的信息，因此还多考虑了手机当时的时间戳和速度，多了两个特征。

因此，训练CNN时的特征即为6X5的矩阵，而训练LSTM时样本的特征为维度为32的向量。

#### 归一化处理

最后一个一定要对数据进行处理的工作就是归一化处理。在本实验的数据集中归一化是必须要做的工作，因为不同特征其单位不同，显然是要做归一化的。本实验中主要使用了sklearn包中preprocessing的StandardScaler()进行归一化，是一种利用数据的均值和方差归一化方法。在使用此方法之前我还是尝试使用最值归一化的方法，但是其效果不如StandardScaler。

实际的归一化过程是要将单个样本的特征排成行向量，不同的样本在不同行，一列下是对应的同一种特征。将所有的数据都这样排列然后对于单个列（特征）作归一化。

归一化的效果非常明显，在不使用归一化的情况下模型几乎很难学习到有用信息，而使用了归一化后模型的预测效果骤增。

#### 经纬度转换

训练时我们需要利用到基站的经纬度作为特征和手机所在的经纬度作为标签数据。而在实际训练中，用经纬度显然是不合适的，最大的问题在于两个经纬度点之间的距离不是通过简单的欧几里得距离公式进行计算，需要考虑到地球的弧度。因此在实验中应该将经纬度转为平面投影坐标系，即UTM坐标系。事实上使用UTM坐标训练和效果会大幅度提升。

同样，再换为UTM后，还要对其进行归一化，不再赘述。

#### 缺省值处理

实验数据中存在-999和-1的缺省值，对于缺省值是一定要处理的。处理的方法主要考虑了以下几种方法：

**剔除：**将存在缺省值的样本剔除出。但是在本实验中这种方法显然是不可行的，首先第一个问题是剔除后数据集数量会骤减，构成数据集数量不够的问题。二是缺省数据是广泛存在的，模型应该有处理这种数据的能力。并且测试数据中也有缺省值，总不能将其也剔除。

**归零**：归零是另外一种方法，神经元接受零数据可以视为不使用。但是考虑到数据集中缺省值过多，会出现过多的0，因此也没采用这个方法。

**补值**：本实验中主要使用了补值的方法，主要考虑到手机的信号数据是连续获得的，因此其此时的数据很可能与之前的数据非常接近。因此将上一个对应位置的非缺省数据补到缺省位置可能是一个有效的方法。



### 数据集划分

为了更好地进行训练和验证模型效果，需要将数据集划分为训练集和测试集(验证集)。为了方便起见，没有进行交叉验证。但是为了保证训练质量，基本的思想是需要保证划分的公平，即接近的数据不能集中取，要尽量分散一些。

由于CNN和LSTM的样本特征不同，两种神经网络划分数据集的方法有略微不同，后面会具体提到。

### 神经网络结构和参数调整

神经网络结构和参数也是影响模型效果的重要因素，实验中需要不断调整网络结构和参数以得到理想的结果。但是实际上由于对CNN和LSTM的深层了解较少，调参多半是凭经验和感觉。因此调参主要的方法就是简单的控制变量，并且由于机器的限制，跑一些复杂的网络需要的时间较久，也给调参带来了很大不便。

### 结果检查

为了对预测结果进行检查，需要用到一些指标，包括中位误差，平均误差，90%误差。其中中位误差是较为重要的参考指标。同时为了能够更直观地判断结果好坏，需要把结果投影到地图上。由于实际上这些数据都是从校园内获得，并且一般来说手机(人)的位置都在道路上，因此如果大多点都在校园内的路上，则说明预测结果不错。

## 实验内容

接下来分别对五个实验项来进行简述，主要着重于每个题目中相较其他实验特殊的地方：

### A实验

#### 过程

A实验是一个CNN回归问题。首先对于五个实验，其数据预处理中只有特征选择有不同，对缺省值的处理、经纬度转换和归一化处理都是一样的。前面提到过，CNN使用的特征式6X5，这对于A实验和B实验也是一样的。

对于数据集划分，A实验和B实验都使用sklearn自带的train_test_split方法，该函数在划分训练集和测试集有着很好的效果。

搭建网络的过程中，首先由于是一个回归的问题，因此lose函数要使用mse(mean squard error)。

对于CNN，卷积层和全连接层的选择非常重要。由于考虑到特征数量较少，并没有使用池化层。卷积层在选择中，首先是考虑层数。还是由于本实验中特征比较简单，不宜使用过多层卷积层，实际上三层以上情况就会变得很差，所以最终我选择了两层。对于卷积核的大小和数量，同样也是由于特征较少不易过大。最终经过调参实验选择了2x2的卷积核，并且数量每层都在20左右。

之后我们需要通过全连接层将输出维度降到2，显然这个过程应该是逐渐降低的。经过估算卷积层展开后输出有50000多维，因此我们逐渐将其减小。最终实验中一共使用了六层全连接层。

在实验中有一个关于激活函数的特别的发现。在除了最后一层的其它层之后我都加了'relu'激活层，而在最后一个输出层(输出2维)，使用了'linear'激活函数，即线性处理（只有bias），使用后发现训练和预测效果大幅度提升，这有一定可能与回归问题的本身有关。

最后是关于优化器，在本实验中优化器基本上都使用最常用的Adam。在使用Adam的时候可以对其学习率进行一些调整，有可能会关系到拟合的过程。

最终网络结构：

```python
#训练模型
model = Sequential()
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

model.add(Conv2D(filters = 25,kernel_size=(2,2),activation="relu",input_shape=(feature_row,feature_no,1)))
model.add(Conv2D(filters = 10,kernel_size=(2,2),activation="relu"))

model.add(Flatten())
model.add(Dense(output_dim=1024,activation="relu"))

model.add(Dense(output_dim=512,activation="relu"))

model.add(Dense(output_dim=256,activation="relu"))
model.add(Dense(output_dim=128,activation="relu"))
model.add(Dense(output_dim=64,activation="relu"))
model.add(Dense(output_dim=32,activation="relu"))
model.add(Dense(output_dim=2,activation="linear"))

model.compile(optimizer=adam,loss='mse')
model.fit(x_train,y_train,batch_size=32,epochs=50)
model.save('./a_1.CNN')
```

#### 结果

误差：

| 中位误差 | 平均误差 | 90%误差 |
| -------- | -------- | ------- |
| 13.84    | 23.32    | 46.90   |

CDF图：

![image-20190621110719857](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621110719857.png)

地图投影：

![image-20190621040646951](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621040646951.png)

### B实验

#### 过程

B实验是一个CNN多分类实验。重点在于划分格子数。划分的格子数如果太少，即每个格子太大，虽然容易预测对类别，但是误差会较大。格子数太多，一是预测会有难度，二是训练时间会过长。因此我们要取一个合适的中间值。在实验中我使用过几个取值，发现只要大概在某个区间范围内其结果都不错。基本上划分的格子长宽在30M左右均可。最终我选择了60x60的格子分法。

B实验的其他部分与A实验类似，稍微有不同的地方在于网络结构。首先卷积层的选择与A实验一样，由于输出是3600(60x60)，因此全连接层不必要太多。在本实验中优化器我选择了'rmsprop'(实际选择Adam差别不大)，最后要记得分类是使用'categorical-crossentropy'的损失函数。

最终网络结构如下：

```python
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
```

#### 结果

误差：

| 中位误差 | 平均误差 | 90%误差 |
| -------- | -------- | ------- |
| 8.84     | 26.85    | 43.07   |

CDF:

![image-20190621110835964](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621110835964.png)

地图投影：

![image-20190621040859073](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621040859073.png)

### C实验

#### 过程

C实验是一个LSTM回归问题。前面提到了LSTM的特征处理与CNN不一样，主要是其要以一个序列一起放入网络中。这里一个很重要的参数就是序列的长度。在选择序列的长度中也是过长或过短都是不好的，根据不断地调参后我最终选择了6作为长度，以其为长度的预测效果还算不错。

接着一个对于LSTM很重要的问题就是序列的补齐。在本实验中，我们要将所有MR样本按照轨迹进行划分，再在同一个轨迹下分成长度为6的序列。这里我直接将所有轨迹数据进行训练而不是分成四个手机每个一个模型，因为个人觉得分成四个意义不是很大，而且这样可能会导致数据量减少。因为LSTM接受的每个序列都是要等长的。由于我们要按照轨迹分类，那么每个轨迹的长度是不等的，其有可能会有最后不够组成长度为6的序列的部分，这就需要特殊的处理。

在实际处理中，我选择了将每个最后剩下的部分补齐方法。补齐的数据来源于本轨迹中前一个序列的后若干个数据，将其补齐成为6的序列。最终我们将所有轨迹的所有序列按照时间顺序组成数据集进行训练。

之后我们要对数据集进行划分。由于LSTM的特殊性，其划分与CNN稍有不同。我们的划分实际是对于每个轨迹的，在每个轨迹中随机挑选一些序列作为测试集。其实其本质与CNN划分数据集是一样的，只是实际实现需要自行写函数而不能利用sklearn提供的函数。

最后是关于网络结构。与CNN不同，LSTM一般只需一个LSTM层和一层全连接层即可，同时LSTM的训练往往epochs数要远多于CNN的次数。CNN一般50次即可，但是LSTM一般需要上百次。然后与CNN一致的是最后一层也是选择'linear'作为激活函数，优化器选择Adam。不同的是，LSTM一个需要调整的参数就是LSTM层Units（隐藏单元）的个数，最终我经过调参选择了64*6(序列长度)。

最终网络结构如下：

```python
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)
model = Sequential()
model.add(LSTM(64*steps, input_shape = (steps,x_train.shape[2]),activation='relu'))
model.add(Dense(output_dim=steps*2, activation="linear"))
model.compile(optimizer=adam,loss='mse')
model.fit(x_train,y_train,epochs=120, batch_size=64)
model.save('./c.LSTM')
```

#### 结果

| 中位误差 | 平均误差 | 90%误差 |
| -------- | -------- | ------- |
| 39.25    | 60.93    | 140.69  |

CDF：

![image-20190621111000468](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621111000468.png)

地图投影：

![image-20190621041134557](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621041134557.png)

### D实验

#### 过程

D实验是一个LSTM多分类问题。D实验基本都零碎地运用到了前面实验的思想，没什么特别特殊的地方。分类的格子基准沿用B实验的，其特征处理过程也与C实验类似，网络结构除损失函数与最后一层的激活函数之外也都与C类似。需要注意的是由于LSTM是按序列分的，因此在构建网络的过程中需要加入Reshape层。

网络结构如下：

```python
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
model.save('./d.LSTM')
```

#### 结果

误差：

| 中位误差 | 平均误差 | 90%误差 |
| -------- | -------- | ------- |
| 39.19    | 82.51    | 228.82  |

CDF：

![image-20190621111106100](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621111106100.png)

地图投影：

![image-20190621041217559](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621041217559.png)

### E实验

#### 过程

E实验是一个CNN/LSTM混合模型多分类问题。虽然看起来有点复杂，但是其本质是前面几个模型的综合，运用之前实验的基础即可。

其本质的过程就是运用实验B得到的CNN多分类模型先对整个训练集进行预测，但得到的是概率向量，以这个概率向量作为后一个LSTM网络的特征，对其进行与D实验类似的轨迹序列划分，再预测其所属的格子id。因此其本质上就是B实验和D实验的综合，绝大多数过程和网络结构、参数都与这两个实验一致，不再赘述。

#### 结果

误差：

| 中位误差 | 平均误差 | 90%误差 |
| -------- | -------- | ------- |
| 10.74    | 38.21    | 56.57   |

CDF:

![image-20190621111308644](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621111308644.png)

地图投影：

![image-20190621041311911](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621041311911.png)

### F实验

#### 过程

F实验是一个autoencoder和LSTM混合模型多分类问题。其与E实验非常类似，不同的地方在于用autoencoder代替了CNN做了LSTM的前处理过程。autoencoder通过一个encode和decode的过程对训练集重新编码，并且得到与原来维度相同的新训练集，再将其投入LSTM中进行训练。因此我们只需要设计autoencoder的网络，后面的LSTM可以直接沿用D实验的网络结构。

网络结构：

```python
#autoencoder编码
inputs = Input(shape=(steps, feature_no))
encoded = LSTM(90)(inputs)
decoded = RepeatVector(steps)(encoded)
decoded = LSTM(feature_no, return_sequences=True)(decoded)
print(x_train.shape)
sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
adam_ = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-8, amsgrad=False)

sequence_autoencoder.compile(loss='mse', optimizer=adam_)
sequence_autoencoder.fit(x_train, x_train, epochs=100, batch_size=32, shuffle=True)
sequence_autoencoder.save('./autoencoder')
seq_train = sequence_autoencoder.predict(x_train)

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
model.fit(seq_train,y_classes,epochs=130, batch_size=64)
model.save('./f.LSTM')
```

#### 结果

误差：

| 中位误差 | 平均误差 | 90%误差 |
| -------- | -------- | ------- |
| 88.10    | 137.65   | 357.89  |

CDF:

![image-20190621123457562](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621123457562.png)

地图投影：

![image-20190621123925810](/Users/liangchengwei/学习/大三下/数据分析和数据挖掘/饶伟雄作业/predict/assets/image-20190621123925810.png)

## 实验总结

| Experiment          | Median Error | Mean Error | 90% Error |
| ------------------- | ------------ | ---------- | --------- |
| CNN Regression      | 13.84        | 23.32      | 46.90     |
| CNN Classification  | 8.84         | 26.85      | 43.07     |
| LSTM Regression     | 39.25        | 60.93      | 140.69    |
| LSTM Classification | 39.19        | 82.51      | 228.82    |
| CNN + LSTM          | 10.74        | 38.21      | 56.57     |
| Autoencoder + LSTM  | 88.10        | 137.65     | 357.89    |
|                     |              |            |           |

从以上的实验中可以总结以下几点：

1. 本实验可以体现在深度学习中数据预处理有着非常重要的地位，其中在本实验中数据归一化有着决定成败的作用，选择合适的数据归一化方法可以大幅度提升模型预测效果；
2. 对于处理地球上的经纬度位置问题，应该将经纬度转换成UTM坐标；
3. 在这个实验中出现了缺省值的处理。对于一个缺省值，比较好的方法是将其补充起来而不是考虑将其丢弃。
4. 在本实验中，显然使用栅格法是更好的方法，选择合适的格子数，将回归问题转换成分类问题，能够很好地控制预测误差；
5. CNN的结构要复杂一些，而LSTM在较简单的情况不需要太复杂的结构，但其对于参数非常敏感，不好进行控制；
6. 合理地混合使用网络模型，能够取得好的效果。
7. 在使用autoencoder+LSTM时出现了比较严重的过拟合。