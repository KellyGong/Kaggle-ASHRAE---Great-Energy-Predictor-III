# Kaggle-ASHRAE---Great-Energy-Predictor-III
记录本人第一次参加kaggle比赛的经历

## 比赛结果

[比赛地址](https://www.kaggle.com/c/ashrae-energy-prediction)

[比赛数据集](https://www.kaggle.com/c/ashrae-energy-prediction/data)

[比赛结果](https://www.kaggle.com/c/ashrae-energy-prediction/leaderboard) 🥈 82 /3614

![avatar](https://github.com/KellyGong/Kaggle-ASHRAE---Great-Energy-Predictor-III/raw/master/pic/profile.png)

## Abstract

本次比赛的目标是要解决能源消耗预测问题（预测给定建筑物，在指定时间下，指定能源类型的能源消耗量）。我们使用基于决策树算法的分布式梯度提升框架LightGBM，从过去一段时间内的建筑物相关信息（建筑物面积、建筑年代、建筑物用途等）、建筑物所处位置的天气信息（温度、大气压强、降水量等）、能源消耗读数（标签）以及特征工程加入的一些特征学习模型。我们通过学习得到的LightGBM模型，预测测试集上的最终结果。采用不同的策略训练得到多个LightGBM模型，最终通过模型融合的方法来获得在测试集上更好的效果。

## 数据集
![avatar](https://github.com/KellyGong/Kaggle-ASHRAE---Great-Energy-Predictor-III/raw/master/pic/meter_reading(target).png)

本次比赛的目标是要解决能源消耗预测问题（预测给定建筑物，在指定时间下，指定能源类型的能源消耗量）。1448个建筑物的详细信息存储在building_meta数据中，这些建筑物分布在15个地方。这15个地方的天气信息存储在weather_train/test数据中。weather_train数据集存储的是训练数据中的时间戳对应时间下当地的天气信息，test对应的是测试集。训练集中会给出最终的能源消耗读数（meter_reading），测试集中则需要预测给定情况下的能源消耗读数（meter_reading）。
最终采用的评估方法为RMSLE。

## 整体思路

### 使用LightGBM + 模型融合

LightGBM使用的是基于Histogram的决策树算法，并使用带深度限制的Leaf_wise的叶子生长策略，具有轻量化、速度快的特点，并直接支持类别特征。

### [Kernel 1：ASHRAE: Half and Half （RMSLE 1.1）](https://www.kaggle.com/rohanrao/ashrae-half-and-half)

特征工程：给了一个holidays列表，加入一个is_holiday的列，若该天在holidays中，则is_holiday的值为1，否则为0。

训练方式：将原训练集切割（原训练集按照时间排序，非随机切割）成大小相同的两部分，训练出两个LightGBM模型。最终用这两个模型预测最终的test集结果做平均。


### [Kernel 2：ASHRAE: KFold LightGBM - without leak （RMSLE 1.08）](https://www.kaggle.com/aitude/ashrae-kfold-lightgbm-without-leak-1-08)

(1) 训练集存在一些异常的数据，meter_reading一直为0，因此先将满足 
(building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")
这部分数据去掉。

(2) 对于每个缺失（NaN）的数据，按照一定的分组，使用分组的平均值来填充。相比起直接粗暴地将整列的平均值直接填入，可能效果会更好些。

(3) 删去了"timestamp","sea_level_pressure", "wind_direction", “wind_speed","year_built","floor_count"一些缺失数据较多的列。

(4) 由于不同建筑物的square_feet相差较大，为了使数据不太离散，做了取对数处理。

训练方式：将原训练集切割成大小相同的三部分，训练出三个LightGBM模型。最终用这三个模型预测最终的test集结果做平均。

### [Kernel 3：ASHRAE: Highway Kernel Route2（1.03）](https://www.kaggle.com/yamsam/ashrae-highway-kernel-route2)


### 模型融合

使用几个模型的结果，分别取一定的比重加权求和得到最终结果，以泄露的数据作为标签，计算评测指标RMSLE。取RMSLE最小的一组作为结果。


## 最终结果
在88%的公开数据上的最好RMSLE = 0.957。

