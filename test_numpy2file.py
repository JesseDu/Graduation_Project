import numpy as np
import random
import pandas as pd


#X:含label的数据集：分割成训练集和测试集
#test_size:测试集占整个数据集的比例
def trainTestSplit(data_x,data_y,test_size=0.2):#data_x,data_y是ndarray
    data_size=data_x.shape[0]
    total_index=list(range(data_size))
    test_index = random.sample(total_index,int(test_size*data_size))
    train_index = list(set(total_index)-set(test_index))
    print("train index:",train_index,"\n test index:",test_index)


    train_x = data_x[train_index]
    test_x = data_x[test_index]
    train_y = data_y[train_index]
    test_y = data_y[test_index]
    # print("train_x:",train_x,"\n train_y:",train_y,"\n test_x:",test_x,"\n test_y:",test_y)
    return train_x,train_y,test_x,test_y

#data_x是列表，将其对data_x进行补齐，返回ndarray
def padding(data_x,PAD_TOKEN,SEQUENCE_LENGTH,ATTRIBUTE_NUM):
    X_lengths = [len(item) for item in data_x]
    print('X_lengths before padding:', X_lengths)
    batch_size = len(data_x)
    padded_X = np.ones((batch_size, SEQUENCE_LENGTH, ATTRIBUTE_NUM)) * PAD_TOKEN  # 矩阵和列表不同，要求每行每列元素个数相同
    for i, x_len in enumerate(X_lengths):
        sequence = data_x[i]
        x_len = SEQUENCE_LENGTH if x_len >= SEQUENCE_LENGTH else x_len
        X_lengths[i] = x_len
        padded_X[i, 0:x_len] = sequence[:x_len]  # 截取前xxx个，可修改
    print('X_lengths after padding:', X_lengths)
    # print('padded_X:', padded_X)
    # print(type(padded_X))  # <class 'numpy.ndarray'>
    return padded_X


#padding之前，不能保存，因为是变长数据，无法用矩阵表示
#padding在此部分完成
#此部分还进行训练集测试集分割，设置比例
if __name__ == '__main__':
    # data = np.arange(40).reshape((4, 5, 2))
    print('start')
    # df = pd.read_csv('data/test.txt')
    # df.drop(['INDEX'],axis = 1,inplace=True)#注意，换了数据集之后，此行删去
    # df.set_index(['ID','TIME'],inplace=True)

    df = pd.read_csv('data/merge_train_new.txt')
    df.set_index(['ID', 'TIME'], inplace=True)

    # print(df.index.levels[0])  # 取第一级索引Int64Index([1265, 2583, 6965, 29165], dtype='int64', name='ID')
    # 注意！索引排序了

    data_x = []#list
    data_y = []
    for index1 in df.index.levels[0]:
        data_x.append(df.ix[index1].values[:, :-1])
        data_y.append(df.ix[index1].values[0, -1])

    PAD_TOKEN = 0
    SEQUENCE_LENGTH = 200  # 最长序列长度
    ATTRIBUTE_NUM = 3  # 数据集属性个数（除去ID和时间)
    padded_X = padding(data_x,PAD_TOKEN,SEQUENCE_LENGTH,ATTRIBUTE_NUM)

    #padded_X是'numpy.ndarray'类型
    print("saving padded_X to file")
    np.save("train_data_x.npy", padded_X)
    np.save("train_data_y.npy", np.array(data_y))
    print("loading from file")
    read_data_x = np.load("train_data_x.npy")
    read_data_y = np.load("train_data_y.npy")

    # print(read_data_x)
    # print(read_data_y)

    train_x, train_y, test_x, test_y = trainTestSplit(read_data_x, read_data_y, test_size=0.2)
    # print("saving train-test data to file")
    np.save("train_x.npy",train_x)
    np.save("train_y.npy", train_y)
    np.save("test_x.npy", test_x)
    np.save("test_y.npy", test_y)

    print("loading train-test data from file")
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
    test_x = np.load("test_x.npy")
    test_y = np.load("test_y.npy")
    print("train_x:", train_x, "\n train_y:", train_y, "\n test_x:", test_x, "\n test_y:", test_y)
    print("train_x shape:", train_x.shape, "\n train_y shape:", train_y.shape, "\n test_x shape:", test_x.shape, "\n test_y shape:", test_y.shape)






