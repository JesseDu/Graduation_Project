import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data as Data
import time




INPUT_SIZE=3
LR = 0.001               # learning rate
EPOCH = 100
class MYMODULE(nn.Module):
    def __init__(self, lstm_hidden_units=20, embedding_dim=5):
        super(MYMODULE,self).__init__()

        self.lstm_hidden_units = lstm_hidden_units
        self.embedding_dim = embedding_dim
        # don't count the padding tag for the classifier output
        self.__build_model()

    def __build_model(self):
        self.embedding = nn.Sequential(
            nn.Linear(INPUT_SIZE,self.embedding_dim),
            nn.Tanh(),
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_hidden_units,
            num_layers=1,
            batch_first=True,
        )
        # output layer which projects back to tag space

        self.hidden_to_tag = nn.Linear(self.lstm_hidden_units, 2)#此处应该是softmax

    # def init_hidden(self):
    #     # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
    #     hidden_a = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
    #     hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
    #     if self.hparams.on_gpu:
    #         hidden_a = hidden_a.cuda()
    #         hidden_b = hidden_b.cuda()
    #     hidden_a = Variable(hidden_a)
    #     hidden_b = Variable(hidden_b)
    #     return (hidden_a, hidden_b)

    def forward(self, X):
        # self.hidden = self.init_hidden()
        batch_size, seq_len, attribute_dim = X.size()
        X = X.view(-1,attribute_dim)
        X = self.embedding(X)
        X = X.view(batch_size, seq_len, self.embedding_dim)
        # X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        lstm_out, (h_n, h_c) = self.lstm(X, None)#？？？？
        # X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)
        # X = X.contiguous()
        #只取最后时刻的输出
        encoder = lstm_out[:,-1,:]#(batch_size,embedding_dim)，取完就是这个维度，不需要再view
        # encoder = lstm_out.view(-1,lstm_out.shape[2])#(batch_size,embedding_dim)
        X = self.hidden_to_tag(encoder)
        # X = F.log_softmax(X, dim=1)#注意：在pytorch中若模型使用CrossEntropyLoss这个loss函数，则不应该在最后一层再使用softmax进行激活。
        # X = X.view(batch_size, seq_len, self.nb_tags)
        return X

PAD_TOKEN = 0
SEQUENCE_LENGTH = 200#最长序列长度
ATTRIBUTE_NUM = 3#数据集属性个数（除去ID和时间)
MINIBATCH_SIZE = 128 #mini-batch


if __name__ == '__main__':
    # df = pd.read_csv('data/test.txt')
    # df.drop(['INDEX'],axis = 1,inplace=True)#注意，换了数据集之后，此行删去
    # df.set_index(['ID','TIME'],inplace=True)
    time_start = time.clock()
    print("程序开始：",time_start)

    print("loading train-test data from file")
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
    test_x = np.load("test_x.npy")
    test_y = np.load("test_y.npy")
    # print("train_x:", train_x, "\n train_y:", train_y, "\n test_x:", test_x, "\n test_y:", test_y)
    print("train_x shape:", train_x.shape, "\ntrain_y shape:", train_y.shape, "\ntest_x shape:", test_x.shape,
          "\ntest_y shape:", test_y.shape)
    # train_x shape: (7436, 200, 3)
    # train_y shape: (7436,)
    # test_x shape: (1858, 200, 3)
    # test_y shape: (1858,)
    #取前7436个做训练集，后1858个作为测试集

    X_tensor_train = torch.Tensor(train_x)
    Y_tensor_train = torch.LongTensor(train_y)
    print('X_tensor_train:',X_tensor_train,'X_tensor_train size:',X_tensor_train.size())
    print('Y_tensor_train:',Y_tensor_train,'Y_tensor_train size:',Y_tensor_train.size())

    X_tensor_test = torch.Tensor(test_x)
    Y_tensor_test = torch.Tensor(test_y)
    # Y_test = torch.Tensor(data_y[:2000]).numpy()
    print('X_tensor_test:', X_tensor_test,'X_tensor_test size:',X_tensor_test.size())
    print('Y_tensor_test:', Y_tensor_test,'Y_tensor_test size:',Y_tensor_test.size())

    # X_lengthstensor = torch.Tensor(X_lengths)
    # torch_dataset_train = Data.TensorDataset(X_tensor_train,Y_tensor_train,X_lengthstensor)
    torch_dataset_train = Data.TensorDataset(X_tensor_train, Y_tensor_train)
    # torch_dataset_test = Data.TensorDataset(X_tensor_test,Y_tensor_test,X_lengthstensor)
    train_loader = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=1           # set multi-work num read data
    )

    # for epoch in range(3):
    #     # 1 epoch go the whole data
    #     for step, (batch_x, batch_y,batch_x_lengths) in enumerate(train_loader):
    #         # here to train your model
    #         print('\n\n epoch: ', epoch, '| step: ', step, '| batch x: ', batch_x.numpy(), '| batch_y: ', batch_y.numpy(),'| batch_x_length:',batch_x_lengths.numpy())
    #         print('batch_x_size:',batch_x.size(),'batch_y_size:',batch_y.size())

    time_dataprocessfinish = time.clock()
    print("数据处理完成,用时：",time_dataprocessfinish - time_start)

    ###模型训练
    mymodule = MYMODULE()
    print(mymodule)

    optimizer = torch.optim.Adam(mymodule.parameters(), lr=LR,weight_decay = 0.15)
    loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

    # training and testing
    print("开始训练：")
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(train_loader):        # gives batch data
            # batch_x = batch_x.view(-1, 28, 28)              # reshape x to (batch, time_step, input_size)

            output = mymodule(batch_x)                               #(batch_size,2)
            loss = loss_func(output, batch_y)                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients

            if step % 5 == 0:
                test_output = mymodule(X_tensor_test)                   # (samples, time_step, input_size)
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()#torch.max()第二个参数，表示按行取最大值返回相应的列索引
                print('pred_y中预测为1个数：',(pred_y == 1).astype(int).sum())
                accuracy = float((pred_y == Y_tensor_test.numpy()).astype(int).sum()) / float(Y_tensor_test.numpy().size)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

    print("finish")

    time_finish = time.clock()
    print("训练模型用时：",time_finish - time_dataprocessfinish)
    print("程序总运行时间：",time_finish - time_start)
    # # print 10 predictions from test data
    # test_output = rnn(test_x[:10].view(-1, 28, 28))
    # pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    # print(pred_y, 'prediction number')
    # print(test_y[:10], 'real number')


