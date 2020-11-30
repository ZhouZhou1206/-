     #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:13:08 2018
two-user autoencoder in paper
@author: musicbeer
"""

import torch     #使用torch框架
from torch import nn    # nn是neural network神经网络的缩略词，这句话的意思是从torch里边把nn拿出来用
import numpy as np      # Numpy 是基础科学计算库，这里用“np数组”指代Numpy数组

NUM_EPOCHS =100         #  epoch的意思是时代。这个变量的意思大概是时代的数目
BATCH_SIZE = 32         #  batch的意思是批，分批，这个变量的意思大概是一批的大小是32
USE_CUDA = False
parm1=4                 #parm这个单词的意思是参数定义，这里是定义两个值为4的参数
parm2=4
M = 2**parm2#one-hot coding feature dim
k = np.log2(M)          #这里是np调用计算库log后然后赋值给k
k = int(k)              #把k强制类型转换为int型
n_channel =parm1#compressed feature dim     #把Parm1赋值给n_channel
R = k/n_channel
CHANNEL_SIZE = M
train_num=8000          # 训练集的数目设置为8000
test_num=50000          # 测试集的数目设置为50000

class RTN(nn.Module):
    def __init__(self, in_channels, compressed_dim):     # compressed 压缩的 dim维度
        super(RTN, self).__init__()

        self.in_channels = in_channels              #实例化

        self.encoder1 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, compressed_dim),
        )
             # nn.Sequential()是一个有序的容器，神经网络模块将按照传入构造器的顺序依次被添加到计算图中执行，
             #同时以神经网络模块为元素的有序字典也可以作为传入参数
             #torch.nn.Linear()的功能是定义一个线性变换(连同偏置)，即定义一个这样的运算:y=xw^T+b
             #tanh函数解决了Sigmoid函数的不是zero-centered输出问题，但梯度消失（gradient vanishing）的问题和幂运算的问题仍然存在。
             #在本论文中tanh激活后是一个密集的输出层与线性激活
             #encoder编码器，decoder解码器
        self.decoder1 = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, in_channels)
        )
        
        # 以上的编码是编码器1和解码器1，下边的是编码器2，解码器2
        self.encoder2 = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, compressed_dim),
        )

        self.decoder2 = nn.Sequential(
            nn.Linear(compressed_dim, in_channels),
            nn.Tanh(),
            nn.Linear(in_channels, in_channels)
        )
    def encode_signal1(self, x):
        x1=self.encoder1(x)
        #x1 = (self.in_channels ** 2) * (x1 / x1.norm(dim=-1)[:, None])
        return x1
    #编码器信号函数1
    def encode_signal2(self, x):
        x1=self.encoder2(x)
        #x2 = (self.in_channels ** 2) * (x1 / x1.norm(dim=-1)[:, None])
        return x1 
    #编码器信号函数2
    def decode_signal1(self, x):
        return self.decoder1(x)
    #解码器信号函数1
    def decode_signal2(self, x):
        return self.decoder2(x)
    #解码器信号函数2 
    
    # AWGN是高斯白噪声，
    def mixedAWGN(self, x1,x2,ebno):     #每个二进制bit能量与噪声能量谱密度的比值（dB）。一般以该参数作为性能衡量的横坐标
        x1 = (self.in_channels ** 0.5) * (x1 / x1.norm(dim=-1)[:, None])
        # bit / channel_use
        communication_rate = R
        # Simulated Gaussian noise.
        noise1 = Variable(torch.randn(*x1.size()) / ((2 * communication_rate * ebno) ** 0.5))
       #Variable就是 变量 的意思。实质上也就是可以变化的量，区别于int变量，它是一种可以变化的变量，这正好就符合了反向传播，参数更新的属性。
        #tensor不能反向传播，variable可以反向传播。
        x2 = (self.in_channels ** 0.5) * (x2 / x2.norm(dim=-1)[:, None])
        # Simulated Gaussian noise.
        noise2 = Variable(torch.randn(*x2.size()) / ((2 * communication_rate * ebno) ** 0.5))
        print("############################",ebno)
        
        signal1=x1+noise1+x2
        signal2=x1+x2+noise2
        return signal1,signal2
   
    def forward(self, x1,x2):
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        # Normalization.
        x1 = (self.in_channels **0.5) * (x1 / x1.norm(dim=-1)[:, None])
        x2 = (self.in_channels **0.5) * (x2 / x2.norm(dim=-1)[:, None])

        # 7dBW to SNR.
        training_signal_noise_ratio =  5.01187  #训练集信噪比

        # bit / channel_use
        communication_rate = R

        # Simulated Gaussian noise. 模拟高斯噪声
        noise1 = Variable(torch.randn(*x1.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        noise2 = Variable(torch.randn(*x2.size()) / ((2 * communication_rate * training_signal_noise_ratio) ** 0.5))
        signal1=x1+noise1+x2
        signal2=x1+x2+noise2
        
        decode1 = self.decoder1(signal1)
        decode2 = self.decoder2(signal2)

        return decode1,decode2

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam,RMSprop  #只是个警告，问题不大
    # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。
    #它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳
    import torch.utils.data as Data
    model = RTN(CHANNEL_SIZE, compressed_dim=n_channel)
    if USE_CUDA: model = model.cuda()
    train_labels1 = (torch.rand(train_num) * CHANNEL_SIZE).long()
    #torch。rand（）返回服从均匀分布的初始化后的tenosr，外形是其参数size。
    train_data1 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels1)
    train_labels2 = (torch.rand(train_num) * CHANNEL_SIZE).long()
    train_data2 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=train_labels2)
    train_labels= torch.cat((torch.unsqueeze(train_labels1,1), torch.unsqueeze(train_labels2,1)), 1)
    train_data=torch.cat((train_data1, train_data2), 1)
    #torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
    #使用torch.cat((A,B),dim)时，除拼接维数dim数值可不同外其余维数数值需相同，方能对齐。
    test_labels1 = (torch.rand(test_num) * CHANNEL_SIZE).long()
    test_data1 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels1)
    #torch.eye(n,m,out)这个函数主要是为了生成对角线全1，其余部分全0的二维数组。n行数，m列数,out输出类型
    test_labels2 = (torch.rand(test_num) * CHANNEL_SIZE).long()
    test_data2 = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels2) #index注释，索引
    test_labels= torch.cat((torch.unsqueeze(test_labels1,1), torch.unsqueeze(test_labels2,1)), 1)
    test_data=torch.cat((test_data1, test_data2), 1)
     #torch.cat是将两个张量（tensor）拼接在一起，cat是concatnate的意思，即拼接，联系在一起。
    #dataset = Data.TensorDataset(data_tensor =  train_data, target_tensor = train_labels)#这句是原来的代码报错了，上一行是改过的代码，可能是新旧版的差异，语句上的不同导致的报错
    dataset = Data.TensorDataset(train_data,train_labels)
    #datasettest = Data.TensorDataset(data_tensor =  test_data, target_tensor = test_labels)  # 这句是原来的代码报错了，上一行是改过的代码，可能是新旧版的差异，语句上的不同导致的报错
    datasettest = Data.TensorDataset(test_data,test_labels)
    #TensorDataset 可以用来对 tensor 进行打包，就好像 python 中的 zip 功能。该类通过每一个 tensor 的第一个维度进行索引。因此，该类中的 tensor 第一维度必须相等。
    train_loader = Data.DataLoader(dataset = dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
    test_loader = Data.DataLoader(dataset =  datasettest, batch_size = test_num, shuffle = True, num_workers = 2)
    #数据加载器，组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。它可以对我们上面所说的数据集Dataset作进一步的设置。

    optimizer = Adam(model.parameters(),lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    a=0.5
    b=0.5
    for epoch in range(NUM_EPOCHS):     #empoch纪元
       for step, (x, y) in enumerate(train_loader):  #enumerate列举，枚举
        b_x1 = Variable(x[:,0:CHANNEL_SIZE])   
        b_y1 = Variable(x[:,0:CHANNEL_SIZE])  
        b_label1 = Variable(y[:,0])               
        b_x2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_y2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_label2 = Variable(y[:,1])               
        decoded1,decoded2 = model(b_x1,b_x2)
        loss1 = loss_fn(decoded1, b_label1)      
        loss2 = loss_fn(decoded2, b_label2)      
        loss=loss1*a+loss2*b

        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()  
        a=loss1/(loss1+loss2)
        #a=a.data[0]               #这句话出现报错，又是版本的问题
        a=a.item()   #这句话是对a=a.data[0]的修改版
        b=loss2/(loss2+loss1)                  # apply gradients
        #b=b.data[0]
        b=b.item()
        if step % 100 == 0:
           # print('Epoch: ', epoch, '| train loss: %.4f, L1:%.4f,L2: %.4f,a: %.4f, (1-a):%.4f' % (loss.data[0],loss1.data[0],loss2.data[0],a,b))
           #这句话依然是data【0】的报错，下边是修改版
            print('Epoch: ', epoch, '| train loss: %.4f, L1:%.4f,L2: %.4f,a: %.4f, (1-a):%.4f' % (loss.item(),loss1.item(),loss2.item(),a,b))
    import numpy as np
    EbNodB_range = list(frange(0,15.5,0.5))
    ber1 = [None]*len(EbNodB_range)     
    ber2 = [None]*len(EbNodB_range)          
    for n in range(0,len(EbNodB_range)):
     EbNo=10.0**(EbNodB_range[n]/10.0)
     for step, (x, y) in enumerate(test_loader):
        b_x1 = Variable(x[:,0:CHANNEL_SIZE])   
        b_y1 = Variable(x[:,0:CHANNEL_SIZE])  
        b_label1 = Variable(y[:,0])              
        b_x2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_y2 = Variable(x[:,CHANNEL_SIZE:CHANNEL_SIZE*2])   
        b_label2 = Variable(y[:,1])
        encoder1=model.encode_signal1(b_x1)
        encoder2=model.encode_signal2(b_x2)
        encoder1,encoder2=model.mixedAWGN(encoder1,encoder2,EbNo)
        decoder1=model.decode_signal1(encoder1)
        decoder2=model.decode_signal2(encoder2)
        pred1=decoder1.data.numpy()
        pred2=decoder2.data.numpy()
        label1=b_label1.data.numpy()
        label2=b_label2.data.numpy()
        pred_output1 = np.argmax(pred1,axis=1)
        pred_output2 = np.argmax(pred2,axis=1)
        no_errors1 = (pred_output1 != label1)
        no_errors2 = (pred_output2 != label2)
        no_errors1 =  no_errors1.astype(int).sum()
        no_errors2 =  no_errors2.astype(int).sum()
        ber1[n] = no_errors1 / test_num
        ber2[n]=no_errors2 / test_num 
        print ('SNR:',EbNodB_range[n],'BER1:',ber1[n],'BER2:',ber2[n])

#    
## ploting ber curve
    import matplotlib.pyplot as plt
    plt.plot(EbNodB_range, ber1, 'bo',label='Autoencoder1(4,4)')
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc='upper right',ncol = 1)
    
    plt.plot(EbNodB_range, ber2, 'bo',label='Autoencoder2(4,4)',color='r')
    plt.yscale('log')
    plt.xlabel('SNR Range')
    plt.ylabel('Block Error Rate')
    plt.grid()
    plt.legend(loc='upper right',ncol = 1)



#            
#            
#    import matplotlib.pyplot as plt
#    test_labels = torch.linspace(0, CHANNEL_SIZE-1, steps=CHANNEL_SIZE).long()
#    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
#    #test_data=torch.cat((test_data, test_data), 1)
#    test_data=Variable(test_data)
#    x=model.encode_signal1(test_data)
#    x = (n_channel**0.5) * (x / x.norm(dim=-1)[:, None])
#    plot_data=x.data.numpy()
#    plt.scatter(plot_data[:,0],plot_data[:,1],color='r')
#    plt.axis((-2.5,2.5,-2.5,2.5))
#    #plt.grid()
#
#    scatter_plot = []
#
#    scatter_plot = np.array(scatter_plot)
#    print (scatter_plot.shape)
#    
#    test_labels = torch.linspace(0, CHANNEL_SIZE-1, steps=CHANNEL_SIZE).long()
#    test_data = torch.sparse.torch.eye(CHANNEL_SIZE).index_select(dim=0, index=test_labels)
#    #test_data=torch.cat((test_data, test_data), 1)
#    test_data=Variable(test_data)
#    x=model.encode_signal2(test_data)
#    x = (n_channel**0.5) * (x / x.norm(dim=-1)[:, None])
#    plot_data=x.data.numpy()
#    plt.scatter(plot_data[:,0],plot_data[:,1])
#    plt.axis((-2.5,2.5,-2.5,2.5))
#    plt.grid()
#   # plt.show()
#    scatter_plot = []
##
##    scatter_plot = np.array(scatter_plot)
#    plt.show()