import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, hidden_layers, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, hidden_layers, bidirectional=True)
        self.embedding = nn.Linear(hidden_layers * 2, nOut)

    def forward(self, input):
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, param):
        super(CRNN, self).__init__()
        # We need to ensure that image height is multiple of 16
        # because our filters are set accordingly
        if param['imgH'] % 16 != 0:
            raise ValueError("Image height must be a multiple of 16!")
        # This config has been extracted from CNN-LSTM Research Paper
        kernel_size = [3, 3, 3, 3, 3, 3, 2]
        padding_size = [1, 1, 1, 1, 1, 1, 0]
        stride_size = [1, 1, 1, 1, 1, 1, 1]
        n_maps = [64, 128, 256, 256, 512, 512, 512]
        # Starting with an empty layered sequential nn model, we'll add modules below
        cnn = nn.Sequential()
        # The convulation activation function
        def conv_activation(i, relu=True):
            # Input for i = 0 is 1 because our image is gray scale having only one channel
            nIn = 1 if i == 0 else n_maps[i - 1] 
            nOut = n_maps[i]
            cnn.add_module('Convulation_{0}'.format(i), nn.Conv2d(nIn, nOut, kernel_size[i], stride_size[i], padding_size[i]))
            if relu:
                cnn.add_module('Relu_{0}'.format(i), nn.ReLU(True))
            else:
                cnn.add_module('BatchNorm_{0}'.format(i), nn.BatchNorm2d(nOut))
        # Adding CNN Modules
        conv_activation(0)
        cnn.add_module('MaxPooling_0', nn.MaxPool2d(2, 2))  # 64x16x64
        conv_activation(1)
        cnn.add_module('MaxPooling_1', nn.MaxPool2d(2, 2))  # 128x8x32
        conv_activation(2, False)
        conv_activation(3)
        cnn.add_module('MaxPooling_2', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        conv_activation(4, False)
        conv_activation(5)
        cnn.add_module('MaxPooling_3', nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        conv_activation(6, False)  # 512x1x16
        self.cnn = cnn
        # self.rnn = nn.Sequential()
        # Here we are using 256 hidden layers
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256*2, 256, 256),
            BidirectionalLSTM(256, 256, param['n_classes']))


    def forward(self, input):
        # Convulation Features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        if h != 1:
            raise ValueError("The height of convulation must be 1!")
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        output = output.transpose(1,0) #Tbh to bth
        return output