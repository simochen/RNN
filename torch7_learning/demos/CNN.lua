require 'torch'
require 'nn'
require 'optim'
mnist = require 'mnist'
--返回了一个mnist对象，获得mnist图像数据的代码如下：
fullset = mnist.traindataset()
testset = mnist.testdataset()
