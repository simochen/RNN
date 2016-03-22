require 'torch'		--Torch
require 'nn'		--神经网络库
require 'gnuplot'	--画图的库
--之前使用的优化算法是最原始的梯度下降法。Torch提供了先进的优化算法，在optim库中
require 'optim'

--1. 生成数据
month = torch.range(1,10)	--1维Tensor, 元素的值为从1到10
price = torch.Tensor{28993, 29110, 29436, 30791, 33384, 36762, 39900, 39972, 40230, 40146}

--2. 建立MLP模型()
model = nn.Sequential()
--由于Sigmoid函数在远离零点的位置几乎梯度为0，需要将输入输出数据调整到-1~1区间附近
--scale输入数据，输入应除以10，即在输入进入前先乘0.1
model:add(nn.MulConstant(0.1))
--添加输入层(1个节点)和第一个隐藏层(3个节点)
model:add(nn.Linear(1,3))
--添加一个Sigmoid层，它的节点个数会自动和前一层的输出个数保持一致
model:add(nn.Sigmoid())
--添加第二个隐藏层(3个节点)
model:add(nn.Linear(3,3))
--再添加一个Sigmoid层
model:add(nn.Sigmoid())
--添加输出层(1个节点)
model:add(nn.Linear(3,1))
--scale输出数据，实际输出数据应除以50000，或在预测输出值后乘50000
model:add(nn.MulConstant(50000))

--3. 定义训练评价标准
criterion = nn.MSECriterion()

--4. 训练之前，对数据做预处理
--把月份和价格的数据从1维Tensor转换为2维的10*1 Tensor
month_train = month:reshape(10, 1)
price_train = price:reshape(10, 1)
--第1个维度10是batch size，也就是每次迭代放进去计算的数据的个数。
--因为只有10个数据，就全部放进去训练，这样的做法叫做full-batch
--如果每次只放一部分数据进去，叫做mini-batch
--第2个维度1，表示数据是1维的

--5. 循环迭代(优化)
--5.1把model里面的参数取出来方便随时调用
--w是model里面所有可调参数的集合，dl_dw是每个参数对loss的偏导数
w, dl_dw = model:getParameters()
--*这里的w和dl_dw都相当于C++里的“引用”，一旦对它们进行操作，模型里的参数也会随之改变

--5.2定义优化函数的目标函数
--优化函数的调用方法有一点特殊，需要你先提供一个目标函数，这个函数相当于C++里的回调函数，他的输入是一组网络权重参数w，输出有两个，第一个是网络使用参数w时，其输出结果与实际结果之间的差别，也可以叫loss损失，另一个是w中每个参数对于loss的偏导数。
feval = function(w_new)  
	if w ~= w_new then w:copy(w_new) end  
	dl_dw:zero()	--偏导数置0
	
	price_predict = model:forward(month_train)	--model正向传播
	loss = criterion:forward(price_predict, price_train)	--criterion正向传播
	gradient = criterion:backward(price_predict, price_train)  --criterion反向传播
	model:backward(month_train, gradient)  	--model反向传播
	return loss, dl_dw  
end

--5.3优化迭代
--有了这个目标函数，优化迭代的过程只需要调用optim.rprop(feval, w, params) --rprop是一种改进的梯度下降法，它只看梯度的方向，不管大小，只要方向不变，它会无限的增大步长，所以他速度非常快。
params = {  
   learningRate = 1e-2  
}  
  
for i=1,3000 do  
   optim.rprop(feval, w, params)  
  
   if i%10==0 then		--每10次迭代用gnuplot画出结果
      gnuplot.plot({month, price}, {month_train:reshape(10), price_predict:reshape(10)})  
   end  
end  

--6. 正向传播，预测11、12月份的房价
month_predict = torch.range(1,12)
local price_predict = model:forward(month_predict:reshape(12,1))
print(price_predict)
--可视化数据
gnuplot.pngfigure('plot.png')
gnuplot.plot({month, price}, {month_predict, price_predict})
gnuplot.plotflush()
