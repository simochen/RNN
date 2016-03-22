require 'torch'		--Torch
require 'nn'		--神经网络库
require 'gnuplot'	--画图的库

--生成数据
month = torch.range(1,10)	--1维Tensor, 元素的值为从1到10
price = torch.Tensor{28993, 29110, 29436, 30791, 33384, 36762, 39900, 39972, 40230, 40146}

--建立线性回归模型(1输入，1输出)
model = nn.Linear(1, 1)

--定义训练评价标准
criterion = nn.MSECriterion()

--训练之前，对数据做预处理
--把月份和价格的数据从1维Tensor转换为2维的10*1 Tensor
month_train = month:reshape(10, 1)
price_train = price:reshape(10, 1)
--第1个维度10是batch size，也就是每次迭代放进去计算的数据的个数。
--因为只有10个数据，就全部放进去训练，这样的做法叫做full-batch
--如果每次只放一部分数据进去，叫做mini-batch
--第2个维度1，表示数据是1维的

--循环迭代,固定步骤
for i = 1, 1000 do
	price_predict = model:forward(month_train)	--model正向传播
	err = criterion:forward(price_predict, price_train)	--criterion正向传播
	print(i, err)
	--model内部的偏导数清零，否则每次将会累加偏导数
	model:zeroGradParameters()
	gradient = criterion:backward(price_predict, price_train) --criterion反向传播
	model:backward(month_train, gradient)
	model:updateParameters(0.01)
end

--正向传播，预测11、12月份的房价
month_predict = torch.range(1,12)
local price_predict = model:forward(month_predict:reshape(12,1))
print(price_predict)
--可视化数据
gnuplot.pngfigure('plot.png')
gnuplot.plot({month, price}, {month_predict, price_predict})
gnuplot.plotflush()
