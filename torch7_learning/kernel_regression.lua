require 'torch'
require 'gnuplot'

local nData = 10	-- Number of data samples
local kWidth = 1	-- Kernel width(lambda)

local xTrain = torch.linspace(-1, 1, nData)
local yTrain = torch.pow(xTrain, 2)
local yTrain = yTrain + torch.mul(torch.randn(nData), 0.1)	-- Gaussian noise

local function phi(x, y)
	return torch.exp(-(1/kWidth)*torch.sum(torch.pow(x-y, 2)))
end

local Phi = torch.Tensor(nData, nData)
for i = 1, nData do
	for j = 1, nData do
		Phi[i][j] = phi(xTrain[{{i}}], xTrain[{{j}}])
	end
end

local regularizer = torch.mul(torch.eye(nData), 0.001)
local theta = torch.inverse((Phi:t()*Phi) + regularizer) * Phi:t() * yTrain

local nTestData = 100	-- Number of test data samples
local xTest = torch.linspace(-1, 1, nTestData)

local PhiTest = torch.Tensor(nData, nTestData)
for i = 1, nData do
	for j = 1, nTestData do
		PhiTest[i][j] = phi(xTrain[{{i}}], xTest[{{j}}])
	end
end

local yPred = PhiTest:t() * theta

gnuplot.pngfigure('plot.png')
gnuplot.plot({'Data', xTrain, yTrain, '+'}, {'Prediction', xTest, yPred, '-'})
gnuplot.plotflush()