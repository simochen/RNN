local matio = require 'matio'
data = matio.load('multi.mat')
multi_y = matio.load('multi_data.mat','multi_y')
print(multi_y) 