local t = torch.Tensor({
	{1, 2, 3},
	{4, 5, 6},
	{7, 8, 9}
})

--extract the middle column form t
local col = t:narrow(2,2,1)
print(col)

local col = t:sub(1,-1,2,2)
print(col)

local col = t:select(2,2)
print(col)

local col = t[{{},2}]
print(col)