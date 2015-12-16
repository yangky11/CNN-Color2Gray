require 'torch'
require 'paths'
require 'optim'
paths.dofile('opts.lua')
require 'image'
require 'nn'
--require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

local rgbImg = image.load(opts.inp)
local labImg = image.rgb2lab(rgbImg)--:cuda()
--local labImg = rgbImg
print(string.format('image size: %dx%d\n------------------', labImg:size(2), labImg:size(3)))

-- create the model and training criterion
paths.dofile('model.lua')
model, criterion = createModel()

-- compute the distances
paths.dofile('dist.lua')
local delta = dist(labImg)

-- define the loss function to optimize
local function loss(x)
    local output = model:forward(x)
    local f = criterion:forward(output, delta)
    model:backward(x, criterion:backward(output, delta))
    return f, model.gradInput
end

-- train the model
optimState = {
    learningRate = opts.LR,
    momentum = opts.momentum,
}
local gray = torch.mean(rgbImg, 1)
for i = 1, opts.iter do
    local _, err = opts.optimAlgo(loss, gray, optimState)
    print(err[1])
end

image.save(opts.oup, gray)