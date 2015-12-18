require 'torch'
require 'paths'
paths.dofile('opts.lua')
require 'image'

--require 'cunn'
torch.setdefaulttensortype('torch.FloatTensor')

local rgbImg = image.load(opts.inp)
local labImg = image.rgb2lab(rgbImg)--:cuda()
--local labImg = rgbImg
print(string.format('image size: %dx%d\n------------------', labImg:size(2), labImg:size(3)))

-- create the model and training criterion
paths.dofile('model.lua')
model, criterion = createModel()

require 'color2gray'
local gray = color2gray(labImg, torch.mean(rgbImg, 1), model, criterion)


print(string.format('the result saved to %s', opts.oup))
image.save(opts.oup, gray)
