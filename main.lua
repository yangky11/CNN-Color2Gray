require 'torch'
require 'paths'
paths.dofile('opts.lua')
if opts.cuda then
    require 'cutorch'
end
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

-- create the model and training criterion
paths.dofile('model.lua')
model, criterion = createModel()

require 'color2gray'
if opts.webcam then
    local cv = require 'cv'
    require 'cv.highgui'
    require 'cv.videoio'
    local cap = cv.VideoCapture({device=0})
    assert(cap:isOpened(), 'failed to open the camera')
    cv.namedWindow({winname='CNN-Color2Gray', flags=cv.WINDOW_AUTOSIZE})
    local _, frame = cap:read({})
    while true do       
        frame = frame:float()
        print(frame:size())
        os.exit()
        local gray = color2gray(frame, model, criterion, false); 
        --image.display(frame)
        --image.save(opts.oup, gray)
        print(gray:size())
        os.exit()
        --cv.imshow({winname='CNN-Color2Gray', image=gray})
        --cap:read({frame})
    end
else
    local rgbImg = image.load(opts.inp)
    print(string.format('image size: %dx%d\n------------------', rgbImg:size(2), rgbImg:size(3)))
    local gray = color2gray(rgbImg, model, criterion)
    print(string.format('the result saved to %s', opts.oup))
    image.save(opts.oup, gray)
    image.display(rgbImg)
    image.display(gray)
end
