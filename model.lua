require 'nn'


function createModel()
    local conv
    if opts.cuda then
        require 'cudnn'
        conv = cudnn.SpatialConvolution(1, 8, 3, 3, 1, 1, 1, 1)
    else
        conv = nn.SpatialConvolution(1, 8, 3, 3, 1, 1, 1, 1)
    end
    
    conv.weight:zero()
    conv.weight:sub(1, -1, 1, -1, 2, 2, 2, 2):add(1)
    conv.weight[{1, 1, 1, 1}] = -1
    conv.weight[{2, 1, 1, 2}] = -1
    conv.weight[{3, 1, 1, 3}] = -1
    conv.weight[{4, 1, 2, 1}] = -1
    conv.weight[{5, 1, 2, 3}] = -1
    conv.weight[{6, 1, 3, 1}] = -1
    conv.weight[{7, 1, 3, 2}] = -1
    conv.weight[{8, 1, 3, 3}] = -1
    conv.bias:zero()
    if opts.cuda then
        conv:resetWeightDescriptors() 
    end
    
    local criterion = nn.MSECriterion(true)
    if opts.cuda then
        conv:cuda()
        criterion:cuda()
    end
    
    return conv, criterion
end
