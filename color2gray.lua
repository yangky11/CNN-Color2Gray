require 'optim'


function color2gray(labImg, init, model, criterion, silent)

    silent = silent or false

    -- compute the distances
    require 'dist'
    local delta = dist(labImg)
    
    -- define the loss function to optimize
    local function loss(x)
        local output = model:forward(x)
        local f = criterion:forward(output, delta)
        model:backward(x, criterion:backward(output, delta))
        local grad = model.gradInput:clone()
        grad:add(-grad:mean())
        return f, grad
    end
    
    -- train the model
    optimState = {
        learningRate = opts.LR,
        momentum = opts.momentum,
    }
    local gray = init:clone()
    local errors = {}
    local cnt = 0
    for i = 1, opts.maxiter do
        local _, err = opts.optimAlgo(loss, gray, optimState)
        if #errors >= 1 and err[1] >= errors[#errors] then
            cnt = cnt + 1
            if cnt == 5 then
                if not silent then print('\nloss failed to decrease, stopped earlier') end
                break
            end
        end
        errors[#errors + 1] = err[1]
        if not silent then xlua.progress(i, opts.maxiter) end
    end
    if not silent then print(string.format('final loss: %f', errors[#errors])) end
    
    -- normalize the result
    gray:add(-gray:min())
    gray:div(gray:max())

    return gray
end