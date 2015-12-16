require 'math'


function dist(img)
    local deltaL = model:forward(img:narrow(1, 1, 1)):clone()
    local deltaA = model:forward(img:narrow(1, 2, 1)):clone()
    local deltaB = model:forward(img:narrow(1, 3, 1)):clone()
    if opts.test then
        paths.dofile('test.lua')
        test(img[1], deltaL)
        test(img[2], deltaA)
        test(img[3], deltaB)
    end
    local normDeltaC = torch.sqrt(torch.cmul(deltaA, deltaA) + torch.cmul(deltaB, deltaB))
    local function crunch(x) return torch.tanh(x / opts.alpha) * opts.alpha end
    local delta = deltaL:clone()
    local case23 = torch.lt(torch.abs(deltaL), crunch(normDeltaC))
    local sign = torch.sign(deltaA * torch.cos(opts.theta * math.pi / 180) + deltaB * torch.sin(opts.theta * math.pi / 180))
    delta[case23] = crunch(torch.cmul(sign, normDeltaC)[case23])
    return delta
end