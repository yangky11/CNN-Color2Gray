function test(img, delta)
    assert(img:dim() == 2 and delta:dim() == 3 and delta:size(1) == 8
         and img:size(1) == delta:size(2) and img:size(2) == delta:size(3))
    local height = img:size(1)
    local width = img:size(2)
    for i = 1, 8 do
        if i == 1 then
            u, v = -1, -1
        elseif i == 2 then
            u, v = -1, 0
        elseif i == 3 then
            u, v = -1, 1
        elseif i == 4 then
            u, v = 0, -1
        elseif i == 5 then
            u, v = 0, 1
        elseif i == 6 then
            u, v = 1, -1
        elseif i == 7 then
            u, v = 1, 0
        else
            u, v = 1, 1
        end
        for j = 1, height do
            for k = 1, width do
                if 1 <= j + u and j + u <= height and 1 <= k + v and k + v <= height then
                    assert(delta[{i, j, k}] == img[{j, k}] - img[{j + u, k + v}]
                            , string.format('testing failed: %f ~= %f', delta[{i, j, k}], img[{j, k}] - img[{j + u, k + v}]))
                end
            end
        end    
    end
end