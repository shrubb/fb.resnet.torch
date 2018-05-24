require 'nn'
require 'cudnn'

local SpatialConvolution = cudnn.SpatialConvolution
local SpatialDilatedConvolution = cudnn.SpatialDilatedConvolution
local SpatialFullConvolution = cudnn.SpatialFullConvolution
local ReLU = cudnn.ReLU
local SpatialBatchNormalization = cudnn.SpatialBatchNormalization
local SpatialMaxPooling = cudnn.SpatialMaxPooling

collectgarbage()

local function createModel(opt)
    local
    function invertedResidualBlock(nInputPlane, nOutputPlane, expansionFactor, stride)
        stride = stride or 1
        local dilation = 1
        local nBlockPlane = nInputPlane*expansionFactor

        local mainBranch = nn.Sequential()
        
        if nInputPlane ~= nBlockPlane then
        -- expand
        mainBranch
            :add(SpatialConvolution(nInputPlane, nBlockPlane, 1,1))
            :add(SpatialBatchNormalization(nBlockPlane))
            :add(ReLU(true))
        end

        mainBranch
            -- depthwise 3x3
            :add(SpatialDilatedConvolution(
                    nBlockPlane, nBlockPlane, 3,3, stride,stride,
                    dilation,dilation, dilation,dilation, nBlockPlane))
            :add(SpatialBatchNormalization(nBlockPlane))
            :add(ReLU(true))

            -- contract
            :add(SpatialConvolution(nBlockPlane, nOutputPlane, 1,1))
            :add(SpatialBatchNormalization(nOutputPlane))
        
        if stride == 1 and nInputPlane == nOutputPlane then
            return nn.Sequential()
                :add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(mainBranch))
                :add(nn.CAddTable())
        else
            return mainBranch
        end
    end

    local
    function invertedResidualBlockStage(nInputPlane, stageConfig)

        local expansionFactor, nOutputPlane, nBlocks, stride = table.unpack(stageConfig)
        local retval = nn.Sequential()

        for blockIdx = 1,nBlocks do
            retval:add(
                invertedResidualBlock(nInputPlane, nOutputPlane, expansionFactor, stride))

            nInputPlane = nOutputPlane
            -- only apply stride to first block
            if blockIdx == 1 then stride = 1 end
        end

        return retval
    end

    local model = nn.Sequential()
        :add(SpatialConvolution(3, 32, 3,3, 2,2, 1,1))
        :add(SpatialBatchNormalization(32))
        :add(ReLU(true))

    -- {expansionFactor, nOutputPlane, number of blocks, stride}
    local blocksConfig = {
        {1,  16, 1, 1},
        {6,  24, 2, 2},
        {6,  32, 3, 2},
        {6,  64, 4, 2},
        {6,  96, 3, 1},
        {6, 160, 3, 2},
        {6, 320, 1, 1},
    }

    local currentNPlanes, currentDownsampling = 32, 2

    for _, stageConfig in ipairs(blocksConfig) do
        model:add(invertedResidualBlockStage(currentNPlanes, stageConfig, currentDilationRate))
        currentNPlanes = stageConfig[2] -- nOutputPlane
        currentDownsampling = currentDownsampling * stageConfig[4] -- stride
    end

    -- ************************ functions for changing output stride ***********************

    function model:setOutputStride(outputStride)
        assert(outputStride == 8 or outputStride == 16 or outputStride == 32)

        local currentDownsampling, currentDilation = 2, 1
        local dilationGrid = {1,1,1,1,1,1,1,1,1,1}
        local previousBlockDilation = 1

        for stageIdx = 5,10 do
            local stage = self:get(stageIdx)
            assert(torch.type(stage) == 'nn.Sequential')

            local firstConv3x3 = stage:get(1):get(4)
            assert(firstConv3x3.kH == 3 and firstConv3x3.kW == 3)
            -- if the current stage changes downsampling rate (i.e., has `stageConfig[4] > 1`), then
            if firstConv3x3.dH > 1 or (firstConv3x3.dilationH or 1) > previousBlockDilation then
                -- (1) update our local tracking variables
                if currentDownsampling == outputStride then
                    -- if we have reached target downsampling rate, apply dilation instead of stride
                    currentDilation = currentDilation * 2
                else
                    currentDownsampling = currentDownsampling * 2
                end

                previousBlockDilation = firstConv3x3.dilationH
            end

            -- (2) possibly change current stage's stride/dilation
            if (firstConv3x3.dilationH or 1) ~= currentDilation then
                if currentDilation == 1 then
                    firstConv3x3.dH = 2
                    firstConv3x3.dW = 2
                    firstConv3x3.dilationH = 1
                    firstConv3x3.dilationW = 1
                    firstConv3x3.padH = 1
                    firstConv3x3.padW = 1
                else
                    assert(#stage <= #dilationGrid)
                    for blockIdx = 1,#stage do
                        local block = stage:get(blockIdx)
                        local conv3x3
                        if torch.type(block:get(1)) == 'nn.ConcatTable' then
                            conv3x3 = block:get(1):get(2):get(4)
                        else
                            conv3x3 = block:get(4)
                        end 
                        assert(torch.type(conv3x3):find('Convolution'))

                        conv3x3.dH = 1
                        conv3x3.dW = 1
                        conv3x3.dilationH = dilationGrid[blockIdx] * currentDilation
                        conv3x3.dilationW = dilationGrid[blockIdx] * currentDilation
                        conv3x3.padH = conv3x3.dilationH
                        conv3x3.padW = conv3x3.dilationW
                    end
                end
            end
        end

        return self
    end

    model:apply(
        function(m)
            if torch.typename(m):find('Convolution') and m.bias then m:noBias() end
            if torch.typename(m):find('Convolution') then
                local std = 0.09
                m.weight:apply(function()
                    local retval = 3*std
                    while math.abs(retval) > 2*std do
                        retval = torch.normal(0, std)
                    end
                    return retval
                end)
            end
            if torch.typename(m):find('BatchNorm') then m.momentum = 0.003 end
        end)

    local outputStride = 32
    model:setOutputStride(outputStride)

    if opt.dataset ~= 'cifar10' then
        model:add(cudnn.SpatialAveragePooling(7,7, 1,1))
    end
    model:add(nn.View(currentNPlanes):setNumInputDims(3))
    model:add(nn.Dropout(0.2))
    model:add(nn.Linear(currentNPlanes, 1000))
    model:add(cudnn.LogSoftMax())

    model:type(opt.tensorType)
    model:get(1).gradInput = nil
    return model
end

return createModel
