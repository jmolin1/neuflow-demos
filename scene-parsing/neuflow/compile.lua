#!/usr/bin/env torch
-- this script compiles a SpatialFovea network for neuFlow
-- should be defined globally before calling:
--  + network
--  + opt.width
--  + opt.height

-- init neuflow
require 'neuflow'
require 'xlua'

xrequire('nnx',true)

-- use packers
require 'neuflow/PyramidPacker'
require 'neuflow/PyramidUnPacker'

----------------------------------------------------------------------
-- INIT: initialize the neuFlow context
-- a mem manager, the dataflow core, and the compiler
--
-- ARGS: parse user arguments


--local platform = 'xilinx_ml605'
local platform = 'xilinx_ml605_tbsp'
nf = neuflow.init {
--  global_msg_level = 'detailled',
   prog_name   = 'scene-parsing',
   platform    = platform,
}

-- channels
nchannels = 3

-- get scales
scales = {}
for i,ratio in ipairs(network.modules[1].ratios) do
   table.insert(scales, 1/ratio)
end

-- create all submodules
convnet=foveanet.processors[1]


convnet.modules[3]=nn.SpatialSubSampling(16,2,2,2,2)
convnet.modules[6]=nn.SpatialSubSampling(64,2,2,2,2)
convnet.modules[3].bias:fill(0)
convnet.modules[3].weight:fill(1)
convnet.modules[6].bias:fill(0)
convnet.modules[6].weight:fill(1)

preproc = foveanet.preProcessors[1]
packer = nn.PyramidPacker(convnet, scales, true)
unpacker = nn.PyramidUnPacker(convnet)

-- generate fake input (with correct dimension)
pyramid = torch.Tensor(nchannels,opt.height/opt.downsampling,opt.width/opt.downsampling)
pyramid = packer:forward(pyramid)
pyramid:resize(nchannels, pyramid:size(2)+1, pyramid:size(3)) -- pad for round subsamples
pyramid:zero()
stackedfeatures = torch.Tensor()

-- compile convnet for neuflow
nf:beginLoop('main') do

   input_dev = nf:copyFromHost(pyramid)
   output_dev = nf:compile(convnet, input_dev)
   outputs = nf:copyToHost(output_dev)

end nf:endLoop('main')

-- load code
nf:sendReset()
nf:loadBytecode()


-- generate forward function
foveanet = {}
foveanet.forward
   = function(self,inp)
        -- (1)
        p:start('pyramid')
        local rpyramid, coordinates = packer:forward(inp)
        p:lap('pyramid')

        -- (2)
        p:start('normalize')
        local normed = preproc:forward(rpyramid)
        pyramid:resize(nchannels,normed:size(2)+1,normed:size(3)):zero()
        pyramid:narrow(2,1,normed:size(2)):copy(normed)
        p:lap('normalize')
        -- (3)
        p:start('network inference')

        debug = 0
        if  debug==1 then
          outputs = convnet:forward(pyramid)
        else
           nf:copyToDev(pyramid)
           nf:copyFromDev(outputs)
        end
        p:lap('network inference')

        -- (4)
        p:start('unpack')
        local features = unpacker:forward(outputs, coordinates)
        stackedfeatures:resize(features[1]:size(1)*#features,
                               features[1]:size(2), features[1]:size(3)):zero()
        for i = 1,#features do
           if i == 1 then
              stackedfeatures:narrow(1,(i-1)*features[1]:size(1)+1,features[1]:size(1)):copy(
                 features[i])
           else
              image.scale(features[i],
                          stackedfeatures:narrow(1,(i-1)*features[1]:size(1)+1,features[1]:size(1)),
                          'simple')
           end
        end
        p:lap('unpack')
        return stackedfeatures
     end


