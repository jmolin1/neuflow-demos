require 'neuflow'

-- user-defined variables
--local platform  = 'xilinx_ml605'
local platform  = 'pico_m503'
local prog_name = title 

-- network architecture
local custom_network = false
if custom_network then
   -- design your custom network below ------------
   network = nn.Sequential()
   network:add(nn.SpatialConvolution(3,32,7,7))
   network:add(nn.Tanh())
   --      ...
   ------------------------------------------------
else
   -- otherwise, load trained network
   network = torch.load('unsupervised.net')
end

local input   = torch.Tensor(3,options.height/options.downs,options.width/options.downs)
local nf      = neuflow.init {prog_name = prog_name, platform = platform}
nf:beginLoop('main') do
   input_dev  = nf:copyFromHost(input)
   output_dev = nf:compile(network, input_dev)
   outputs    = nf:copyToHost(output_dev)
end nf:endLoop('main')

nf.forward = function(nf,input)
                nf:copyToDev(input)
                nf:copyFromDev(outputs)
                local outputs_p = addpooler:forward(outputs)
                return outputs_p
             end

-- reset neuflow and execute bytecode
nf:sendReset()
nf:loadBytecode()

return nf
