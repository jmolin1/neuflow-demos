#!/usr/bin/env torch
--------------------------------------------------------------------------
-- Test your custom network on neuFlow
--
-- This script provides simple and easy interface to test your network
-- on neuFlow hardware. Either you can design a network 
-- by modifying this script or point out an existing network.
--------------------------------------------------------------------------
require 'neuflow'
require 'qt'
require 'qtwidget'
require 'xlua'
xrequire('inline',true)
xrequire('nnx',true)
xrequire('camera',true)
xrequire('image',true)

-- parse options
op = xlua.OptionParser('%prog [options]')
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, specify the camera index: /dev/videoIDX', default=0}
op:option{'-n', '--network', action='store', dest='network', 
          help='path to existing network', default='no-existing-network'}
opt,args = op:parse()

-- add profiler
profiler = xlua.Profiler()

-- user-defined variables
local platform  = 'xilinx_ml605'
--local platform  = 'pico_m503'
local prog_name = 'Network-on-neuFlow'
local ichannel  = 3
local iwidth    = 320
local iheight   = 240

--------------------------------------------------------------------------
-- init neuflow and define hardware loop ---------------------------------
if opt.network == 'no-existing-network' then
   -- design your custom network below -------------------------
   network = nn.Sequential()
   network:add(nn.SpatialConvolutionMap(nn.tables.random(ichannel,32,1),7,7))
   network:add(nn.Tanh())
   network:add(nn.Abs())
   network:add(nn.SpatialSubSampling(32,2,2,2,2))
   network:add(nn.SpatialConvolutionMap(nn.tables.random(32,32,4),7,7))
   network:add(nn.Tanh())
   network:add(nn.Abs())
   network:add(nn.SpatialSubSampling(32,2,2,2,2))
   -- design your custom network above -------------------------
else
   -- load an existing network
   network = torch.load(opt.network)
end

-- init an input size and a loop on neuFlow
input   = torch.Tensor(ichannel,iheight,iwidth) -- fix the input size of network
nf      = neuflow.init {prog_name = prog_name, platform = platform}
nf:beginLoop('main') do                         -- init loop on neuflow
   input_dev  = nf:copyFromHost(input)
   output_dev = nf:compile(network, input_dev)
   outputs    = nf:copyToHost(output_dev)
end nf:endLoop('main')

-- define 'forward' class to apply data to hardware
nf.forward = function(nf,input)
                nf:copyToDev(input)
                nf:copyFromDev(outputs)
                return outputs
             end

-- reset neuflow and execute bytecode
nf:sendReset()
nf:loadBytecode()

--------------------------------------------------------------------------
-- process the network and retrieve output ------------------------------- 

-- apply an input to a feed-forward network
camera = image.Camera{}
function process()
   img    = image.scale(camera:forward(), iwidth, iheight)
   profiler:start('on-board-processing')
   result = nf:forward(img)
   profiler:lap('on-board-processing')
end

-- display an output image on GUI
function display()
   win:gbegin()
   win:showpage()
   image.display{image=img, win=win}
   win:gend()
end

-- open GUI for a display
if not win then
   win = qtwidget.newwindow(input:size(3), input:size(2), prog_name)
end

-- repeat whole process with a fixed time interval
timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
qt.connect(timer,
           'timeout()',
           function()
              process()
              display()
              collectgarbage()
              timer:start()
           end)
timer:start()
