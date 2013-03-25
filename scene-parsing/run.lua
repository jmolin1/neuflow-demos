#!/usr/bin/env torch -m
------------------------------------------------------------
-- a scene segmenter, base on a ConvNet trained end-to-end
-- to predict class distributions at dense locations.
--
-- Clement Farabet
--

require 'xlua'
require 'torch'
require 'qt'
require 'qtwidget'
require 'inline'
require 'ffmpeg'
require 'imgraph'
require 'nnx'
require 'segmtools'

xrequire('camera',true)
xrequire('image',true)

-- parse args
op = xlua.OptionParser('%prog [options]')
op:option{'-v', '--video', action='store', dest='video',
          help='video file to process'}
op:option{'-n', '--network', action='store', dest='network',
          help='path to existing [trained] network'}
op:option{'-s', '--save', action='store', dest='save',
          help='path to save segmented video'}
op:option{'-lib', '--ffmpeglib', action='store_true', 
          dest='useffmpeglib', default=false,
          help='use ffmpeglib module to read frames directly from video'}
op:option{'-k', '--seek', action='store', dest='seek',
          help='seek number of seconds', default=0}
op:option{'-f', '--fps', action='store', dest='fps',
          help='number of frames per second', default=10}
op:option{'-t', '--time', action='store', dest='seconds',
          help='length to process (in seconds)', default=10}
op:option{'-w', '--width', action='store', dest='width',
          help='resize video, width', default=320}
op:option{'-h', '--height', action='store', dest='height',
          help='resize video, height', default=256}
op:option{'-z', '--zoom', action='store', dest='zoom',
          help='display zoom', default=1}
op:option{'-tk', '--task', action='store', dest='task',
          help='determine which classes to use', default='stanford'}
op:option{'-m', '--method', action='store', dest='method',
          help='parsing method: dense | centroids', default='dense'}
op:option{'-nf', '--neuflow', action='store_true', dest='neuflow',
          help='compute convnet using neuflow', default=false}   --false
op:option{'-fst', '--fast', action='store_true', dest='fast',
          help='use all sorts of tricks to be the fastest possible', default=false}
op:option{'-cf', '--confidence', action='store', dest='confidence',
          help='min confidence to produce results', default=0.3}
op:option{'-ds', '--downsampling', action='store', dest='downsampling',
          help='downsample input frame for processing', default=2}
op:option{'-cr', '--crop', action='store', dest='crop',
          help='crop region in main image for processing', default=nil}
op:option{'-c', '--camera', action='store', dest='camidx',
          help='if source=camera, specify the camera index: /dev/videoIDX', default=0}
opt,args = op:parse()



opt.downsampling = tonumber(opt.downsampling)

offx = 0
offy = 0

if opt.crop then
   dofile('split.lua')
   opt.crop = split(opt.crop,',')
   for i = 1,4 do 
      opt.crop[i] = tonumber(opt.crop[i])
   end
   -- network needs input divisible by 16 
   -- FIXME make this internal and cleaner
   for i = 3,4 do 
      if not ((opt.crop[i] % 16) == 0) then
         opt.crop[i] = opt.crop[i] + 16 - (opt.crop[i] % 16)
      end
   end
   offx = opt.crop[1]
   offy = opt.crop[2]
end

-- summary
op:summarize()

-- do everything in float
torch.setdefaulttensortype('torch.FloatTensor')

-- use profiler
p = xlua.Profiler()

-- classes
if opt.task == 'stanford' then
   classes = {'unknown','sky','tree','road','grass','water','building',
              'mountain','object'}

   colormap = imgraph.colormap{[1]={0.0000, 0.0000, 0.0000},
                               [2]={0.0706, 0.7255, 0.9412},
                               [3]={0.2157, 0.8275, 0.1843},
                               [4]={0.3294, 0.6824, 0.4902},
                               [5]={0.2235, 0.9804, 0.0314},
                               [6]={0.2196, 0.3608, 0.7961},
                               [7]={0.7216, 0.7216, 0.2902},
                               [8]={0.6667, 0.4588, 0.3490},
                               [9]={0.9647, 0.0471, 0.3216}}

   defaultnet = 'stanford.net'

elseif opt.task == 'siftflow' then
   classes = {'unknown',
              'awning', 'balcony', 'bird', 'boat', 'bridge', 'building', 'bus',
              'car', 'cow', 'crosswalk', 'desert', 'door', 'fence', 'field',
              'grass', 'moon', 'mountain', 'person', 'plant', 'pole', 'river',
              'road', 'rock', 'sand', 'sea', 'sidewalk', 'sign', 'sky',
              'staircase', 'streetlight', 'sun', 'tree', 'window'}

   colormap = imgraph.colormap{[1] ={0.0, 0.0, 0.0},
                               [2] ={0.5, 0.5, 0.5}, -- awning
                               [3] ={0.9, 0.3, 0.3}, -- balcony
                               [4] ={0.8, 0.3, 0.2}, -- bird
                               [5] ={0.4, 0.4, 0.8}, -- boat
                               [6] ={0.5, 0.9, 0.9}, -- bridge
                               [7] ={0.7, 0.7, 0.3}, -- building
                               [8] ={0.4, 0.7, 0.8}, -- bus
                               [9] ={0.4, 0.4, 0.8}, -- car
                               [10]={0.8, 0.6, 0.6}, -- cow
                               [11]={0.9, 0.7, 0.9}, -- crosswalk
                               [12]={0.9, 0.9, 0.5}, -- desert
                               [13]={0.5, 0.3, 0.0}, -- door
                               [14]={0.6, 0.5, 0.1}, -- fence
                               [15]={0.7, 0.7, 0.1}, -- field
                               [16]={0.0, 0.9, 0.0}, -- grass
                               [17]={0.0, 0.2, 0.2}, -- moon
                               [18]={0.7, 0.5, 0.3}, -- mountain
                               [19]={1.0, 0.0, 0.3}, -- person
                               [20]={0.3, 0.7, 0.1}, -- plant
                               [21]={0.4, 0.2, 0.2}, -- pole
                               [22]={0.1, 0.4, 0.9}, -- river
                               [23]={0.3, 0.3, 0.3}, -- road
                               [24]={0.5, 0.4, 0.2}, -- rock
                               [25]={0.8, 0.8, 0.5}, -- sand
                               [26]={0.1, 0.1, 0.9}, -- sea
                               [27]={0.5, 0.5, 0.5}, -- sidewalk
                               [28]={1.0, 0.1, 0.1}, -- sign
                               [29]={0.0, 0.7, 0.9}, -- sky
                               [30]={0.9, 0.4, 0.3}, -- staircase
                               [31]={0.1, 1.0, 0.1}, -- streetlight
                               [32]={1.0, 1.0, 0.0}, -- sun
                               [33]={0.2, 0.8, 0.1}, -- tree
                               [34]={0.1, 0.6, 1.0}} -- window

   defaultnet = 'siftflow.net'
else
   error 'unknown task'
end

-- fast?
if opt.fast then
   opt.method = 'centroids'
end

-- load pre-trained network from disk
network = torch.load(opt.network or defaultnet)
network:type(torch.getdefaulttensortype())

-- replace classifier (2nd module) by SpatialClassifier
foveanet = network.modules[1]
classifier1 = network.modules[2]
classifier = nn.SpatialClassifier(classifier1)
network.modules[2] = classifier

-- neuflow?
if opt.neuflow then require 'neuflow/compile' end

-- load video
if opt.video then
   if opt.video:find('jpg') or opt.video:find('png') then
      local i = image.load(opt.video)
      i = image.scale(i, opt.width, opt.height)
      video = {}
      video.forward = function()
                        return i
                      end
   elseif opt.video:find('.lua$') then
      -- pass a script as the video for live cameras.  Allows for
      -- complicated stitching and simple single camera scenarios.
      dofile(opt.video) 
      elseif opt.useffmpeglib then
      print("Using ffmpeglib")
      require 'ffmpeglib'
      ffmpeglib.init()
      video = {}
      video.fp = ffmpeg.open(opt.video,opt.width,opt.height)
      video.frame = torch.Tensor()
      video.nframes = 0
      video.forward = 
         function ()
             video.nframes = video.nframes + 1 
             video.frame.ffmpeg.getFrame(video.fp,video.frame) 
             return video.frame
         end 
   else
      -- old style video
      video = ffmpeg.Video{path=opt.video, 
                        width=opt.width, height=opt.height,
                        fps=opt.fps, length=opt.seconds, seek=opt.seek, 
                        encoding='jpg',
                        delete=false}
   end
else 
  camera = image.Camera{}
end

-- setup GUI (external UI file)
if not win or not widget then
   if tonumber(opt.width) > 2*opt.height then
      win = qtwidget.newwindow(opt.width*opt.zoom, opt.height*opt.zoom*2,
                               'End-to-End Learned Scene Segmenter')
   else
      win = qtwidget.newwindow(opt.width*opt.zoom*2, opt.height*opt.zoom,
                               'End-to-End Learned Scene Segmenter')
   end
end

-- gaussian (a gaussian, really, is always useful)
gaussian = image.gaussian(3)

-- softmax
softmax = nn.SoftMax()

-- process function
function process()
   -- (1) grab frame
   p:start('get next frame')
   --fframe = video:forward()
   if opt.video then
      fframe = video:forward()
   else
      fframe = camera:forward()
   end

   local width = opt.width
   local height = opt.height
   if opt.crop then
      width = opt.crop[3]
      height = opt.crop[4]
      cframe = 
         fframe:narrow(3,opt.crop[1],width):narrow(2,opt.crop[2],height) 
   else 
      cframe = fframe
   end
   if opt.downsampling > 1 then
      width  = width/opt.downsampling
      height = height/opt.downsampling
      frame = image.scale(cframe, width, height)
   else
      frame = cframe:clone()
   end
   p:lap('get next frame')
   -- (2) process frame through foveanet
   p:start('extract features')
   features = foveanet:forward(frame)
   p:lap('extract features')

   -- (3) compute graph on input image
   p:start('construct and cut edge-weighted graph')
   frame_smoothed = image.convolve(frame, gaussian, 'same')
   if opt.fast then
      frame_smoothed = image.scale(frame_smoothed, frame_smoothed:size(3)/2, frame_smoothed:size(2)/2)
      graph = imgraph.graph(frame_smoothed)
      mstsegm = imgraph.segmentmst(graph, 2, 50)
      mstsegm = image.scale(mstsegm, frame:size(3), frame:size(2), 'simple')
   else
      graph = imgraph.graph(frame_smoothed)
      mstsegm = imgraph.segmentmst(graph, 2, 50)
   end
   p:lap('construct and cut edge-weighted graph')

   -- (3) compute class distributions, either densely, or per component
   if opt.method == 'dense' then
      -- (a) compute class distributions
      p:start('compute class distributions')
      distributions = classifier:forward(features)
      p:lap('compute class distributions')

      -- crap
      distributions = nn.SpatialClassifier(nn.SoftMax()):forward(distributions)
      --for _,i in ipairs{3,4,6,10,12,14,15,16,17,18,22,24,25,26,9,28} do
      --   distributions[i]:mul(0.05)
      --end

      -- (b) upsample the distributions
      p:start('rescale distributions')
      distributions = image.scale(distributions, frame:size(3), frame:size(2), 'simple')
      p:lap('rescale distributions')

      -- (c) pool the predicted distributions into the segmentation
      p:start('delineate distributions')
      distributions, components, hcomponents = imgraph.histpooling(distributions, mstsegm, true)
      p:lap('delineate distributions')

      -- (d) winner take all
      p:start('winner take-all')
      _,winners = torch.max(distributions,1)
      winner = winners[1]
      p:lap('winner take-all')

   elseif opt.method == 'centroids' then
      -- (a) resize segmentation & labelmap to feature map size
      p:start('rescale segmentation')
      ssegm = image.scale(mstsegm, features:size(3), features:size(2), 'simple')
      p:lap('rescale segmentation')

      -- (b) extract components from feature maps
      p:start('extract components info')
      components = imgraph.extractcomponents(ssegm)
      p:lap('extract components info')

      -- (c) process each component through classifier
      p:start('classify components')
      components.id2class = {}
      components.id2confidence = {}
      --components.id2distribution = {}
      featvectors = featvectors or torch.Tensor()
      for i = 1,components:size() do
         local vector = features:select(3, components.centroid_x[i]):select(2, components.centroid_y[i])
         featvectors:resize(components:size(), vector:size(1))
         featvectors[i] = vector
      end
      classvectors = softmax:forward(classifier1:forward(featvectors))
      for i = 1,components:size() do
         local max,argmax = classvectors[i]:max(1)
         argmax = argmax[1]; max = max[1]
         if max > opt.confidence then
            components.id2class[components.id[i]] = {argmax}
            components.id2confidence[components.id[i]] = {max}
            --components.id2distribution[components.id[i]] = classvectors[i]
         else
            components.id2class[components.id[i]] = {1}
            components.id2confidence[components.id[i]] = {0}
            --components.id2distribution[components.id[i]] = classvectors[i]
         end
      end
      -- make sure the max ID exists in those tables (the reason it
      -- can sometimes not exist is because we extract them from the
      -- downsampled segm)
      maxid = mstsegm:max()
      components.id2class[maxid] = components.id2class[maxid] or {1}
      components.id2confidence[maxid] = components.id2confidence[maxid] or {0}
      --components.id2distribution[maxid] = components.id2distribution[maxid] or randn(#classes)
      p:lap('classify components')

      -- (d) generate final parse
      p:start('colorize segmentation (parse)')
      winner = imgraph.colorize(mstsegm, imgraph.colormap(components.id2class, 1))[1]
      confmap = imgraph.colorize(mstsegm, imgraph.colormap(components.id2confidence, 1))[1]
      --distributions = imgraph.colorize(mstsegm, imgraph.colormap(components.id2distribution))
      p:lap('colorize segmentation (parse)')

   else
      error('<parser> unknown method: ' .. opt.method .. ' must be one of: dense | centroids')
   end
   collectgarbage()
end

-- display function
function display()
   -- (1) colorize classes
   colored, colormap = imgraph.colorize(winner, colormap)

   -- (1b) if confidence map, then multiply
   if confmap then
      colored = image.rgb2hsl(colored)
      colored[2]:cmul(confmap)
      colored[3]:cmul(confmap)
      colored = image.hsl2rgb(colored)
   end
   -- (2) display raw input
   if frame:size(3) > 2*frame:size(2) then
      image.display{image=fframe, win=win, y=fframe:size(2)*opt.zoom,
                    zoom=opt.zoom, min=0, max=1}
   else
      image.display{image=fframe, win=win, x=fframe:size(3)*opt.zoom,
                    zoom=opt.zoom, min=0, max=1}
   end

   -- map just the processed part back into the whole image
   if opt.crop then
      if opt.downsampling > 1 then
         colored = image.scale(colored,opt.crop[3],opt.crop[4])
      end 
      local tmp = torch.Tensor():resizeAs(fframe):fill(0.5)
      local ctmp = tmp:narrow(3,opt.crop[1],opt.crop[3]):narrow(2,opt.crop[2],opt.crop[4])
      ctmp:copy(colored)
      colored = tmp
   else
      if opt.downsampling > 1 then
         colored = image.scale(colored,fframe:size(3),fframe:size(2))
      end 
   end


   if opt.multiplicative then
      colored:mul(0.2)
      colored:cmul(fframe)
   else
      colored:add(fframe)
   end
   -- (3) overlay segmentation on input frame
   image.display{image=colored, win=win,
                 zoom=opt.zoom, min=0, max=colored:max()}

   -- (4) print classes
   if opt.method == 'centroids' then
      for i = 1,components:size() do
         components.centroid_x[i] = components.centroid_x[i] * 4
         components.centroid_y[i] = components.centroid_y[i] * 4
         components.surface[i] = components.surface[i] * 16
      end
   end
   segmtools.overlayclasses{win=win, classes=classes, 
                            components=components,
                            zoom=opt.zoom*opt.downsampling, 
                            minsize=200, font=12,
                            offx = offx, offy = offy
                         }

   -- (5) save ?
   if opt.save then
      local t = win:image():toTensor(3)
      local fname = opt.save .. string.format('/frame_%05d.jpg',times)
      sys.execute(string.format('mkdir -p %s',opt.save))
      print('saving:'..fname)
      image.save(fname,t)
   end
end

-- setup gui
timer = qt.QTimer()
timer.interval = 10
timer.singleShot = true
times = 0
qt.connect(timer,
           'timeout()',
           function()
              print('=== processing new frame ===')
              p:start('total','fps')
              times = times + 1
              p:start('process')
              process()
              p:lap('process')
              p:start('display')
              win:gbegin()
              win:showpage()
              display()
              win:gend()
              p:lap('display')
              timer:start()
              collectgarbage()
              p:lap('total')
              p:printAll()
              print('')
           end)
timer:start()
