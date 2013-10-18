#!/usr/bin/env torch
------------------------------------------------------------
-- Proto-object based saliency algorithm
--
-- Code written by: Jamal Molin
--

require 'torch'
require 'image'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'nn'
require 'inline'
require 'camera'
require 'neuflow'

----------------------
--Set default tensor--
----------------------
torch.setdefaulttensortype('torch.DoubleTensor')

--------------------------
--Initialize filter size--
--------------------------
MAX_FILTER_SIZE = 9;

-----------------
--SETUP NEUFLOW--
-----------------
--local platform = arg[1] or 'pico_m503'
local platform = 'pico_m503'
nf = neuflow.init {
   prog_name   = 'proto-object',
   platform    = platform
}

-------------------------------------------------------------------
--------------------------HELPER FUNCTIONS-------------------------
-------------------------------------------------------------------
--function: Write to File
function writeToFile(data,filename)
   file = torch.DiskFile('/home/jamal/Dropbox/ProtoObject/ProtoObjectStatic_noresize/' .. filename, 'w')
   file:writeObject(data:size())
   file:writeObject(data)
   file:close()
   print('Data written to file: ' .. filename);
end

--function: Pre-Pad Image
function prePadImage(orig_img,kRows,kCols,padValue)
   pad_rows = math.floor(kRows / 2);
   pad_cols = math.floor(kCols / 2);
   
   --initially pad with zeros
   local padder = nn.SpatialZeroPadding(pad_cols,pad_cols,pad_rows,pad_rows)
   padder:forward(orig_img);
   new_image = padder.output;
   
   if (padValue ~= nil) then
      new_image[1][{{},{1,pad_cols}}]:fill(padValue);
      new_image[1][{{},{new_image:size(3) - pad_cols + 1,new_image:size(3)}}]:fill(padValue);
      new_image[1][{{1,pad_rows},{}}]:fill(padValue);
      new_image[1][{{new_image:size(2) - pad_rows + 1,new_image:size(2)},{}}]:fill(padValue);
   else
      
      --pad left/right/top/bottom
      first_col = new_image[1][{{},{pad_cols + 1}}];
      cols_to_pad = torch.expand(first_col,first_col:size(1),pad_cols);
      new_image[1][{{},{1,pad_cols}}] = cols_to_pad:clone();
      
      last_col = new_image[1][{{},{new_image:size(3) - pad_cols}} ];
      cols_to_pad = torch.expand(last_col,last_col:size(1),pad_cols);
      new_image[1][{{},{new_image:size(3) - pad_cols + 1,new_image:size(3)}}] = cols_to_pad:clone();
      
      first_row = new_image[1][{{pad_rows + 1},{}}];
      rows_to_pad = torch.expand(first_row,pad_rows,first_row:size(2));
      new_image[1][{{1,pad_rows},{}}] = rows_to_pad:clone();
      
      last_row = new_image[1][{{new_image:size(2) - pad_rows},{}}];
      rows_to_pad = torch.expand(last_row,pad_rows,last_row:size(2));
      new_image[1][{{new_image:size(2) - pad_rows + 1,new_image:size(2)},{}}] = rows_to_pad:clone();
   end

   return new_image:clone();
end

--function: Degrees to Radians
function deg2rad(angleInDegrees)
   local angleInRadians = torch.mul(angleInDegrees,math.pi/180);
   return angleInRadians;
end

--function: Radians to Degrees
function rad2deg(angleInRadians)
   local angleInDegrees = angleInRadians  * (180/math.pi);
   return angleInDegrees;
end

--function: Normalize Image
function normalizeImage(im,range)
   if(range == nil) then
      range = torch.DoubleTensor({0,1});
   end
   if ( (range[1] == 0) and (range[2] == 0) ) then
      res = torch.DoubleTensor(im);
      return res;
   else
      local mx = torch.max(im);
      local mn = torch.min(im);
      
      if(mx == mn) then
         if mx == 0 then
            res_im = torch.mul(im,0);
         else
            res_im = im - mx + (0.5 * torch.sum(range));
         end
      else
         res_im = (torch.div((im - mn),(mx - mn)) * math.abs(range[2] - range[1])) + torch.min(range);
      end
      return res_im;
   end
end

--function: Safe Divide
function safeDivide(arg1, arg2)
   
   ze = torch.eq(arg2,0):clone();
   arg2[ze] = 1;
   result = torch.cdiv(arg1,arg2):clone();
   result[ze] = 0
   return result:clone();

end

--function: Make Colors
function makeColors(im)

   local r = torch.DoubleTensor(im[1]);
   local g = torch.DoubleTensor(im[2]);
   local b = torch.DoubleTensor(im[3]);
   
   
   local gray = torch.div(torch.add(torch.add(r, g),b), 3);

   local msk = torch.DoubleTensor(gray):clone();
   msk[torch.lt(msk,torch.max(gray) * 0.1)] = 0;
   msk[torch.ne(msk,0)]=1;
  
   r = safeDivide(torch.cmul(r,msk),gray:clone());
   g = safeDivide(torch.cmul(g,msk),gray:clone());
   b = safeDivide(torch.cmul(b,msk),gray:clone());

   local R = r - torch.div(g + b,2);
   R[torch.lt(R,0)] = 0;

   local G = g - torch.div(r + b,2);
   G[torch.lt(G,0)] = 0;
   
   local B = b - torch.div(r + g,2);
   B[torch.lt(B,0)] = 0;

   local Y = torch.div(r + g,2) - torch.div(torch.abs(r - g),2) - b;   
   Y[torch.lt(Y,0)] = 0;   

   return gray:clone(),R:clone(),G:clone(),B:clone(),Y:clone();

end

--function: Generate Channels
function generateChannels(img, params, fil_vals)
   --Get different feature channels
   gray,R,G,B,Y =  makeColors(img);
   
   --Generate color opponency channels
   local rg = R - G;
   local by = B - Y;
   local gr = G - R;
   local yb = Y - B;
      
   --Threshold opponency channels
   rg[torch.lt(rg,0)] = 0;
   gr[torch.lt(gr,0)] = 0;
   by[torch.lt(by,0)] = 0;
   yb[torch.lt(yb,0)] = 0;

   local return_result = {};

   --Set channels
   for c = 1,#defaultParameters.channels do
      --Intensity Channel
      if defaultParameters.channels:sub(c,c) == 'I' then
         return_result[c] = {type = 'Intensity', subtype = { {data = torch.DoubleTensor(gray:size()):copy(gray), type =  'Intensity'} } }

      --Color Channel
      elseif defaultParameters.channels:sub(c,c) == 'C' then
            return_result[c] = {type = 'Color', subtype = { {data = torch.DoubleTensor(rg:size()):copy(rg), type = 'Red-Green Opponency'}, 
                                                            {data = torch.DoubleTensor(gr:size()):copy(gr), type = 'Green-Red Opponency'},
                                                            {data = torch.DoubleTensor(by:size()):copy(by), type = 'Blue-Yellow Opponency'},
                                                            {data = torch.DoubleTensor(yb:size()):copy(yb), type = 'Yellow-Blue Opponency'} } }
      --Orientation Channel
      elseif defaultParameters.channels:sub(c,c) == 'O' then
         
            return_result[c] = {type = 'Orientation', subtype = { {data = torch.DoubleTensor(gray:size()):copy(gray), ori = 0, type = 'Orientation', filval = fil_vals[1][1]}, 
                                                                  {data = torch.DoubleTensor(gray:size()):copy(gray), ori = math.pi/4, type = 'Orientation', filval = fil_vals[1][2]},
                                                                  {data = torch.DoubleTensor(gray:size()):copy(gray), ori = math.pi/2, type = 'Orientation', filval = fil_vals[1][3]},
                                                                  {data = torch.DoubleTensor(gray:size()):copy(gray), ori = 3 * math.pi/4, type = 'Orientation', filval = fil_vals[1][4]} } }
      end
   end
   
   return return_result;
   
end

--function: Make Pyramid
function makePyramid(img, params)
   local depth = params.maxLevel;
   local pyr = {};
   local curr_width = img:size(2);
   local curr_height = img:size(1);
   
   pyr[1] = {data = img:clone()};
   for level = 2,depth do
      if params.csPrs.downSample == 'half' then
         curr_width = math.ceil(curr_width * 0.7071)
         curr_height = math.ceil(curr_height * 0.7071)
--         pyr[level] = {data = image.scale(pyr[level-1].data:clone(),curr_width,curr_height,'simple')};
         pyr[level] = {data = torch.add(torch.mul(pyr[level-1].data:clone(), 1/level),2)};
      elseif params.csPrs.downSample == 'full' then
         curr_width = math.ceil(curr_width * 0.5)
         curr_height = math.ceil(curr_height * 0.5)
--         pyr[level] = {data = image.scale(pyr[level-1].data:clone(),curr_width,curr_height,'simple')};
         pyr[level] = {data = torch.add(torch.mul(pyr[level-1].data:clone(), 1/level),2)};

      else
         print('Please specify if downsampling should be half or full octave');
      end
      
   end
   --image.display{image = pyr[6].data};
   return pyr;
   
end

--function Mesh Grid
function meshGrid(x_array,y_array)
   
   --Get size
   xsize = torch.numel(x_array);
   ysize = torch.numel(y_array);

   --Meshgrid
   x = torch.expand(x_array,ysize,xsize):clone();
   y = torch.expand(y_array,ysize,xsize):clone();
   
   return x,y;
end

--function: Make Even Orientation Cells
function makeEvenOrientationCells(theta,lambda,sigma,gamma)
   --Decode inputs
   local sigma_x = sigma;
   local sigma_y = sigma/gamma;
   
   --Bounding box
   local nstds = 1;
   local xmax = math.max(math.abs(nstds*sigma_x*math.cos(math.pi/2-theta)),math.abs(nstds*sigma_y*math.sin(math.pi/2-theta)));
   xmax = math.ceil(math.max(1,xmax));
   local ymax = math.max(math.abs(nstds*sigma_x*math.sin(math.pi/2-theta)),math.abs(nstds*sigma_y*math.cos(math.pi/2-theta)));
   ymax = math.ceil(math.max(1,ymax));
   
   --Meshgrid
   local xmin = -xmax;
   local ymin = -ymax;
   local xsize = xmax*2 + 1;
   local ysize = ymax*2 + 1;
   local x_array = torch.DoubleTensor(1,xsize);
   local i = xmin-1; x_array:apply(function() i=i+1;return i end);
   local y_array = torch.DoubleTensor(ysize,1);
   local i = ymin-1; y_array:apply(function() i=i+1;return i end);
   x,y = meshGrid(x_array,y_array);

   --Rotation
   local x_theta = torch.add(torch.mul(x,math.cos(math.pi/2-theta)), torch.mul(y,math.sin(math.pi/2-theta)));
   local y_theta = torch.add(torch.mul(x,-1*math.sin(math.pi/2-theta)), torch.mul(y,math.cos(math.pi/2-theta)));

   local msk = torch.exp(torch.add(torch.div(torch.pow(x_theta,2),sigma_x^2),torch.div(torch.pow(y_theta,2),sigma_y^2)):mul(-0.5)):mul(1/(2*math.pi*sigma_x*sigma_y)):cmul(torch.cos(torch.mul(x_theta,2*math.pi/lambda)));
   msk = msk - torch.mean(msk);
   
   return msk;
end

--function: Edge Even Pyramid
function edgeEvenPyramid(map,params,fil_vals)
   local prs = params.evenCellsPrs;
   
   --Initialize newMap
   local newMap = {};
   for i = prs.minLevel,prs.maxLevel do
      newMap[i] = { orientation = {} };
   end
   for level = prs.minLevel,prs.maxLevel do
      temp_data = torch.DoubleTensor(map[level].data):clone();
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      
      input = torch.squeeze(temp_data:clone());
      timerr = torch.Timer();
      nf:copyToDev(input)
      nf:copyFromDev(outputs)
      print('Time elapse = ' .. timerr:time().real);

      for local_ori = 1,prs.numOri do
         --Evmsk = makeEvenOrientationCells(prs.oris[local_ori],prs.lambda,prs.sigma,prs.gamma);
         --setup filter
         --module = nn.SpatialConvolution(1,1,Evmsk:size()[2],Evmsk:size()[1],1,1);         
         --module.bias = torch.zero(module.bias);
         --module.weight[1] = torch.DoubleTensor(1,Evmsk:size()[2],Evmsk:size()[1]):copy(Evmsk);         
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
        
         --module:forward(temp_data);
         
         --h = outputs[2]:clone();
         --image.display{image = h}
         --writeToFile(h,'luaimage.txt')
         --print('outputs',outputs[1]:size());
         --print(fil_vals[1][local_ori]);
         --val = fil_vals[1][local_ori];
         newMap[level].orientation[local_ori] = {ori = prs.oris[local_ori], data =  torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals[1][local_ori]])};
         --print(outputs[1])
         --print('asdfasdf;lakjsd');
         --val[32] = 3
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
      end
   end
   --image.display{image = newMap[2].orientation[1].data}
   return newMap;

end

--function: Make Odd Orientation Cells
function makeOddOrientationCells(theta,lambda,sigma,gamma)
   --Decode inputs
   local sigma_x = sigma;
   local sigma_y = sigma/gamma;
   
   --Bounding box
   local nstds = 1;
   local xmax = math.max(math.abs(nstds*sigma_x*math.cos(math.pi/2-theta)),math.abs(nstds*sigma_y*math.sin(math.pi/2-theta)));
   xmax = math.ceil(math.max(1,xmax));
   local ymax = math.max(math.abs(nstds*sigma_x*math.sin(math.pi/2-theta)),math.abs(nstds*sigma_y*math.cos(math.pi/2-theta)));
   ymax = math.ceil(math.max(1,ymax));
   
   --Meshgrid
   local xmin = -xmax;
   local ymin = -ymax;
   local xsize = xmax*2 + 1;
   local ysize = ymax*2 + 1;
   local x_array = torch.DoubleTensor(1,xsize);
   local i = xmin-1; x_array:apply(function() i=i+1;return i end);
   local y_array = torch.DoubleTensor(ysize,1);
   local i = ymin-1; y_array:apply(function() i=i+1;return i end);
   x,y = meshGrid(x_array,y_array);
   
   --Rotation
   local x_theta = torch.add(torch.mul(x,math.cos(math.pi/2-theta)), torch.mul(y,math.sin(math.pi/2-theta)));
   local y_theta = torch.add(torch.mul(x,-1*math.sin(math.pi/2-theta)), torch.mul(y,math.cos(math.pi/2-theta)));

   local msk1 = torch.exp(torch.add(torch.div(torch.pow(x_theta,2),sigma_x^2),torch.div(torch.pow(y_theta,2),sigma_y^2)):mul(-0.5)):mul(1/(2*math.pi*sigma_x*sigma_y)):cmul(torch.sin(torch.mul(x_theta,2*math.pi/lambda)));
   msk1 = torch.DoubleTensor(msk1 - torch.mean(msk1)):clone();
   
   local msk2 = torch.exp(torch.add(torch.div(torch.pow(x_theta,2),sigma_x^2),torch.div(torch.pow(y_theta,2),sigma_y^2)):mul(-0.5)):mul(1/(2*math.pi*sigma_x*sigma_y)):cmul(torch.sin(torch.mul(x_theta,2*math.pi/lambda) + math.pi));
   msk2 = torch.DoubleTensor(msk2 - torch.mean(msk2)):clone();
   
   return msk1, msk2;
end

--function: Edge Odd Pyramid
function edgeOddPyramid(map,params,fil_vals)
   local prs = params.oddCellsPrs;
   
   --Initialize newMap
   local newMap1 = {}
   local newMap2 = {}
   
   for i = prs.minLevel,prs.maxLevel do
      newMap1[i] = { orientation = {} }
      newMap2[i] = { orientation = {} }
   end
   for level = prs.minLevel,prs.maxLevel do
      temp_data = torch.DoubleTensor(map[level].data):clone();
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      input = torch.squeeze(temp_data:clone());
      nf:copyToDev(input)
      nf:copyFromDev(outputs)
      --Oddmsk1, Oddmsk2 = makeOddOrientationCells(prs.oris[local_ori],prs.lambda,prs.sigma,prs.gamma);
      --setup filter
      --module = nn.SpatialConvolution(1,1,Oddmsk1:size()[2],Oddmsk1:size()[1],1,1);         
      --module.bias = torch.zero(module.bias);
      --module.weight[1] = torch.DoubleTensor(1,Oddmsk1:size()[2],Oddmsk1:size()[1]):copy(Oddmsk1);
      for local_ori = 1,prs.numOri do


         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         --temp_data = torch.DoubleTensor(map[level].data):clone();
         --temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),FILTER_SIZE_PAD,FILTER_SIZE_PAD);
         --module:forward(temp_data);
         
         --input = temp_data:clone();
         --image.display{image = input}
         --writeToFile(outputs[2]:clone(),'luaimage.txt');
         newMap1[level].orientation[local_ori] = {ori = prs.oris[local_ori], data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals[1][local_ori]])};
         --newMap2[level].orientation[local_ori] = {ori = prs.oris[local_ori], data = {1,2;3,4} };
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
      end
   end
   
   return newMap1,newMap2;
end


--function: Make Complex Edge
function makeComplexEdge(EPyr, OPyr)
   local cPyr = {};

   for i = 1,#EPyr do
      cPyr[i] = { orientation = {} }
   end

   for level = 1,#EPyr do
      for local_ori = 1,#EPyr[level].orientation do
         cPyr[level].orientation[local_ori] = {data = torch.sqrt(torch.pow(torch.Tensor(EPyr[level].orientation[local_ori].data),2) + torch.pow(torch.Tensor(OPyr[level].orientation[local_ori].data),2)):clone() };
      end
   end
   
   return cPyr;
end

--function: Gabor Pyramid
function gaborPyramid(pyr,ori,params,fil_val)
   depth = params.maxLevel;
   gaborPrs = params.gaborPrs;
   --Evmsk = makeEvenOrientationCells(ori,gaborPrs.lambda,gaborPrs.sigma,gaborPrs.gamma);
   local gaborPyr = {};
   for i = 1,depth do
      gaborPyr[i] = { data = {} }
   end
   --local module = nn.SpatialConvolution(1,1,Evmsk:size()[2],Evmsk:size()[1],1,1);         
   --module.bias = torch.zero(module.bias);
   --module.weight[1] = torch.DoubleTensor(1,Evmsk:size()[2],Evmsk:size()[1]):copy(Evmsk);
   for level = 1,depth do
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      temp_data = torch.DoubleTensor(pyr[level].data):clone();
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      input = torch.squeeze(temp_data:clone());
      nf:copyToDev(input)
      nf:copyFromDev(outputs)
--module:forward(temp_data);
      gaborPyr[level].data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_val]);
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
   end
   
   return gaborPyr;
end

--function Make Gauss
function makeGauss(dim1,dim2,sigma_1,sigma_2,theta)
   local x0 = 0;
   local y0 = 0;
   norm = 1;
   local msk = torch.DoubleTensor(dim1:size()[1],dim2:size()[1]):zero();
   
   local xmax = dim1[#dim1];
   local ymax = dim2[#dim2];
   
   local xmin = -xmax;
   local ymin = -ymax;
   local xsize = xmax*2 + 1;
   local ysize = ymax*2 + 1;
   local x_array = torch.DoubleTensor(1,xsize);
   local i = xmin-1; x_array:apply(function() i=i+1;return i end);
   local y_array = torch.DoubleTensor(ysize,1);
   local i = ymin-1; y_array:apply(function() i=i+1;return i end);
   
   X,Y = meshGrid(x_array,y_array);

   local a = math.cos(theta)^2/2/sigma_1^2 + math.sin(theta)^2/2/sigma_2^2;
   local b = -math.sin(theta)^2/4/sigma_1^2 + math.sin(2*theta)^2/4/sigma_2^2;
   local c = math.sin(theta)^2/2/sigma_1^2 + math.cos(theta)^2/2/sigma_2^2;
   
   if norm then
      msk = torch.mul(torch.exp(torch.mul(torch.mul(torch.pow(X-x0,2),a) + torch.mul(torch.cmul(X-x0,Y-y0),2*b) + torch.mul(torch.pow(Y-y0,2),c),-1)),1/(2*math.pi*sigma_1*sigma_2)) ;
      --print(msk)
   else
      msk = torch.exp(torch.mul(torch.mul(torch.pow(X-x0,2),a) + torch.mul(torch.cmul(X-x0,Y-y0),2*b) + torch.mul(torch.pow(Y-y0,2),c),-1));
   end
   
   return msk;
end

--function: Make Center Surround
function makeCenterSurround(std_center, std_surround)
   --get dimensions
   local center_dim = math.ceil(3*std_center);
   local surround_dim = math.ceil(3*std_surround);
   --create gaussians
   local idx_center = torch.range(-center_dim,center_dim,1);
   local idx_surround = torch.range(-surround_dim,surround_dim,1);
   local msk_center = makeGauss(idx_center,idx_center,std_center,std_center,0);
   local msk_surround = makeGauss(idx_surround,idx_surround,std_surround,std_surround,0);
   --difference of gaussians
   local msk = torch.mul(msk_surround,-1);
   local temp = torch.add(msk[{{surround_dim+1-center_dim,surround_dim+1+center_dim},{surround_dim+1-center_dim,surround_dim+1+center_dim}}]:clone(),msk_center:clone() );
   msk[{{surround_dim+1-center_dim,surround_dim+1+center_dim},{surround_dim+1-center_dim,surround_dim+1+center_dim}}] = temp:clone();
   msk = msk - (torch.sum(msk) / (msk:size()[1] * msk:size()[2]))

   return msk;
end

--function: CS Pyramid
function csPyramid(pyr,params,fil_val)
   depth = params.maxLevel;
   csPrs = params.csPrs;
   --CSmsk = makeCenterSurround(csPrs.inner,csPrs.outer);
   local csPyr = {};
   for i = 1,depth do
      csPyr[i] = { data = {} }
   end
   --local module = nn.SpatialConvolution(1,1,CSmsk:size()[2],CSmsk:size()[1],1,1);         
   --module.bias = torch.zero(module.bias);
   --module.weight[1] = torch.DoubleTensor(1,CSmsk:size()[2],CSmsk:size()[1]):copy(CSmsk);
   for level = 1,depth do
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      temp_data = torch.DoubleTensor(pyr[level].data):clone();
      temp_data = prePadImage(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
      input = torch.squeeze(temp_data:clone());
      nf:copyToDev(input)
      nf:copyFromDev(outputs)
      --module:forward(temp_data);
      --print('imageSize after',module.output:size())
      csPyr[level].data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_val]);
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
   end
   
   return csPyr;
end

--funciton: Separate Pyramids
function separatePyr(pyr)
   pyr1 = {};
   pyr2 = {};
   for i = 1,#pyr do
      pyr1[i] = { data = {} }
      pyr2[i] = { data = {} }
   end
   for level = 1,#pyr do
      pyr1[level].data = torch.DoubleTensor(pyr[level].data);
      pyr2[level].data = torch.DoubleTensor(pyr[level].data) * -1;
      pyr1[level].data[torch.lt(pyr1[level].data,0)] = 0;
      pyr2[level].data[torch.lt(pyr2[level].data,0)] = 0;
   end

   return pyr1,pyr2;
end

--function: Sum pyramids
function sumPyr(pyr1,pyr2)
   pyr = {};
   
   for i = 1,#pyr1 do
      if (pyr1[1].orientation ~= nil) then
         pyr[i] = {orientation = {}};
         for o = 1,#pyr1[i].orientation do
            if (pyr1[i].orientation[o].ori ~= nil) then               
               if (pyr1[i].orientation[1].invmsk ~= nil) then
                  pyr[i].orientation[o] = {ori = {}, invmsk = -1};
               end
               pyr[i].orientation[o] = {ori = {}};
            end
         end

      elseif (pyr1[1].data ~= nil) then
         pyr[i] = { data = {} };
   
      elseif (pyr1[1].hData ~= nil) then
         pyr[i] = { hData = {}, vData = {} };
      end
   end

   if (#pyr1 == 0) or (pyr1 == nil) then
      pyr = torch.DoubleTensor(pyr2);
   else
      if (pyr1[1].orientation ~= nil) then
         for level = 1,#pyr1 do
            for ori = 1,#pyr1[level].orientation do
               pyr[level].orientation[ori].data = torch.DoubleTensor(pyr1[level].orientation[ori].data + pyr2[level].orientation[ori].data):clone();
               if (pyr1[level].orientation[1].ori ~= nil) then
                  pyr[level].orientation[ori].ori = pyr1[level].orientation[ori].ori;
                  if (pyr1[level].orientation[1].invmsk ~= nil) then
                     pyr[level].orientation[ori].invmsk = pyr1[level].orientation[ori].invmsk;
                  end
               end
            end
         end
      
      elseif (pyr1[1].data ~= nil) then
         for level = 1,#pyr1 do
            pyr[level].data = torch.DoubleTensor(pyr1[level].data + pyr2[level].data):clone();
         end
            
      elseif (pyr1[1].hData ~= nil) then
         pyr[level].hData = torch.DoubleTensor(pyr1[level].hData + pyr2[level].hData):clone();
         pyr[level].vData = torch.DoubleTensor(pyr1[level].vData + pyr2[level].vData):clone();
      else
         print("Error in pyramid summation");
      end
   end

   return pyr;
end

--function: Clamp Data
function clamp(data,bottom,top)
   if(bottom ~= nil) then
      data[torch.lt(data,bottom)] = bottom;
   end
   
   if(top ~= nil) then
       data[torch.gt(data,top)] = top;
   end
   
   return data;
end

--function: (Get) Mex Local Maxima
function mexLocalMaxima(data,thresh)
   refData = torch.squeeze(torch.DoubleTensor(data)):clone();
   temp_data = torch.squeeze(torch.DoubleTensor(data)):clone();
   end_row = refData:size()[1];
   end_col = refData:size()[2];
   refData = refData[{{2,end_row-1},{2,end_col-1}}];

   and_true_val = 5;
   sum1 = torch.add( torch.ge(refData,temp_data[{{1,end_row-2},{2,end_col-1}}]), torch.ge(refData,temp_data[{{3,end_row},{2,end_col-1}}]));
   sum2 = torch.add(sum1,torch.ge(refData,temp_data[{{2,end_row-1},{1,end_col-2}}]))
   sum3 = torch.add(sum2, torch.ge(refData,temp_data[{{2,end_row-1},{3,end_col}}]));
   sum4 = torch.add(sum3,torch.ge(refData,thresh));
   localMax = torch.eq(sum4,and_true_val);
   maxData = refData[localMax]:clone();
   
   if(torch.numel(maxData) > 0) then
      lm_avg = torch.mean(maxData);
      lm_sum = torch.sum(maxData);
      lm_num = torch.numel(maxData);
   else
      print("Error in Mex Local Maxima");
      lm_avg = 0;
      lm_sum = 0;
      lm_num = 0;
   end
   return lm_avg, lm_num, lm_sum;
end

--function: Max Normalize Local Max
function maxNormalizeLocalMax(data,minmax)
   if (minmax == nil) then
      minmax = torch.DoubleTensor({0,10});
   end

   temp_data = torch.DoubleTensor(data);
   temp_data = clamp(temp_data,0);

   data = normalizeImage(temp_data,minmax);
   
   if (minmax[1] == minmax[2]) then
      thresh = 0.6;
   else
      thresh = minmax[1] + ((minmax[2] - minmax[1]) / 10);
   end
   
   lm_avg,lm_num,lm_sum = mexLocalMaxima(data,thresh);
   --print('lm_avg',lm_avg);
   --print('lm_num',lm_num);
   --print('lm_sum',lm_sum);

   if(lm_num > 1) then
      result = data * ((minmax[2] - lm_avg)^2);
   elseif (lm_num == 1) then
      result = data * (minmax[2]^2);
   else
      result = data;
   end

   return result:clone();
end

--function: Normalize CS Pyramids (2)
function normCSPyr2(csPyr1,csPyr2)
   newPyr1 = {};
   newPyr2 = {};
   for i = 1,#csPyr1 do
      newPyr1[i] = { data = {} }
      newPyr2[i] = { data = {} }
   end
   
   for level = 1,#csPyr1 do
      temp = sumPyr(csPyr1,csPyr2);
      norm = maxNormalizeLocalMax(temp[level].data,torch.DoubleTensor({0,10}));
      if (torch.max(temp[level].data) ~= 0) then
         scale = torch.max(norm) / torch.max(temp[level].data);
      else
         scale = 0;
      end
      
      newPyr1[level].data = csPyr1[level].data:clone() * scale;
      newPyr2[level].data = csPyr2[level].data:clone() * scale;
   end

   return newPyr1,newPyr2;
end

function fact (n)
      if n == 0 then
        return 1
      else
        return n * fact(n-1)
      end
end

--function: Modified Besseli function
function besseli(Z,finalZ,k)
   --local finalZ = torch.DoubleTensor(Z:size()):zero();
   --for k = 0,15 do
   --   if (k == 0) then
   --      divisor = 1;
   --   else
   --      divisor = divisor * k;
   --   end
   --   final_divisor = math.pow(divisor,2)
   if (k < 15) then
      final_divisor = math.pow(fact(k),2);
      finalZ = finalZ + torch.div(torch.pow(torch.div(torch.pow(Z,2),4),k),final_divisor);
      return besseli(torch.DoubleTensor(Z),torch.DoubleTensor(finalZ),k+1);
   else
      return torch.DoubleTensor(finalZ);
   end
end

--function: Modified Besseli function
function besseli2(Z)
   local finalZ = torch.DoubleTensor(Z:size()):zero();
   for k = 0,15 do
      divisor = math.pow(fact(k),2);
      finalZ = finalZ + torch.div(torch.pow(torch.div(torch.pow(Z,2),4),k),divisor);
   end
   
   return torch.DoubleTensor(finalZ);
end

--function: Make Von Mises
function makeVonMises(R0, theta0, dim1, dim2)
   local msk1 = torch.DoubleTensor(dim1:size()[1],dim2:size()[1]):zero();
   local msk2 = torch.DoubleTensor(dim1:size()[1],dim2:size()[1]):zero();  

   sigma_r = R0/2;
   B = R0;
   
   if (dim2[1] == torch.min(dim2)) then
      dim2_temp = {};
      for i = 0,dim2:size()[1]-1 do
         dim2_temp[i + 1] = dim2[dim2:size()[1] - i];
      end
      dim1 = torch.DoubleTensor(1,dim1:size()[1]):copy(dim1)
      dim2 = torch.DoubleTensor(dim2_temp);
      dim2 = torch.DoubleTensor(dim2:size()[1],1):copy(dim2);
   else
      dim1 = torch.DoubleTensor(1,dim1:size()[1]):copy(dim1)
      dim2 = torch.DoubleTensor(dim2:size()[1],1):copy(dim2);
   end
   
   --make grid
   X,Y = meshGrid(dim1,dim2);
   
   --convert to polar coordinates
   R = torch.sqrt(torch.pow(X,2) + torch.pow(Y,2));
   theta = torch.atan2(Y,X);

   --make mask
   -----------
   -----------besseli
   msk1 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - (theta0)),B)),besseli(R-R0,torch.DoubleTensor(R:size()):zero(),0))
   --msk1 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - theta0),B)),besseli2(R-R0))
   -----------
   -----------

   msk1 = torch.div(msk1,torch.max(msk1));
   -----------
   -----------besseli
   msk2 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - (theta0 + math.pi)),B)),besseli(R-R0,torch.DoubleTensor(R:size()):zero(),0))
   --msk2 = torch.cdiv(torch.exp(torch.mul(torch.cos(theta - (theta0 + math.pi)),B)),besseli2(R-R0))
   -----------
   -----------
   msk2 = torch.div(msk2,torch.max(msk2));

   return msk1,msk2;
end

--function: Von Mises Pyramid
function vonMisesPyramid(map, vmPrs, fil_vals1, fil_vals2)
   local pyr1 = {};
   local pyr2 = {};
   local msk1 = {};
   local msk2 = {};
   for l = vmPrs.minLevel,vmPrs.maxLevel do
      pyr1[l] = { orientation = {} };
      pyr2[l] = { orientation = {} };
      msk1[l] = { orientation = {} };
      msk2[l] = { orientation = {} };
   end

   --dim1 = {};
   --idx = 1;
   --for i=-3*vmPrs.R0,3*vmPrs.R0 do
   --   dim1[idx] = i;
   --   idx = idx + 1;
   --end
   --dim1 = torch.DoubleTensor(dim1):clone();
   --dim2 = dim1:clone();

      --msk_1,msk_2 = makeVonMises(vmPrs.R0, vmPrs.oris[ori] + (math.pi / 2), dim1, dim2);
      --setup filters
      --module1 = nn.SpatialConvolution(1,1,msk_1:size()[2],msk_1:size()[1],1,1);         
      --module1.bias = torch.zero(module1.bias);
      --module1.weight[1] = torch.DoubleTensor(1,msk_1:size()[2],msk_1:size()[1]):copy(msk_1);
      
      --module2 = nn.SpatialConvolution(1,1,msk_2:size()[2],msk_2:size()[1],1,1);         
      --module2.bias = torch.zero(module2.bias);
      --module2.weight[1] = torch.DoubleTensor(1,msk_2:size()[2],msk_2:size()[1]):copy(msk_2);
   for level = vmPrs.minLevel,vmPrs.maxLevel do
      
      if(#map[level].data ~= 0) then
         temp_data = torch.DoubleTensor(map[level].data):clone();
         temp_data = prePadImage(torch.DoubleTensor(temp_data:size()):copy(temp_data),MAX_FILTER_SIZE,MAX_FILTER_SIZE,0);
         
         --module1:forward(temp_data:clone());   
         input = torch.squeeze(temp_data:clone());           
         nf:copyToDev(input);
         nf:copyFromDev(outputs);
         for ori = 1,vmPrs.numOri do
            --if(torch.numel(map[level].data) < torch.numel(msk_1)) then
            --print("no convolution since msk is larger than image");
            --pyr1[level].orientation[ori] = {data = torch.DoubleTensor(map[level].data):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
            --pyr2[level].orientation[ori] = {data = torch.DoubleTensor(map[level].data):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
            --else
            -------------------------------
            -------------------------------
            -------------------------------
            -------------------------------
            
            pyr1[level].orientation[ori] = {data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals1[1][ori]]), ori = vmPrs.oris[ori] + (math.pi / 2)};
            -------------------------------
            -------------------------------
            -------------------------------
            -------------------------------
            --module2:forward(temp_data);
            pyr2[level].orientation[ori] = {data = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[fil_vals2[1][ori]]), ori = vmPrs.oris[ori] + (math.pi / 2)};
            -------------------------------
            -------------------------------
            -------------------------------
            -------------------------------
            --end              
            --msk1[level].orientation[ori] = {data = torch.DoubleTensor(msk_2):clone(), ori =  vmPrs.oris[ori] + (math.pi / 2)};
            --msk2[level].orientation[ori] = {data = torch.DoubleTensor(msk_1):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
            msk1[level].orientation[ori] = {data_fil_val = fil_vals1[1][ori], ori =  vmPrs.oris[ori] + (math.pi / 2)};
            msk2[level].orientation[ori] = {data_fil_val = fil_vals2[1][ori], ori = vmPrs.oris[ori] + (math.pi / 2)};
         end
         else
            print('Map is empty at specified level.');
         end
   end
   
   return pyr1,msk1,pyr2,msk2;
end

--function: Von Mises Sum
function vonMisesSum(csPyr, vmPrs,vmfil_vals1,vmfil_vals2)
   local maxLevel = vmPrs.maxLevel;
   --local maxLevel = 5
   local map1 = {};
   local map2 = {};
   
   --create pyramid of center surround convoled with von Mises distribution
   vmPyr1, msk1, vmPyr2, msk2 = vonMisesPyramid(csPyr,vmPrs,vmfil_vals1,vmfil_vals2);   
   for level = 1, maxLevel do
      map1[level] = { orientation = {} };
      map2[level] = { orientation = {} };
      for ori = 1,vmPrs.numOri do
         map1[level].orientation[ori] = {data = {}};
         map2[level].orientation[ori] = {data = {}};
      end
   end
   
   --sum convolved output across different spatial scales
   for minL = 1,maxLevel do
      mapLevel = minL;
      for l = minL,maxLevel do
         for ori = 1,vmPrs.numOri do
            if (l == minL) then
               map1[minL].orientation[ori].data = torch.DoubleTensor(vmPyr1[mapLevel].orientation[ori].data:size()):zero();
               map2[minL].orientation[ori].data = torch.DoubleTensor(vmPyr2[mapLevel].orientation[ori].data:size()):zero();
            end
            temp = image.scale(vmPyr1[l].orientation[ori].data[1],vmPyr1[mapLevel].orientation[ori].data:size()[3],vmPyr1[mapLevel].orientation[ori].data:size()[2],'simple'):clone();
            --map1[minL].orientation[ori].data = torch.add(map1[minL].orientation[ori].data, torch.mul(temp,math.pow(0.5,l-1))):clone();
            map1[minL].orientation[ori].data:add(math.pow(0.5,l-1),temp)

            temp = image.scale(vmPyr2[l].orientation[ori].data[1],vmPyr1[mapLevel].orientation[ori].data:size()[3],vmPyr1[mapLevel].orientation[ori].data:size()[2],'simple');
            map2[minL].orientation[ori].data = torch.add(map2[minL].orientation[ori].data, torch.mul(temp,math.pow(0.5,l-1))):clone();
         end
      end   
      --image.display{image = map1[minL].orientation[1].data}   
   end
   
   return map1,msk1,map2,msk2;
end

--function: Border Pyramid
function borderPyramid(csPyrL,csPyrD,cPyr,params,VM_fil_vals1,VM_fil_vals2)
   bPrs = params.bPrs;
   vmPrs = params.vmPrs;
      
   bPyr1 = {};
   bPyr2 = {};
   bPyr3 = {};
   bPyr4 = {};
   for level = bPrs.minLevel, bPrs.maxLevel do
      bPyr1[level] = {orientation = {}};
      bPyr2[level] = {orientation = {}};
      bPyr3[level] = {orientation = {}};
      bPyr4[level] = {orientation = {}};
      for ori = 1,bPrs.numOri do
         bPyr1[level].orientation[ori] = {data = {}};
         bPyr2[level].orientation[ori] = {data = {}};
         bPyr3[level].orientation[ori] = {data = {}};
         bPyr4[level].orientation[ori] = {data = {}};
      end
   end
   
   --convolve center surround with von Mises distributiona nd, for every orientation, sum across all spatial scales greater or equal to scale 1
   vmL1, msk1, vmL2, msk2 = vonMisesSum(csPyrL,vmPrs, VM_fil_vals1, VM_fil_vals2);
   vmD1, csmsk1, vmD2, csmsk2 = vonMisesSum(csPyrD, vmPrs, VM_fil_vals1, VM_fil_vals2);
   --image.display{image = vmD2[2].orientation[2].data}
   
   --calculate border ownership and grouping responses
   for level = bPrs.minLevel,bPrs.maxLevel do
      for ori_cnt = 1,bPrs.numOri do
         --create border ownership for light objects (on center CS results)
         bpyr1_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmL1[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmD2[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr1_temp[torch.lt(bpyr1_temp,0)]=0;
         bPyr1[level].orientation[ori_cnt] = {data = bpyr1_temp:clone(), ori = msk1[level].orientation[ori_cnt].ori, invmsk = msk1[level].orientation[ori_cnt].data_fil_val};
         
         bpyr2_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmL2[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmD1[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr2_temp[torch.lt(bpyr2_temp,0)]=0;
         bPyr2[level].orientation[ori_cnt] = {data = bpyr2_temp:clone(), ori = msk2[level].orientation[ori_cnt].ori + math.pi, invmsk = msk2[level].orientation[ori_cnt].data_fil_val};
         
         --create border ownership for dark objects (off center cs results)
         bpyr3_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmD1[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmL2[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr3_temp[torch.lt(bpyr3_temp,0)]=0;
         bPyr3[level].orientation[ori_cnt] = {data = bpyr3_temp:clone(), ori = msk1[level].orientation[ori_cnt].ori, invmsk = msk1[level].orientation[ori_cnt].data_fil_val};
         
         bpyr4_temp = torch.cmul(cPyr[level].orientation[ori_cnt].data,torch.add(torch.mul(vmD2[level].orientation[ori_cnt].data,bPrs.alpha) - torch.mul(vmL1[level].orientation[ori_cnt].data,bPrs.CSw),1));
         bpyr4_temp[torch.lt(bpyr4_temp,0)]=0;
         bPyr4[level].orientation[ori_cnt] = {data = bpyr4_temp:clone(), ori = msk2[level].orientation[ori_cnt].ori + math.pi, invmsk = msk2[level].orientation[ori_cnt].data_fil_val};
      end
   end
   
   return bPyr1,bPyr2,bPyr3,bPyr4;
end

--function: Make Border Ownership
function makeBorderOwnership(im_channels, params,even_fil_vals,odd_fil_vals,cs_fil_val,vm1_fil_vals,vm2_fil_vals)
   local map = {};
   local imPyr = {};
   local EPyr = torch.DoubleTensor();
   local OPyr = {};
   local b1Pyr = {};
   local b2Pyr = {};
 
   for level = 1,#im_channels do
      b1Pyr[level] = {subtype = {},subname = {}, type = {}};
      b2Pyr[level] = {subtype = {},subname = {}, type = {}};
   end

   --EXTRACT EDGES
   for m = 1,#im_channels do

      for sub = 1,#im_channels[m].subtype do
         map = torch.DoubleTensor(im_channels[m].subtype[sub].data):clone();
         imPyr = makePyramid(map,params);
         --writeToFile(imPyr[1].data:clone(),'luaimage.txt');
         --writeToFile(imPyr[2].data:clone(),'luaimage2.txt');
         --writeToFile(imPyr[3].data:clone(),'luaimage3.txt');
         ------------------
         --Edge Detection--
         ------------------
         EPyr = edgeEvenPyramid(imPyr,params,even_fil_vals);
         OPyr, o = edgeOddPyramid(imPyr,params,odd_fil_vals);
         cPyr = makeComplexEdge(EPyr,OPyr);
         if m == 1 and sub == 1 then
            writeToFile(cPyr[1].orientation[1].data,'luaimage.txt');
         end

         ----------------------
         --Make Image Pyramid--
         ----------------------
         if(im_channels[m].subtype[sub].type == "Orientation") then
            csPyr = gaborPyramid(imPyr,im_channels[m].subtype[sub].ori,params,im_channels[m].subtype[sub].filval);
         else
            csPyr = csPyramid(imPyr,params,cs_fil_val);
         end

         csPyrL,csPyrD = separatePyr(csPyr);
         csPyrL,csPyrD = normCSPyr2(csPyrL,csPyrD);
         -----------------------------------------------
         --Generate Border Ownership and Grouping Maps--
         -----------------------------------------------
         bPyr1_1, bPyr2_1, bPyr1_2, bPyr2_2 = borderPyramid(csPyrL,csPyrD,cPyr,params,vm1_fil_vals,vm2_fil_vals);
         b1Pyr[m].subtype[sub] = sumPyr(bPyr1_1,bPyr1_2);
         b2Pyr[m].subtype[sub] = sumPyr(bPyr2_1,bPyr2_2);

         if (im_channels[m].subtype[sub].type == "Orientation") then
            b1Pyr[m].subname[sub] = rad2deg(im_channels[m].subtype[sub].ori) .. " deg";
            b2Pyr[m].subname[sub] = rad2deg(im_channels[m].subtype[sub].ori) .. " deg";
         else
            b1Pyr[m].subname[sub] = im_channels[m].subtype[sub].type;
            b2Pyr[m].subname[sub] = im_channels[m].subtype[sub].type;
         end
         --image.display{image = b2Pyr[m].subtype[sub][1].orientation[1].data}

      end
      b1Pyr[m].type = im_channels[m].type;
      b2Pyr[m].type = im_channels[m].type;
   end
   
   return b1Pyr,b2Pyr;
end

--function: Grouping Pryamid Max Difference
function groupingPyramidMaxDiff(bpyrr1,bpyrr2,params)
   bPrs = params.bPrs;
   giPrs = params.giPrs;
   w = giPrs.w_sameChannel;
   
   gPyr1 = {};
   gPyr2 = {};
   
   for l = bPrs.minLevel,bPrs.maxLevel do
      gPyr1[l] = {orientation = {}};
      gPyr2[l] = {orientation = {}};
      for o = 1,bPrs.numOri do
         gPyr1[l].orientation[o] = {data = {}};
         gPyr2[l].orientation[o] = {data = {}};
      end
   end
   
   --module = nn.SpatialConvolution(1,1,bpyrr1[1].orientation[1].invmsk:size()[2],bpyrr1[1].orientation[1].invmsk:size()[1],1,1);         
   --module.bias = torch.zero(module.bias);

   --calculate border ownership and grouping responses
   for level = bPrs.minLevel, bPrs.maxLevel do
      bTemp1 = torch.DoubleTensor(bPrs.numOri,bpyrr1[level].orientation[1].data:size(2),bpyrr1[level].orientation[1].data:size(3)):zero();
      for ori = 1,bPrs.numOri do
         bTemp1[ori][{{},{}}] = torch.squeeze(torch.abs(bpyrr1[level].orientation[ori].data - bpyrr2[level].orientation[ori].data)):clone();
      end
      m1, m_ind1 = torch.max(bTemp1,1);
      m_ind1 = m_ind1:type('torch.DoubleTensor');
      
      for ori = 1,bPrs.numOri do
         m_i1 = torch.squeeze(m_ind1:clone())
         m_i1[torch.ne(m_i1,ori)] = 0;
         m_i1[torch.eq(m_i1,ori)] = 1;
         
         invmsk1 = bpyrr1[level].orientation[ori].invmsk;
         invmsk2 = bpyrr2[level].orientation[ori].invmsk;
         --print("invmsk1", invmsk1);
         --print("invmsk2", invmsk2);
         
         --print(m_i1)
         --if level == 1 and ori == 1 then
         --image.display{image = m_i1}
         --end
         b1p = torch.cmul(m_i1,torch.DoubleTensor(bpyrr1[level].orientation[ori].data - bpyrr2[level].orientation[ori].data));
         b1n = torch.DoubleTensor(b1p:clone() * -1);         
         b1p[torch.lt(b1p,0)] = 0;
         b1p[torch.ne(b1p,0)] = 1;
         b1n[torch.lt(b1n,0)] = 0;
         b1n[torch.ne(b1n,0)] = 1;
         
         --if level == 1 and ori == 1 then
         --image.display{image = b1p}
         --image.display{image = b1n}
         --end

         --module.weight[1] = torch.DoubleTensor(1,invmsk1:size()[2],invmsk1:size()[1]):copy(invmsk1);
         
         temp_data1_1 = torch.DoubleTensor(torch.cmul(bpyrr1[level].orientation[ori].data,b1p)):clone();         
         temp_data1_1 = prePadImage(torch.DoubleTensor(1,temp_data1_1:size()[2],temp_data1_1:size()[3]):copy(temp_data1_1),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
         
         --module:forward(temp_data1_1);
         input = torch.squeeze(temp_data1_1:clone());           
         nf:copyToDev(input);
         nf:copyFromDev(outputs);
         out1_1 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk1]);
         
         temp_data1_2 = torch.DoubleTensor(torch.cmul(bpyrr2[level].orientation[ori].data,b1p * w)):clone();
         temp_data1_2 = prePadImage(torch.DoubleTensor(1,temp_data1_2:size()[2],temp_data1_2:size()[3]):copy(temp_data1_2),MAX_FILTER_SIZE,MAX_FILTER_SIZE, torch.max(temp_data1_2));
         --module:forward(temp_data1_2);
         
         input = torch.squeeze(temp_data1_2:clone());           
         nf:copyToDev(input);
         nf:copyFromDev(outputs);
         out1_2 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk1]);
         
         --module.weight[1] = torch.DoubleTensor(1,invmsk2:size()[2],invmsk2:size()[1]):copy(invmsk2);
         
         temp_data2_1 = torch.DoubleTensor(torch.cmul(bpyrr2[level].orientation[ori].data,b1n)):clone();
         temp_data2_1 = prePadImage(torch.DoubleTensor(1,temp_data2_1:size()[2],temp_data2_1:size()[3]):copy(temp_data2_1),MAX_FILTER_SIZE,MAX_FILTER_SIZE);
         --module:forward(temp_data2_1);
         input = torch.squeeze(temp_data2_1:clone());           
         nf:copyToDev(input);
         nf:copyFromDev(outputs);
         out2_1 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk2]);
         
         temp_data2_2 = torch.DoubleTensor(torch.cmul(bpyrr1[level].orientation[ori].data,b1n * w)):clone();
         temp_data2_2 = prePadImage(torch.DoubleTensor(1,temp_data2_2:size()[2],temp_data2_2:size()[3]):copy(temp_data2_2),MAX_FILTER_SIZE,MAX_FILTER_SIZE, torch.max(temp_data2_2));
         --module:forward(temp_data2_2);
         input = torch.squeeze(temp_data2_2:clone());           
         nf:copyToDev(input);
         nf:copyFromDev(outputs);
         out2_2 = torch.DoubleTensor(1,outputs:size(2),outputs:size(3)):copy(outputs[invmsk2]);
         
         
         final1 = torch.DoubleTensor(out1_1 - out1_2):clone();
         final2 = torch.DoubleTensor(out2_1 - out2_2):clone();
         
         final1[torch.lt(final1,0)] = 0;
         final2[torch.lt(final2,0)] = 0;

         gPyr1[level].orientation[ori].data = final1:clone();
         gPyr2[level].orientation[ori].data = final2:clone();
      end
   end
   
   return gPyr1, gPyr2;
end

--function Merge Level
function mergeLevel(pyrr)
   local newPyr = {};
   local temp = {};
   
   for l = 1,#pyrr do
      newPyr[l] = {data = {}};
   end

   for level = 1,#pyrr do
      if (pyrr[1].orientation[1] ~= nil) then
         temp = torch.DoubleTensor(pyrr[level].orientation[1].data:size()):zero();
         for ori =1,#pyrr[level].orientation do
            temp = torch.add(temp,pyrr[level].orientation[ori].data);
         end
         newPyr[level].data = temp:clone();
      end
   end
   
   temp = nil;
   
   return newPyr;

end

--function: Make Grouping
function makeGrouping(b1Pyrr,b2Pyrr,params)
   
   local gPyr = {};
   local gPyr1_1 = {};
   local gPyr2_1 = {};
   local g11 = {};
   local g21 = {};
   
   for m = 1,#b1Pyrr do
      gPyr1_1[m] = {subtype = {}};
      gPyr2_1[m] = {subtype = {}};
      gPyr[m] = {type = {}, subtype = {}};
   end

   for m = 1,#b1Pyrr do
      print("\nAssigning Grouping on " .. b1Pyrr[m].type .. " channel:\n");
      for sub = 1,#b1Pyrr[m].subtype do
         print("Subtype " .. sub .. " of " .. #b1Pyrr[m].subtype);
         print(b1Pyrr[m].subname[sub] .. "\n");
         gPyr1_1[m].subtype[sub], gPyr2_1[m].subtype[sub] = groupingPyramidMaxDiff(b1Pyrr[m].subtype[sub],b2Pyrr[m].subtype[sub],params);    
         
         g11 = mergeLevel(gPyr1_1[m].subtype[sub]);
         g21 = mergeLevel(gPyr2_1[m].subtype[sub]);
         --if m==1 and sub == 1 then
          --  image.display{image = g21[1].data}
          --  image.display{image = g21[2].data}
         --end

         gPyr[m].subtype[sub] = sumPyr(g11,g21);
      end
      gPyr[m].type = b1Pyrr[m].type;
   end
   
   return gPyr;
end

--function: Itti et. al. Normalization
function ittiNorm(gPyrr,collapseLevel)
   CM = {};
   for mm = 1,#gPyrr do
      CM[mm] = {data = {}};
   end
   
   for m = 1,#gPyrr do
      FM = {};
      if(gPyrr[m].type ~= "Orientation") then
         print("no orientation");
         for l = 1,#gPyrr[m].subtype[1] do
            FM[l] = { data = {}}
         end

         for level = 1,#gPyrr[m].subtype[1] do
            --print(#gPyrr[m].subtype[1])
            FM[level].data = torch.DoubleTensor(gPyrr[m].subtype[1][level].data:size(2),gPyrr[m].subtype[1][level].data:size(3)):zero();
            --print(FM[1].data:size());
            for sub = 1,#gPyrr[m].subtype do
               --print(sub);
               --print('IN HERE');
               temp_normalized = maxNormalizeLocalMax(gPyrr[m].subtype[sub][level].data:clone(),torch.DoubleTensor({0,10}));
               --if(m == 2) then
               --  writeToFile(temp_normalized,'temp_' .. sub .. '.txt');
               --end

               FM[level].data = torch.add(FM[level].data:clone(),temp_normalized:clone()):clone();
               --if m ==2 then
               --   image.display{image = FM[level].data}
               --end

            end
         end

         CML = {};
         for l = 1,#FM do
            CML[l] = {data = FM[l].data:clone()};
         end
         --image.display{image = CML[1].data}         
         CM[m].data = torch.DoubleTensor(FM[collapseLevel].data:size()):zero();
         for l = 1,#CML do
            temp_resized = image.scale(CML[l].data, FM[collapseLevel].data:size(2), FM[collapseLevel].data:size(1),'bilinear'):clone();
            CM[m].data = torch.add(CM[m].data,temp_resized:clone());
         end

      elseif (gPyrr[m].type == "Orientation") then
         print("in orientation");
         FM = gPyrr[m].subtype;
         CM[m].data = torch.DoubleTensor(FM[1][collapseLevel].data:size()):zero();
         CML = {};
         FML = {};
         
         for sub = 1,#FM do
            temp = {};
            for i = 1,#FM[m] do
               temp[i] = {data = i + sub};
            end
            FML[sub] = temp;
         end
         
         --print(FML)

         for sub = 1,#FM do
            CML[sub] = {data = torch.DoubleTensor(FM[1][collapseLevel].data:size()):zero()};
            

            for l = 1,#FM[m] do
               temp_norm = maxNormalizeLocalMax(FM[sub][l].data,torch.DoubleTensor({0,10}));
               FML[sub][l] = {data = temp_norm:clone()};
               temp_resize = image.scale(FML[sub][l].data,FM[1][collapseLevel].data:size(3), FM[1][collapseLevel].data:size(2),'bilinear');
               CML[sub].data = torch.add(CML[sub].data:clone(),temp_resize:clone());
            end
            CM[m].data = torch.add(CM[m].data,CML[sub].data);
         end

      else
         print("Please ensure algorithm operates on known feature types");
      end
   end
   
   h = {data = torch.DoubleTensor(CM[1].data:size()):zero()};
   
   for m = 1,#CM do
      temp_normd = maxNormalizeLocalMax(CM[m].data:clone(),torch.DoubleTensor({0,10}));
      temp_normd = torch.mul(temp_normd,1/3);
      h.data = torch.add(h.data,temp_normd);
   end
   
   return h;
   
end

--function: Calc Sigma
function calcSigma(r,x)
   local sigma1 = (r^2) / (4 * math.log(x)) * (1-(1/(x^2)));
   sigma1 = math.sqrt(sigma1);
   local sigma2 = x * sigma1;
   
   return sigma1,sigma2;
end

--function: Get Default Parameters
function getDefaultParameters(mxLevel)

   local local_minLevel = 1;
   local local_maxLevel = mxLevel;
   local local_downsample = 'half';
   local ori = torch.DoubleTensor({0,45});
   
   local local_oris = deg2rad(torch.DoubleTensor({ori[1], ori[2], ori[1] + 90, ori[2] + 90}));
   local local_lambda = 4;
   local local_odd_lambda = 3;
   local local_even_lambda = 3;
   
   local local_gabor_lambda = 8;
   
   local local_sigma1,local_sigma2 = calcSigma(1,2);
   local msk = makeCenterSurround(local_sigma1,local_sigma2);
   local temp = msk[{ {math.ceil(msk:size()[1] / 2)},{} }];
   temp[torch.gt(temp,0)] = 1;
   temp[torch.lt(temp,0)] = -1;
   temp_length = torch.numel(temp);
   zc = temp[{{}, {math.ceil(msk:size()[2] / 2), temp_length - 1}, {} }] - temp[{{}, {math.ceil(msk:size()[1] / 2) + 1, temp_length}}];
   temp_R0 = torch.eq(torch.abs(zc),2);
   idx = 1;
   val = torch.numel(temp_R0);
   while (idx <= val) do
      temp_val = (temp_R0[1][idx]);
      if( temp_val == 1) then
         local_R0 = idx;
      end
      idx = idx + 1;
   end
      
   params =  {
      channels = 'ICO',
      maxLevel = local_maxLevel,
      evenCellsPrs = {minLevel = local_minLevel,
                      maxLevel = local_maxLevel,
                      oris = local_oris,
                      numOri = local_oris:size(1),
                      lambda = local_even_lambda,
                      sigma = 0.56 * local_even_lambda,
                      gamma = 0.5
                   },

      oddCellsPrs = {minLevel = local_minLevel,
                      maxLevel = local_maxLevel,
                      oris = local_oris,
                      numOri = local_oris:size(1),
                      lambda = local_odd_lambda,
                      sigma = 0.56 * local_odd_lambda,
                      gamma = 0.5
                  },
      
      gaborPrs = {lambda = local_gabor_lambda,
                  sigma = 0.4 * local_gabor_lambda,
                  gamma = 0.8,
                  oris = torch.DoubleTensor({0,math.pi/4,math.pi/2,3*math.pi/4}),
                  numOri = 4
                 },

      csPrs = { downSample = local_downsample,
                inner = local_sigma1,
                outer = local_sigma2,
                depth = local_maxLevel
             },

      bPrs = { minLevel = local_minLevel,
               maxLevel = local_maxLevel,
               numOri = local_oris:size(1),
               alpha = 1,
               oris = local_oris,
               CSw = 1
            },

      vmPrs = { minLevel = local_minLevel,
                maxLevel = local_maxLevel,
                oris = local_oris,
                numOri = local_oris:size(1),                
                R0 = local_R0
             },
      
      giPrs = { w_sameChannel = 1
            }
   }
   
   return params;
end


-------------------------
-----MAIN FUNCTION-------
-------------------------
function runProtoSal(input_img,parameters,tmp_nf)
   --print("Starting Proto-object Based Saliency")
   --defaultParameters = getDefaultParameters(5);
   
   --Read in image from file (column,row) / (x,y)
   --im = image.loadJPG('/home/jamal/ProtoObject/soccer.jpg')
   
   --Normalize image
   im = normalizeImage(input_img);
   
   --Generate channels from image
   im_channels = generateChannels(im,parameters);
   
   b1Pyr_final, b2Pyr_final = makeBorderOwnership(im_channels,parameters);
   
   gPyr_final = makeGrouping(b1Pyr_final, b2Pyr_final, parameters);
   
   --image.display{image = gPyr_final[1].subtype[1][1].data}
   h_final = ittiNorm(gPyr_final,1);
   
   return h_final.data:clone();
   --writeToFile(h_final.data:clone(),'sal_out_map.txt');
   
   --image.display{image = h_final.data}
end
------------------------------------------------------------------------------
------------------------------------------------------------------------------
--------------------------------MAIN CODE-------------------------------------
------------------------------------------------------------------------------

--function: setup filter for 10 x 10
function setupFilter(filter, filter_size)
   
   if torch.numel(filter) > 81 then
      print('ERROR - Filter is greater than 10 x 10!');
      return nil;

   elseif torch.numel(filter) == 81 then
      return torch.DoubleTensor(1,filter:size(1),filter:size(2)):copy(filter);

   else
      curr_rows = filter:size(1);
      curr_cols = filter:size(2);
      pad_cols = (filter_size - curr_cols) / 2;
      pad_rows = (filter_size - curr_rows) / 2;
      
      padder = nn.SpatialZeroPadding(pad_cols,pad_cols,pad_rows,pad_rows)
      filter = torch.DoubleTensor(1,filter:size(1),filter:size(2)):copy(filter)
      padder:forward(filter); 
      new_filter = padder.output:clone();
      
      return new_filter;
   end
end

------------------
--GET PARAMETERS--
------------------
defaultParameters = getDefaultParameters(1);

----------------------------------------------
--SETUP FILTER VALUE IN SPATIAL CONVOLUTIONS--
----------------------------------------------
EVMSK = torch.DoubleTensor(1,defaultParameters.evenCellsPrs.numOri):zero();
ODDMSK = torch.DoubleTensor(1,defaultParameters.oddCellsPrs.numOri):zero();
GABORMSK = torch.DoubleTensor(1,defaultParameters.gaborPrs.numOri):zero();
CSMSK = 0;
VMSK1 = torch.DoubleTensor(1,defaultParameters.vmPrs.numOri):zero();
VMSK2 = torch.DoubleTensor(1,defaultParameters.vmPrs.numOri):zero();

---------------------------------
--SETUP ALL FILTERS FOR NEUFLOW--
---------------------------------
NUM_OF_FILTERS = 1 + torch.numel(EVMSK) + torch.numel(ODDMSK) + torch.numel(GABORMSK) + torch.numel(VMSK1) + torch.numel(VMSK2);

network = nn.Sequential()
conv = nn.SpatialConvolution(1,NUM_OF_FILTERS,MAX_FILTER_SIZE,MAX_FILTER_SIZE,1,1);
conv.bias = torch.zero(conv.bias)

--Initialize filter_cnt
filter_cnt = 1;

--Setup Even Pyramid mask (4 orientations)
prs_even = defaultParameters.evenCellsPrs;
for local_ori = 1,prs_even.numOri do
   Evmsk = makeEvenOrientationCells(prs_even.oris[local_ori],prs_even.lambda,prs_even.sigma,prs_even.gamma);
   Evmsk_final = setupFilter(torch.DoubleTensor(Evmsk),MAX_FILTER_SIZE);
   conv.weight[filter_cnt] = torch.DoubleTensor(Evmsk_final);
   EVMSK[1][local_ori] = filter_cnt;
   filter_cnt = filter_cnt + 1;
end

--Setup Odd Pyramid mask (4 orientations)
prs_odd = defaultParameters.oddCellsPrs;
for local_ori = 1,prs_odd.numOri do
   Oddmsk1, Oddmsk2 = makeOddOrientationCells(prs_odd.oris[local_ori],prs_odd.lambda,prs_odd.sigma,prs_odd.gamma);
   Oddmsk_final = setupFilter(torch.DoubleTensor(Oddmsk1),MAX_FILTER_SIZE);
   conv.weight[filter_cnt] = torch.DoubleTensor(Oddmsk_final);
   ODDMSK[1][local_ori] = filter_cnt;
   filter_cnt = filter_cnt + 1;
end

--Setup Gabor/Orientation mask (4 orientations)
prs_gabor = defaultParameters.gaborPrs;
for local_ori = 1,prs_gabor.numOri do
   Gabormsk = makeEvenOrientationCells(prs_gabor.oris[local_ori],prs_gabor.lambda,prs_gabor.sigma,prs_gabor.gamma);
   Gabormsk_final = setupFilter(torch.DoubleTensor(Gabormsk),MAX_FILTER_SIZE);
   conv.weight[filter_cnt] = torch.DoubleTensor(Gabormsk_final);
   GABORMSK[1][local_ori] = filter_cnt;
   filter_cnt = filter_cnt + 1;
end

--Setup Center Surround mask (1 center surround)
prs_cs = defaultParameters.csPrs;
CSmsk = makeCenterSurround(prs_cs.inner,prs_cs.outer);
CSmsk_final = setupFilter(torch.DoubleTensor(CSmsk),MAX_FILTER_SIZE);
conv.weight[filter_cnt] = torch.DoubleTensor(CSmsk_final);
CSMSK = filter_cnt;
filter_cnt = filter_cnt + 1;

--Setup Von Mises masks (4 orientations, 2 types)
prs_vm = defaultParameters.vmPrs;
dim1 = {};
idx = 1;
for i=-3*prs_vm.R0,3*prs_vm.R0 do
   dim1[idx] = i;
   idx = idx + 1;
end
dim1 = torch.DoubleTensor(dim1):clone();
dim2 = dim1:clone();

for local_ori = 1,prs_vm.numOri do
   VMmsk1,VMmsk2 = makeVonMises(prs_vm.R0, prs_vm.oris[local_ori] + (math.pi / 2), dim1, dim2);
   VMmsk1_final = setupFilter(torch.DoubleTensor(VMmsk1),MAX_FILTER_SIZE);
   VMmsk2_final = setupFilter(torch.DoubleTensor(VMmsk2),MAX_FILTER_SIZE);
   
   conv.weight[filter_cnt] = torch.DoubleTensor(VMmsk1_final);
   VMSK1[1][local_ori] = filter_cnt;
   filter_cnt = filter_cnt + 1;
   
   conv.weight[filter_cnt] = torch.DoubleTensor(VMmsk2_final);
   VMSK2[1][local_ori] = filter_cnt;
   filter_cnt = filter_cnt + 1;
end
--conv.weight[1] = torch.DoubleTensor(9,9):fill(0);
--conv.weight[1] = 1;
--conv.weight[1][1][5][5] = 1;
--print(conv.weight[2]);
--image.display{image = conv.weight[1]}
network:add(conv)

-----------------------------
--COMPILE FILTERS ONTO FPGA--
-----------------------------
--Input image size here determines size of all input images to fpga/neuflow
input_im = image.load('/home/jamal/ProtoObject/soccer.jpg');
input = image.scale(input_im, input_im:size(3) + MAX_FILTER_SIZE - 1, input_im:size(2) + MAX_FILTER_SIZE - 1)
input = input[1]:clone();

-- loop over the main code
nf:beginLoop('main') do

   -- send data to device
   input_dev = nf:copyFromHost(input)

   -- compile network
   output_dev = nf:compile(network, input_dev)

   -- send result back to host
   outputs = nf:copyToHost(output_dev)

end nf:endLoop('main')

----------------------------------------------------------------------
-- LOAD: load the bytecode on the device, and execute it
--
nf:sendReset()
nf:loadBytecode()

----------------------------------------------------------------------
-- EXEC: this part executes the host code, and interacts with the dev
--

-- profiler
--p = nf.profiler

-- zoom
zoom = 0.5

-- try to initialize camera, or default to Lena
--if xlua.require 'camera' then
--   camera = image.Camera{}
--end

-- process loop
--image.display{image = outputs}
function process()
  -- p:start('whole-loop','fps')
   
   -----------------------
   --USE CAMERA AS INPUT--
   -----------------------
   --   if camera then
   --      p:start('get-camera-frame')
   --      frameRGB = camera:forward()
   --      input_im = image.scale(frameRGB, 278, 458)
   --input = image.rgb2y(frameRGB)
   --     image.display(input_im)
   --     p:lap('get-camera-frame')
   --  end

   -----------------------------------
   --RUN PROTO-OBJECT SALIENCY MODEL--
   -----------------------------------
   --Normalize Image 
   im = normalizeImage(input_im);   
   --Generate channels from image
   im_channels = generateChannels(im,defaultParameters,GABORMSK);   
   --Compute Border Ownership
   b1Pyr_final, b2Pyr_final = makeBorderOwnership(im_channels,defaultParameters,EVMSK,ODDMSK,CSMSK,VMSK1,VMSK2);
   --Compute Grouping
   gPyr_final = makeGrouping(b1Pyr_final, b2Pyr_final, defaultParameters);
   --Compute itti Normalization
   h_final = ittiNorm(gPyr_final,1);

   image.display{image = h_final.data};
   --writeToFile(h_final.data:clone(),'luaimage.txt');
   -----------------------------------
   -----------------------------------
   --win:gbegin()
   --win:showpage()

   --p:start('display')
   --image.display{image=outputs, win=win, min=-1, max=1, zoom=zoom}
  -- p:lap('display')

  -- p:lap('whole-loop')

  -- p:displayAll{painter=win, x=outputs:size(3)*4*zoom+10, y=outputs:size(2)*2*zoom+40, font=12}
   --win:gend()
end

----------------------------------------
-- GUI: setup user interface / display--
----------------------------------------

--if not win then
--   win = qtwidget.newwindow(outputs:size(3)*6*zoom, outputs:size(2)*3*zoom, 'Filter Bank')
--end

----------------------------
--USE ONLY IF USING CAMERA--
----------------------------
--timer = qt.QTimer()
--timer.interval = 10
--timer.singleShot = true
--qt.connect(timer,
--           'timeout()',
--           function()
--              process()
--              collectgarbage()
--              timer:start()
--           end)
--timer:start()

---------------------------
--USE IF NOT USING CAMERA--
---------------------------
process()
