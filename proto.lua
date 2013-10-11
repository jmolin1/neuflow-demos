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

--require 'helperFunctions'
torch.setdefaulttensortype('torch.DoubleTensor')
-------------------------------------------------------------------
--------------------------HELPER FUNCTIONS-------------------------
-------------------------------------------------------------------

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
   
   arg2[torch.eq(arg2,0)] = 1
   result = torch.cdiv(arg1,arg2)
   result[torch.eq(arg2,0)] = 0
   return result;

end

--function: Make Colors
function makeColors(im)

   local r = im[1];
   local g = im[2];
   local b = im[3];

   local gray = torch.div(r:clone() + g:clone() + b:clone(), 3);

   local msk = torch.DoubleTensor(gray):clone();
   msk[torch.lt(msk,torch.max(gray) * 0.1)] = 0;
   msk[torch.ne(msk,0)]=1;
  
   r = safeDivide(torch.cmul(r,msk),gray);
   g = safeDivide(torch.cmul(g,msk),gray);
   b = safeDivide(torch.cmul(b,msk),gray);

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
function generateChannels(img, params)
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
         
            return_result[c] = {type = 'Orientation', subtype = { {data = torch.DoubleTensor(gray:size()):copy(gray), ori = 0, type = 'Orientation'}, 
                                                                  {data = torch.DoubleTensor(gray:size()):copy(gray), ori = math.pi/4, type = 'Orientation'},
                                                                  {data = torch.DoubleTensor(gray:size()):copy(gray), ori = math.pi/2, type = 'Orientation'},
                                                                  {data = torch.DoubleTensor(gray:size()):copy(gray), ori = 3 * math.pi/4, type = 'Orientation'} } }
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
   
   pyr[1] = {data = img};
   for level = 2,depth do
      if params.csPrs.downSample == 'half' then
         curr_width = math.ceil(curr_width * 0.7071)
         curr_height = math.ceil(curr_height * 0.7071)
         pyr[level] = {data = image.scale(pyr[level-1].data,curr_width,curr_height,'simple')};
      elseif params.csPrs.downSample == 'full' then
         curr_width = math.ceil(curr_width * 0.5)
         curr_height = math.ceil(curr_height * 0.5)
         pyr[level] = {data = image.scale(pyr[level-1].data,curr_width,curr_height,'simple')};
      else
         print('Please specify if downsampling should be half or full octave');
      end
      
   end
   
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
function edgeEvenPyramid(map,params)
   local prs = params.evenCellsPrs;   
   
   --Initialize newMap
   local newMap = {};
   for i = prs.minLevel,prs.maxLevel do
      newMap[i] = { orientation = {} };
   end

   for local_ori = 1,prs.numOri do
      Evmsk = makeEvenOrientationCells(prs.oris[local_ori],prs.lambda,prs.sigma,prs.gamma);
      --setup filter
      module = nn.SpatialConvolution(1,1,Evmsk:size()[2],Evmsk:size()[1],1,1);         
      module.bias = torch.zero(module.bias);
      module.weight[1] = torch.DoubleTensor(1,Evmsk:size()[2],Evmsk:size()[1]):copy(Evmsk);         
      for level = prs.minLevel,prs.maxLevel do
         ------------------------------------------------------------------------------ DONE - NO PADDING
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         temp_data = torch.DoubleTensor(map[level].data):clone();
         module:forward(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data));
         --print(module.output[1][100][100])
         newMap[level].orientation[local_ori] = {ori = prs.oris[local_ori], data =  torch.DoubleTensor(module.output):clone()};
         --print(newMap[level].orientation[local_ori].data[1][100][100])
         --if ((level == 2) and (local_ori == 1)) then
         --   image.display{image = newMap[level].orientation[local_ori].data}
         --end
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
function edgeOddPyramid(map,params)
   local prs = params.oddCellsPrs;
   
   --Initialize newMap
   local newMap1 = {}
   local newMap2 = {}
   
   for i = prs.minLevel,prs.maxLevel do
      newMap1[i] = { orientation = {} }
      newMap2[i] = { orientation = {} }
   end
   
   for local_ori = 1,prs.numOri do
      Oddmsk1, Oddmsk2 = makeOddOrientationCells(prs.oris[local_ori],prs.lambda,prs.sigma,prs.gamma);
      --setup filter
      module = nn.SpatialConvolution(1,1,Oddmsk1:size()[2],Oddmsk1:size()[1],1,1);         
      module.bias = torch.zero(module.bias);
      module.weight[1] = torch.DoubleTensor(1,Oddmsk1:size()[2],Oddmsk1:size()[1]):copy(Oddmsk1);
      for level = prs.minLevel,prs.maxLevel do
         ------------------------------------------------------------------------------DONE - NO PADDING
         ------------------------------------------------------------------------------
         ------------------------------------------------------------------------------
         temp_data = torch.DoubleTensor(map[level].data):clone();
         module:forward(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data));
         newMap1[level].orientation[local_ori] = {ori = prs.oris[local_ori], data = torch.DoubleTensor(module.output):clone() };
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
function gaborPyramid(pyr,ori,params)
   depth = params.maxLevel;
   gaborPrs = params.gaborPrs;
   Evmsk = makeEvenOrientationCells(ori,gaborPrs.lambda,gaborPrs.sigma,gaborPrs.gamma);
   local gaborPyr = {};
   for i = 1,depth do
      gaborPyr[i] = { data = {} }
   end
   local module = nn.SpatialConvolution(1,1,Evmsk:size()[2],Evmsk:size()[1],1,1);         
   module.bias = torch.zero(module.bias);
   module.weight[1] = torch.DoubleTensor(1,Evmsk:size()[2],Evmsk:size()[1]):copy(Evmsk);
   for level = 1,depth do
      ------------------------------------------------------------------------------DONE - NO PADDING
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      temp_data = torch.DoubleTensor(pyr[level].data):clone();
      module:forward(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data))      
      gaborPyr[level].data = torch.DoubleTensor(module.output):clone();
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
function csPyramid(pyr,params)
   depth = params.maxLevel;
   csPrs = params.csPrs;
   CSmsk = makeCenterSurround(csPrs.inner,csPrs.outer);
   local csPyr = {};
   for i = 1,depth do
      csPyr[i] = { data = {} }
   end
   local module = nn.SpatialConvolution(1,1,CSmsk:size()[2],CSmsk:size()[1],1,1);         
   module.bias = torch.zero(module.bias);
   module.weight[1] = torch.DoubleTensor(1,CSmsk:size()[2],CSmsk:size()[1]):copy(CSmsk);
   for level = 1,depth do
      ------------------------------------------------------------------------------DONE - NO PADDING
      ------------------------------------------------------------------------------
      ------------------------------------------------------------------------------
      temp_data = torch.DoubleTensor(pyr[level].data):clone();
      module:forward(torch.DoubleTensor(1,temp_data:size()[1],temp_data:size()[2]):copy(temp_data))      
      csPyr[level].data = torch.DoubleTensor(module.output):clone();
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
      pyr[i] = { data = {} }
   end

   if (#pyr1 == 0) or (pyr1 == nil) then
      pyr = torch.DoubleTensor(pyr2);
   else
      if (pyr1[1].data ~= nil) then
         for level = 1,#pyr1 do
            pyr[level].data = torch.DoubleTensor(pyr1[level].data + pyr2[level].data):clone();
         end
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
   sum1 = torch.add( torch.ge(refData,temp_data[{{1,end_row-2},{2,end_col-1}}]), torch.ge(refData,temp_data[{{3,end_row},{2,end_col-1}}]), torch.ge(refData,temp_data[{{2,end_row-1},{1,end_col-2}}]));
   sum2 = torch.add( sum1, torch.ge(refData,temp_data[{{2,end_row-1},{3,end_col}}]), torch.ge(refData,thresh) );
   localMax = torch.eq(sum2,and_true_val);
   maxData = refData[localMax]:clone();
   
   if(torch.numel(maxData) > 0) then
      lm_avg = torch.mean(maxData);
      lm_sum = trch.sum(maxData);
      lm_num = torch.numel(maxData);
   else
      --print("Error in Mex Local Maxima");
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
   
   if(lm_num > 1) then
      result = data * ((minmax[2] - lm_avg)^2);
   elseif (lm_num == 1) then
      result = data * (minmax[2]^2);
   else
      result = data;
   end

   return result;
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
function vonMisesPyramid(map, vmPrs)
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

   dim1 = {};
   idx = 1;
   for i=-3*vmPrs.R0,3*vmPrs.R0 do
      dim1[idx] = i;
      idx = idx + 1;
   end
   dim1 = torch.DoubleTensor(dim1):clone();
   dim2 = dim1:clone();

   for ori = 1,vmPrs.numOri do
      msk_1,msk_2 = makeVonMises(vmPrs.R0, vmPrs.oris[ori] + (math.pi / 2), dim1, dim2);
      --setup filters
      module1 = nn.SpatialConvolution(1,1,msk_1:size()[2],msk_1:size()[1],1,1);         
      module1.bias = torch.zero(module.bias);
      module1.weight[1] = torch.DoubleTensor(1,msk_1:size()[2],msk_1:size()[1]):copy(msk_1);
      
      module2 = nn.SpatialConvolution(1,1,msk_2:size()[2],msk_2:size()[1],1,1);         
      module2.bias = torch.zero(module.bias);
      module2.weight[1] = torch.DoubleTensor(1,msk_2:size()[2],msk_2:size()[1]):copy(msk_2);

      for level = vmPrs.minLevel,vmPrs.maxLevel do
         if(#map[level].data ~= 0) then
            if(torch.numel(map[level].data) < torch.numel(msk_1)) then
               print("no convolution since msk is larger than image");
               pyr1[level].orientation[ori] = {data = torch.DoubleTensor(map[level].data):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
               pyr2[level].orientation[ori] = {data = torch.DoubleTensor(map[level].data):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
            else
               ------------------------------- DONE - NO PADDING
               -------------------------------
               -------------------------------
               -------------------------------
               temp_data = torch.DoubleTensor(map[level].data):clone();
               module1:forward(temp_data:clone());   
               pyr1[level].orientation[ori] = {data = torch.DoubleTensor(module1.output):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
               ------------------------------- DONE - NO PADDING
               -------------------------------
               -------------------------------
               -------------------------------
               temp_data = torch.DoubleTensor(map[level].data):clone();
               module2:forward(torch.DoubleTensor(temp_data:size()):copy(temp_data));
               pyr2[level].orientation[ori] = {data = torch.DoubleTensor(module2.output):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
               -------------------------------
               -------------------------------
               -------------------------------
               -------------------------------
            end              
            msk1[level].orientation[ori] = {data = torch.DoubleTensor(msk_2):clone(), ori =  vmPrs.oris[ori] + (math.pi / 2)};
            msk2[level].orientation[ori] = {data = torch.DoubleTensor(msk_1):clone(), ori = vmPrs.oris[ori] + (math.pi / 2)};
         else
            print('Map is empty at specified level.');
         end
      end
   end
   
   return pyr1,msk1,pyr2,msk2;
end

--function: Von Mises Sum
function vonMisesSum(csPyr, vmPrs)
   local maxLevel = vmPrs.maxLevel;
   --local maxLevel = 5
   local map1 = {};
   local map2 = {};
   
   for level = 1, maxLevel do
      for ori = 1,vmPrs.numOri do
         map1[level] = { orientation = {} };
         map2[level] = { orientation = {} };
      end
   end
   
   --create pyramid of center surround convoled with von Mises distribution
   vmPyr1, msk1, vmPyr2, msk2 = vonMisesPyramid(csPyr,vmPrs);   
   for level = 1, maxLevel do
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
function borderPyramid(csPyrL,csPyrD,cPyr,params)
   bPrs = params.bPrs;
   vmPrs = params.vmPrs;
   
   vmL1, msk1, vmL2, msk2 = vonMisesSum(csPyrL,vmPrs);
   image.display{image = vmL1[2].orientation[2].data}
   vmD1, csmsk1, vmD2, csmsk2 = vonMisesSum(csPyrD, vmPrs);
   
end

--function: Make Border Ownership
function makeBorderOwnership(im_channels, params)
   local map = {};
   local imPyr = {};
   local EPyr = torch.DoubleTensor();
   local OPyr = {};

   --EXTRACT EDGES
   for m = 1,#im_channels do

      for sub = 1,#im_channels[m].subtype do
         map = torch.DoubleTensor(im_channels[m].subtype[sub].data):clone();
         imPyr = makePyramid(map,params);
         ------------------
         --Edge Detection--
         ------------------
         EPyr = edgeEvenPyramid(imPyr,params);
         OPyr, o = edgeOddPyramid(imPyr,params);
         cPyr = makeComplexEdge(EPyr,OPyr);
         ----------------------
         --Make Image Pyramid--
         ----------------------
         if(im_channels[m].subtype[sub].type == "Orientation") then
            csPyr = gaborPyramid(imPyr,im_channels[m].subtype[sub].ori,params);
         else
            csPyr = csPyramid(imPyr,params);
         end

         csPyrL,csPyrD = separatePyr(csPyr);
         csPyrL,csPyrD = normCSPyr2(csPyrL,csPyrD);
         -----------------------------------------------
         --Generate Border Ownership and Grouping Maps--
         -----------------------------------------------
         print(m,sub)
         bPyr1_1, bPyr2_1, bPyr1_2, bPyr2_2 = borderPyramid(csPyrL,csPyrD,cPyr,params);
         
      end

   end

end

--function: Degrees to Radians
function deg2rad(angleInDegrees)
   local angleInRadians = torch.mul(angleInDegrees,math.pi/180);
   return angleInRadians;
end

--function: Calc Sigma
function calcSigma(r,x)
   local sigma1 = (r^2) / (4 * math.log(x)) * (1-(1/(x^2)));
   sigma1 = math.sqrt(sigma1);
   local sigma2 = x * sigma1;
   
   return sigma1,sigma2;
end

--function: Get Default Parameters
function getDefaultParameters()

   local local_minLevel = 1;
   local local_maxLevel = 10;
   local local_downsample = 'half';
   local ori = torch.DoubleTensor({0,45});
   
   local local_oris = deg2rad(torch.DoubleTensor({ori[1], ori[2], ori[1] + 90, ori[2] + 90}));
   local local_lambda = 4;
   local local_odd_lambda = 4;
   local local_even_lambda = 4;
   
   local local_gabor_lambda = 8;
   
   local local_sigma1,local_sigma2 = calcSigma(2,3);
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
                      lambda = local_lambda,
                      sigma = 0.56 * local_even_lambda,
                      gamma = 0.5
                   },

      oddCellsPrs = {minLevel = local_minLevel,
                      maxLevel = local_maxLevel,
                      oris = local_oris,
                      numOri = local_oris:size(1),
                      lambda = local_lambda,
                      sigma = 0.56 * local_odd_lambda,
                      gamma = 0.5
                  },
      
      gaborPrs = {lambda = local_gabor_lambda,
                  sigma = 0.4 * local_gabor_lambda,
                  gamma = 0.8
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

--print("Starting Proto-object Based Saliency")

defaultParameters = getDefaultParameters();

--Read in image from file (column,row) / (x,y)
im = image.loadJPG('/home/jamal/ProtoObject/test.jpg')

--Normalize image
im = normalizeImage(im);

--Generate channels from image
--print("Generating color channels according to Itti et al (1998)");
im_channels = generateChannels(im,defaultParameters);
--print(im_channels[2].type);
--Evmsk = makeEvenOrientationCells(defaultParameters.evenCellsPrs.oris[4],defaultParameters.gaborPrs.lambda,defaultParameters.gaborPrs.sigma,defaultParameters.gaborPrs.gamma);
--print(Evmsk:size())
--print(Evmsk);
--image.display{image = Evmsk}
--print("starting convolution");
--module = nn.SpatialConvolution(1, 1, 7, 7);
--module.bias = torch.zero(module.bias);
--a = torch.DoubleTensor(3,3):zero();
--a[1][1] = 1;
--a[2][1] = 0;
--a[3][1] = -1;
--a[1][2] = 2;
--a[2][2] = 0;
--a[3][2] = -2;
--a[1][3] = -1;
--a[2][3] = 0;
--a[3][3] = 1;

--module.weight[1] = Evmsk;
--module:forward(image.rgb2y(im))
--image.display{image=module.output}
--print(module.output[1][100][100])
--print(image.rgb2y(im))orch.DoubleTensor(3,4):zero()
--d = torch.DoubleTensor(1,3,4):copy(torch.DoubleTensor(3,4))
--print(d)
makeBorderOwnership(im_channels,defaultParameters);
