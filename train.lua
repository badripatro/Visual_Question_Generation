------------------------------------------------------------------------------------
--  Torch Implementation of Stack attention based Networks for Visual  Question generation
--  momentum= 0.9, learning_rate=4e-4, batch_size=100, lr_decay= no 
--  dim_embed=512,dim_hidden= 512,dim_image= 4096.
--  th train -gpuid 1
------------------------------------------------------------------------------------

require 'nn'
require 'torch'
require 'rnn'
require 'optim' --this is for only log only not for update parameter
require 'misc.LanguageModel'
require 'misc.optim_updates'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local Multimodal=require 'misc.multimodal'
require 'xlua'
-------------------------------------------------------
-- require 'misc.TripletNet'
-- require 'misc.DistanceRatioCriterion'
require 'misc.TripletCriterion'
-----------------------------------------------------------------------------
colour = require 'trepl.colorize'
require 'trepl'
colour = require 'trepl.colorize'
local b = colour.blue
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings
cmd:option('-input_img_train_h5','data/img_train_fc7.h5','path to the h5file containing the image feature')
cmd:option('-input_img_test_h5','data/img_test_fc7.h5','path to the h5file containing the image feature')
cmd:option('-input_ques_h5','data/coco_data_prepro.h5','path to the h5file containing the preprocessed dataset')
cmd:option('-input_json','data/coco_data_prepro.json','path to the json file containing additional info and vocab')

-- Data input settings
cmd:option('-knn_idx_value_train_input', 'data/knn/knn_trainvalid_idx_file_transpose.h5', 'knn_idx_value_train_input')
cmd:option('-knn_idx_value_test_input', 'data/knn/knn_test_idx_file_transpose.h5', ' knn_idx_value_test_input')
-- cmd:option('-input_cap_train_h5','data/cap_train_fc7.h5','path to the h5file containing the image feature')
-- cmd:option('-input_cap_test_h5','data/cap_test_fc7.h5','path to the h5file containing the image feature')

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-feature_type', 'VGG', 'VGG or Residual')

-- Model settings
cmd:option('-batch_size',100,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'the encoding size of each token in the vocabulary, and the image.')
cmd:option('-att_size',512,'size of sttention vector which refer to k in paper')
cmd:option('-emb_size',512,'the size after embeeding from onehot')
cmd:option('-rnn_layers',1,'number of the rnn layer')

-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',0.00001,'learning rate')--0.0001,--0.0002,--0.005
cmd:option('-learning_rate_decay_start', -1, 'at what iteration to start decaying learning rate? (-1 = dont)')--learning_rate_decay_start', 100,
cmd:option('-learning_rate_decay_every', 5000, 'every how many iterations thereafter to drop LR by half?')---learning_rate_decay_every', 1500,
cmd:option('-momentum',0.9,'momentum')
cmd:option('-optim_alpha',0.8,'alpha for adagrad/rmsprop/momentum/adam')--optim_alpha',0.99
cmd:option('-optim_beta',0.999,'beta used for adam')--optim_beta',0.995
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1250)
cmd:option('-drop_prob_lm', 0.5, 'strength of drop_prob_lm in the Language Model RNN')


-- Evaluation/Checkpointing
cmd:text('===>Save/Load Options')
cmd:option('-save',               'Results', 'save directory')
cmd:option('-checkpoint_dir', 'Results/checkpoints', 'folder to save checkpoints into (empty = this folder)')
cmd:option('-language_eval', 1, 'Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
cmd:option('-val_images_use', 31225, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')
cmd:option('-save_checkpoint_every', 2500, 'how often to save a model checkpoint?')
cmd:option('-losses_log_every', 200, 'How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-backend', 'cudnn', 'nn|cudnn')
cmd:option('-id', '1', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-seed', 1234, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')


cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

opt = cmd:parse(arg)



---------------------------------------------------------------------
--Step 4: create directory and log file
------------------------------------------------------------------
------------------------- Output files configuration -----------------
os.execute('mkdir -p ' .. opt.save) -- to create result folder  save folder
cmd:log(opt.save .. '/Log_cmdline.txt', opt)  --save log file in save folder
--os.execute('cp ' .. opt.network .. '.lua ' .. opt.save)  -- to copy network to the save file path

-- to save model parameter
os.execute('mkdir -p ' .. opt.checkpoint_dir) 

-- to save log
local err_log_filename = paths.concat(opt.save,'ErrorProgress')
local err_log = optim.Logger(err_log_filename)

-- to save log
local triplet_err_log_filename = paths.concat(opt.save,'TripletErrorProgress')
local triplet_err_log = optim.Logger(triplet_err_log_filename)

-- to save log
local lang_stats_filename = paths.concat(opt.save,'language_statstics')
local lang_stats_log = optim.Logger(lang_stats_filename)

-------------------------------------------------------------------------------
-- Create the Data Loader instance
-------------------------------------------------------------------------------
-- dataloader
local dataloader = dofile('misc/dataloader.lua')
dataloader:initialize(opt)
collectgarbage()
------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
local loaded_checkpoint
local lmOpt
-- intialize language model
if string.len(opt.start_from) > 0 then
  -- load protos from file
  print('initializing weights from ' .. opt.start_from)
  local loaded_checkpoint = torch.load(opt.start_from)
  protos = loaded_checkpoint.protos
  
  local lm_modules = protos.lm:getModulesList()
  for k,v in pairs(lm_modules) do net_utils.unsanitize_gradients(v) end
  
  protos.crit = nn.LanguageModelCriterion() -- not in checkpoints, create manually

else
        -- create protos from scratch
        print('Building the model from scratch...')
        -- intialize language model
        lmOpt = {}
        lmOpt.vocab_size = dataloader:getVocabSize()
        lmOpt.input_encoding_size = opt.input_encoding_size
        lmOpt.rnn_size = opt.rnn_size
        lmOpt.num_layers = 1
        lmOpt.drop_prob_lm = opt.drop_prob_lm
        lmOpt.seq_length = dataloader:getSeqLength()
        lmOpt.batch_size = opt.batch_size 
        lmOpt.emb_size= opt.input_encoding_size
        lmOpt.hidden_size = opt.input_encoding_size
        lmOpt.att_size = opt.att_size
        lmOpt.num_layers = opt.rnn_layers

end

-- Design Model From scratch
---------------------------------------------------------------------------------------------
-- Encoding Part 

        -- Caption feature embedding
        --protos.emb = nn.emb_net(lmOpt) -- because problem in sharing network
        protos.emb = nn.Sequential()
                :add(nn.LookupTableMaskZero(lmOpt.vocab_size, lmOpt.input_encoding_size))
                :add(nn.Dropout(0.5))
                :add(nn.SplitTable(1, 2))
                :add(nn.Sequencer(nn.FastLSTM(lmOpt.input_encoding_size, lmOpt.rnn_size):maskZero(1)))
                :add(nn.Sequencer(nn.FastLSTM(lmOpt.rnn_size, lmOpt.rnn_size):maskZero(1)))
                :add(nn.SelectTable(-1))

        -- Image feature embedding
        protos.cnn = nn.Sequential()
                :add(nn.Linear(4096,opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
              
        -- Fusion feature embedding        
        protos.fuse = nn.Sequential()
		            :add(Multimodal.AcatB(opt.input_encoding_size,opt.input_encoding_size,opt.input_encoding_size,0.5))				
                :add(nn.BatchNormalization(opt.input_encoding_size))
                :add(nn.Tanh())
                :add(nn.Dropout(0.5))
                :add(nn.Linear(opt.input_encoding_size, opt.input_encoding_size))
		    
---------------------------------------------------------------------------------------------
-- Decoding Part 

        -- Question feature embedding
        protos.lm = nn.LanguageModel(lmOpt)
       
        -- criterion for the language model
        protos.crit = nn.LanguageModelCriterion()

--print('model',protos)
print('seq_length',lmOpt.seq_length)
---------------------------------------------------------------------------------------
print('ship everything to GPU...')
-- ship everything to GPU, maybe
if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local eparams, grad_eparams = protos.emb:getParameters()
local cparams, grad_cparams = protos.cnn:getParameters()
local fparams, grad_fparams = protos.fuse:getParameters()
local lparams, grad_lparams = protos.lm:getParameters()

eparams:uniform(-0.1, 0.1)
cparams:uniform(-0.1, 0.1) 
fparams:uniform(-0.1, 0.1)
lparams:uniform(-0.1, 0.1) 

if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  eparams:copy(loaded_checkpoint.eparams)
  cparams:copy(loaded_checkpoint.cparams)
  fparams:copy(loaded_checkpoint.fparams)
  lparams:copy(loaded_checkpoint.lparams)

end

print('total number of parameters in Question embedding net: ', eparams:nElement())
assert(eparams:nElement() == grad_eparams:nElement())

print('total number of parameters in Image  embedding net: ', cparams:nElement())
assert(cparams:nElement() == grad_cparams:nElement())


print('total number of parameters in fuse embedding net: ', fparams:nElement())
assert(fparams:nElement() == grad_fparams:nElement())

print('total number of parameters of language Generating model ', lparams:nElement())
assert(lparams:nElement() == grad_lparams:nElement())



collectgarbage() 

---------------------------------------------------------------
-- Triplet net
---------------------------------------------------------------
CreateTriplet = function(Net)
  
  convNetPos = Net:clone('weight', 'bias', 'gradWeight', 'gradBias')
  convNetNeg = Net:clone('weight', 'bias', 'gradWeight', 'gradBias')

  -- Parallel container
  prl = nn.ParallelTable()
  prl:add(Net)
  prl:add(convNetPos)
  prl:add(convNetNeg)
  print(b('Cloneing Image embedding network:')); print(prl)
  return prl
end


local img_TripletNet =  CreateTriplet(protos.cnn)
local cap_TripletNet =  CreateTriplet(protos.emb)
-- local cap_TripletNet =  nn.MapTable():add(protos.emb)
local fuse_TripletNet =  CreateTriplet(protos.fuse)
-- local fuse_TripletNet =  nn.MapTable():add(protos.fuse)
local triplet_criterion = nn.TripletCriterion()
---------------------------------------------------------------
-- Triplet net
---------------------------------------------------------------
print '==> Image Triplet Network'
print(img_TripletNet)
print '==> Caption Triplet Network'
print(cap_TripletNet)
print '==> Loss'
print(triplet_criterion)
-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(split)
  protos.emb:evaluate()
  protos.cnn:evaluate()
  protos.fuse:evaluate()
  protos.lm:evaluate()
	
	dataloader:resetIterator(2)-- 2 for test and 1 for train
	
        local verbose = utils.getopt(evalopt, 'verbose', false) -- to enable the prints statement  entry.image_id, entry.caption
        local val_images_use = utils.getopt(evalopt, 'val_images_use', true)

        local n = 0
        local loss_sum = 0
        local loss_evals = 0
        local right_sum = 0

        total_num = dataloader:getDataNum(2) -- 2 for test and 1 for train-- this will provide total number of example in the image 
        
        local predictions = {}
        local vocab = dataloader:getVocab()
          
  while true do
        --local data = loader:getBatch{batch_size = opt.batch_size, split = split}
        local batch = dataloader:next_batch_eval(opt)
        --print('Ques_cap_id In eval batch[3]',batch[3])
        local data = {}
        data.images=batch[1]-- check this in dataloader return sequence
        data.questions=batch[2]
        data.caption=batch[4]
        data.ques_id=batch[3]
     -------------------------------------------------------------------------------------
      	n = n + data.images:size(1)
      	xlua.progress(n, total_num)
    
        --------------------------------------------------------------------------------------
        local decode_question=data.questions:t()
        -------------------------------------------------------------------------------------------------------------------
        --Forward the Caption word feature through word embedding
        local cap_feat =protos.emb:forward(data.caption)

        -- forward the ConvNet on images 
        local img_feat=protos.cnn:forward(data.images)


        --Fusion on Image embedding and Caption Embedding 
        local fuse_feat = protos.fuse:forward({img_feat,cap_feat})

        -- forward the language model
        local logprobs = protos.lm:forward({fuse_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats


        -- forward the language model criterion
        local loss = protos.crit:forward(logprobs, decode_question)

        -------------------------------------------------------------------------------------------------------------------
 
        loss_sum = loss_sum + loss
        loss_evals = loss_evals + 1

        -- forward the model to also get generated samples for each image
        local seq = protos.lm:sample(fuse_feat)
        local sents = net_utils.decode_sequence(vocab, seq)
        for k=1,#sents do
                local entry = {image_id = data.ques_id[k], question = sents[k]} -- change here
                table.insert(predictions, entry) -- to save all the alements
                -------------------------------------------------------------------------
                -- for print log
                if verbose then
                        print(string.format('image %s: %s', entry.image_id, entry.question))
                end
                ------------------------------------------------------------------------
        end
        if n >= total_num then break end 
        if n >= opt.val_images_use then break end
   
  end
  ------------------------------------------------------------------------
  -- for blue,cider score
  local lang_stats
  if opt.language_eval == 1 then
          lang_stats = net_utils.language_eval(predictions, opt.id)
          local score_statistics = {epoch = epoch, statistics = lang_stats}
          print('Current language statistics',score_statistics)
  end
   ------------------------------------------------------------------------       
   -- write a (thin) json report-- for save image id and question print in json format
  local question_filename = string.format('%s/question_checkpoint_epoch%d', opt.checkpoint_dir, epoch)
  utils.write_json(question_filename .. '.json', predictions) -- for save image id and question print in json format
  print('wrote json checkpoint to ' .. question_filename .. '.json')

------------------------------------------------------------------------
  return loss_sum/loss_evals, predictions, lang_stats

end

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local te_optim_state = {}  --- to mentain state in optim
local tc_optim_state = {}  --- to mentain state in optim
local tf_optim_state = {}  --- to mentain state in optim

local function lossFun()

  protos.emb:training()
  protos.cnn:training()
  protos.fuse:training()
  protos.lm:training()
----------------------------------------------------------------------------
-- Get batch of data 
-----------------------------------------------------------------------------
        local batch = dataloader:next_batch(opt)        
        local data = {}
        data.images=batch[1]
        data.questions=batch[2]
        data.caption=batch[3]
        data.ques_id= batch[4]

        data.images_R=batch[5]
        data.images_NR=batch[6]        

        data.caption_R=batch[7]
        data.caption_NR=batch[8]
-------------------------------------------------------
        local decode_question= data.questions:t()   
---------------------------------------------------------------------------------------
-- Triplet Net Forward pass 
----------------------------------------------------------------------------------------
      --Triplet Caption Embedding features 
        local triplet_cap_emb =cap_TripletNet :forward({data.caption,data.caption_R,data.caption_NR})
        --Triplet Image Embedding features 
        local triplet_img_emb =img_TripletNet :forward({data.images,data.images_R,data.images_NR})
        --Triplet Fusion Embedding features
        local triplet_fuse_emb =fuse_TripletNet:forward({{triplet_img_emb[1],triplet_cap_emb[1]},{triplet_img_emb[2],triplet_cap_emb[2]},{triplet_img_emb[3],triplet_cap_emb[3]}})
        -- Triplet loss
        local triplet_loss=triplet_criterion:forward(triplet_fuse_emb)
---------------------------------------------------------------------------------------
-- Triplet Net Backward pass 
----------------------------------------------------------------------------------------
      grad_eparams:zero()  
      grad_cparams:zero() 
      grad_fparams:zero()  
      ------------------------------------------------------------------------------------------------------------------------------------
       local  dtriplet_loss=triplet_criterion:backward(triplet_fuse_emb)
       local  dtriplet_fuse =fuse_TripletNet:backward({{triplet_img_emb[1],triplet_cap_emb[1]},{triplet_img_emb[2],triplet_cap_emb[2]},{triplet_img_emb[3],triplet_cap_emb[3]}},dtriplet_loss)
       local dtriplet_img_emb =img_TripletNet:backward({data.images,data.images_R,data.images_NR},{dtriplet_fuse[1][1],dtriplet_fuse[2][1],dtriplet_fuse[3][1]})
       local dtriplet_cap_emb =cap_TripletNet :backward({data.caption,data.caption_R,data.caption_NR},{dtriplet_fuse[1][2],dtriplet_fuse[2][2],dtriplet_fuse[3][2]})
      ------------------------------------------------------------------------------------------------------------------------------------
      -------------------------------------------------------------------------------------
      --Triplet Net update weight
      ----------------------------------------------------------------------------------------
        rmsprop(eparams, grad_eparams, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, te_optim_state)
        rmsprop(cparams, grad_cparams, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, tc_optim_state)
        rmsprop(fparams, grad_fparams, opt.learning_rate, opt.optim_alpha, opt.optim_epsilon, tf_optim_state)
        
        -- sgdm(eparams, grad_eparams, opt.learning_rate, opt.momentum, te_optim_state)
        -- sgdm(cparams, grad_cparams, opt.learning_rate, opt.momentum, tc_optim_state) 
        -- sgdm(fparams, grad_fparams, opt.learning_rate, opt.momentum, tf_optim_state)
  
        -----------------------------------------------------------------------------
----------------------------------------------------------------------------
--  Main Net Forward pass
-----------------------------------------------------------------------------
        --Forward the Caption word feature through word embedding
        local cap_feat =protos.emb:forward(data.caption)

        -- forward the ConvNet on images 
        local img_feat=protos.cnn:forward(data.images)


        --Fusion on Image embedding and Caption Embedding 
        local fuse_feat = protos.fuse:forward({img_feat,cap_feat})

        -- forward the language model
        local logprobs = protos.lm:forward({fuse_feat, decode_question}) -- data.questions=data.labels, img_feat=expanded_feats


        -- forward the language model criterion
        local loss = protos.crit:forward(logprobs, decode_question)     
-----------------------------------------------------------------------------
--  Main Net Backward pass
-----------------------------------------------------------------------------
    grad_eparams:zero()  
    grad_cparams:zero() 
    grad_fparams:zero()  
    grad_lparams:zero() 
  	
        -- backprop criterion
        local dlogprobs = protos.crit:backward(logprobs, decode_question)
        -- backprop language model
        local d_lm_feats, ddummy = unpack(protos.lm:backward({fuse_feat, decode_question}, dlogprobs))
        -- backprop the CNN
        local dummy_img_feats = protos.cnn:backward(data.images, d_lm_feats)
        -- backprop the Caption Embedding
        local dummy_img_feats = protos.emb:backward(data.images, d_lm_feats)
  -----------------------------------------------------------------------------
  local losses = { total_loss = loss,triplet_loss=triplet_loss }
  return losses
end

-------------------------------------------------------------------------------
--Step 13:--Log Function
-------------------------------------------------------------------------------
function printlog(epoch,ErrTrain,ErrTest,triplet_error)
 	------------------------------------------------------------------------------
	-- log plot
	paths.mkdir(opt.save)
	err_log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
	err_log:style{['Training Error'] = '-', ['Test Error'] = '-'}
	err_log:plot()
	
	triplet_err_log:add{['Training Triplet Error']= triplet_error* 100}
	triplet_err_log:style{['Training Triplet Error'] = '-'}
	triplet_err_log:plot()
	---------------------------------------------------------------------------------
	---------------------------------------------------------------------------------
	if paths.filep(opt.save..'/ErrorProgress.eps') or paths.filep(opt.save..'/accuracyProgress.eps') then
		-----------------------------------------------------------------------------------------------------------
		-- convert .eps file as .png file
		local base64im
		do
			os.execute(('convert -density 200 %s/ErrorProgress.eps %s/ErrorProgress.png'):format(opt.save,opt.save))
			os.execute(('openssl base64 -in %s/ErrorProgress.png -out %s/ErrorProgress.base64'):format(opt.save,opt.save))
			local f = io.open(opt.save..'/ErrorProgress.base64')
			if f then base64im = f:read'*all' end
		end
		-- this is for Triplet Error
		local base64im_tr
		do
			os.execute(('convert -density 200 %s/TripletErrorProgress.eps %s/TripletErrorProgress.png'):format(opt.save,opt.save))
			os.execute(('openssl base64 -in %s/TripletErrorProgress.png -out %s/TripletErrorProgress.base64'):format(opt.save,opt.save))
			local f = io.open(opt.save..'/TripletErrorProgress.base64')
			if f then base64im_tr = f:read'*all' end
		end
		
		-----------------------------------------------------------------------------------------------------------------------
		-- to display in .html file
		local file = io.open(opt.save..'/report.html','w')
		file:write('<h5>Training data size:  '..total_train_example ..'\n')
		file:write('<h5>Validation data size:  '..total_num ..'\n')
		file:write('<h5>batchSize:  '..opt.batch_size..'\n')
		file:write('<h5>LR:  '..opt.learning_rate..'\n')
		file:write('<h5>optimization:  '..opt.optim..'\n')
		file:write('<h5>drop_prob_lm:  '..opt.drop_prob_lm..'\n')


		file:write(([[
		<!DOCTYPE html>
		<html>
		<body>
		<title>%s - %s</title>
		<img src="data:image/png;base64,%s">
		<h4>optimState:</h4>
		<table>
		]]):format(opt.save,epoch,base64im))
	
	--[[	for k,v in pairs(optim_state) do
			if torch.type(v) == 'number' then
			 	file:write('<tr><td>'..k..'</td><td>'..v..'</td></tr>\n')
			end
		end --]]

		file:write'</table><pre>\n'
		file:write'</pre></body></html>'
		file:close()
	end
--[[
	if opt.visualize then
		require 'image'
		local weights = EmbeddingNet:get(1).weight:clone()
		--win = image.display(weights,5,nil,nil,nil,win)
		image.saveJPG(paths.concat(opt.save,'Filters_epoch'.. epoch .. '.jpg'), image.toDisplayTensor(weights))
	end
--]]
	return 1	
end

-------------------------------------------------------------------------------
--Step 12:--Training Function
-------------------------------------------------------------------------------
local e_optim_state = {}  --- to mentain state in optim
local c_optim_state = {}  --- to mentain state in optim
local f_optim_state = {}  --- to mentain state in optim
local l_optim_state = {}  --- to mentain state in optim

local grad_clip = 0.1
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch) -- for lr decay
local learning_rate = opt.learning_rate


total_train_example = dataloader:getDataNum(1) -- for lr decay
train_nbatch=math.ceil(total_train_example /opt.batch_size)


function Train()
	count_sum=0  -- Cannt be make local bcz it is insisde the function and other function are using this.
	local iter=1	
	local ave_loss = 0  --for iter_log_print  train error
	err=0
	local triplet_err = 0	
	local triplet_err_local=0 	-- for iter_log_print  train error

	while iter <= train_nbatch do
		-- Training loss/gradient
		local losses = lossFun()
		err=err+ losses.total_loss
		ave_loss = ave_loss + losses.total_loss
		triplet_err=triplet_err+losses.triplet_loss
		triplet_err_local=triplet_err_local+losses.triplet_loss
		---------------------------------------------------------
		
		-- decay the learning rate  
		if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
                        local frac = (iter - opt.learning_rate_decay_start) / opt.learning_rate_decay_every
                        local decay_factor = math.pow(0.5, frac)
                        learning_rate = learning_rate * decay_factor -- set the decayed rate
		end
		---------------------------------------------------------
		if iter % opt.losses_log_every == 0 then
			ave_loss = ave_loss / opt.losses_log_every
			print(string.format('epoch:%d  iter %d: %f, %f, %f', epoch, iter, ave_loss,triplet_err_local,learning_rate, timer:time().real))
			ave_loss = 0
			triplet_err_local=0
			collectgarbage()
		end
		---------------------------------------------------------
    -- perform a parameter update
    if opt.optim == 'sgd' then
      sgdm(eparams, grad_eparams, learning_rate, opt.momentum, e_optim_state)
      sgdm(cparams, grad_cparams, learning_rate, opt.momentum, c_optim_state) 
      sgdm(fparams, grad_fparams, learning_rate, opt.momentum, f_optim_state)
      sgdm(lparams, grad_lparams, learning_rate, opt.momentum, l_optim_state)          
    elseif opt.optim == 'rmsprop' then
      rmsprop(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      rmsprop(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      rmsprop(fparams, grad_fparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, f_optim_state)
      rmsprop(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
    elseif opt.optim == 'adam' then
      adam(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, e_optim_state)
      adam(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, c_optim_state)
      adam(fparams, grad_fparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, f_optim_state)
      adam(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_beta, opt.optim_epsilon, l_optim_state)
    elseif opt.optim == 'sgdm' then
      sgdm(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      sgdm(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      sgdm(fparams, grad_fparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, f_optim_state)
      sgdm(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
    elseif opt.optim == 'sgdmom' then
      sgdmom(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      sgdmom(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      sgdmom(fparams, grad_fparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, f_optim_state)
      sgdmom(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)
    elseif opt.optim == 'adagrad' then
      adagrad(eparams, grad_eparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, e_optim_state)
      adagrad(cparams, grad_cparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, c_optim_state)
      adagrad(fparams, grad_fparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, f_optim_state)
      adagrad(lparams, grad_lparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, l_optim_state)	        				
		else
			error('bad option opt.optim')
		end
		---------------------------------------------------------
		iter = iter + 1
                if iter % 10 == 0 then collectgarbage() end -- good idea to do this once in a while, i think
                if loss0 == nil then loss0 = losses.total_loss end
                if losses.total_loss > loss0 * 20 then
                        print('loss seems to be exploding, quitting.')
                        break
                end

	end
	return err/train_nbatch, triplet_err/train_nbatch
end

local best_score_Bleu_1
local best_score_Bleu_2
local best_score_Bleu_3
local best_score_Bleu_4
local best_score_ROUGE_L
local best_score_METEOR
local best_score_CIDEr
--local best_score_SPICE
-------------------------------------------------------------------------------
--Step 14:-- Main loop
-------------------------------------------------------------------------------
epoch = 1  -- made gloobal ,bcz inside training function, it is used
print '\n==> Starting Training\n'
while epoch ~= opt.epoch do
	
	print('Epoch ' .. epoch)
  --local val_loss, val_predictions, lang_stats = eval_split(2)
  local ErrTrain,triplet_error = Train()
	
	print('Checkpointing. Calculating validation accuracy..')
	local val_loss, val_predictions, lang_stats = eval_split(2)
	print('------------------------------------------------------------------------')
	print('Training Error:  ', ErrTrain ,'Validation loss: ', val_loss, 'Triplet Error',triplet_error)
	
	
	local result=printlog(epoch,ErrTrain,val_loss,triplet_error)
	-----------------------------------------------------------

	-- for print best score
	-- write the full model checkpoint as well if we did better than ever	        
        local current_score_Bleu_1
        local current_score_Bleu_2
        local current_score_Bleu_3
        local current_score_Bleu_4
        local current_score_ROUGE_L
        local current_score_METEOR
        local current_score_CIDEr
        --local current_score_SPICE
       
        
        if lang_stats then
                -- use CIDEr score for deciding how well we did                
                current_score_Bleu_1 = lang_stats['Bleu_1']
                current_score_Bleu_2 = lang_stats['Bleu_2']
                current_score_Bleu_3 = lang_stats['Bleu_3']
                current_score_Bleu_4 = lang_stats['Bleu_4']
                current_score_ROUGE_L = lang_stats['ROUGE_L']
                current_score_METEOR = lang_stats['METEOR']
                current_score_CIDEr = lang_stats['CIDEr']
               -- current_score_SPICE = lang_stats['SPICE']
                
        else
                -- use the (negative) validation loss as a score
                
                current_score_Bleu_1 = -val_loss
                current_score_Bleu_2 = -val_loss
                current_score_Bleu_3 = -val_loss
                current_score_Bleu_4 =-val_loss
                current_score_ROUGE_L = -val_loss
                current_score_METEOR = -val_loss
                current_score_CIDEr = -val_loss
                --current_score_SPICE = -val_loss
        end


        
        if best_score_Bleu_1 == nil or current_score_Bleu_1 > best_score_Bleu_1 then
                best_score_Bleu_1 = current_score_Bleu_1
        end
        
        if best_score_Bleu_2 == nil or current_score_Bleu_2 > best_score_Bleu_2 then
                best_score_Bleu_2 = current_score_Bleu_2
        end
        
        if best_score_Bleu_3 == nil or current_score_Bleu_3 > best_score_Bleu_3 then
                best_score_Bleu_3 = current_score_Bleu_3
        end
        
        if best_score_Bleu_4 == nil or current_score_Bleu_4 > best_score_Bleu_4 then
                best_score_Bleu_4 = current_score_Bleu_4
        end
        
        if best_score_ROUGE_L == nil or current_score_ROUGE_L > best_score_ROUGE_L then
                best_score_ROUGE_L = current_score_ROUGE_L
        end
        
        if best_score_METEOR == nil or current_score_METEOR > best_score_METEOR then
                best_score_METEOR = current_score_METEOR
        end
        
        if best_score_CIDEr == nil or current_score_CIDEr > best_score_CIDEr then
                best_score_CIDEr = current_score_CIDEr
        end
        
         --if best_score_SPICE == nil or current_score_SPICE > best_score_SPICE then
         --       best_score_SPICE = current_score_SPICE
       -- end
        
        print('-----------------------------------------------------------------------------------------')
         print('current_Bleu_1:', current_score_Bleu_1,'current_Bleu_2:', current_score_Bleu_2,'current_Bleu_3:', current_score_Bleu_3,'current_Bleu_4:', current_score_Bleu_4) 
         print('current_ROUGE_L:', current_score_ROUGE_L, 'current_METEOR:',current_score_METEOR, 'And current_CIDEr:',current_score_CIDEr) 
        print('-----------------------------------------------------------------------------------------')
         print('best_Bleu_1:', best_score_Bleu_1,'best_Bleu_2:', best_score_Bleu_2,'best_Bleu_3:', best_score_Bleu_3,'best_Bleu_4:', best_score_Bleu_4) 
         print('best_ROUGE_L:', best_score_ROUGE_L, 'best_METEOR:',best_score_METEOR, 'And best_CIDEr:',best_score_CIDEr) 
         print('-----------------------------------------------------------------------------------------')
         --print('Current language statistics',lang_stats)      
          ----------------------------------------------------------------------------------------
        -- for print log      
        lang_stats_log:add{['Bleu_1']= current_score_Bleu_1, ['Bleu_2'] = current_score_Bleu_2,['Bleu_3'] = current_score_Bleu_3,['Bleu_4'] = current_score_Bleu_4,['ROUGE_L'] = current_score_ROUGE_L,['METEOR'] = current_score_METEOR,['CIDEr'] = current_score_CIDEr}

        lang_stats_log:style{['Bleu_1']= '-', ['Bleu_2'] = '-',['Bleu_3'] = '-',['Bleu_4'] = '-',['ROUGE_L'] = '-',['METEOR'] = '-',['CIDEr'] = '-'}

        lang_stats_log:plot()	
	-----------------------------------------------------------------------------------
                
        ---------------------------------------------------------------------------------------------------------------------------------------------
        local model_save_filename = string.format('%s/model_epoch%d.t7', opt.checkpoint_dir, epoch)
        if epoch % 100==0 then --dont save on very first iteration
                torch.save(model_save_filename, {eparams=eparams, cparams=cparams,fparams=fparams,lparams=lparams, lmOpt=lmOpt})  -- vocabulary mapping is included here, so we can use the checkpoint 
        end
	print('Saving current checkpoint to:', model_save_filename)

	epoch = epoch+1
end
