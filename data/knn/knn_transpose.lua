------------------------------------------------------------------------------------
--  Torch Implementation of Stack attention based Networks for Visual  Question generation
--  momentum= 0.9, learning_rate=4e-4, batch_size=100, lr_decay= no 
--  dim_embed=512,dim_hidden= 512,dim_image= 4096.
--  th train -gpuid 1
------------------------------------------------------------------------------------

require 'nn'
require 'torch'
require 'hdf5'
require 'xlua'
-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')


-- Data input settings
cmd:option('-knn_idx_value_train_input', 'data/knn/knn_trainvalid_idx_file.h5', 'knn_idx_value_train_input')
cmd:option('-knn_idx_value_test_input', 'data/knn/knn_test_idx_file.h5', ' knn_idx_value_test_input')
--output transpose file
cmd:option('-out_name_train', 'data/knn/knn_trainvalid_idx_file_transpose.h5', 'output name train')
cmd:option('-out_name_test', 'data/knn/knn_test_idx_file_transpose.h5', 'output name test')
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

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
print('-------------------------------start loading Index ----------------------------')

	print('DataLoader loading  knn index list file : ', opt.knn_idx_value_train_input)
 	knn_train_index_list = hdf5.open(opt.knn_idx_value_train_input, 'r')


	print('DataLoader loading  knn index list file : ', opt.knn_idx_value_test_input)
 	knn_test_index_list = hdf5.open(opt.knn_idx_value_test_input, 'r')
 	
 	
        batch_knn_train_index_list=knn_train_index_list:read('/images_train'):all()
        print('batch_knn_train_index_list',batch_knn_train_index_list:size())
        batch_knn_train_index_list=batch_knn_train_index_list:t()
        print('After transpose batch_knn_train_index_list',batch_knn_train_index_list:size())
        
        
        
        batch_knn_test_index_list=knn_test_index_list:read('/images_test'):all()
        print('batch_knn_test_index_list',batch_knn_test_index_list:size())
        batch_knn_test_index_list=batch_knn_test_index_list:t()
        print('After transpose batch_knn_test_index_list',batch_knn_test_index_list:size())



local train_h5_file = hdf5.open(opt.out_name_train, 'w')
train_h5_file:write('/images_train', batch_knn_train_index_list)
train_h5_file:close()

local test_h5_file = hdf5.open(opt.out_name_test, 'w')
test_h5_file:write('/images_test', batch_knn_test_index_list)
test_h5_file:close()    


