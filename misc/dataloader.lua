------------------------------------------------------------------------------
-- this is dataloader for sequential example reading
---------------------------------------------------------------------------------------
require 'hdf5'
cjson = require 'cjson'
utils = require 'misc/utils'

local dataloader = {}

function dataloader:initialize(opt)
    print('Reading ' .. opt.input_json)
    local file = io.open(opt.input_json, 'r')
    local text = file:read()
    file:close()
    local params = cjson.decode(text)
    for k,v in pairs(params) do self[k] = v end
    self['vocab_size'] = 0 for i,w in pairs(self['ix_to_word']) do self['vocab_size'] = self['vocab_size'] + 1 end

------------------------------------------------------------------------------------------------------
-- this is for getting image information
  
         if opt['input_img_train_h5'] ~= nil then
                print('Reading DataLoader loading h5 image file:' .. opt['input_img_train_h5'])
                self.h5_img_file_train = hdf5.open(opt['input_img_train_h5'], 'r')
               
               --for Knn feat
                print('Reading DataLoader knn loading h5 image file:' .. opt['knn_idx_value_train_input'])
                self.knn_train_index_list = hdf5.open(opt['knn_idx_value_train_input'], 'r')

                -- --for caption feat, total 3.7k caption feat are present which is same as total number of image
                -- print('Reading DataLoader loading h5 image file:' .. opt['input_cap_train_h5'])
                -- self.h5_cap_file_train = hdf5.open(opt['input_cap_train_h5'], 'r')

        end
        if opt['input_img_test_h5'] ~= nil then
                print('Reading  DataLoader loading h5 image file: ' .. opt['input_img_test_h5'])
                self.h5_img_file_test  = hdf5.open(opt['input_img_test_h5'], 'r')
               
                --for Knn feat
                print('Reading DataLoader loading h5 Knn  file:' .. opt['knn_idx_value_test_input'])
                self.knn_test_index_list = hdf5.open(opt['knn_idx_value_test_input'], 'r')

                -- --for caption feat, total 1.25k caption feat are present which is same as total number of image
                -- print('Reading DataLoader loading h5 image file:' .. opt['input_cap_test_h5'])
                -- self.h5_cap_file_test = hdf5.open(opt['input_cap_test_h5'], 'r')
        end

----------------------------------------------------------------------------------------------------------
        -- this is getting question information 
        print ('DataLoader loading h5 question file: ',opt.input_ques_h5)
        local qa_data = hdf5.open(opt.input_ques_h5, 'r')

       -- if split == 'train' then
       -- split is not required bcz here , u have chanaged variale name as like ques_train from ques and ques_test from ques which implecite indicate split
                -- image
                self['im_list_train']   = qa_data:read('/img_pos_train'):all()  --or self['im_list_train'] =self.im_list_train
                -- question
                self['ques_train']      = qa_data:read('/ques_train'):all()
                self['ques_len_train']  = qa_data:read('ques_length_train'):all()
                --self['ques_train']      = utils.right_align(self['ques_train'], self['ques_len_train'])-- you will get  bad argument #1 to 'unpack' (table expected, got nil)--unpack(self.state[t-1])}
                self['ques_id_train']   = qa_data:read('/question_id_train'):all()
                -- caption
                self['cap_train']       = qa_data:read('/cap_train'):all()
                self['cap_len_train']   = qa_data:read('cap_length_train'):all()
                self['cap_train']       = utils.right_align(self['cap_train'], self['cap_len_train'])
                -- answer
               -- self['ans_train']       = qa_data:read('/answers'):all()
                self['train_id']  = 1
                self.seq_length = self.ques_train:size(2)
                
                -- to print complete size of each split
                print('self[ques_train]:size(1)',self['ques_train']:size(1))
                
        --elseif split == 'test' then 
        -- split is not required bcz here , u have chanaged variale name as like ques_train from ques and ques_test from ques which implecite indicate split
        
                -- image
                self['im_list_test']   = qa_data:read('/img_pos_test'):all()
                -- question
                self['ques_test']      = qa_data:read('/ques_test'):all()
                self['ques_len_test']  = qa_data:read('ques_length_test'):all()
                --self['ques_test']      = utils.right_align(self['ques_test'], self['ques_len_test'])
                self['ques_id_test']   = qa_data:read('/question_id_test'):all()
                -- caption
                self['cap_test']       = qa_data:read('/cap_test'):all()
                self['cap_len_test']   = qa_data:read('cap_length_test'):all()
                self['cap_test']       = utils.right_align(self['cap_test'], self['cap_len_test'])
                -- answer
                --self['ans_test']       = qa_data:read('/answers_test'):all()
                self['test_id']   = 1
                -- to print complete size of each split
                print('self[ques_test]:size(1)',self['ques_test']:size(1))
        --end
        qa_data:close()
end

-----------------------------------------------------------------------------------------------------------------------------------------------------
function dataloader:next_batch(opt)
    local start_id = self['train_id'] -- start id , and it  it wiil be remember for next batch
    --print("self['ques_train']:size(1)",self['ques_train']:size(1))
    if start_id + opt.batch_size - 1 <= self['ques_train']:size(1) then 
        end_id = start_id + opt.batch_size - 1        
    else 
        self['train_id'] =1  --reset train id to 1
        start_id = self['train_id']
        end_id = start_id + opt.batch_size - 1 
        print('end of epoch')    
    end
    
--------------------------------------------------------------------------------------------------------
    local iminds = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  question index
    local qinds = torch.LongTensor(end_id - start_id + 1):fill(0) -- to keep track of  question index
    local im    = torch.LongTensor(opt.batch_size, 4096):fill(0)   --14, 14, 512):fill(0)  --chanaged for fc7 -- for store img batch of size 14x14x512 changed to 4096
    -----------------------------------------------------------------
    local iminds_R = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  question index
    local iminds_NR = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  question index

    local im_R    = torch.LongTensor(opt.batch_size, 4096):fill(0)   --14, 14, 512):fill(0)for attention  --chanaged to 4096 for fc7  -- for store img batch of size 14x14x512 changed to 4096
    local im_NR    = torch.LongTensor(opt.batch_size, 4096):fill(0)   --14, 14, 512):fill(0)for attention  --chanaged to 4096 for fc7 -- for store img batch of size 14x14x512 changed to 4096
    

    local capinds_R = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  Caption index
    local capinds_NR = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  Caption index
    local cap_R    = torch.LongTensor(opt.batch_size, 26):fill(0)   --26 sequence length --chanaged for caption -- for store img batch of size 14x14x512 changed to 4096
    local cap_NR    = torch.LongTensor(opt.batch_size, 26):fill(0)  --26 sequence length --chanaged for caption -- for store img batch of size 14x14x512 changed to 4096

    local batch_knn_train_index_list = torch.LongTensor(opt.batch_size, 200):fill(0)
    local iter = torch.LongTensor(end_id - start_id + 1):fill(0) -- to keep track of  question index
    local iminds_knn = torch.LongTensor(end_id - start_id + 1):fill(0)-- to keep track of  question index
    ------------------------------------------------------------

    for i = 1, end_id - start_id + 1 do    
        qinds[i] = start_id + i - 1               -- this  is for sequential
        iminds[i] = self['im_list_train'][qinds[i]] -- extract image id from image list 
        im[i] =  self.h5_img_file_train:read('/images_train'):partial({iminds[i],iminds[i]},{1,4096}) --{1,14},{1,14},{1,512}) --chanaged for fc7
        
        --------------------------------------------------------------
        --to find one batch 
        iter[i] = start_id + i - 1               -- this  is for sequential
        iminds_knn[i] = self['im_list_train'][iter[i]]        -- extract image id from image list  
        batch_knn_train_index_list[i]=self.knn_train_index_list:read('/images_train'):partial(iminds_knn[i],{1,200})
       
        iminds_R[i] = batch_knn_train_index_list[i][2]                -- extract image id from image list
        iminds_NR[i] = batch_knn_train_index_list[i][200]               -- extract image id from image list  
        
        im_R[i] =  self.h5_img_file_train:read('/images_train'):partial({iminds_R[i],iminds_R[i]},{1,4096})         --{1,14},{1,14},{1,512}) --chanaged for fc7
        im_NR[i] =  self.h5_img_file_train:read('/images_train'):partial({iminds_NR[i],iminds_NR[i]},{1,4096})      --{1,14},{1,14},{1,512}) --chanaged for fc7
        

        --   --2 for caption
        -- capinds_R[i] = batch_knn_train_index_list[i][2]                -- extract image id from image list
        -- capinds_NR[i] = batch_knn_train_index_list[i][200]               -- extract image id from image list  
        -- cap_R[i] =  self.h5_cap_file_train:read('/images_train'):partial({capinds_R[i],capinds_R[i]},{1,26})         --{1,14},{1,14},{1,512}) --chanaged for fc7
        -- cap_NR[i] =  self.h5_cap_file_train:read('/images_train'):partial({capinds_NR[i],capinds_NR[i]},{1,26})      --{1,14},{1,14},{1,512}) --chanaged for fc7
        
        --------------------------------------------------------------       
    end

    local ques    = self['ques_train']:index(1, qinds)
    local cap     = self['cap_train']:index(1, qinds)
    local ques_id = self['ques_id_train']:index(1, qinds)

    if opt.gpuid >= 0 then
        im     = im:cuda()
        ques   = ques:cuda()        
        cap    = cap:cuda()
        im_R   = im_R:cuda()
        im_NR  = im_NR:cuda()

        --2 for caption
        cap_R  = cap:cuda() -- for caption related input
        cap_NR = cap:cuda() -- for caption  non related input

    end

        self['train_id'] = self['train_id'] + end_id - start_id + 1   -- self['test_id']=  self.test_id both have same meaning
    return {im, ques,cap,ques_id,im_R,im_NR,cap_R,cap_NR}
end

function dataloader:next_batch_eval(opt)
    local start_id = self['test_id']
    local end_id = math.min(start_id + opt.batch_size - 1, self['ques_test']:size(1))  --here it do sequential basic because it will check complete data set

    local iminds = torch.LongTensor(end_id - start_id + 1):fill(0)
    local qinds = torch.LongTensor(end_id - start_id + 1):fill(0)
    local im    = torch.LongTensor(end_id - start_id + 1, 4096):fill(0)   --14, 14, 512):fill(0)--chanaged for fc7

    for i = 1, end_id - start_id + 1 do
        qinds[i] = start_id + i - 1
        iminds[i] = self['im_list_test'][qinds[i]]
        im[i] = self.h5_img_file_test:read('/images_test'):partial({iminds[i],iminds[i]},{1,4096}) --{1,14},{1,14},{1,512}) --chanaged for fc7
    end


    local ques    = self['ques_test']:index(1, qinds)
    local ques_id = self['ques_id_test']:index(1, qinds)
    local cap     = self['cap_test']:index(1, qinds)

    if opt.gpuid >= 0 then
        im     = im:cuda()
        ques   = ques:cuda()
        cap    = cap:cuda()
    end
    self['test_id'] = self['test_id'] + end_id - start_id + 1   -- self['test_id']=  self.test_id both have same meaning

    return {im, ques, ques_id,cap}
end
function dataloader:getVocab(opt)
     return self.ix_to_word
end

function dataloader:getVocabSize()
    return self['vocab_size'] -- or self.vocab_size
end

function dataloader:resetIterator(split)
        if split ==1 then 
                self['train_id'] = 1
        end
        if split ==2  then
                self['test_id']=1
        end
end


function dataloader:getDataNum(split)
        if split ==1 then 
               return self['ques_train']:size(1)
        end
        if split ==2  then
             return  self['ques_test']:size(1)
        end
end

function dataloader:getSeqLength()
  return self.seq_length
end


return dataloader
