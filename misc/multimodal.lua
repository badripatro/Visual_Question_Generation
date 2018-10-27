require 'nn'
require 'nngraph'

local Multimodel = {}

-----------------------------------------------------------------------------
--Joint Model :Hardmard product   
-----------------------------------------------------------------------------
-- to multiply a with b= AXB
function Multimodel.AmulB(nhA,nhB,nhcommon,dropout) 
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
	local output=nn.CMulTable()({qc,ic});
	return nn.gModule({q,i},{output});
end
-----------------------------------------------------------------------------
--Joint Model :Joint concatinate  
-----------------------------------------------------------------------------
-- to multiply a with b= A_B
function Multimodel.AcatB(nhA,nhB,nhcommon,dropout)
        dropout = dropout or 0 
        local q=nn.Identity()();
        local i=nn.Identity()();
        local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
        local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
        local joint_output=nn.JoinTable(2)({qc,ic});-- will get twice the dim vector
        local output=nn.Tanh()(nn.Linear(2*nhcommon,nhcommon)(nn.Dropout(dropout)(joint_output)));
        return nn.gModule({q,i},{output});
end

-----------------------------------------------------------------------------
--Joint Model :Add elementry 
-----------------------------------------------------------------------------
-- to multiply a with b= A+B
function Multimodel.AaddB(nhA,nhB,nhcommon,dropout) 
	dropout = dropout or 0 
	local q=nn.Identity()();
	local i=nn.Identity()();
	local qc=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(q)));
	local ic=nn.Tanh()(nn.Linear(nhB,nhcommon)(nn.Dropout(dropout)(i)));
	local output=nn.CAddTable()({qc,ic});
	return nn.gModule({q,i},{output});
end
-----------------------------------------------------------------------------
--Attention : Staack Attention  
-----------------------------------------------------------------------------
-- Apply Attention between image with language= A atten B
--m=196,d=1024,
--input_size=1024, att_size=512, img_seq_size=196, output_size=1000, drop_ratio=0.5
function Multimodel.stack_atten(input_size, att_size, img_seq_size, output_size, drop_ratio)
	local inputs = {}
	local outputs = {}
	table.insert(inputs, nn.Identity()()) 
	table.insert(inputs, nn.Identity()()) 

	local ques_feat = inputs[1]	-- [batch_size, d]
	local img_feat = inputs[2]	-- [batch_size, m, d]

	-- Stack 1
	local ques_emb_1 = nn.Linear(input_size, att_size)(ques_feat)   -- [batch_size, att_size]
	local ques_emb_expand_1 = nn.Replicate(img_seq_size,2)(ques_emb_1)         -- [batch_size, m, att_size]
	local img_emb_dim_1 = nn.Linear(input_size, att_size, false)(nn.View(-1,input_size)(img_feat)) -- [batch_size*m, att_size]
	local img_emb_1 = nn.View(-1, img_seq_size, att_size)(img_emb_dim_1)        		          -- [batch_size, m, att_size]
	local h1 = nn.Tanh()(nn.CAddTable()({ques_emb_expand_1, img_emb_1}))
	local h1_drop = nn.Dropout(drop_ratio)(h1)	                     	  -- [batch_size, m, att_size]
	local h1_emb = nn.Linear(att_size, 1)(nn.View(-1,att_size)(h1_drop))  -- [batch_size * m, 1]
	local p1 = nn.SoftMax()(nn.View(-1,img_seq_size)(h1_emb))       -- [batch_size, m]
	local p1_att = nn.View(1,-1):setNumInputDims(1)(p1)             -- [batch_size, 1, m]
	-- Weighted Sum
	local img_Att1 = nn.MM(false, false)({p1_att, img_feat})	    -- [batch_size, 1, d]
	local img_att_feat_1 = nn.View(-1, input_size)(img_Att1)	    -- [batch_size, d]
	local u1 = nn.CAddTable()({ques_feat, img_att_feat_1})	    -- [batch_size, d]


	-- Stack 2
	local ques_emb_2 = nn.Linear(input_size, att_size)(u1)          -- [batch_size, att_size] 
	local ques_emb_expand_2 = nn.Replicate(img_seq_size,2)(ques_emb_2) 	    -- [batch_size, m, att_size]
	local img_emb_dim_2 = nn.Linear(input_size, att_size, false)(nn.View(-1,input_size)(img_feat)) -- [batch_size*m, att_size]
	local img_emb_2 = nn.View(-1, img_seq_size, att_size)(img_emb_dim_2)			          -- [batch_size, m, att_size]
	local h2 = nn.Tanh()(nn.CAddTable()({ques_emb_expand_2, img_emb_2}))
	local h2_drop = nn.Dropout(drop_ratio)(h2)    	          -- [batch_size, m, att_size]
	local h2_emb = nn.Linear(att_size, 1)(nn.View(-1,att_size)(h2_drop)) -- [batch_size * m, 1]
	local p2 = nn.SoftMax()(nn.View(-1,img_seq_size)(h2_emb))       -- [batch_size, m]
	local p2_att = nn.View(1,-1):setNumInputDims(1)(p2)             -- [batch_size, 1, m]
	-- Weighted Sum
	local img_Att2 = nn.MM(false, false)({p2_att, img_feat})        -- [batch_size, 1, d]
	local img_att_feat_2 = nn.View(-1, input_size)(img_Att2)        -- [batch_size, d]
	local u2 = nn.CAddTable()({u1, img_att_feat_2})		                  -- [batch_size, d]


	-- Final Answer Prdict
	--local score = nn.Linear(input_size, output_size)(u2)	    -- [batch_size, 1000]  shifted to main finction
	-- table.insert(outputs, score)
	table.insert(outputs, u2)   

	return nn.gModule(inputs, outputs)
end

-----------------------------------------------------------------------------
--Word Embedding 
-----------------------------------------------------------------------------

function Multimodel.wordembedding(vocab_size, embedding_size, conv_size, seq_length)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 

    local seq = inputs[1]

    local embed = nn.Dropout(0.5)(nn.Tanh()(nn.LookupTableMaskZero(vocab_size, embedding_size)(seq)))
    
    table.insert(outputs, embed)
    
    return nn.gModule(inputs, outputs)
end

-----------------------------------------------------------------------------
--Phrase Embedding 
-----------------------------------------------------------------------------
function Multimodel.phraseembedding(conv_size,embedding_size, seq_length)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 

    local embed = inputs[1]

    local unigram = cudnn.TemporalConvolution(embedding_size, conv_size, 1, 1, 0)(embed)
    local bigram = cudnn.TemporalConvolution(embedding_size, conv_size, 2, 1, 1)(embed)
    local trigram = cudnn.TemporalConvolution(embedding_size,conv_size,3, 1, 1)(embed)

    local bigram = nn.Narrow(2,1,seq_length)(bigram)

    local unigram_dim = nn.View(-1, seq_length, conv_size, 1)(unigram)
    local bigram_dim = nn.View(-1, seq_length, conv_size, 1)(bigram)
    local trigram_dim = nn.View(-1, seq_length, conv_size, 1)(trigram)

    local feat = nn.JoinTable(4)({unigram_dim, bigram_dim, trigram_dim})
    local max_feat = nn.Dropout(0.5)(nn.Tanh()(nn.Max(3, 3)(feat)))

    table.insert(outputs, max_feat)

    return nn.gModule(inputs, outputs)
end


-----------------------------------------------------------------------------
--multiplication
-----------------------------------------------------------------------------
function Multimodel.mulAttention(nhA,nhcommon,dropout)
--f=A+ l[(AxB) - (AxC)+(AxBxC)]
    
  	local A = nn.Identity()();   --batch_size*1000 -- A=feat_anchor,B=feat_pos,C=feat_neg
	local B = nn.Identity()() ;  --batch_size*1000
	local C = nn.Identity()();   --batch_size*1000
	
	local Relavent_context=nn.CMulTable()({A,B});	--batch_size*1000 
	local Irrelevent_context=nn.CMulTable()({A,C}); --batch_size*1000 --acquaintance=a person one knows slightly, but who is not a close friend
	local Common_context=nn.CMulTable()({Relavent_context,C}); --batch_size*1000 --axbxc
	local oc=nn.CSubTable()({Relavent_context,Irrelevent_context});
	local output_1=nn.CAddTable()({Common_context,oc});
	local ic=nn.Tanh()(nn.Linear(nhA,nhcommon)(nn.Dropout(dropout)(output_1)));
	local output=nn.CAddTable()({A,ic});

	return nn.gModule({A,B,C},{output});

end


return Multimodel

