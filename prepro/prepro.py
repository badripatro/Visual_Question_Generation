"""
Preoricess a raw json dataset into hdf5/json files.

Caption: Use NLTK or split function to get tokens. 
"""
"""
python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000


python prepro_vqa.py --input_train_json ../data/vqa_raw_train.json --input_test_json ../data/vqa_raw_test.json --num_ans 1000

python prepro_vqa.py

To get the question features. --num_ans specifiy how many top answers you want to use during training. You will also see some question and answer statistics in the terminal output. This will generate two files in data/ folder, vqa_data_prepro.h5 and vqa_data_prepro.json.


For question generation , here i am commenting  " filter_question" during encoding 
"""

import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json
import spacy.en 

import re
import math


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(imgs, params):
  
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['question']
        if params['token_method'] == 'nltk':
            txt = word_tokenize(s)
        #elif params['token_method'] == 'spacy':
        #    txt = [token.norm_ for token in params['spacy'](s)]
        else:
        	txt = tokenize(s)
        img['processed_tokens'] = txt
        if i < 10: print txt
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def prepro_caption(imgs, params):
  
    # preprocess all the question
    print 'example processed tokens:'
    for i,img in enumerate(imgs):
        s = img['caption']
        if params['token_method'] == 'nltk':
            txt_c = word_tokenize(s)
	    # elif params['token_method'] == 'spacy':
            # txt_c = [token.norm_ for token in params['spacy'](s)]
        else:
        	txt_c = tokenize(s)

        img['processed_tokens_caption'] = txt_c
        if i < 10: print txt_c
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %  (i, len(imgs), i*100.0/len(imgs)) )
            sys.stdout.flush()   
    return imgs

def build_vocab_question(imgs, params):
    # build vocabulary for question and answers.

    count_thr = params['word_count_threshold']

    # count up the number of words
    counts = {}
    for img in imgs:
        for w in img['processed_tokens']:
            counts[w] = counts.get(w, 0) + 1
        for w in img['processed_tokens_caption']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top words and their counts:'
    print '\n'.join(map(str,cw[:20]))

    # print some stats
    total_words = sum(counts.itervalues())
    print 'total words:', total_words
    bad_words = [w for w,n in counts.iteritems() if n <= count_thr]
    vocab = [w for w,n in counts.iteritems() if n > count_thr]   # will incorpate vocab for  both caption and question
    bad_count = sum(counts[w] for w in bad_words)
    print 'number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts))
    print 'number of words in vocab would be %d' % (len(vocab), )
    print 'number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words)

    # additional special UNK token we will use below to map infrequent words to
    print 'inserting the special UNK token'
    vocab.append('UNK')
  
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_caption']
        caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt_c]
        img['final_caption'] = caption   

    return imgs, vocab

def apply_vocab_question(imgs, wtoi):  ## this is for val or test question and caption 
    # apply the vocab on test.
    for img in imgs:
        txt = img['processed_tokens']
        question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in txt]
        img['final_question'] = question
        txt_c = img['processed_tokens_caption']
        caption = [w if w in wtoi else 'UNK' for w in txt_c]
        img['final_caption'] = caption  

    return imgs

def get_top_answers(imgs, params):
    counts = {}
    for img in imgs:
        ans = img['ans'] 
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count,w) for w,count in counts.iteritems()], reverse=True)
    print 'top answer and their counts:'    
    print '\n'.join(map(str,cw[:20]))
    
    vocab = []
    for i in range(params['num_ans']):
        vocab.append(cw[i][1])

    return vocab[:params['num_ans']]

def encode_question(imgs, params, wtoi):

    max_length = params['max_length']
    N = len(imgs)

    label_arrays = np.zeros((N, max_length), dtype='uint32')
    label_length = np.zeros(N, dtype='uint32')
    question_id = np.zeros(N, dtype='uint32')
    question_counter = 0
    
    caption_arrays = np.zeros((N, max_length), dtype='uint32') # will store encoding caption words
    caption_length = np.zeros(N, dtype='uint32')# will store encoding caption words
       
    
    for i,img in enumerate(imgs):
        question_id[question_counter] = img['ques_cap_id']
        label_length[question_counter] = min(max_length, len(img['final_question'])) # record the length of this question sequence
        caption_length[question_counter] = min(max_length, len(img['final_caption'])) # record the length of this caption sequence        
        question_counter += 1
        for k,w in enumerate(img['final_question']):
            if k < max_length:
                label_arrays[i,k] = wtoi[w]
        for k,w in enumerate(img['final_caption']):         ## this is for caption
            if k < max_length:
                caption_arrays[i,k] = wtoi[w]            
  
    return label_arrays, label_length, question_id, caption_arrays, caption_length

################################# Related to ANS  #############################################
def encode_answer(imgs, atoi):
    N = len(imgs)
    ans_arrays = np.zeros(N, dtype='uint32')

    for i, img in enumerate(imgs):
        ans_arrays[i] = atoi.get(img['ans'], -1) # -1 means wrong answer.

    return ans_arrays

def encode_mc_answer(imgs, atoi):
    N = len(imgs)
    mc_ans_arrays = np.zeros((N, 18), dtype='uint32')

    for i, img in enumerate(imgs):
        for j, ans in enumerate(img['MC_ans']):
            mc_ans_arrays[i,j] = atoi.get(ans, 0)
    return mc_ans_arrays

def filter_question(imgs, atoi):
    new_imgs = []
    for i, img in enumerate(imgs):
        if img['ans'] in atoi:
            new_imgs.append(img)

    print 'question number reduce from %d to %d '%(len(imgs), len(new_imgs))
    return new_imgs


############################################################################## '../../badripatro/VQA/workspace_project/contex_attention_vqa/VQA/Images/mscoco/'+ 
def get_unqiue_img(imgs):
    count_img = {}
    N = len(imgs)
    img_pos = np.zeros(N, dtype='uint32')
    for img in imgs:
        count_img[img['file_path']] = count_img.get(img['file_path'], 0) + 1

    unique_img = [w for w,n in count_img.iteritems()]
    imgtoi = {w:i+1 for i,w in enumerate(unique_img)} # add one for torch, since torch start from 1.


    for i, img in enumerate(imgs):
        img_pos[i] = imgtoi.get(img['file_path'])

    return unique_img, img_pos


def main(params):
    if params['token_method'] == 'spacy':
        print 'loading spaCy tokenizer for NLP'
        params['spacy'] = spacy.en.English(data_dir=params['spacy_data'])

    imgs_train = json.load(open(params['input_train_json'], 'r'))
    imgs_test = json.load(open(params['input_test_json'], 'r'))

    # tokenization and preprocessing training question
    imgs_train = prepro_question(imgs_train, params)
    # tokenization and preprocessing testing question
    imgs_test = prepro_question(imgs_test, params)

    ## this is newly added for caption 
    # tokenization and preprocessing training caption
    imgs_train = prepro_caption(imgs_train, params)
    # tokenization and preprocessing testing caption
    imgs_test = prepro_caption(imgs_test, params)


    # create the vocab for question
    imgs_train, vocab = build_vocab_question(imgs_train, params)
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table

    ques_train, ques_length_train, question_id_train, cap_train, cap_length_train = encode_question(imgs_train, params, wtoi) ##newly added for cap

    imgs_test = apply_vocab_question(imgs_test, wtoi) # this is for create vocb for test and val 
    ques_test, ques_length_test, question_id_test, cap_test, cap_length_test = encode_question(imgs_test, params, wtoi) ##newly added for cap

    # get the unique image for train and test
    unique_img_train, img_pos_train = get_unqiue_img(imgs_train)
    unique_img_test, img_pos_test = get_unqiue_img(imgs_test)

    # create output h5 file for training set.
    N = len(imgs_train)
    f = h5py.File(params['output_h5'], "w")
	## for train information
    f.create_dataset("ques_train", dtype='uint32', data=ques_train)
    f.create_dataset("ques_length_train", dtype='uint32', data=ques_length_train)
    #f.create_dataset("answers", dtype='uint32', data=ans_train)
    f.create_dataset("question_id_train", dtype='uint32', data=question_id_train)#this is actually the ques_cap_id
    f.create_dataset("img_pos_train", dtype='uint32', data=img_pos_train)
    f.create_dataset("cap_train", dtype='uint32', data=cap_train)
    f.create_dataset("cap_length_train", dtype='uint32', data=cap_length_train)

    
	## for test information
    f.create_dataset("ques_test", dtype='uint32', data=ques_test)
    f.create_dataset("ques_length_test", dtype='uint32', data=ques_length_test)
    #f.create_dataset("answers_test", dtype='uint32', data=ans_test)
    f.create_dataset("question_id_test", dtype='uint32', data=question_id_test)
    f.create_dataset("img_pos_test", dtype='uint32', data=img_pos_test)
    #  f.create_dataset("MC_ans_test", dtype='uint32', data=MC_ans_test)
    f.create_dataset("cap_test", dtype='uint32', data=cap_test)
    f.create_dataset("cap_length_test", dtype='uint32', data=cap_length_test)

    f.close()
    print 'wrote ', params['output_h5']

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab
    #out['ix_to_ans'] = itoa
    out['unique_img_train'] = unique_img_train
    out['unique_img_test'] = unique_img_test
    json.dump(out, open(params['output_json'], 'w'))
    print 'wrote ', params['output_json']

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_train_json', default='cocoAaai_attention_train_val.json', help='input json file to process into hdf5')
    parser.add_argument('--input_test_json', default='cocoAaai_attention_test.json', help='input json file to process into hdf5')
    parser.add_argument('--num_ans', default=1000, type=int, help='number of top answers for the final classifications.')

    parser.add_argument('--output_json', default='coco_data_prepro.json', help='output json file')
    parser.add_argument('--output_h5', default='coco_data_prepro.h5', help='output h5 file')
  
    # options
    parser.add_argument('--max_length', default=52, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=0, type=int, help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--token_method', default='nltk', help='token method. set "spacy" for unigram paraphrasing')    
    parser.add_argument('--num_test', default=0, type=int, help='number of test images (to withold until very very end)')
    parser.add_argument('--spacy_data', default='spacy_data', help='location of spacy NLP model')
    parser.add_argument('--batch_size', default=10, type=int)

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print 'parsed input parameters:'
    print json.dumps(params, indent = 2)
    main(params)
