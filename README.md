# Visual_Question_Generation

Torch implementation of an "Multimodal Differential Network for Visual Question Generation" .
### Training Step:

    1. Download VQG dataset from MicrosoftVQG site.
    2. Create train,val and test json file.
    3. Preprocess the MSCOCO image file using prepro/prepro_img.lua for joint model and prepro/prepro_img_att.lua for attention model.
    4. Find the exemplar(Supporting and oppsing) Image using /data/knn_image.m
    5. Run : th training.lua
