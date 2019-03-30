# Visual_Question_Generation

Torch implementation of ["Multimodal Differential Network for Visual Question Generation"](https://arxiv.org/pdf/1808.03986).
### Training Step:

    1. Download VQG dataset from MicrosoftVQG site.
    2. Create train,val and test json file.
    3. Preprocess the MSCOCO image file using prepro/prepro_img.lua for joint model and prepro/prepro_img_att.lua for attention model.
    4. Find the exemplar(Supporting and oppsing) Image using /data/knn_image.m
    5. Run : th train.lua
    
 
### Reference

If you use this code as part of any published research, please acknowledge the following paper

```
@InProceedings{D18-1434,
  author = 	"Patro, Badri Narayana
		and Kumar, Sandeep
		and Kurmi, Vinod Kumar
		and Namboodiri, Vinay",
  title = 	"Multimodal Differential Network for Visual Question Generation",
  booktitle = 	"Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"4002--4012",
  location = 	"Brussels, Belgium",
  url = 	"http://aclweb.org/anthology/D18-1434"
}
```
    
    
# Acknowledgements    
This codebase is based on [Neural Talk2](https://github.com/karpathy/neuraltalk2) repository by [Andrej Karpathy](https://github.com/karpathy), [coco-caption](https://github.com/tylin/coco-caption) repository by [Tsung-Yi Lin](https://github.com/tylin) and [TripletNet](https://github.com/eladhoffer/TripletNet) repository by [Elad Hoffer](https://github.com/eladhoffer).

