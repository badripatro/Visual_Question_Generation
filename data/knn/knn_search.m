% h5create('traindatafile.h5','/DS1',[10 5])
% h5create('testdatafile.h5','/DS1',[3 5])
% 
% x=abs(randn(10,5))+10;
% y=abs(randn(3,5))+10;
% 
% 
% h5write('traindatafile.h5','/DS1', x);
% h5write('testdatafile.h5','/DS1', y);
disp('start of Knn search program')
%xx_trans=h5read('input/img_data_vgg_train.h5','/images_train');
%yy_trans=h5read('input/img_data_vgg_test.h5','/images_test');
disp('start of training transpose')
xx_trans=h5read('cocotalk_trainval.h5','/images_train');
yy_trans=h5read('cocotalk_test.h5','/images_test');
xx=xx_trans;
disp('start of testing transpose')
yy=yy_trans;

disp('start of training knn')
[idx_train,d_train]=knnsearch(xx,xx,'k',16); %16 nearest neighbor.

disp('Train: start of writing to file')
h5create('output/knn_train_idx_file.h5','/images_train',[215200 16])% total no of training image* no of nearest neighbor
h5write('output/knn_train_idx_file.h5','/images_train', idx_train); %provide nearest neighbor for each traininng image

h5create('output/knn_train_dist_file.h5','/images_train',[215200 16])% total no of training image* no of nearest neighbor
h5write('output/knn_train_dist_file.h5','/images_train', d_train); %provide nearest neighbor for each traininng image



disp('start of source=training and target=testing knn search')
[idx_test,d_test]=knnsearch(xx,yy,'k',16);

disp('test: start of writing to file')
h5create('output/knn_test_idx_file.h5','/images_test',[121400 16])% total no of test image* no of nearest neighbor in training image
h5write('output/knn_test_idx_file.h5','/images_test', idx_test);

h5create('output/knn_test_dist_file.h5','/images_test',[121400 16])% total no of test image* no of nearest neighbor in training image
h5write('output/knn_test_dist_file.h5','/images_test', d_test);

disp('--------------------End of Knn search program--------------------')


%h5disp('knnfile.h5');
%data = h5read('knnfile.h5','/DS1');