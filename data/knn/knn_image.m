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
xx_trans=h5read('img_train_fc7.h5','/images_train');
yy_trans=h5read('img_test_fc7.h5','/images_test');
disp('start of training transpose')
xx=xx_trans';


%yy = yy(1:10000 ,:);
disp('start of kd tree buid')
MdlKDT = KDTreeSearcher(xx)
%MdlKDT


disp('start of training knn')
[idx_train,d_train]=knnsearch(MdlKDT,xx,'K',1000); %32 nearest neighbor.

disp('Train: start of writing to file')
h5create('knn_train_idx_file.h5','/images_train',[34537 1000])% total no of training image* no of nearest neighbor
h5write('knn_train_idx_file.h5','/images_train', idx_train); %provide nearest neighbor for each traininng image

h5create('knn_train_dist_file.h5','/images_train',[34537 1000])% total no of training image* no of nearest neighbor
h5write('knn_train_dist_file.h5','/images_train', d_train); %provide nearest neighbor for each traininng image

disp('--------------------End of Training  Knn search program--------------------')

disp('start of Validation knn')
disp('start of source=training and target=testing knn search')
[idx_test,d_test]=knnsearch(MdlKDT,yy,'K',1000);

disp('test: start of writing to file')
h5create('knn_test_idx_file.h5','/images_test',[15729 1000])% total no of test image* no of nearest neighbor in training image
h5write('knn_test_idx_file.h5','/images_test', idx_test);

h5create('knn_test_dist_file.h5','/images_test',[15729 1000])% total no of test image* no of nearest neighbor in training image
h5write('knn_test_dist_file.h5','/images_test', d_test);

disp('--------------------End of Validation Knn search program--------------------')


%h5disp('knnfile.h5');
%data = h5read('knnfile.h5','/DS1');