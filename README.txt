1. read datasets
2. resize all datasets to 32x32
3. use deep normalized probenet
4. train probenet on each dataset for 10 epochs to get DCN (5 is enough), lr = 0.0001
5. train proxylessnas on each dataset to get optimal architecture - keep for last
6. find closest dataset and do transfer learning


https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/

shibuiwilliam/ENAS-Keras - works
melodyguan/enas - memory error
carpedm20/ENAS-pytorch - not implemented, abandon
tensorflow/adanet
titu1994/neural-architecture-search

enas trainer | done
enas tranferlearner build | change transfer learner interfaces | done
child_network_name | done
datagen | done
dataflow gen | done
model_path -> best_model_path
use copy_tree directly | not important

child_epochs		search_epochs 	batch_size		child_init_filters		time per child epoch 		total training time per dataset
10					30				128				64						100s						30000s = 8.33hrs
10					30				256				32						70s							21000s = 5.83hrs
5					30				256				32						60s							9000s = 2.5 hrs
5					10				256				32						65s							3250s = 0.9 hrs
