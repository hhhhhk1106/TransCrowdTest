import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')


'''please set your dataset path'''
try:
    my_train_path = '/root/TransCrowdTest/dataset/train_data/images/'
    my_test_path = '/root/TransCrowdTest/dataset/test_data/images/'

    train_list = []
    for filename in os.listdir(my_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(my_train_path + filename)

    train_list.sort()
    np.save('./npydata/my_train.npy', train_list)

    test_list = []
    for filename in os.listdir(my_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(my_test_path + filename)
    test_list.sort()
    np.save('./npydata/my_test.npy', test_list)

    print("generate my image list successfully", len(train_list), len(test_list))
except:
    print("The my dataset path is wrong. Please check you path.")
