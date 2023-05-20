"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number # This parameter determines the digit that defines the label of a bag. If a bag contains at least one instance of this digit, the label of the bag is positive; otherwise, it's negative.
        self.mean_bag_length = mean_bag_length # This parameter specifies the average number of images (instances) in each bag.
        self.var_bag_length = var_bag_length #This parameter specifies the variance of the bag length
        self.num_bag = num_bag #This parameter determines the total number of bags in the dataset. In this case,
        self.train = train #This parameter specifies whether the generated dataset is for training or testing

        self.r = np.random.RandomState(seed)
        #The actual value of self.r will be an instance of numpy.random.RandomState, 
        #so it's not a simple numerical value. You can use it to generate random numbers

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          #applying two transformations to the data: converting it to a PyTorch Tensor and normalizing it.
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
            # preparing the MNIST training dataset to be used in a PyTorch model.
            # It's loading the data, applying necessary transformations, and setting up a DataLoader to handle mini-batches.
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            #starts a loop that iterates over the dataset. At each iteration, a batch of data (batch_data) and the corresponding labels (batch_labels) are loaded. 
            all_imgs = batch_data
            all_labels = np.squeeze(batch_labels.numpy())

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            # print(bag_length)

            if bag_length < 1:
                bag_length = 1
            
            #This line is randomly choosing two numbers from the list self.target_number without replacement.
            #np.random.choice() 第一个参数可以是int类型，意味着从[0,int）选取
            target_num = np.squeeze(np.random.choice(np.squeeze(self.target_number), 2, replace=False))
            
            #getting the numbers from self.target_number that are not in target_num.
            #此处由于target_num中没有9，因此结果永远是9
            r_target_num = np.squeeze(np.setdiff1d(np.squeeze(self.target_number), target_num))
            #r_target_num = array(9)
            
            #getting the indices of the elements in self.target_number that are equal to r_target_num.
            #由于等号左右都是array(9),因此选择9在的索引，indice=0
            class_num = np.squeeze(np.where(np.squeeze(self.target_number)==r_target_num))
            
            if self.train:
                r_index = np.squeeze(np.arange(all_labels.shape[0]))
                for iter in range(target_num.size):
                    r_idx = np.squeeze(np.where(all_labels[r_index]==target_num[iter]))
                    r_index = np.squeeze(np.setdiff1d(r_index, r_index[r_idx]))

                indices = np.squeeze(np.random.choice(r_index, bag_length, replace=False))
            else:
                r_index = np.squeeze(np.arange(all_labels.shape[0]))
                for iter in range(target_num.size):
                    r_idx = np.squeeze(np.where(all_labels[r_index]==target_num[iter]))
                    r_index = np.squeeze(np.setdiff1d(r_index, r_index[r_idx]))
                    
                indices = np.squeeze(np.random.choice(r_index, bag_length, replace=False))

            labels_in_bag = all_labels[indices]

            labels_in_bag = torch.from_numpy(labels_in_bag).type(torch.LongTensor)
            r_target_num = torch.from_numpy(r_target_num).type(torch.LongTensor)
            class_num = torch.from_numpy(class_num).type(torch.LongTensor)
            
            labels_in_bag[labels_in_bag != r_target_num] = 0
            labels_in_bag[labels_in_bag == r_target_num] = class_num+1

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.min(len_bag_list_train), np.max(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.min(len_bag_list_test), np.max(len_bag_list_test)))
    
    # This script is a good example of how to adapt a traditional single-instance dataset into a multiple-instance learning scenario
