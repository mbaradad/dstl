from odo.backends.sparksql import chunk_file

from data.dataset import Dataset
from numpy.random import RandomState
from data_iterator import DataIterator

def load_data(dataset, train_crop_size=(224, 224), one_hot=False,
              batch_size=10,
              horizontal_flip=False,
              rng=RandomState(0)):

    if isinstance(batch_size, int):
        batch_size = [batch_size] * 3

    d_train = Dataset(train=True)
    train_generator = d_train.cropped_generator(chunk_size=batch_size, crops_per_image=10, subset='train')
    val_generator = d_train.cropped_generator(chunk_size=batch_size, crops_per_image=10, subset='val')

    train_iter = DataIterator(train_generator, d_train.get_n_samples('train', 10))
    val_iter = DataIterator(val_generator, d_train.get_n_samples('val', 10))
    #
    test_iter = DataIterator(None)

    return train_iter, val_iter, test_iter

    '''
    train_iter = CamvidDataset(which_set='train',
                               batch_size=batch_size[0],
                               seq_per_video=0,
                               seq_length=0,
                               crop_size=train_crop_size,
                               horizontal_flip=horizontal_flip,
                               get_one_hot=False,
                               get_01c=False,
                               overlap=0,
                               use_threads=True,
                               rng=rng)

    val_iter = CamvidDataset(which_set='val',
                             batch_size=batch_size[1],
                             seq_per_video=0,
                             seq_length=0,
                             crop_size=None,
                             get_one_hot=False,
                             get_01c=False,
                             shuffle_at_each_epoch=False,
                             overlap=0,
                             use_threads=True,
                             save_to_dir=False)

    test_iter = CamvidDataset(which_set='test',
                              batch_size=batch_size[2],
                              seq_per_video=0,
                              seq_length=0,
                              crop_size=None,
                              get_one_hot=False,
                              get_01c=False,
                              shuffle_at_each_epoch=False,
                              overlap=0,
                              use_threads=True,
                              save_to_dir=False)
    '''
