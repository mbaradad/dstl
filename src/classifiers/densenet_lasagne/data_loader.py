from data.dataset import Dataset
from numpy.random import RandomState
from data_iterator import DataIterator

def load_data(dataset, train_crop_size=(224, 224), one_hot=False,
              batch_size=10,
              horizontal_flip=False,
              rng=RandomState(0)):

    d_train = dataset
    train_generator = d_train.cropped_generator(chunk_size=batch_size, crop_size=train_crop_size, subset='train')
    val_generator = d_train.cropped_generator(chunk_size=batch_size, crop_size=train_crop_size, subset='val')

    train_iter = DataIterator(train_generator, d_train.get_n_samples('train', train_crop_size), d_train, batch_size)
    val_iter = DataIterator(val_generator, d_train.get_n_samples('val', train_crop_size), d_train, batch_size)
    #
    test_iter = DataIterator(None, 0, d_train, batch_size)

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
