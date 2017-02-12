from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, MaxPooling2D, Merge, Deconv2D

def BN_ReLU_Conv(inputs, n_filters, filter_size=3, dropout_p=0.2):
    """
    Apply successivly BatchNormalization, ReLu nonlinearity, Convolution and Dropout (if dropout_p > 0) on the inputs
    """
    l = BatchNormalization()(inputs)
    l = Activation('relu')(l)
    l = Conv2D(n_filters, filter_size, filter_size, border_mode='same')(l)
    if dropout_p != 0.0:
        l = Dropout(dropout_p)(l)
    return l


def TransitionDown(inputs, n_filters, dropout_p=0.2):
    """ Apply first a BN_ReLu_conv layer with filter size = 1, and a max pooling with a factor 2  """

    l = BN_ReLU_Conv(inputs, n_filters, filter_size=1, dropout_p=dropout_p)
    l = MaxPooling2D()(l)

    return l
    # Note : network accuracy is quite similar with average pooling or without BN - ReLU.
    # We can also reduce the number of parameters reducing n_filters in the 1x1 convolution


def TransitionUp(skip_connection, block_to_upsample, n_filters_keep):
    """
    Performs upsampling on block_to_upsample by a factor 2 and concatenates it with the skip_connection """

    # Upsample
    l = ConcatLayer(block_to_upsample)
    l = Deconv2D(l, n_filters_keep, filter_size=3, stride=2,
                      crop='valid', W=HeUniform(gain='relu'), nonlinearity=linear)
    # Concatenate with skip connection
    l = ConcatLayer([l, skip_connection], cropping=[None, None, 'center', 'center'])

    return l
    # Note : we also tried Subpixel Deconvolution without seeing any improvements.
    # We can reduce the number of parameters reducing n_filters_keep in the Deconvolution


def SigmoidLayer(inputs, n_classes):
    """
    Performs 1x1 convolution followed by softmax nonlinearity
    The output will have the shape (batch_size  * n_rows * n_cols, n_classes)
    """

    l = Conv2D(inputs, n_classes, 1, border_mode='same')


    l = Activation('sigmoid')

    return l

    # Note : we also tried to apply deep supervision using intermediate outputs at lower resolutions but didn't see
    # any improvements. Our guess is that FC-DenseNet naturally permits this multiscale approach