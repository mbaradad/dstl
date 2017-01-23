from keras.engine import Layer, InputSpec
from keras import initializations, regularizers
from keras import backend as K

#!!!!Change axis to adapt for different than dim_ordering=theano and backend=tf!!!!
class Bias(Layer):
    '''
    Simple bias layer
    '''
    def __init__(self, axis=1, momentum = 0.9, beta_init='zero',  **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializations.get(beta_init)
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.beta]

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        #TODO: possibly do better than this. We have to init the bias term to a big value, and this, though ugly,
        #does the trick
        #ln(1 / 224 / 224) = -10.82
        out = x + 12*K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum}
        base_config = super(Bias, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))