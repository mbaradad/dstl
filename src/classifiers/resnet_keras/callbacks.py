from keras.callbacks import Callback
import warnings
from utils.utils import save_layer_image
import tensorflow as tf
from pkg_resources import parse_version
import keras.backend as K

class PlotLayer(Callback):

    def __init__(self, data, layer_name, prefix=''):
        super(Callback, self).__init__()

        self.data = data
        self.layer_name = layer_name
        self.epoch_count = 0
        self.prefix = prefix

    def save_images(self, prefix=''):
      try:
          save_layer_image(self.model, self.data, self.layer_name, prefix=self.prefix +'_' +prefix)
      except Exception as inst:
        raise Exception('Error on plot layer callback: ' + str(inst))

    def on_train_begin(self, logs=None):
      self.save_images(prefix='train_begin_')

    def on_epoch_end(self, epoch, logs=None):
      self.save_images(prefix='epoch_' + str(self.epoch_count) + '_')
      self.epoch_count = self.epoch_count + 1





class MyTensorBoard(Callback):
    """Tensorboard basic visualizations.

    This callback writes a log for TensorBoard, which allows
    you to visualize dynamic graphs of your training and test
    metrics, as well as activation histograms for the different
    layers in your model.

    TensorBoard is a visualization tool provided with TensorFlow.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:
    ```
    tensorboard --logdir=/full_path_to_your_logs
    ```
    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/versions/master/how_tos/summaries_and_tensorboard/index.html).

    # Arguments
        log_dir: the path of the directory where to save the log
            files to be parsed by Tensorboard
        histogram_freq: frequency (in epochs) at which to compute activation
            histograms for the layers of the model. If set to 0,
            histograms won't be computed.
        write_graph: whether to visualize the graph in Tensorboard.
            The log file can become quite large when
            write_graph is set to True.
    """

    def __init__(self, generator, log_dir='./logs',
                 histogram_freq=0,
                 write_graph=True,
                 write_images=False):
        super(MyTensorBoard, self).__init__()
        if K.backend() != 'tensorflow':
            raise RuntimeError('TensorBoard callback only works '
                               'with the TensorFlow backend.')
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.merged = None
        self.write_graph = write_graph
        self.write_images = write_images
        self.generator = generator

    def set_model(self, model):
        self.model = model
        self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            i = 0
            for layer in self.model.layers:
                j = 0
                if hasattr(layer, 'trainable') and layer.trainable:
                    for weight in layer.weights:
                        if hasattr(tf, 'histogram_summary'):
                            tf.histogram_summary(str(i) + '_' + str(j) + '_' + weight.name, weight)
                        else:
                            tf.summary.histogram(str(i) + '_' + str(j) + '_' + weight.name, weight)
                        j+= 1
                if hasattr(layer, 'output'):
                    if hasattr(tf, 'histogram_summary'):
                        tf.histogram_summary('{}_out'.format(str(i) + '_' + layer.name),
                                             layer.output)
                    else:
                        tf.summary.histogram('{}_out'.format(str(i) + '_' + layer.name),
                                             layer.output)
                    if self.write_images:
                        if 'mask_output' in layer.name:
                            for k in range(10):
                                w_img = K.abs(layer.output[:,k:k+1])

                                if hasattr(tf, 'image_summary'):
                                    tf.image_summary(str(i) + '_' + layer.name + '_output_' + str(k),
                                                     tf.transpose(w_img, [0, 2, 3, 1]))
                                else:
                                    tf.summary.image(str(i) + '_' + layer.name + '_output_' + str(k),
                                                     tf.transpose(w_img, [0, 2, 3, 1]))
                        else:
                            w_img = K.expand_dims(K.sum(K.abs(layer.output), axis=1), 1)

                            if hasattr(tf, 'image_summary'):
                                tf.image_summary(str(i) + '_' + layer.name + '_output', tf.transpose(w_img,[0,2,3,1]))
                            else:
                                tf.summary.image(str(i) + '_' + layer.name + '_output', tf.transpose(w_img,[0,2,3,1]))
                    i += 1

        if hasattr(tf, 'merge_all_summaries'):
            self.merged = tf.merge_all_summaries()
        else:
            self.merged = tf.summary.merge_all()

        if self.write_graph:
            if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
                self.writer = tf.summary.FileWriter(self.log_dir,
                                                    self.sess.graph)
            elif parse_version(tf.__version__) >= parse_version('0.8.0'):
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph)
            else:
                self.writer = tf.train.SummaryWriter(self.log_dir,
                                                     self.sess.graph_def)
        else:
            if hasattr(tf, 'summary') and hasattr(tf.summary, 'FileWriter'):
                self.writer = tf.summary.FileWriter(self.log_dir)
            else:
                self.writer = tf.train.SummaryWriter(self.log_dir)

    def on_train_begin(self, logs=None):
        self.on_epoch_end(-1, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.generator and self.histogram_freq:
            batch = self.generator.next()
            tb_data = [batch[0][0], batch[0][1]]

            if epoch % self.histogram_freq == 0:
                # TODO: implement batched calls to sess.run
                # (current call will likely go OOM on GPU)
                if self.model.uses_learning_phase:
                    cut_v_data = len(self.model.inputs)
                    val_data = tb_data[:cut_v_data] + [0]
                    tensors = self.model.inputs + [K.learning_phase()]
                else:
                    val_data = tb_data
                    tensors = self.model.inputs
                feed_dict = dict(zip(tensors, val_data))
                result = self.sess.run([self.merged], feed_dict=feed_dict)
                summary_str = result[0]
                self.writer.add_summary(summary_str, epoch + 1)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

