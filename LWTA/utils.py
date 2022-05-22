import tensorflow as tf
import tensorflow.keras as keras


class PrintDot(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end = '')


# MAPE calculation: eps makes it more stable
def mean_absolute_percentage_error(y_true, y_pred, eps = 1e-3):
    diff = tf.math.abs((y_true - y_pred) / tf.clip_by_value(tf.math.abs(y_true), eps,
                                                            tf.math.abs(y_true)))

    return 100. * tf.reduce_mean(diff)