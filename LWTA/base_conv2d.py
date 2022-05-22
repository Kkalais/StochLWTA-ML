import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
from tensorflow.keras import layers
from tensorflow.python.util import tf_inspect
from tensorflow.keras.layers import BatchNormalization

from LWTA.distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from LWTA.distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample
from LWTA.bit_precision import compute_reduced_weights
from LWTA.base import LwtaClassifier as lwta_clf

class SB_Conv2d(tf.keras.layers.Layer):
    """
    Main class for the custom convolutional layers.
    """
    def __init__(self,
                 ksize,
                 padding='SAME',
                 strides=[1, 1, 1, 1],
                 bias=True,
                 sbp=False,
                 temp_bern=0.67,
                 temp_cat=0.67,
                 tau = 1e-3,
                 activation='lwta',
                 deterministic=False,
                 regularizer=None,
                 batch_norm = True,
                 **kwargs):
        """
        Initialize the layer with some parameters.
        @param ksize: list, [h, l, K,U] the size of the used kernel. h and l are the height
        and the length of the window, K is the number of blocks and U the number of competitors for LWTA
        @param padding: str, the padding to use.
        @param strides: tuple, the strides to use in the conv operation
        @param bias: boolean, flag to use an additional bias term
        @param sbp: boolean, flag to use the IBP prior
        @param temp_bern: float, the temperature of the posterior relaxation for the bernoulli distribution
        @param temp_cat: float, the temperature of the posterior relaxation for the categorical distribution
        @param tau: float, the cut-off threshold for the utility of the components
        @param activation: str, the activation to use. Supported: relu, maxout, lwta and none
        @param deterministic: boolean, if True obtain point estimates for the weights, otherwise infer a gaussian
        distribution
        @param regularizer: tensorflow regularizer, regularizer to use for the weights of the layer
        @param batch_norm: boolean, if True employ a batch norm layer.
        @param kwargs:
        """

        super(SB_Conv2d, self).__init__(**kwargs)

        self.tau = tau
        self.ksize = ksize
        self.U = ksize[-1]
        self.padding = padding
        self.strides = strides
        self.bias = bias
        self.sbp = sbp
        self.temp_bern = temp_bern
        self.temp_cat = temp_cat
        self.activation = activation
        self.deterministic = deterministic

        if deterministic:# and activation!='lwta':
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if activation != 'lwta':
            self.ksize = [self.ksize[0], self.ksize[1], self.ksize[2]*self.ksize[3], 1]

        self.batch_norm = batch_norm


    def build(self, input_shape):
        """
        Build the custom layer. Essentially define all the necessary parameters for training.
        The resulting definition depend on the initialization function, e.g. if we use the IBP, e.t.c.
        @param input_shape: tf.shape, the shape of the inputs
        @return: nothing, this is an internal call when building the model
        """

        self.mW = self.add_weight(shape=(self.ksize[0], self.ksize[1], input_shape[3], self.ksize[-2] * self.ksize[-1]),
                                  initializer=tf.keras.initializers.glorot_normal(),
                                  trainable=True,
                                  regularizer = self.regularizer,
                                  dtype=tf.float32,
                                  name='mW1')

        if not self.deterministic:
            self.sW = self.add_weight(
                shape=(self.ksize[0], self.ksize[1], input_shape[3], self.ksize[-2] * self.ksize[-1]),
                trainable=True,
                initializer = tf.keras.initializers.RandomNormal(-5, 1e-2),
#                 initializer=tf.constant_initializer(-5.),
                constraint=lambda x: tf.clip_by_value(x, -7., x),
                dtype=tf.float32,
                name='sW1')

        # variables and construction for the stick breaking process
        if self.sbp:
            # posterior concentrations for the Kumaraswamy distribution
            self.conc1 = self.add_weight(shape=([self.ksize[-2]]),
                                         initializer=tf.constant_initializer(2.),
                                         constraint=lambda x: tf.clip_by_value(x, -6., x),
                                         dtype=tf.float32,
                                         trainable=True,
                                         name='sb_t_u_1')

            self.conc0 = self.add_weight(shape=([self.ksize[-2]]),
                                         initializer=tf.constant_initializer(0.5453),
                                         constraint=lambda x: tf.clip_by_value(x, -6., x),
                                         dtype=tf.float32,
                                         trainable=True,
                                         name='sb_t_u_2')

            # posterior bernooulli (relaxed) probabilities
            self.t_pi = self.add_weight(shape=([self.ksize[-2]]),
                                        initializer=tf.compat.v1.initializers.random_uniform(4., 5.),
                                        constraint=lambda x: tf.clip_by_value(x, -7., 600.), \
                                        dtype=tf.float32,
                                        trainable=True,
                                        name='sb_t_pi')

        self.biases = 0.
        if self.bias:
            self.biases = self.add_weight(shape=(self.ksize[-2] * self.ksize[-1],),
                                          #initializer=tf.constant_initializer(0.0),
                                          initializer=tf.constant_initializer(0.1),
                                          trainable=True,
                                          name='bias1')

        # set the batch norm for the layer here
        if self.batch_norm:
            self.bn_layer = BatchNormalization()



    ###############################################
    ################## CALL #######################
    ###############################################
    def call(self, inputs, training=None):
        """
        Define what happens when the layer is called with specific inputs. We perform the
        necessary operation and add the kl loss if applicable in the layer's loss.
        @param inputs: tf.tensor, the input to the layer
        @param training: boolean, falag to choose between train and test branches. Initial values is
        none and the value comes from keras.
        @return: tf.tensor, the output of the layer
        """

        layer_loss = 0.
        if training:

            # if not deterministc, use the reparametrization trick for the Gaussian distribution and
            # add the kl loss to the layer's loss.
            if not self.deterministic:

                # reparametrizable normal sample
                sW_softplus = tf.nn.softplus(self.sW)
                eps = tf.stop_gradient(tf.random.normal(self.mW.get_shape()))
                W = self.mW + eps * sW_softplus

                kl_weights = - 0.5 * tf.reduce_mean(2 * sW_softplus - tf.square(self.mW) - sW_softplus ** 2 + 1,
                                                    name='kl_weights')
                tf.summary.scalar('kl_weights', kl_weights)
               
                layer_loss += tf.math.reduce_mean(kl_weights) / inputs.shape[0]
            
            else:

                W = self.mW

            # stick breaking construction
            if self.sbp:
                z, kl_sticks, kl_z = indian_buffet_process(self.t_pi,
                                                           self.temp_bern,
                                                           self.ksize[-1],
                                                           self.conc1, self.conc0)

                layer_loss = layer_loss + kl_sticks
                layer_loss = layer_loss + kl_z

                tf.summary.scalar('kl_sticks', kl_sticks)
                tf.summary.scalar('kl_z', kl_z)

                W = W * z

            
            # convolution operation
            out = tf.nn.conv2d(inputs, W, strides=(self.strides[0], self.strides[1]),
                               padding=self.padding) + self.biases

            # choose the activation
            if self.activation == 'lwta':
                assert self.ksize[-1] > 1, 'The number of competing units should be larger than 1'
              
                x = out
                #########################################################
                # reshape weight for LWTA    
                logits = tf.reshape(x, [-1, x.get_shape()[1], x.get_shape()[2], self.ksize[-2],
                                            self.ksize[-1]])
                
                q = tf.nn.softmax(logits) + 1e-4
                q /= tf.reduce_sum(q, -1, keepdims=True)
                
                log_q = tf.math.log(q + 1e-8)
                
                xi = concrete_sample(q, self.temp_cat)
                # apply activation
                out = logits * xi
                out = tf.reshape(out, tf.shape(input=x))
              
                
                kl_xi = tf.reduce_sum(q * (log_q - tf.math.log(1.0 / self.ksize[-1])), [1,2,3])
                kl_xi = tf.reduce_mean(kl_xi) / x.shape[0]
                #########################################################
                tf.compat.v2.summary.scalar('kl_xi', kl_xi)
                layer_loss = layer_loss + kl_xi

            elif self.activation == 'relu':

                out = tf.nn.relu(out)

            elif self.activation == 'maxout':

                out_re = tf.reshape(out, [-1, out.get_shape()[1], out.get_shape()[2],
                                          self.ksize[-2], self.ksize[-1]])

                out = tf.reduce_max(out_re, -1, keepdims=False)

            else:

                if self.activation != 'none':
                    print('Activation:', self.activation, 'not implemented.')

            if self.batch_norm:
                out = self.bn_layer(out, training = training)

        else:

            W = self.mW

            # if sbp is active calculate mask and draw samples
            if self.sbp:
                # posterior probabilities z
                z, _, _ = indian_buffet_process(self.t_pi, 0.01, self.ksize[-1], tau=1e-2, train=False)

                W = W * z

            # convolution operation
            out = tf.nn.conv2d(inputs, W, strides=(self.strides[0], self.strides[1]),
                               padding=self.padding) + self.biases

            if self.activation == 'lwta':
                # calculate probabilities of activation
                x = out
                logits = tf.reshape(x, [-1, x.get_shape()[1], x.get_shape()[2], self.ksize[-2],
                                            self.ksize[-1]])
                
                q = tf.nn.softmax(logits) + 1e-4
                q /= tf.reduce_sum(q, -1, keepdims=True)
                
                xi = concrete_sample(q, 0.01)
                # apply activation
                out = logits * xi
                out = tf.reshape(out, tf.shape(input=x))

            elif self.activation == 'relu':
                # apply relu
                out = tf.nn.relu(out)

            elif self.activation == 'maxout':

                # apply maxout operation
                out_re = tf.reshape(out, [-1, out.get_shape()[1], out.get_shape()[2],
                                          self.ksize[-2], self.ksize[-1]])
                out = tf.reduce_max(input_tensor=out_re, axis=-1)

            else:
                if self.activation != 'none':
                    print('Activation:', self.activation, 'not implemented.')

            if self.batch_norm:
                out = self.bn_layer(out, training=training)

        self.add_loss(layer_loss)
        return out, layer_loss, self.mW, self.sW**2


"""
num_filters_std = [64, 64, 64];
num_filters_ens = [32, 32, 32];
num_filters_ens_2 = 4;

model_rep_baseline = 1;
model_rep_ens = 2;
"""

##############################################
################# Classifier #################
##############################################
class LwtaClassifier(tf.keras.Model):
    def __init__(self, ksize=[5,5,16,2], activation="lwta", name='LwtaClassifier_2', strides=[2,2], **kwargs):
        super(LwtaClassifier, self).__init__(name = name, **kwargs)
    
        self.sb_1 = SB_Conv2d(ksize=[5,5,16,2], strides=[2,2],activation="lwta")
        self.sb_2 = SB_Conv2d(ksize=[3,3,16,2], strides=[2,2], activation="lwta")
        self.sb_3 = SB_Conv2d(ksize=[3, 3, 16, 2], strides=[2,2], activation="lwta")
        self.sb_4 = SB_Conv2d(ksize=[3, 3, 16, 2], strides=[2,2], activation="none")
        self.sb_5 = lwta_clf(original_dim = [5,1], tau = 5e-2, lwta=True)
    
    def call(self, inputs, train=True):
        out, kl, mw, var = self.sb_1(inputs, training=train)
        out, kl2, mw2, var2 = self.sb_2(out, training=train)
        out, kl3, mw3, var3 = self.sb_3(out, training=train)
        out, kl3, mw3, var3 = self.sb_4(out, training=train)
#         out, kl3, mw3, var3 = self.sb_3(out, training=True)
#         out, kl3, mw3, var3 = self.sb_2(out, training=True)
        out, kl4, mw4, var4 = self.sb_4(out, training=train)
       
        out = layers.Flatten()(out)  #(25,512)
       
        out, _, _ = self.sb_5(out, train=train, activation="lwta")
      
        return out, [],[],0