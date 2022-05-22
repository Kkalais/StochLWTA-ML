import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions
from tensorflow.keras import layers
from LWTA.distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from LWTA.distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample
from LWTA.bit_precision import compute_reduced_weights
from tensorflow.keras import layers
import bma.linear_averaging as linear_averaging
from sklearn.metrics import mean_squared_error, r2_score


BMA_SAMPLING_POINTS = 4


class CustomSBLayer(layers.Layer):
    def __init__(self, K, U, name='custom_layer', tau = 1e-2, bma=False, deterministic=False, **kwargs):
        super(CustomSBLayer, self).__init__(name= name, **kwargs)
        self.K = K
        self.U = U
        self.tau = tau
        self.bma = bma # for Bayesian Model Averaging
        self.deterministic = deterministic
        
    def build(self, input_shape):
        # mean and variance of weights
        self.mW = self.add_weight(
            shape = [int(input_shape[1]), self.K*self.U],
            dtype = tf.float32,
            initializer = tf.keras.initializers.GlorotUniform(), 
            #initializer = tf.keras.initializers.RandomNormal(0, 1e-2),
            name = 'mW',
            trainable = True)
        
        if not self.deterministic:
            self.sW = self.add_weight(
                shape = [int(input_shape[1]), self.K * self.U],
                dtype = tf.float32,
                initializer = tf.keras.initializers.RandomNormal(-5, 1e-2),
                constraint = lambda x: tf.clip_by_value(x, -7., x),
                name = 'sW',
                trainable = True)

        self.bias = self.add_weight(
            shape = [self.K *self.U],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.1),
            name = 'bias',
            trainable = True)

    # Call method will sometimes get used in graph mode,
    # training will get turned into a tensor
    #@tf.function(input_signature=train_step_signature)
    def call(self, inputs, temp_bern=0.67, temp_cat=0.67, activation = 'lwta', train = True):
        
        layer_loss = 0
        if train:
            
            if not self.deterministic:
            # reparametrizable normal sample
                eps = tf.stop_gradient(tf.random.normal(self.mW.get_shape()))
                W = self.mW + eps * tf.nn.softplus(self.sW)

                # add the kl for the weights to the collection
                #kl_weights = tf.reduce_sum(normal_kl(tf.zeros_like(self.mW), tf.ones_like(self.sW),
                #                                     self.mW, tf.nn.softplus(self.sW), W))
                kl_weights = - 0.5 * tf.reduce_mean(
                    2*tf.nn.softplus(self.sW) - tf.square(self.mW) - \
                    tf.nn.softplus(self.sW)**2 + 1, name = 'kl_weights')

                #self.add_loss(kl_weights)
                tf.summary.scalar('kl_weights', kl_weights)

                # we optimize the variational lower bound scaled by the number of data
                # points (so we can keep our intuitions about hyper-params such as the learning rate)
                layer_loss += tf.math.reduce_mean(kl_weights) / inputs.shape[0]
            else:
                W = self.mW
                
            # dense calculation
            lam = tf.matmul(inputs, W) + self.bias

            # activation branches
            if activation == 'lwta':

                assert self.U > 1, 'The number of competing units should be larger than 1'

                # reshape weight for LWTA
                lam_w = tf.reshape(lam, [-1, self.K, self.U])
            
                # calculate probability of activation and some stability operations
                prbs = tf.nn.softmax(lam_w) + 1e-4
                prbs /= tf.reduce_sum(prbs, -1, keepdims = True)

                # relaxed categorical sample
                xi = concrete_sample(prbs, temp_cat)

                # apply activation
                out = lam_w * xi
                out = tf.reshape(out, tf.shape(lam))
                
                # kl for the relaxed categorical variables
               
                kl_xi = -tf.reduce_mean(tf.reduce_sum(concrete_kl(tf.ones([inputs.shape[0], self.K, self.U]) / self.U, prbs), [1,2]))

                #self.add_loss(kl_xi)
                tf.summary.scalar('kl_xi', kl_xi)
                layer_loss += kl_xi

            elif activation == 'relu':

                out = tf.nn.relu(lam)

            elif activation == 'maxout':

                lam_w = tf.reshape(lam, [-1, self.K, self.U])
                out = tf.reduce_max(lam_w, -1)

            elif activation == 'sigmoid':
                out = tf.nn.sigmoid(lam)

            elif activation == 'softmax':
                out = tf.nn.softmax(lam, -1)

            else:
                out = lam

        # test branch. It follows the train branch, but replacing samples with means
        else:

            re = 1.
            layer_loss = 0
            lam = tf.matmul(inputs, re * self.mW) + self.bias
    
            if activation == 'lwta':

                # reshape and calulcate winners
                lam_re = tf.reshape(lam, [-1, self.K, self.U])
                prbs = tf.nn.softmax(lam_re) + 1e-4
                prbs /= tf.reduce_sum(prbs, -1, keepdims = True)

                # apply activation
                out = lam_re * concrete_sample(prbs, 0.01)
                out = tf.reshape(out, tf.shape(lam))

            elif activation == 'relu':
                out = tf.nn.relu(lam)

            elif activation == 'maxout':

                lam_re = tf.reshape(lam, [-1, self.K, self.U])
                out = tf.reduce_max(lam_re, -1)

            elif activation == 'sigmoid':
                out = tf.nn.sigmoid(lam)

            elif activation == 'softmax':
                out = tf.nn.softmax(lam, -1)

            elif activation == 'linear':
                if (self.bma == True) and (train == False) and (self.deterministic == False):
                    r2_scores = []
                    outputs = []
                    
                    """
                    Bayesian Model Averaging (sample 4 points)
                    
                    Running BMA is as simple as fitting a regression model. Estimates will be close to the ones
                    you would obtain from fitting the "true" nested model, and no knowledge of that model is
                    required.
                    """
                    for i in range(0, BMA_SAMPLING_POINTS):
                        
                        eps = tf.stop_gradient(tf.random.normal(self.mW.get_shape()))
                        W = self.mW + tf.nn.softplus(self.sW) * eps
                        y = tf.matmul(inputs, re * W) + self.bias
                        outputs.append(y)
                        y = tf.math.reduce_mean(y, axis=-1)
                        
                        mc3 = linear_averaging.LinearMC3(inputs.numpy(), y.numpy(),
                                                         int(inputs.shape[1])**2, 1/3)
                        mc3.select(niter=10000, method="random") # computes P(M|y,x) posterior prob for each model
                        mc3.estimate() # computes the coeffs

                        computed_weights = mc3.estimates['coefficients']
                        
                        y_hat =  np.dot(inputs, computed_weights[1:inputs.shape[1]+1]) 
                        r2_scores.append(r2_score(y,y_hat))
                    
                    max_index = np.argmax(r2_scores)
                    out = outputs[max_index]
                else:    
                    out = lam
        
        if not self.deterministic:
            return out, layer_loss, self.mW, self.sW**2
        else:
            return out, layer_loss, self.mW, self.mW**2 #the 4th arguments must be equal to 0
        

##############################################
################# Classifier #################
##############################################
class LwtaClassifier(tf.keras.Model):
    def __init__(self, original_dim, latent_dim = [16, 2], tau = 1e-2, dataset="omniglot",
                 bma=False, deterministic=False, name='LwtaClassifier', **kwargs):
        super(LwtaClassifier, self).__init__(name = name, **kwargs)
        
        self.sb_layer = CustomSBLayer(K=16, U=2, tau=tau, name = 'sb_layer', bma=bma, deterministic=deterministic)
        self.sb_layer2 = CustomSBLayer(K=8, U=2, tau=tau, name = 'sb_layer2', bma=bma,deterministic=deterministic)
        #self.sb_layer3 = CustomSBLayer(K=2, U=2, name = 'sb_layer3', lwta=lwta, bma=bma)
        self.sb_logits = CustomSBLayer(original_dim[0], original_dim[1], tau=tau, name = 'logits',
                                       bma=bma, deterministic=deterministic)
        self.batch_norm1 = layers.BatchNormalization()
        self.maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        self.relu = layers.ReLU()
        self.dataset = dataset
        
    def call(self, input, temp_bern = .67, temp_cat = 0.5, activation = 'lwta', train = True):
         
        out = self.batch_norm1(input)
        out = self.relu(out)
        
        if self.dataset == "Imagenet":
            x1, x2 = out.shape
            out = self.maxpool(tf.reshape(out, [x1, int(np.sqrt(x2/3)), int(np.sqrt(x2/3)), 3]))
            out = tf.reshape(out, [out.shape[0], out.shape[1]*out.shape[2]*out.shape[3]])
            
        out, kl, mw, var = self.sb_layer(out, temp_bern, temp_cat, activation = activation, train = train)  
        out, kl2, mw2, var2 = self.sb_layer2(out, temp_bern, temp_cat, activation = activation, train = train)
       # out, kl3, mw3, var3 = self.sb_layer3(out, temp_bern, temp_cat, activation = activation, train = train)
        logits, kl4, mw4, var4 = self.sb_logits(out, temp_bern, temp_cat, activation= 'linear', train = train)
        logits = tf.nn.softmax(logits)
        
        mws = [mw, mw2, mw4]
        vars = [var, var2, var4]
        kls = [kl, kl2, kl4]
        if train:
            for kl in kls:
                self.add_loss(kl)
        
        return logits, mws, vars