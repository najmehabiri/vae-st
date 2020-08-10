import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tf.random.set_seed(1234)

class VAE(tf.keras.Model):
    #subclassing the Model class
    Default = {'encod_size':  None,
               'decod_size': None,
               'inp_shape':None,
               'act_fun': tf.nn.relu, 
               'batch_size': 200,
               'l2' : 0,
               'K' : None
               }
    
    def __init__(self, hyper_p):
        super(VAE, self).__init__()
        self.__dict__.update(self.Default, **hyper_p )
         
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()                  
    
    def posterior(self, args):
        z_mu , z_var, z_nu = args
        return tfp.distributions.Independent(tfp.distributions.StudentT(loc = z_mu , scale= z_var, df = z_nu ),
                                            reinterpreted_batch_ndims = 1)
    def prior(self, arg):
        nu = arg
        p_z = tfp.distributions.Independent(tfp.distributions.StudentT(loc = tf.zeros(self.encod_size[-1]),
                                                                            scale =tf.ones(self.encod_size[-1]),
                                                                            df = nu),
                                            reinterpreted_batch_ndims=1)
        return p_z
 

    def make_encoder(self):
        inp = tf.keras.Input(self.inp_shape)
        x = inp
        
        for lay in self.encod_size[:-1]:
            x = tf.keras.layers.Dense(lay, 
                                       kernel_regularizer =tf.keras.regularizers.l2(self.l2),
                                      activation = self.act_fun)(x)
        z_mu = tf.keras.layers.Dense(self.encod_size[-1], activation = 'linear')(x)
        z_var = tf.keras.layers.Dense(self.encod_size[-1], activation = 'softplus')(x)

        @tf.function
        def custom_act(x):
            return tf.nn.softplus(x) + 3. 
 
        nu_lambda = tf.keras.layers.Dense( self.encod_size[-1],
                                         kernel_regularizer =tf.keras.regularizers.l2(self.l2),
                                         kernel_initializer=tf.keras.initializers.RandomUniform(
                                             minval= 2. , maxval=30),
                                         activation=custom_act,
                                         name = 'z_nu')(x)
         
        return tf.keras.models.Model(inp, [z_mu, z_var, nu_lambda], name= 'Encoder')
        
        
    def make_decoder(self):
        decoder = tf.keras.Sequential(name= 'Decoder')

        for lay in self.decod_size:
            
            decoder.add(tf.keras.layers.Dense(lay,
                                      kernel_regularizer =tf.keras.regularizers.l2(self.l2),
                                      activation = self.act_fun))
            
        decoder.add(tf.keras.layers.Dense(self.inp_shape, name='output'))
        decoder.add(tfp.layers.IndependentBernoulli(self.inp_shape, tfp.distributions.Bernoulli.logits))
 
        return decoder


    def call(self, inputs): 
        inp , self.mask = inputs 
        z_mu, z_var, z_nu = self.encoder(inp)
        q_z = self.posterior([z_mu, z_var, z_nu])
        p_z = self.prior(z_nu)

        z = q_z.sample(self.K)
        #z =  tf.reshape(z, [self.K*self.batch_size, self.encod_size[-1]])
        self.kl =  q_z.log_prob(z) - p_z.log_prob(z)
        reconstructed = self.decoder(z)

        return reconstructed

    
   
    def kl_loss(self, x, x_v):
        return tf.reduce_mean(self.kl)

    def total_loss(self, x, x_v):
        nll = self.reco_loss(x, x_v)
        kl = self.kl_loss(x, x_v)
        return nll + kl

        
    def _log_prob(self, px_z, event, mask):
        logits = px_z
        log_probs0, log_probs1 = -tf.math.softplus(logits),  -tf.math.softplus(-logits),
        event = tf.cast(event, log_probs0.dtype)
        masked_log = tf.multiply(tf.math.multiply_no_nan(log_probs0, 1 - event) +
                     tf.math.multiply_no_nan(log_probs1, event),  mask)
        out =  tf.reduce_sum(masked_log, axis=[-1]) 
        return out


    def reco_loss(self, x, x_v):
        ## Negative log-likelohood
        mask_K = tf.repeat(self.mask[tf.newaxis, :,:], self.K, axis=0)
        recon_loss = - self._log_prob(x_v, x, mask_K)
        return tf.reduce_mean(recon_loss)
 




