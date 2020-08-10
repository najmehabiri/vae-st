import itertools
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle
import sys
import tensorflow as tf
import tensorflow_probability as tfp
import time
from skimage.metrics import structural_similarity as ssim


tf.keras.backend.clear_session() 
tf.random.set_seed(739)
np.random.seed(739)

from vae_st import *



#os.system('rm -rf logs/*')

k = 7


(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()




def plot_train_history(history ):
    plt.figure('Loss plot_{}'.format(model))
    keys = ['kl_loss', 'val_kl_loss','reco_loss', 'val_reco_loss' ]
    for i,k in enumerate(keys):
        plt.plot(history.history[k], label=k)
        plt.legend( loc='upper right')
   


def dis_img(x, indx = None, tit = None):
    
    if not indx.any():
        np.random.seed()
        indx = np.random.choice(range(x.shape[0]) , 25 )

    sq = int(np.sqrt(len(indx)))
    if tit:
        title = tit
    else:
        title = 'outliers_{}'.format(model)
        
    fig, ax = plt.subplots(sq, sq,
                           figsize=(150,100),
                           subplot_kw={'xticks': [], 'yticks': []},
                           num= title )  
    for i, axi in enumerate(ax.flat):
        ind = indx[i]
        axi.imshow(x[ind].reshape(28,28), cmap='gray')
        #axi.set_xlabel('{}'.format(ind))
    #plt.subplots_adjust( wspace=3)
    plt.tight_layout()

    
class TestCallback(tf.keras.callbacks.Callback):
    def __init__(self, x,mask,y):
        self.x = x
        self.mask = mask
        self.y = y
        global eval_test
        eval_test=[] 
    def on_epoch_end(self, epoch, logs={}):
        loss = self.model.evaluate([self.x,self.mask], self.y, verbose=0)
        eval_test.append(loss)
        if epoch%50==0:
            print('\nEpoch {} Testing loss: {}\n'.format(epoch,loss))


            

x_train_org = x_train.reshape(x_train.shape[0],
                              x_train.shape[1] * x_train.shape[2]).astype('float32') /255.
x_test_org = x_test.reshape(x_test.shape[0],
                            x_test.shape[1] * x_test.shape[2]).astype('float32') /255.



mask =  np.load('data/ising_mixed_1.9_5.npz')
train_mask , test_mask = mask['train'] ,  mask['test']

                                                                 
train_mask=train_mask.reshape(train_mask.shape[0],
                              train_mask.shape[1] * train_mask.shape[2]).astype('float32')
test_mask=test_mask.reshape(test_mask.shape[0],
                            test_mask.shape[1] * test_mask.shape[2]).astype('float32')


### Complete data 
# train_mask = np.ones_like(train_mask)
# test_mask = np.ones_like(test_mask)





xmask_train = x_train_org * train_mask
xmask_test = x_test_org * test_mask

# if NAN in mask is represented via Zero
mean_imput = lambda x, mask, mean : x * mask + (1-mask) * mean

xm_train = mean_imput(x_train_org, train_mask, xmask_train.mean(axis=0))
xm_test = mean_imput(x_test_org, test_mask, xmask_test.mean(axis=0))

verb = 0
hyp_param ={ 'inp_shape': xm_train.shape[1],
             'encod_size' : [50, 20],
             'decod_size':  [50],
             'l2':1e-4,
             'act_fun': tf.nn.relu,
             'batch_size' : 500,
             'epochs' : 2,
             'learning_rate' : 1e-3,
             'K':k
}




vae = VAE(hyp_param)

vae.compile(optimizer=tf.keras.optimizers.Adam(lr=hyp_param['learning_rate']),              
            loss = vae.total_loss,
            metrics = [ vae.kl_loss, vae.reco_loss ])


tim = time.strftime("%Y-%m-%d_%H:%M:%S")
log_dir="logs/{}".format(tim) 
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      write_grads = True,
                                                      histogram_freq=1,
)



start = time.time()
history = vae.fit([xm_train,train_mask],
                  xm_train,
                  batch_size=hyp_param['batch_size'],
                  epochs=hyp_param['epochs'],
                  callbacks=[TestCallback(xm_test,test_mask,xm_test)],
                  #callbacks=[TestCallback(xm_test,test_mask,xm_test),tensorboard_callback],
                  shuffle= True,
                  verbose=verb)
end = time.time()
print('TIME:  ', end-start)



#####################
#####################
#### Results &  PLOT
#####################
#####################
plt.close('all')

index = np.random.choice(range(x_test_org.shape[0]) ,  5)
#index = [8274,7704, 4829, 9760,  6804 ]
#index =  [2223, 2223, 3234, 2223, 4469]



## x^ is the reconstructed test set
## x~ is generated samples from decoder

x_hat = vae([xm_test, test_mask])
x_hat_mean = np.mean(x_hat.mean(), axis=0) # x_hat.mean().numpy()[0]



sm_all = [ ssim(x_test_org[i],x_hat_mean[i] , data_range=1.) for i in range(x_test_org.shape[0]) ]
sm_avg = np.round(np.mean(sm_all), 5)
keys = history.history.keys()
print('SSIM', sm_avg )
print('last iter: ', [(k,history.history[k][-1]) for k in keys] )



######################## Change color of mask


masked = xmask_test

masked1 = np.copy(masked)
masked2 = np.copy(masked)
masked3 = np.copy(masked)


from matplotlib import colors
col = colors.to_rgb('maroon')

masked1[test_mask==0] = col[0]#0.9803921568627451
masked2[test_mask==0] = col[1]#0.5019607843137255
masked3[test_mask==0] = col[2]#0.4470588235294118

x_masked = np.dstack([masked1, masked2, masked3])


col_name = ['original','masked','mean imputation','reco_1X','reco_avgX', 'generation']
xh =  x_hat.mean()[0].numpy()

s1 = [ ssim(x_test_org[i], xh[i], data_range=1.) for i in range(x_test_org.shape[0]) ] 
s2 = np.round(np.mean(sm_all), 5) 
print('one sample X ssim: ', s2) 


imags = list(itertools.chain(*[[x_test_org[inx],
                                x_masked[inx],
                                xm_test[inx],
                                xh[inx],
                                x_hat_mean[inx]
                                ] for inx in index]))


#####################
#####################


plt.figure('history_model')
keys = history.history.keys()
for i,k in enumerate(keys):
   plt.plot(history.history[k], label=k)
   plt.legend( loc='upper right')





fig, ax = plt.subplots(5,5, 
                       figsize=(10, 15),
                       subplot_kw={'xticks': [], 'yticks': []},
                       num= 'model_results')
for i, axi in enumerate(ax.flat):
     if imags[i].shape[-1]==3:
         axi.imshow(imags[i].reshape(28,28,3),interpolation='nearest')
     else:
         axi.imshow(imags[i].reshape(28,28),interpolation='nearest', cmap = 'gray' )
for a, col in zip(ax[0], col_name):
    a.set_title(col, fontsize=12)


    
plt.xlabel( (str(hyp_param),'ssim',str(sm_avg)),
            wrap=True, position = (-2,0))# ha = 'center', va ='center' )

name = 'K{}'.format(hyp_param['K'])

""" Saving the results
plt.savefig('results/fashion_{}_{}_miss_jf.eps'.format(name,tim),
             transparent=True,format='eps', dpi=300)
with open('results/hist_{}_{}'.format(name,tim),'wb') as f:
     pickle.dump(history.history, f)
np.savez('results/jf_{}_{}_miss.npz'.format(name,tim),
         hist = eval_test,
         ss = sm_avg)
#"""

plt.show()

msg = "curl 'https://api.simplepush.io/send/4ruCJe/{} SSIM {}'".format(hyp_param['K'],
                                                                          sm_avg)
os.system(msg)
