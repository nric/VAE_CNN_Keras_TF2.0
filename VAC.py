#%%
import numpy as np
import matplotlib.pyplot as plt
import os

import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


#%%
class VAE_CNN:
    def __init__(self,image_size):
        self.image_size = image_size
        
        self.batch_size = 128
        self.kernel_size = 3
        self.filters = 16
        self.latent_dim = 2
        self.epochs = 30

        self.input_shape = (image_size, image_size, 1)
        self.inputs = KL.Input(shape=self.input_shape, name='encoder_input')
        self.encoder, self.shape, self.kl_loss = self._build_encoder()
        self.decoder = self._build_decoder()
        self.vae,self.outputs = self._build_vae()


    def sampling(self,args):
        """Reparametrization trick - return sampled latent vector
        It is not possible to backprob though a stocastic generation layer.
        Therefor we allow the VAE to backprob though the mu and sigma path and
        for the sampling we use samling from an isotropic unit gaussian
        """
        z_mean, z_log_var = args
        batch_size = K.shape(z_mean)[0]
        dimension_size = K.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch_size,dimension_size))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    

    def _build_encoder(self):
        x = self.inputs
        for i in range(2):
            self.filters *= 2
            x = KL.Conv2D(filters=self.filters,
                        kernel_size = self.kernel_size,
                        activation = 'relu',
                        strides = 2,
                        padding = 'same')(x)
        shape = K.int_shape(x) #required for decoder
        x = KL.Flatten()(x)
        x = KL.Dense(16, activation='relu')(x)
        #the outputs of encoder:
        z_mean = KL.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = KL.Dense(self.latent_dim, name='z_log_var')(x)
        #z itself is a random node. But we use the reparametrization trick.
        #so the sampling output is the input
        z = KL.Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        #Instatiate. We also add z_mean and z_log specifically as outputs for predictions and statistics
        encoder = Model(inputs=self.inputs,outputs=[z_mean,z_log_var,z],name='encoder')
        encoder.summary()
        #calc kl_loss here because all the requred vars are here
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return encoder, shape, kl_loss
    

    def _build_decoder(self):
        shape = self.shape
        latent_inputs = KL.Input(shape=(self.latent_dim,), name='z_sampling')
        x = KL.Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
        x = KL.Reshape((shape[1], shape[2], shape[3]))(x)
        for i in range(2):
            x = KL.Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2
        #one more time with filer = 1 to have the correct output shape
        outputs = KL.Conv2DTranspose(filters=1,
                          kernel_size=self.kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)
        #Instatiate
        decoder = Model(inputs=latent_inputs,outputs=outputs,name='decoder')
        decoder.summary()
        return decoder
    

    def _build_vae(self):
        #VAE = encoder + decoder
        outputs = self.decoder(self.encoder(self.inputs)[2]) #the position 2 refers to the latent vector output z
        vae = Model(inputs=self.inputs,outputs=outputs,name='vae')
        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = binary_crossentropy(K.flatten(self.inputs),K.flatten(outputs))
        #alternativly: reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
        reconstruction_loss *= self.image_size * self.image_size
        #calc total loss and add to model
        vae_loss = K.mean(reconstruction_loss + self.kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='rmsprop')
        vae.summary()
        return vae,outputs


    def train(self,x_train,x_test,y_train,y_test):
        self.vae.fit(x_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, None))
        self.vae.save_weights('vae_cnn_mnist.h5')
    

    def load_weights(self,path='vae_cnn_mnist.h5'):
        self.vae.load_weights(path)
    

class VAE_CNN2:
    """This is a nother, probably more elegant implementation of an VAE using Keras in an object.
    The diffrences to the first one are:
    1) Only one Keras.Model is created and compiled.
    2) The KLDivergence is encapsuled in an pass-though layer which makes it easy to use other 
    measurments of disribution equality/divergence.
    3) For the reconstruction loss, a default implicit keras loss fuction is used because
    this term is invariant in every VAE type (I know of...).
    """
    def __init__(self,image_size):
        self.image_size = image_size
        
        self.batch_size = 128
        self.kernel_size = 3
        self.filters = 16
        self.latent_dim = 2
        self.epochs = 10

        self.input_shape = (image_size, image_size, 1)
        self.inputs = KL.Input(shape=self.input_shape, name='encoder_input')
        self.encoder, self.decoder, self.vae = self._build_vae()
    

    def _build_encoder_layers(self):
        x = self.inputs
        for i in range(2):
            self.filters *= 2
            x = KL.Conv2D(filters=self.filters,
                        kernel_size = self.kernel_size,
                        activation = 'relu',
                        strides = 2,
                        padding = 'same')(x)
        self.shape = K.int_shape(x) #required for decoder
        x = KL.Flatten()(x)
        x = KL.Dense(16, activation='relu')(x)
        z_mean = KL.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = KL.Dense(self.latent_dim, name='z_log_var')(x)
        return z_mean, z_log_var



    def _build_decoder_layers(self):
        latent_inputs = KL.Input(shape=(self.latent_dim,), name='z_sampling')
        x = KL.Dense(self.shape[1] * self.shape[2] * self.shape[3], activation='relu')(latent_inputs)
        x = KL.Reshape((self.shape[1], self.shape[2], self.shape[3]))(x)
        for i in range(2):
            x = KL.Conv2DTranspose(filters=self.filters,
                                kernel_size=self.kernel_size,
                                activation='relu',
                                strides=2,
                                padding='same')(x)
            self.filters //= 2
        #one more time with filer = 1 to have the correct output shape
        decoder_out = KL.Conv2DTranspose(filters=1,
                          kernel_size=self.kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)
        decoder = Model(inputs=latent_inputs,outputs=decoder_out,name='decoder')
        decoder.summary()
        return decoder
    
    
    def _build_vae(self):
        z_mu, z_log_var = self._build_encoder_layers()
        z_mu, z_log_var = KLDivergenceLayer()([z_mu,z_log_var])
        z_sigma = KL.Lambda(lambda t: K.exp(0.5*t))(z_log_var)
        #generate the epsilon
        batch_size = K.shape(z_mu)[0]
        dimension_size = K.shape(z_mu)[1]
        epsilon = KL.Input(tensor=K.random_normal(stddev=1.0,shape=(batch_size,dimension_size)))
        #put it together
        z_epsilon = KL.Multiply()([z_sigma,epsilon])
        z = KL.Add()([z_mu,z_epsilon])
        #bring in the decoder
        decoder = self._build_decoder_layers()
        out = decoder(z)
        #make a keras.Model out of it
        vae = Model(inputs=(self.inputs,epsilon),outputs=out)
        #Make the loss function. the total loss is the reconstruction loss 
        #(output compared the input) plus the KLDicergence.
        #The KL divergence is added py the KLDivergenceLayer we defined.
        def nll(y_true,y_pred):
            #define keras loss function. Negative Log Likelihood
            return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
        #compile and return        
        vae.compile(optimizer='rmsprop', loss=nll)
        vae.summary()
        #return also the endoer and decoder in the model as accessible object for plotting and predicting
        encoder = Model(self.inputs,z_mu)
        return encoder, decoder, vae
    

    def train(self,x_train,x_test,y_train,y_test):
        self.vae.fit(x_train,
                x_train,
                shuffle=True,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(x_test, x_test))
        self.vae.save_weights('vae_cnn2_mnist.h5')
    

    def load_weights(self,path='vae_cnn2_mnist.h5'):
        self.vae.load_weights(path)
    


class KLDivergenceLayer(tf.keras.layers.Layer):
    """Identity transform layer that adds the KL divergence to the final loss.
    This obvisually can be done by just adding the loss in the model definition. But
    a key benefit of encapsulating the divergence in an auxiliary layer is that 
    we can easily implement and swap in other divergences, such as the χ-divergence 
    or the α-divergence.
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - 0.5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(K.mean(kl_batch), inputs=inputs)
        return inputs


#%%
#The rest ist just plotting. The plot is from the keras VAE example on github
def plot_results(encoder,decoder,data,batch_size=128,model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean = encoder.predict(x_test,batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


#%%
if __name__ == "__main__":
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    vae = VAE_CNN2(image_size)
    vae.train(x_train,x_test,y_train,y_test)
    #alternativly, uncomment the next just for predicting and plotting (no training): 
    #vae.load_weights()

    plot_results(vae.encoder,vae.decoder, (x_test,y_test), batch_size=vae.batch_size, model_name="vae_cnn")

