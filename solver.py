import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.layers as layers
DELTA_CLIP = 50.0


class BSDESolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde
       
        self.model = NonsharedModel(config, bsde)
        #self.y_init = self.model.y_init

        try:
            lr_schedule = config.net_config.lr_schedule
        except AttributeError:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.net_config.lr_boundaries, self.net_config.lr_values)     
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in tqdm(range(self.net_config.num_iterations+1)):
            if step % self.net_config.logging_frequency == 0:
                loss = self.loss_fn(valid_data, training=False).numpy()
                y_init = self.model.y_init.numpy()[0]
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, elapsed_time])
                if self.net_config.verbose:
                    #logging.info("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                    #    step, loss, y_init, elapsed_time))
                    print("step: %5u,    loss: %.4e, Y0: %.4e,   elapsed time: %3u" % (
                        step, loss, y_init, elapsed_time))
            data = self.bsde.sample(self.net_config.batch_size)
            self.train_step(data)            
        return np.array(training_history)

    def loss_fn(self, inputs, training):

        dw, x = inputs
        # y_terminal, y_barrier, x_barrier = self.model(inputs, training)
        y_terminal= self.model(inputs, training)
        
        if self.eqn_config.eqn_name == "Swap" :
            delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, tf.gather(x, self.eqn_config.schedule_float, axis=2))
        elif self.eqn_config.eqn_name == "BarrierOption" :
            # barrier_check =  np.ones([self.net_config.batch_size, self.eqn_config.dim, self.eqn_config.num_time_interval])  # stesse dimensioni di x_sample
            # barrier_check = x > self.eqn_config.barrier
            # b = tf.cast(barrier_check, tf.float64)
            # check = tf.math.reduce_sum(b, axis=2)  # dimension = num_sample x self.dim
            # y_terminal = y_terminal * (check // self.eqn_config.num_time_interval)
            # for i in range(1024):
            #     for j in range(1):
            #         for k in range(201):
            #             if not barrier_check[i,j,k]:
            #                 x[i,j,k:] = x[i,j,k-1]
            #                 print(y_terminal[i, j, k].numpy())
            #                 y_terminal[i,j,k:].assign(y_terminal[i, j, k-1])
            #                 print(y_terminal[i, j, k].numpy())
            #                 break
            # the problem with the above is that y is [1024,1], so it is not possible to stop the learning of it through time 
            #delta = y_barrier_terminal - self.bsde.g_tf(self.bsde.total_time, x_barrier[:, :, -1])
            
            delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x)
        
        else : 
            delta = y_terminal - self.bsde.g_tf(self.bsde.total_time, x[:, :, -1])
        
        # use linear approximation outside the clipped range
        loss = tf.reduce_mean(tf.where(tf.abs(delta) < DELTA_CLIP, tf.square(delta),
                                    2 * DELTA_CLIP * tf.abs(delta) - DELTA_CLIP ** 2))
        loss += 1000*(tf.maximum(self.model.y_init[0]-self.net_config.y_init_range[1],0)+tf.maximum(self.net_config.y_init_range[0]-self.model.y_init[0],0))
        return loss
        

    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss_fn(inputs, training)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad

    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))     


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde       
        self.dim = bsde.dim
        self.y_init = tf.Variable(np.random.uniform(low=self.net_config.y_init_range[0],
                                                    high=self.net_config.y_init_range[1],
                                                    size=[1]),dtype=self.net_config.dtype
                                  )
        self.z_init = tf.Variable(np.random.uniform(low=-.1, high=.1,
                                                    size=[1, self.eqn_config.dim]),dtype=self.net_config.dtype
                                  )        
        
        self.subnet = [FeedForwardSubNet(config,bsde.dim) for _ in range(self.bsde.num_time_interval-1)]
       
    def call(self, inputs, training):
        dw, x = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
        y = all_one_vec * self.y_init
        y_new = y
        # x_new = x
        # x_barrier = x
        # y_barrier = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)
        
        ####################################################################################
        if self.eqn_config.eqn_name == "BarrierOption" :
        
            # Create a boolean mask indicating elements greater than the barrier
            mask = tf.math.greater(x, self.eqn_config.barrier)
            first_false_index_initial = tf.argmax(tf.math.logical_not(mask), axis=-1, output_type=tf.int32)
            # Handle the case where all values are True
            last_index = tf.shape(mask)[-1] - 1
            first_false_index = tf.where(tf.reduce_all(mask, axis=-1), last_index, first_false_index_initial)
            equality_tensor = ~tf.math.equal(first_false_index_initial, first_false_index)
            first_false_index = tf.cast(first_false_index, tf.int32)
            first_false_index = tf.expand_dims(first_false_index, axis=-1)
            
            range_tensor = tf.range(tf.shape(x)[-1], dtype=tf.int32)
            range_tensor = tf.expand_dims(range_tensor, axis=0)
            
            barrier_check = tf.less(range_tensor, first_false_index)
            barrier_check = tf.cast(barrier_check, tf.float64)
            
            # Extract the indices from tensor 'first_false_index'
            first_false_index = tf.where(tf.equal(first_false_index, 0), 1, first_false_index)
            indices = first_false_index[..., 0] - 1
            # Take into accout whether all are TRUE in mask
            indices = indices + tf.multiply(tf.ones(shape=(self.net_config.valid_size, 1), dtype=tf.int32), tf.cast(equality_tensor, dtype=tf.int32))
            # Use tf.gather to select values from 'x' based on indice
            values_to_repeat = tf.gather(x, indices, batch_dims=2, axis=-1)
            # Get the number of times to repeat each value
            num_repeats = tf.shape(x)[-1] - indices
            # Flatten 'values_to_repeat' and 'num_repeats' for tf.repeat
            values_flat = tf.reshape(values_to_repeat, [-1])
            num_repeats_flat = tf.reshape(num_repeats, [-1])
            # Use tf.repeat to repeat each value according to the specified number of repeats
            expanded_values = tf.repeat(values_flat, repeats=num_repeats_flat)
            # Reshape to (total_repeats, 1)
            updates = tf.reshape(expanded_values, (-1, 1)) 

            # Create indices to update in 'x'
            idx1 = tf.range(tf.shape(x)[0], dtype=tf.int32)
            idx1 = tf.repeat(idx1, repeats=num_repeats_flat)
            idx1 = tf.reshape(idx1, (-1, 1))            
            idx2 = tf.zeros((tf.size(updates),), dtype=tf.int32)
            idx2 = tf.reshape(idx2, (-1, 1))
            # Create a range for each element in 'indices'
            ranges = tf.range(0, 201, dtype=tf.int32)
            matrix = indices + ranges
            reshaped_tensor = tf.reshape(matrix, (-1, 1))
            idx3 = reshaped_tensor[reshaped_tensor <= 200]
            idx3 = tf.reshape(idx3, (-1, 1))
            modified_indices = tf.stack([idx1, idx2, idx3], axis=-1)  
            
            # Update 'x' 
            x = tf.tensor_scatter_nd_update(x, modified_indices, updates)
            
            
            # Use tf.gather to select values from 'x' based on indices
            values_to_repeat = tf.gather(dw, indices, batch_dims=2, axis=-1)
            # Get the number of times to repeat each value
            num_repeats = tf.shape(dw)[-1] - indices
            # Flatten 'values_to_repeat' and 'num_repeats' for tf.repeat
            values_flat = tf.reshape(values_to_repeat, [-1])
            num_repeats_flat = tf.reshape(num_repeats, [-1])
            # Use tf.repeat to repeat each value according to the specified number of repeats
            expanded_values = tf.repeat(values_flat, repeats=num_repeats_flat)
            # Reshape to (total_repeats, 1)
            updates_dw = tf.reshape(expanded_values, (-1, 1)) 
            
            # Check if the third element is not equal to 200
            condition = tf.math.not_equal(modified_indices[:, :, 2], 200)
            # Use boolean indexing to filter out arrays with third element equal to 200
            filtered_indices = tf.boolean_mask(modified_indices, condition, axis=0)
            filtered_indices = tf.expand_dims(filtered_indices, axis=1)
            
            # Update 'dw' 
            dw = tf.tensor_scatter_nd_update(dw, filtered_indices, updates_dw)
        
        ####################################################################################           
            
        for t in range(0, self.bsde.num_time_interval-1):
            
            y_new = (y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True))
            
            if self.eqn_config.eqn_name == "BarrierOption" : 
                # # come andrebbe fatto secondo il paper
                # barrier_check_3rd_dim = tf.squeeze(barrier_check, axis=-1)
                # y_barrier = y*(all_one_vec-barrier_check[:,:,t])+y_barrier*barrier_check[:,:,t]  
                # break
            
                # come faro io
                y = y*(all_one_vec-barrier_check[:,:,t])+y_new*barrier_check[:,:,t]
                # print(y)
                
            else:
                y = y_new
                      
            try:          
                z = self.subnet[t](x[:, :, t + 1], training) / self.bsde.dim
            except TypeError:
                z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=training) / self.bsde.dim
            

        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
        # y_barrier = y_barrier - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x_barrier[:, :, -2], y_barrier, z) + \
        #       tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
                     
        return y#, y_barrier

    def predict_step(self, data):
        dw, x = data[0]
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(dw)[0], 1]), dtype=self.net_config.dtype)
        y = all_one_vec * self.y_init
        y_new = all_one_vec * self.y_init
        z = tf.matmul(all_one_vec, self.z_init)        
        
        history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
        history = history.write(0,y)
                
        if self.eqn_config.eqn_name == "BarrierOption" :
        
            # Create a boolean mask indicating elements greater than the barrier
            mask = tf.math.greater(x, self.eqn_config.barrier)
            # Get the first index if it exists, otherwise use a default value (e.g., [0, 0])
            first_false_index_initial = tf.argmax(tf.math.logical_not(mask), axis=-1, output_type=tf.int32)
            # Handle the case where all values are True
            last_index = tf.shape(mask)[-1] - 1
            first_false_index = tf.where(tf.reduce_all(mask, axis=-1), last_index, first_false_index_initial)
            equality_tensor = ~tf.math.equal(first_false_index_initial, first_false_index)
            first_false_index = tf.cast(first_false_index, tf.int32)
            first_false_index = tf.expand_dims(first_false_index, axis=-1)
            
            range_tensor = tf.range(tf.shape(x)[-1], dtype=tf.int32)
            range_tensor = tf.expand_dims(range_tensor, axis=0)
            
            barrier_check = tf.less(range_tensor, first_false_index)
            barrier_check = tf.cast(barrier_check, tf.float64)
            
            # Extract the indices from tensor 'first_false_index'
            first_false_index = tf.where(tf.equal(first_false_index, 0), 1, first_false_index)
            indices = first_false_index[..., 0] - 1
            # Take into accout whether all are TRUE in mask
            indices = indices + tf.multiply(tf.ones(shape=(self.net_config.valid_size, 1), dtype=tf.int32), tf.cast(equality_tensor, dtype=tf.int32))
            # Use tf.gather to select values from 'x' based on indices
            values_to_repeat = tf.gather(x, indices, batch_dims=2, axis=-1)
            # Get the number of times to repeat each value
            num_repeats = tf.shape(x)[-1] - indices
            # Flatten 'values_to_repeat' and 'num_repeats' for tf.repeat
            values_flat = tf.reshape(values_to_repeat, [-1])
            num_repeats_flat = tf.reshape(num_repeats, [-1])
            # Use tf.repeat to repeat each value according to the specified number of repeats
            expanded_values = tf.repeat(values_flat, repeats=num_repeats_flat)
            updates = tf.reshape(expanded_values, (-1, 1)) 

            # Create indices to update in 'x'
            idx1 = tf.range(tf.shape(x)[0], dtype=tf.int32)
            idx1 = tf.repeat(idx1, repeats=num_repeats_flat)
            idx1 = tf.reshape(idx1, (-1, 1))            
            idx2 = tf.zeros((tf.size(updates),), dtype=tf.int32)
            idx2 = tf.reshape(idx2, (-1, 1))
            # Create a range for each element in 'indices'
            ranges = tf.range(0, 201, dtype=tf.int32)
            matrix = indices + ranges
            reshaped_tensor = tf.reshape(matrix, (-1, 1))
            idx3 = reshaped_tensor[reshaped_tensor <= 200]
            idx3 = tf.reshape(idx3, (-1, 1))
            modified_indices = tf.stack([idx1, idx2, idx3], axis=-1)  
            
            # Update 'x' 
            x = tf.tensor_scatter_nd_update(x, modified_indices, updates)
            
            
            
            # Use tf.gather to select values from 'x' based on indices
            values_to_repeat = tf.gather(dw, indices, batch_dims=2, axis=-1)
            # Get the number of times to repeat each value
            num_repeats = tf.shape(dw)[-1] - indices
            # Flatten 'values_to_repeat' and 'num_repeats' for tf.repeat
            values_flat = tf.reshape(values_to_repeat, [-1])
            num_repeats_flat = tf.reshape(num_repeats, [-1])
            # Use tf.repeat to repeat each value according to the specified number of repeats
            expanded_values = tf.repeat(values_flat, repeats=num_repeats_flat)
            # Reshape to (total_repeats, 1)
            updates_dw = tf.reshape(expanded_values, (-1, 1)) 
            
            # Check if the third element is not equal to 200
            condition = tf.math.not_equal(modified_indices[:, :, 2], 200)
            # Use boolean indexing to filter out arrays with third element equal to 200
            filtered_indices = tf.boolean_mask(modified_indices, condition, axis=0)
            filtered_indices = tf.expand_dims(filtered_indices, axis=1)
            
            # Update 'dw' 
            dw = tf.tensor_scatter_nd_update(dw, filtered_indices, updates_dw)
        
        for t in range(0, self.bsde.num_time_interval-1):
            
            y_new = y - self.bsde.delta_t * (
                self.bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + tf.reduce_sum(z * dw[:, :, t], 1, keepdims=True)
            
            history = history.write(t+1,y_new)
            
            if self.eqn_config.eqn_name == "BarrierOption" : 
                y = y*(all_one_vec-barrier_check[:,:,t])+y_new*barrier_check[:,:,t]
            else:
                y = y_new
                   
            try:          
                z = self.subnet[t](x[:, :, t + 1], training=False) / self.bsde.dim
            except TypeError:
                z = self.subnet(tf.concat([time_stamp[t+1]*all_one_vec,x[:, :, t + 1]],axis=1), training=False) / self.bsde.dim
        # terminal time
        y = y - self.bsde.delta_t * self.bsde.f_tf(time_stamp[-1], x[:, :, -2], y, z) + \
            tf.reduce_sum(z * dw[:, :, -1], 1, keepdims=True)
    
        history = history.write(self.bsde.num_time_interval,y)
        history = tf.transpose(history.stack(),perm=[1,2,0])
        return dw,x,history      

    def simulate_path(self,num_sample):
        return self.predict(num_sample)[2]           


class FeedForwardSubNet(tf.keras.Model):
    def __init__(self, config,dim):
        super(FeedForwardSubNet, self).__init__()        
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None,)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim
        self.dense_layers.append(tf.keras.layers.Dense(dim, activation=None))

    def call(self, x, training):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense """
        x = self.bn_layers[0](x, training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i+1](x, training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        return x

### univeral neural networks instead of one neural network at each time point
def get_universal_neural_network(input_dim):    
    input = layers.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)    
    for i in range(5):
        x = layers.Dense(input_dim+10,'relu',False)(x)
        x = layers.BatchNormalization()(x)
    output = layers.Dense(input_dim-1,'relu')(x)
    #output = layers.Dense(2*dim,'relu')(x)
    return tf.keras.Model(input,output)
'''
def get_universal_neural_network(input_dim,num_neurons=20,num_hidden_blocks=4):
    
    input = tf.keras.Input(shape=(input_dim,))
    x = layers.BatchNormalization()(input)
    s = layers.Dense(num_neurons,activation='relu',use_bias=False)(x)
    s = layers.BatchNormalization()(s)
    for i in range(num_hidden_blocks-1):        
        z = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(s)])
        z = Add_bias(num_neurons)(z)
        z = layers.Activation(tf.nn.sigmoid)(z)
       
        g = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(s)])
        g = Add_bias(num_neurons)(g)
        g = layers.Activation(tf.nn.sigmoid)(g)
        r = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(s)])
        r = Add_bias(num_neurons)(r)
        r = layers.Activation(tf.nn.sigmoid)(r)
        h = layers.add([layers.Dense(num_neurons,None,False)(x),layers.Dense(num_neurons,None,False)(layers.multiply([s,r]))])
        h = Add_bias(num_neurons)(h)
        h = layers.Activation(tf.nn.relu)(h)
        s = layers.add([layers.multiply([1-g,h]),layers.multiply([z,s])])
        s = layers.BatchNormalization()(s)
    
    output = layers.Dens e(input_dim-1,None)(s)
    return tf.keras.Mode l(input,output)
'''
      
class Add_bias(tf.keras.layers.Layer):
    def __init__(self,units):        
        super(Add_bias, self).__init__()       
        self.units = units
    def build(self, input_shape):              
        self.b = self.add_weight(shape=(self.units,),
                                initializer='zeros',
                                trainable=True)
    def call(self, inputs):
        return inputs + self.b


    

