import tensorflow as tf
import mitdeeplearning as mdl
import numpy
import matplotlib.pyplot as plt

# sport = tf.constant('tennis', tf.string)
# number = tf.constant(1.34567890237535, tf.float64)
#
# print("'sport' is a {}-d tensor".format(tf.rank(sport).numpy()))
# print("'number' is a {}-d tensor".format(tf.rank(number).numpy()))
# print(tf.shape(sport))
# print(tf.shape(number))


# sports = tf.constant(["Tennis", "Basketball"], tf.string)
# numbers = tf.constant([3.141592, 1.414213, 2.71821], tf.float64)
#
# print("`sports` is a {}-d Tensor with shape: {}".format(tf.rank(sports).numpy(), tf.shape(sports)))
# print("`numbers` is a {}-d Tensor with shape: {}".format(tf.rank(numbers).numpy(), tf.shape(numbers)))
# #tf.rank и tf.shape предназначены для работы с объектами типа tf.Tensor.


# matrix = tf.constant([[1, 2], [3, 4], [5, 6], [8, 9]], tf.int64)
# #не прямоугольная матрица не может стать тензором.
# # matrix1 = [[1,2],[3,4]]
# matrix2 = tf.constant([[1, 1], [1, 1], [1, 1]], tf.int64)
# assert isinstance(matrix, tf.Tensor), "matrix must be a tf Tensor object"
# assert tf.rank(matrix2).numpy() == 2
# print(matrix2)
# print(matrix2.numpy())


# images = tf.zeros([10, 256, 256, 3])
# assert isinstance(images, tf.Tensor), "matrix must be a tf Tensor object"
# assert tf.rank(images).numpy() == 4, "matrix must be of rank 4"
# assert tf.shape(images).numpy().tolist() == [10, 256, 256, 3], "matrix is incorrect shape"
# matrix3 = tf.zeros([2,3,4])
# print(matrix3)
# print(matrix3.numpy())


# matrix = tf.constant([[1, 2], [3, 4], [5, 6], [8, 9]], tf.int64)
# row_vector = matrix[1]
# column_vector = matrix[:,1] #[:, :] literally means [all rows, all columns].
# scalar = matrix[0, 1]
#
# print("`row_vector`: {}".format(row_vector.numpy()))
# print("`column_vector`: {}".format(column_vector.numpy()))
# print("`scalar`: {}".format(scalar.numpy()))


# a = tf.constant(15)
# b = tf.constant(61)
# c1 = tf.add(a,b)
# c2 = a + b
# print(c1)
# print(c2)


### Defining Tensor computations ###

# Construct a simple computation function
# def func(a,b):
#   '''TODO: Define the operation for c, d, e (use tf.add, tf.subtract, tf.multiply).'''
#   c = tf.add(a,b)
#   d = tf.subtract(b,1)
#   e = tf.multiply(c,d)
#   return e
#
# # Consider example values for a,b
# a, b = 1.5, 2.5
# # Execute the computation
# e_out = func(a,b)
# print(e_out)


# ### Defining a network Layer ###
#
# # n_output_nodes: number of output nodes
# # input_shape: shape of the input
# # x: input to the layer
#
# class OurDenseLayer(tf.keras.layers.Layer):
#   def __init__(self, n_output_nodes):
#     super(OurDenseLayer, self).__init__()
#     self.n_output_nodes = n_output_nodes
#
#   def build(self, input_shape):
#     d = int(input_shape[-1])
#     # Define and initialize parameters: a weight matrix W and bias b
#     # Note that parameter initialization is random!
#     self.W = self.add_weight(name="weight", shape=[d, self.n_output_nodes]) # note the dimensionality
#     self.b = self.add_weight(name="bias", shape=[1, self.n_output_nodes]) # note the dimensionality
#
#   def call(self, x):
#     '''TODO: define the operation for z (hint: use tf.matmul)'''
#     z = tf.add(tf.matmul(x,self.W), self.b)
#
#     '''TODO: define the operation for out (hint: use tf.sigmoid)'''
#     y = tf.sigmoid(z)
#     return y
#
# # Since layer parameters are initialized randomly, we will set a random seed for reproducibility
# tf.keras.utils.set_random_seed(1)
# layer = OurDenseLayer(3)
# layer.build((1,2))
# x_input = tf.constant([[1,2.]], shape=(1,2))
# y = layer.call(x_input)
#
# # test the output!
# print(y.numpy())
# mdl.lab1.test_custom_dense_layer_output(y)


# ### Defining a neural network using the Sequential API ###
#
# # Import relevant packages
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
#
# # Define the number of outputs
# n_output_nodes = 3
#
# # First define the model
# model = Sequential()
#
# '''TODO: Define a dense (fully connected) layer to compute z'''
# # Remember: dense layers are defined by the parameters W and b!
# # You can read more about the initialization of W and b in the TF documentation :)
# # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense?version=stable
# dense_layer = tf.keras.layers.Dense(n_output_nodes)
#
# # Add the dense layer to the model
# model.add(dense_layer)
#
# # Test model with example input
# x_input = tf.constant([[1,2.]], shape=(1,2))
#
# '''TODO: feed input into the model and predict the output!'''
# model_output = model(x_input)
# print(model_output)


# ### Defining a model using subclassing ###
#
# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense
# class SubclassModel(tf.keras.Model):
#
#   # In __init__, we define the Model's layers
#   def __init__(self, n_output_nodes):
#     super(SubclassModel, self).__init__()
#     '''TODO: Our model consists of a single Dense layer. Define this layer.'''
#     self.dense_layer = tf.keras.layers.Dense(n_output_nodes)
#
#   # In the call function, we define the Model's forward pass.
#   def call(self, inputs):
#     return self.dense_layer(inputs)
#
# n_output_nodes = 3
# model = SubclassModel(n_output_nodes)
#
# x_input = tf.constant([[1,2.]], shape=(1,2))
#
# print(model.call(x_input))


### Defining a model using subclassing and specifying custom behavior ###

# from tensorflow.keras import Model
# from tensorflow.keras.layers import Dense
# class IdentityModel(tf.keras.Model):
#
#   # As before, in __init__ we define the Model's layers
#   # Since our desired behavior involves the forward pass, this part is unchanged
#   def __init__(self, n_output_nodes):
#     super(IdentityModel, self).__init__()
#     self.dense_layer = tf.keras.layers.Dense(n_output_nodes, activation='sigmoid')
#
#   '''TODO: Implement the behavior where the network outputs the input, unchanged,
#       under control of the isidentity argument.'''
#   def call(self, inputs, isidentity=False):
#     x = self.dense_layer(inputs)
#     if isidentity:
#         return inputs
#     else:
#         return x
#     '''TODO: Implement identity behavior'''
#
# n_output_nodes = 3
# model = IdentityModel(n_output_nodes)
#
# x_input = tf.constant([[1, 2.]], shape=(1, 2))
# '''TODO: pass the input into the model and call with and without the input identity option.'''
# out_activate = model(x_input, isidentity=False)
# out_identity = model(x_input, isidentity=True)
#
# print("Network output with activation: {}; network identity output: {}".format(out_activate.numpy(), out_identity.numpy()))


### Gradient computation with GradientTape ###

# # y = x^2
# # Example: x = 3.0
# x = tf.Variable(3.0)
#
# # Initiate the gradient tape
# with tf.GradientTape() as tape:
#   # Define the function
#   y = x * x
# # Access the gradient -- derivative of y with respect to x
# dy_dx = tape.gradient(y, x)
#
# assert dy_dx.numpy() == 6.0


### Function minimization with automatic differentiation and SGD ###

# Initialize a random value for our initial x
# x = tf.Variable(tf.random.normal([1]))
# print("Initializing x={}".format(x.numpy()))
#
# learning_rate = 1e-2 # learning rate for SGD
# history = []
# # Define the target value
# x_f = 4
#
# # We will run SGD for a number of iterations. At each iteration, we compute the loss,
# #   compute the derivative of the loss with respect to x, and perform the SGD update.
# for i in range(500):
#   with tf.GradientTape() as tape:
#     '''TODO: define the loss as described above'''
#     loss = (x-x_f)**2
#
#   # loss minimization using gradient tape
#   grad = tape.gradient(loss, x) # compute the derivative of the loss with respect to x
#   new_x = x - learning_rate*grad # sgd update
#   x.assign(new_x) # update the value of x
#   history.append(x.numpy()[0])
#
# # Plot the evolution of x as we optimize towards x_f!
# plt.plot(history)
# plt.plot([0, 500],[x_f,x_f])
# plt.plot([100,200], [1,3])
# plt.legend(('Predicted', 'True', 'random line'))
# plt.xlabel('Iteration')
# plt.ylabel('x value')
# plt.show()