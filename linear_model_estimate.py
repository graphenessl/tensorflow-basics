# TensorFlow core: The lowest level of API for TensorFlow

# Tensors: An array of values of [n]-rank
[1., 2., 3.]                        # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]]        # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]]    # a rank 3 tensor with shape [2, 1, 3]

# Import TensorFlow:
import tensorflow as tf

# Graph nodes
frequency = tf.constant(12500.0, dtype=tf.float32)
loudness  = tf.constant(75.0, dtype=tf.float32)

# Run a session. Without running a session, the [nodes] are NOT evaluated
session = tf.Session()
print(session.run([frequency, loudness]))

# Adding tensors together
sum = tf.add(frequency, loudness)

print(sum)
print(session.run([sum]))

# Setting non-literal values to tensors, which can be added later
hydrogen_atoms = tf.placeholder(tf.float32)
nitrogen_atoms = tf.placeholder(tf.float32)

water_molecule = hydrogen_atoms + nitrogen_atoms # '+' provides a shortcut for tf.add(a, b) - SEPARATE GRAPH!

# Add values
print(session.run(water_molecule, {hydrogen_atoms: 2, nitrogen_atoms: 1}))
print(session.run(water_molecule, {hydrogen_atoms: [4, 2], nitrogen_atoms: [5, 5]}))

# The model: Line
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# To initialize all the variables in a TensorFlow program, you must explicitly call a special operation:
init = tf.global_variables_initializer()
session.run(init)

# Since x is a placeholder, we can evaluate linear_model for several values of x simultaneously as follows:
testing_data = [1, 2, 3, 4]
desired_data = [10, 20, 30, 40]

# Change values of initial variables, which have been set in the linear model
fixW = tf.assign(W, [1])
fixb = tf.assign(b, [0])
session.run([fixW, fixb])

print("Input data (x):")
print(testing_data)

print("Desired data:")
print(desired_data)

print("Calculated:")
print(session.run(linear_model, { x: testing_data }))

#=== Loss estimation function ===
# We've created a model, but we don't know how good it is yet.
# To evaluate the model on training data, we need a [y] placeholder to provide the desired values,
# and we need to write a loss function.
# A loss function measures how far apart the current model is from the provided data.
# We'll use a standard loss model for linear regression, which sums the squares
# of the deltas between the current model and the provided data.
# linear_model - y creates a vector where each element is the corresponding
# example's error delta. We call tf.square to square that error.
# Then, we sum all the squared errors to create a single scalar that abstracts
# the error of all examples using tf.reduce_sum:

# Apply loss function
y              = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss           = tf.reduce_sum(squared_deltas)

print("Squared errors:")
print(session.run(loss, {x: testing_data, y: desired_data}))

#=== Reducing error functions ===
# TensorFlow provides optimizers that slowly change each variable in order to
# minimize the loss function. The simplest optimizer is gradient descent.
# It modifies each variable according to the magnitude of the derivative of loss
# with respect to that variable. In general, computing symbolic derivatives
# manually is tedious and error-prone. Consequently, TensorFlow can automatically
# produce derivatives given only a description of the model using the function
# tf.gradients. For simplicity, optimizers typically do this for you.
optimizer   = tf.train.GradientDescentOptimizer(0.01)
train       = optimizer.minimize(loss)

# reset values to incorrect defaults.
session.run(init)

# Machine learning: Minimize the error
for i in range(1000):
  session.run(train, { x: testing_data, y: desired_data })

print(session.run([W,b]))
#curr_W, curr_b, curr_loss = session.run([W, b, loss], { x: desired_data, y: testing_data })
#print("[W]: %s [b]: %s [loss]: %s" %(curr_W, curr_b, curr_loss))
