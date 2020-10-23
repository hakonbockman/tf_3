import numpy as np
import tensorflow as tf
from tensorflow import keras


'''
    FREEZING LAYERS: UNDERSTANDING THE TRAINABLE ATTRIBUTE

    Looking at weights on different TYPE of layers in keras, to understand which weights
    are trainable and which are not.
'''

# Dense 
layer = keras.layers.Dense(3) # 2 trainable weights (kernel & Bias)
layer.build((None, 4)) # Creates the weights

print('weighs: ', len(layer.weights))
print('trainable_weights: ', len(layer.trainable_weights))
print('non_trainable_weights: ', len(layer.non_trainable_weights))

# Batch normalization
layer = keras.layers.BatchNormalization() # 2 trainable and 2 non-trainable weights
layer.build((None, 4)) # Creates the weights

print('weighs: ', len(layer.weights))
print('trainable_weights: ', len(layer.trainable_weights))
print('non_trainable_weights: ', len(layer.non_trainable_weights))

layer = keras.layers.Dense(3) # Again, 2 trainable weights (kernal & bias)
layer.build((None, 4)) # Creates the weights
layer.trainable = False # Freeze layer

print('weights: ', len(layer.weights))
print('traianable_weights: ', len(layer.trainable_weights))
print('non_trainable_weights: ', len(layer.non_trainable_weights))

# When weights are set as "non-trainable" its values are not updated during training.
# This example will show that:

# Makes a model with 2 Dense layers, with different activation rule
layer1 = keras.layers.Dense(3, activation='relu')
layer2 = keras.layers.Dense(3, activation='sigmoid')
model = keras.Sequential(
    [ keras.Input(shape=(3, )), layer1, layer2 ]
)

# Freeze  layer1
layer1.trainable = False

# keep a copy of the weights in layer1 before training for later 
# comparison/reference
initial_layer_weights_values = layer1.get_weights()

# Train the model
model.compile(optimizer='adam', loss='mse')
model.fit(np.random.random((2, 3)), np.random.random((2, 3)))

# Check that the weights of layer1 have not changed during training
final_layer1_weights_values = layer1.get_weights()

# Asserting would raze a warning in output about differneces if there is one.
np.testing.assert_allclose(
    initial_layer_weights_values[0], final_layer1_weights_values[0]
)
np.testing.assert_allclose(
    initial_layer_weights_values[1], final_layer1_weights_values[1]
)

'''
    RECUURSIVE SETTING OF TRAINABLE ATTRIBITE

    if model is set to trainable = False or layer any sub model or layer (i.e all children) will be also non-trainable as well.
'''

inner_model = keras.Sequential(
    [
        keras.Input(shape=(3, )),
        keras.layers.Dense(3, activation='relu'),
        keras.layers.Dense(3, activation='relu'),
    ]
)

model = keras.Sequential(
    [
        keras.Input(shape=(3, )),
        inner_model,
        keras.layers.Dense(3, activation='sigmoid'),
    ]
)

model.trainable = False # Freeze the outer model

assert inner_model.trainable == False   # All layers in the model are now frozen
assert inner_model.layers[0].trainable == False # trainable is propagated recursively

