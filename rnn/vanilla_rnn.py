from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras.utils.vis_utils import plot_model

step = 4

model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1, step), activation="relu"))
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
