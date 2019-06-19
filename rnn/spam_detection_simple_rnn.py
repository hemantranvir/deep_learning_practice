import sys

from keras.layers import SimpleRNN, Embedding, Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.utils import plot_model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./spam_text_message_data.csv")

print(data.head())

print(data.tail())

messages = []
labels = []
for index, row in data.iterrows():
    messages.append(row['Message'])
    if row['Category'] == 'ham':
        labels.append(0)
    else:
        labels.append(1)

messages = np.asarray(messages)
labels = np.asarray(labels)

print("Number of messages: ", len(messages))
print("Number of labels: ", len(labels))

max_words = 10000
max_len = 500

# Ignore all words except the 10000 most common words
tokenizer = Tokenizer()
# Calculate the frequency of words
tokenizer.fit_on_texts(messages)
# Convert text to list of integers
sequences = tokenizer.texts_to_sequences(messages)

# Dict keeping track of words to integer index
word_index = tokenizer.word_index

# Convert the list of sequences(of integers) to 2D array with padding
# maxlen specifies the maximum length of sequence (truncated if longer, padded if shorter)
data = pad_sequences(sequences, maxlen=max_len)

print("data shape: ", data.shape)

## Shuffle data
#np.random.seed(42)
#indices = np.arange(data.shape[0])
#np.random.shuffle(indices)
#data = data[indices]
#labels = labels[indices]

# We will use 80% of data for training & validation(80% train, 20% validation) and 20% for testing
train_samples = int(len(messages)*0.8)

messages_train = data[:train_samples]
labels_train = labels[:train_samples]

messages_test = data[train_samples:len(messages)-2]
labels_test = labels[train_samples:len(messages)-2]

#print('messages 5573: ', messages[5573])
#print('data 5573', data[5573])

embedding_mat_columns=32
# Construct the SimpleRNN model
model = Sequential()
## Add embedding layer to convert one-hot encoding to word embeddings(the model learns the
## embedding matrix during training), embedding matrix has max_words as no. of rows and chosen
## no. of columns
model.add(Embedding(input_dim=max_words, output_dim=embedding_mat_columns, input_length=max_len))

model.add(SimpleRNN(32))
#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#callback = TensorBoard(log_dir='./Graph', histogram_freq=0,
#                       write_graph=True, write_images=True)

# Training the model
history_rnn = model.fit(messages_train, labels_train, epochs=10, batch_size=60, validation_split=0.2)

# Testing the model
pred = model.predict_classes(messages_test)
#with np.printoptions(threshold=np.inf):
print(pred)
acc = model.evaluate(messages_test, labels_test)
print("Test loss is {0:.2f} accuracy is {1:.2f}  ".format(acc[0],acc[1]))

#with np.printoptions(threshold=np.inf):
print(labels_test)

print('type of data: ', type(data))

test_seq = np.zeros((1, 500))
test_seq[0,0] = 53
test_seq[0,1] = 125
test_seq[0,2] = 1497
test_seq[0,3] = 922
pred = model.predict_classes(test_seq)
print(pred)

#test_seq[0,1] = 2143
#test_seq[0,2] = 47
#test_seq[0,3] = 1309
#test_seq[0,4] = 31
#test_seq[0,5] = 212
#test_seq[0,6] = 2287
print('message is: ', messages[2])
pred = model.predict_classes(np.matrix(data[2]))
print(pred)
