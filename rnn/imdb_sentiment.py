import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# parameters for data load
num_words = 10000
maxlen = 500

(train_reviews, train_labels), (test_reviews, test_labels) = imdb.load_data(num_words = num_words)

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}

print('---imdb review---')
print([id2word.get(i-3, '?') for i in train_reviews[10]])
print('---label 0:negative, 1:positive---')
print(train_labels[10])

# pad the sequences with zeros and convert the list of sequences of integers to 2D numpy array
train_reviews = pad_sequences(train_reviews, padding = 'post', maxlen=maxlen)
test_reviews = pad_sequences(test_reviews, padding = 'post', maxlen=maxlen)

print('train shape [0]: ', train_reviews.shape[0])
print('train shape [1]: ', train_reviews.shape[1])

print('test shape [0]: ', test_reviews.shape[0])
print('test shape [1]: ', test_reviews.shape[1])

#train_reviews = np.array(train_reviews).reshape((train_reviews.shape[0], train_reviews.shape[1], 1))
#test_reviews = np.array(test_reviews).reshape((test_reviews.shape[0], test_reviews.shape[1], 1))

embedding_cols = 32
model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_cols, input_length=maxlen))
model.add(SimpleRNN(units=embedding_cols))
model.add(Dense(1, activation='sigmoid'))
#model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

batch_size = 64
epochs = 3

val_news, val_labels = train_reviews[:batch_size], train_labels[:batch_size]
train_reviews, train_labels = train_reviews[batch_size:], train_labels[batch_size:]

model.fit(train_reviews, train_labels, validation_data=(val_news, val_labels), batch_size=batch_size, epochs=epochs)

scores = model.evaluate(test_reviews, test_labels, verbose=0)

print('Test accuracy: ', scores)
