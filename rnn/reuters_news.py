import numpy as np

from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, Dense, SimpleRNN, Activation
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

# parameters for data load
num_words = 30000
maxlen = 50
test_split = 0.3

(train_news, train_labels), (test_news, test_labels) = reuters.load_data(num_words = num_words, maxlen = maxlen, test_split = test_split)

word2id = reuters.get_word_index()
id2word = {i: word for word, i in word2id.items()}
labels_words = ['cocoa', 'grain','veg-oil','earn','acq','wheat','copper','housing','money-supply',
                'coffee','sugar','trade','reserves','ship','cotton','carcass','crude','nat-gas',
                'cpi','money-fx','interest','gnp','meal-feed','alum','oilseed','gold','tin',
                'strategic-metal','livestock','retail','ipi','iron-steel','rubber','heat','jobs',
                'lei','bop','zinc','orange','pet-chem','dlr','gas','silver','wpi','hog','lead']
id2words_labels = {i: word for i, word in enumerate(labels_words)}

print('---news wire---')
print([id2word.get(i-3, '?') for i in train_news[0]])
print('---label---')
print(id2words_labels[train_labels[0]])

# pad the sequences with zeros and convert the list of sequences of integers to 2D numpy array
train_news = pad_sequences(train_news, padding = 'post', maxlen=maxlen)
test_news = pad_sequences(test_news, padding = 'post', maxlen=maxlen)

print('train shape [0]: ', train_news.shape[0])
print('train shape [1]: ', train_news.shape[1])

#train_news = np.array(train_news).reshape((train_news.shape[0], train_news.shape[1], 1))
#test_news = np.array(test_news).reshape((test_news.shape[0], test_news.shape[1], 1))

labels = np.concatenate((train_labels, test_labels))
labels = to_categorical(labels)
train_labels = labels[:1395]
test_labels = labels[1395:]

embedding_cols = 50
def vanilla_rnn():
    model = Sequential()
    model.add(Embedding(input_dim=num_words, output_dim=embedding_cols, input_length=maxlen))
    model.add(SimpleRNN(units=embedding_cols))
    model.add(Dense(46))
    model.add(Activation('softmax'))
    adam = optimizers.Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn = vanilla_rnn, epochs = 10, batch_size = 50, verbose = 1)
model.fit(train_news, train_labels)

y_pred = model.predict(test_news)
y_test_ = np.argmax(test_labels, axis = 1)

print(accuracy_score(y_pred, y_test_))
