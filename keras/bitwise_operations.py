#
# Bitwise operations with Keras
#

from keras.models import Sequential
from keras.layers import Dense
import numpy


EPOCHS = 5000


numpy.random.seed(139)


def build_model(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, init='uniform', activation='relu'))
    model.add(Dense(16, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

    
def train(model, X, Y):
    model.fit(X, Y, nb_epoch=EPOCHS, verbose=0)


def predict(model, X):
    return model.predict(X)


def test(dataset, label):
    dataset = numpy.array(dataset).astype(float)

    input_dim = dataset.shape[1] - 1

    X = dataset[:,0:input_dim]
    Y = dataset[:,-1]

    model = build_model(input_dim)
    train(model, X, Y)
    
    predictions = predict(model, X)

    predictions = numpy.reshape(predictions, (predictions.shape[0]))
    rounded_predictions = numpy.round(predictions, 2)
    
    print "------------ %s ------------" % label
    print "Expected : %s" % str(Y)
    print "Predicted: %s" % str(rounded_predictions)
    print "Real     : %s\n\n" % str(predictions)

    return X, Y, predictions


if __name__ == "__main__":
    # AND
    dataset = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]
    test(dataset, "AND")

    # OR
    dataset = [[0,0,0], [0,1,1], [1,0,1], [1,1,1]]
    test(dataset, "OR")

    # XOR
    dataset = [[0,0,0], [0,1,1], [1,0,1], [1,1,0]]
    test(dataset, "XOR")

    # NOT
    dataset = [[0,1], [1,0]]
    test(dataset, "NOT")
