#
# Sequence prediction with Keras using stateless LSTM (long short-term memory)
# [Tested with Theano and TensorFlow]
#

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


EPOCHS = 500
LAYERS = [100, 200, 50]
DATA_LENGTH = 50
WINDOW_SIZE = 10


def reshape_dataset(dataset, window_size, offset=0):
    dataX, dataY = [], []
    datasetLength = len(dataset)
    for i in range(datasetLength - window_size):
        a = dataset[i + offset:(i + window_size + offset), 0]
        dataX.append(a)
        if i + window_size + offset < datasetLength:
            dataY.append(dataset[i + window_size + offset, 0])

    return numpy.array(dataX), numpy.array(dataY)


def build_model(window_size):
    model = Sequential()
    model.add(LSTM(LAYERS[0], batch_input_shape=(None, window_size, 1), return_sequences=True))
    model.add(LSTM(LAYERS[1]))
    model.add(Dense(LAYERS[2]))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='Adam')

    return model


def train_network(model, X, Y):
    try:
        model.fit(X, Y, nb_epoch=EPOCHS, shuffle=False, verbose=1)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt'


def test(dataset, window_size, label):
    print "------------ %s ------------" % label

    dataset = numpy.array(dataset).astype(float)
    dataset = numpy.reshape(dataset, (len(dataset), 1))

    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(dataset)

    trainX, trainY = reshape_dataset(dataset, window_size)
    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX, testY = reshape_dataset(dataset, window_size, offset=1)
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # Build model
    model = build_model(window_size)

    # Training
    print "Training ... (Press CTRL+C to stop)"
    train_network(model, trainX, trainY)

    print "Predicting..."
    testPredict = model.predict(testX, batch_size=1)
    testY = scaler.inverse_transform(numpy.reshape(testY, (-1, 1)))
    testPredict = scaler.inverse_transform(testPredict)

    # Next ???
    lastSeq = dataset[-5:]
    print "Last 5 values: %s" % str(numpy.reshape(scaler.inverse_transform(lastSeq), (5)))
    print "Next -> %f\n\n" % float(testPredict[-1])

    # Plotting graph
    plt.plot(testY[:500])
    plt.plot(testPredict[:500])
    plt.show()


if __name__ == "__main__":
    # x * sin(x)
    dataset = [x * numpy.sin(x) for x in range(DATA_LENGTH)]
    test(dataset, window_size=WINDOW_SIZE, label="x * sin(x)")

    # cos(x)
    dataset = [numpy.cos(x) for x in range(DATA_LENGTH)]
    test(dataset, window_size=WINDOW_SIZE, label="cos(x)")

    # [0 ... DATA_LENGTH-1]
    dataset = [x for x in range(DATA_LENGTH)]
    test(dataset, window_size=WINDOW_SIZE, label="Simple sequence")

    # Multiples of 3
    dataset = [x * 3 for x in range(DATA_LENGTH)]
    test(dataset, window_size=WINDOW_SIZE, label="Multiples of 3")    
