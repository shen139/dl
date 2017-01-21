#
# Sequence prediction with Keras using Dense
# [Tested with Theano and TensorFlow]
#

import numpy
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


EPOCHS = 500
LAYERS = [10, 10, 20]
DATA_LENGTH = 50
WINDOW_SIZE = 30
PREDICTIONS = 30


def reshape_dataset(dataset, window_size, offset=0):
    dataX, dataY = [], []
    datasetLength = len(dataset)
    for i in range(datasetLength - window_size):
        dataX.append(dataset[i + offset:(i + window_size + offset), 0])
        if i + window_size + offset < datasetLength:
            dataY.append(dataset[i + window_size + offset, 0])

    return numpy.array(dataX), numpy.array(dataY)


def build_model(window_size):
    model = Sequential()
    model.add(Dense(LAYERS[0], input_dim=WINDOW_SIZE))
    model.add(Dense(LAYERS[1]))
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
    testX, testY = reshape_dataset(dataset, window_size, offset=1)

    # Build model
    model = build_model(window_size)

    # Training
    print "Training ... (Press CTRL+C to stop)"
    train_network(model, trainX, trainY)

    print "Predicting..."
    testPredict = model.predict(testX, batch_size=1)
    nextValue = float(testPredict[-1])
    testY = scaler.inverse_transform(numpy.reshape(testY, (-1, 1)))
    testPredict = scaler.inverse_transform(testPredict)

    # Next
    lastSeq = dataset[-5:]
    print "Last 5 values: %s" % str(numpy.reshape(scaler.inverse_transform(lastSeq), (5)))

    # Next PREDICTIONS
    nextDataset = dataset[-window_size:]
    predictOverPredictions = [nextValue, ]

    for x in range(PREDICTIONS):
        nextDataset = nextDataset[-window_size:]
        nextDataset = numpy.append(nextDataset, nextValue)
        nextDataset = numpy.reshape(nextDataset, (len(nextDataset), 1))
        nextDataset, unused = reshape_dataset(nextDataset, window_size, offset=1)
        predictNext = model.predict(nextDataset, batch_size=1)
        nextValue = float(predictNext[-1])
        predictOverPredictions.append(nextValue)

    normPredictOverPredictions = scaler.inverse_transform(numpy.reshape(numpy.array(predictOverPredictions), (-1, 1)))
    print "Next -> %s\n\n" % str(numpy.reshape(normPredictOverPredictions, (PREDICTIONS + 1)))
    testPredictOverPredictions = numpy.empty_like(testPredict[1:])
    testPredictOverPredictions[:, :] = numpy.nan
    testPredictOverPredictions = numpy.append(testPredictOverPredictions, normPredictOverPredictions)

    # Plotting graph
    plt.plot(testY[:500], label='Real Data', color='blue')
    plt.plot(testPredict[:500], label='predictions over Real Data', color='green')
    plt.plot(testPredictOverPredictions[:500], label='Predictions over predictions', color='yellow')
    plt.legend(loc='upper left', shadow=True)
    plt.show()


if __name__ == "__main__":
    # x * sin(x)
    dataset = [x * numpy.sin(x) for x in numpy.arange(-DATA_LENGTH, DATA_LENGTH, 0.5)]
    test(dataset, window_size=WINDOW_SIZE, label="x * sin(x)")

    # cos(x)
    dataset = [numpy.cos(x) for x in numpy.arange(-DATA_LENGTH, DATA_LENGTH, 0.5)]
    test(dataset, window_size=WINDOW_SIZE, label="cos(x)")

    # Parabola
    dataset = [x*x for x in numpy.arange(-DATA_LENGTH, DATA_LENGTH, 0.5)]
    test(dataset, window_size=WINDOW_SIZE, label="Parabola")

    # [0 ... DATA_LENGTH-1]
    dataset = [x for x in range(DATA_LENGTH)]
    test(dataset, window_size=WINDOW_SIZE, label="Simple sequence")

    # Multiples of 3
    dataset = [x * 3 for x in range(DATA_LENGTH)]
    test(dataset, window_size=WINDOW_SIZE, label="Multiples of 3")    
