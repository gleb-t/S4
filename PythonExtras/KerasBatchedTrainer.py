import itertools

import math
import random
import time
from typing import *

import keras
import sklearn.metrics
import numpy as np

import PythonExtras.Normalizer as Normalizer
from PythonExtras import numpy_extras as npe


class KerasBatchedCallback(keras.callbacks.Callback):

    def on_macro_batch_start(self, macroBatch: int, logs=None):
        pass

    def on_macro_batch_end(self, macroBatch: int, logs=None):
        pass


class KerasBatchedLambdaCallback(KerasBatchedCallback):
    def __init__(self, on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None,
                 on_train_begin=None, on_train_end=None, on_macro_batch_start=None, on_macro_batch_end=None):
        super(KerasBatchedCallback, self).__init__()
        emptyCallback = lambda *args: None
        self.on_epoch_begin = on_epoch_begin or emptyCallback
        self.on_epoch_end = on_epoch_end or emptyCallback
        self.on_batch_begin = on_batch_begin or emptyCallback
        self.on_batch_end = on_batch_end or emptyCallback
        self.on_train_begin = on_train_begin or emptyCallback
        self.on_train_end = on_train_end or emptyCallback
        self.on_macro_batch_start = on_macro_batch_start or emptyCallback
        self.on_macro_batch_end = on_macro_batch_end or emptyCallback


class KerasBatchedCallbackList(keras.callbacks.CallbackList):

    def on_macro_batch_start(self, macroBatch: int, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_macro_batch_start'):
                callback.on_macro_batch_start(macroBatch, logs)

    def on_macro_batch_end(self, macroBatch: int, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            if hasattr(callback, 'on_macro_batch_end'):
                callback.on_macro_batch_end(macroBatch, logs)

    def set_validation_data(self, valDataX: List[np.ndarray], valDataY: List[np.ndarray]):
        # Have to interface with Keras here.
        # It expects not only X and Y data, but 'sample weights' and an optional 'learning phase' bool.
        valData = list(itertools.chain(valDataX, valDataY, [np.ones((valDataX[0].shape[0],))]))
        if self.callbacks[0].model._uses_dynamic_learning_phase():
            valData += [0.0]
        for callback in self.callbacks:
            callback.validation_data = valData


class KerasBatchedTrainer:

    class History:

        def __init__(self, metricsPerEpoch: Optional[List[Dict[str, float]]] = None):
            self.metricsPerEpoch = metricsPerEpoch or []

        def add_epoch(self, trainMetrics: Dict[str, float], testMetrics: Dict[str, float]):
            self.metricsPerEpoch.append({
                **trainMetrics,
                **testMetrics
            })

        def get_train_loss_history(self):
            return [d['loss'] for d in self.metricsPerEpoch]

        def get_test_loss_history(self):
            return [d['val_loss'] for d in self.metricsPerEpoch]

    # todo document better
    def __init__(self,
                 model: keras.Model,
                 trainX: Union[npe.LargeArray, List[npe.LargeArray]],
                 trainY: Union[npe.LargeArray, List[npe.LargeArray]],
                 testX: Union[npe.LargeArray, List[npe.LargeArray]],
                 testY: Union[npe.LargeArray, List[npe.LargeArray]],
                 macrobatchSize: int,
                 minibatchSize: int, minibatchSizeEval: int = None,
                 normalizerX: Normalizer = None,
                 featureAxis: int = None,
                 macrobatchPreprocess: Optional[Callable] = None):

        if minibatchSizeEval is None:
            minibatchSizeEval = minibatchSize

        trainX, trainY, testX, testY = self._to_list(trainX), self._to_list(trainY), \
                                       self._to_list(testX), self._to_list(testY)

        # The interface allows multiple outputs, but we don't need that functionality atm.
        if len(trainY) > 1:
            raise NotImplementedError()

        tensorsTrain = itertools.chain(trainX, trainY)
        tensorsTest = itertools.chain(testX, testY)
        if not all((tensor.shape[0] == trainX[0].shape[0] for tensor in tensorsTrain)) or \
           not all((tensor.shape[0] == testX[0].shape[0] for tensor in tensorsTest)):
            raise RuntimeError("The batch dimension of all input and output tensors should have the same size.")

        # Round up the macrobatch size to be a multiple of the minibatch size.
        # This sometimes helps to avoid very small minibatches at the end of an epoch.
        macrobatchSize = int(math.ceil(macrobatchSize / minibatchSize)) * minibatchSize

        self.model = model
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.macrobatchSize = macrobatchSize
        self.minibatchSize = minibatchSize
        self.minibatchSizeEval = minibatchSizeEval
        self.normalizerX = normalizerX
        self.featureAxis = featureAxis

        self.macrobatchPreprocess = macrobatchPreprocess

        self.trainPointNumber = self.trainX[0].shape[0]
        self.testPointNumber = self.testX[0].shape[0]
        self.macrobatchNumberTrain = int(math.ceil(self.trainPointNumber / self.macrobatchSize))
        self.macrobatchNumberTest = int(math.ceil(self.testPointNumber / self.macrobatchSize))

    def fit_normalizer(self, printFunc=print):

        if len(self.trainX) > 1:
            raise NotImplementedError()

        # The normalizer is fit only to the first macrobatch, to save on time and code.
        # This should be sufficient, esp. if the training data is shuffled.
        trainBatchX = self.trainX[0][:min(self.macrobatchSize, self.trainPointNumber)]
        self.normalizerX.fit(trainBatchX, axis=self.featureAxis)
        printFunc("Computed normalization parameters: {}".format(self.normalizerX.get_weights()))

        return self.normalizerX

    def test_init_loss(self, macrobatchNumber: int = None, lossType: str = 'mse', printFunc=print):
        if len(self.trainX) > 1:
            raise NotImplementedError()

        macrobatchNumber = macrobatchNumber or self.macrobatchNumberTrain

        printFunc("Performing init loss check using {} macrobatches.".format(macrobatchNumber))
        if lossType == 'mse':
            # This is rather copy-pasty, cf. train() method.
            loss = 0
            predMean, predVar, dataMean, dataVar = 0, 0, 0, 0
            for macrobatchIndex in range(0, macrobatchNumber):
                dataStart = macrobatchIndex * self.macrobatchSize
                dataEnd = min(dataStart + self.macrobatchSize, self.trainPointNumber)

                # Extract the batch and normalize if needed.
                predictionInput = self.trainX[0][dataStart:dataEnd]
                predictionTarget = self.trainY[0][dataStart:dataEnd]
                if self.normalizerX is not None:
                    if not self.normalizerX.isFit:
                        self.fit_normalizer(printFunc=printFunc)

                    predictionInput = self.normalizerX.scale(predictionInput)

                prediction = self.model.predict(predictionInput, batch_size=self.minibatchSizeEval, verbose=0)
                macrobatchMse = sklearn.metrics.mean_squared_error(predictionTarget.flatten(),
                                                                   prediction.flatten())
                # Normalize to avoid bias.
                normCoef = (dataEnd - dataStart) / min(macrobatchNumber * self.macrobatchSize, self.trainPointNumber)
                # Compute loss and also prediction and data statistics.
                # Technically, for variance, we should multiply by N/N-1 at the end, but it's not crucial here.
                loss += macrobatchMse * normCoef
                predMean += np.mean(prediction) * normCoef
                predVar += np.var(prediction) * normCoef
                dataMean += np.mean(predictionTarget) * normCoef
                dataVar += np.var(predictionTarget) * normCoef
        else:
            raise ValueError("Invalid loss: '{}'".format(lossType))

        # (Not 100% on this): MSE = Var(y - y_hat) + E(y - y_hat)^2
        expectedLoss = predVar + dataVar + (predMean - dataMean) ** 2
        printFunc("Loss at init: {:.3f}. Expected loss: {:.3f} Prediction mean(var): {:.3f} ({:.3f}) "
                  "Data mean(var): {:.3f} ({:.3f})".format(loss, expectedLoss, predMean, predVar, dataMean, dataVar))

        return loss, expectedLoss, predMean, predVar, dataMean, dataVar

    def train(self, epochNumber: int, callbacks: List[keras.callbacks.Callback] = None, printFunc=print) -> History:

        # Construct the callback list, provide parameters that won't change during the training.
        callbackList = KerasBatchedCallbackList(callbacks)
        callbackList.set_params({
            'batch_size': self.minibatchSize,
            'epochs': epochNumber,
            'steps': None,
            'samples': self.trainPointNumber,
            'verbose': False,
            'do_validation': False,
            'metrics': [],
        })
        callbackList.set_model(self.model)
        # Some callbacks (e.g. Tensorboard) require val.data. Provide only one macrobatch to not overflow RAM.
        callbackList.set_validation_data([array[:self.macrobatchSize] for array in self.testX],
                                         [array[:self.macrobatchSize] for array in self.testY])
        callbackList.on_train_begin()

        printFunc("Starting batched training. {} epochs, {:,} macrobatches with up to {:,} points each.".format(
            epochNumber, self.macrobatchNumberTrain, self.macrobatchSize
        ))

        # Main 'learning epoch' loop, which goes through all the data at each step.
        trainingHistory = KerasBatchedTrainer.History()
        for epochIndex in range(0, epochNumber):
            epochTrainMetrics = {}
            timeLoad, timeTraining, timeNormalization = 0, 0, 0
            timeStart = time.time()
            callbackList.on_epoch_begin(epochIndex)
            # Access macrobatches in a different random order every epoch.
            macrobatchDataIndices = list(range(0, self.macrobatchNumberTrain))
            random.shuffle(macrobatchDataIndices)
            # 'Macrobatch' loop, which splits the training data into chunks for out-of-core processing.
            for macrobatchIndex in range(0, self.macrobatchNumberTrain):
                macrobatchDataIndex = macrobatchDataIndices[macrobatchIndex]
                dataStart = macrobatchDataIndex * self.macrobatchSize
                dataEnd = min(dataStart + self.macrobatchSize, self.trainPointNumber)

                # Load the training data chunk from disk.
                t = time.time()
                trainBatchX = [array[dataStart:dataEnd, ...] for array in self.trainX]
                trainBatchY = [array[dataStart:dataEnd, ...] for array in self.trainY]
                timeLoad += time.time() - t

                t = time.time()
                if self.normalizerX is not None:
                    if len(self.trainX) > 1:
                        raise NotImplementedError()

                    if not self.normalizerX.isFit:
                        self.fit_normalizer(printFunc=printFunc)

                    # Scale the training data.
                    assert len(trainBatchX) == 1
                    self.normalizerX.scale(trainBatchX[0], inPlace=True)  # todo Support multiple tensors.

                if self.macrobatchPreprocess:
                    trainBatchX, trainBatchY = self.macrobatchPreprocess(trainBatchX, trainBatchY)

                timeNormalization += time.time() - t
                callbackList.on_macro_batch_start(macrobatchIndex, {'epoch': epochIndex})

                t = time.time()
                # Train the model on the macrobatch. Minibatch size specifies SGD batch size,
                # which is also the chunk size for GPU loads.
                batchHistory = self.model.fit(trainBatchX, trainBatchY,
                                              epochs=1, shuffle=True,
                                              batch_size=self.minibatchSize,
                                              verbose=0)

                timeForMacrobatch = time.time() - t
                timeTraining += timeForMacrobatch

                # Maintain epoch metric values as the mean of macrobatch metrics.
                # Note, that since the model is training progressively,
                # this is not the true metric value. But this is faster and accurate enough.

                # Batches have different sizes -> weight the batch metrics to get an unbiased epoch mean estimate.
                meanNormCoef = (dataEnd - dataStart) / self.trainPointNumber
                for metricName in batchHistory.history.keys():
                    macrobatchMetric = batchHistory.history[metricName][0]  # There's only one epoch.
                    if macrobatchIndex == 0:
                        epochTrainMetrics[metricName] = 0

                    epochTrainMetrics[metricName] += meanNormCoef * macrobatchMetric

                callbackList.on_macro_batch_end(macrobatchIndex, {'epoch': epochIndex, **epochTrainMetrics})

            t = time.time()
            # Perform out-of-core prediction for the test data.
            epochTestMetrics = self.compute_test_metrics()

            timeEvaluation = time.time() - t
            timeTotal = time.time() - timeStart
            timeRest = timeTotal - timeTraining - timeEvaluation - timeNormalization - timeLoad

            trainingHistory.add_epoch(epochTrainMetrics, epochTestMetrics)
            callbackList.on_epoch_end(epochIndex, {**epochTrainMetrics, **epochTestMetrics,
                                                   'time_total': timeTotal, 'time_load': timeLoad,
                                                   'time_train': timeTraining, 'time_norm': timeNormalization,
                                                   'time_val': timeEvaluation, 'time_rest': timeRest})

            # Support the standard EarlyStopping callback.
            # todo Write our own early stopping logic?
            if self.model.stop_training:
                if printFunc:
                    printFunc("Training stop requested on epoch {}.".format(epochIndex))
                break

        callbackList.on_train_end()

        return trainingHistory

    def compute_test_metrics(self) -> Dict[str, float]:
        testMetrics = {}
        for macrobatchIndex in range(0, self.macrobatchNumberTest):
            dataStart = macrobatchIndex * self.macrobatchSize
            dataEnd = min(dataStart + self.macrobatchSize, self.testPointNumber)

            # Extract the batch and normalize if needed.
            predictionInput = [array[dataStart:dataEnd] for array in self.testX]
            if self.normalizerX is not None:
                if len(predictionInput) > 1:
                    raise NotImplementedError()

                predictionInput = self.normalizerX.scale(predictionInput[0])  # todo Support multiple tensors.

            if self.macrobatchPreprocess:
                predictionInput, _ = self.macrobatchPreprocess(predictionInput, None)

            prediction = self.model.predict(predictionInput, batch_size=self.minibatchSizeEval, verbose=0)
            predictionTarget = self.testY[0][dataStart:dataEnd]

            batchMetrics = self._compute_batch_metrics(prediction, predictionTarget)

            # Add up the metrics to compute the average, while normalizing to avoid bias.
            # Rename the metrics to follow Keras' conventions.
            for k, v in batchMetrics.items():
                nameFull = 'val_' + k
                valueNorm = v * ((dataEnd - dataStart) / self.testPointNumber)
                if nameFull in testMetrics:
                    testMetrics[nameFull] += valueNorm
                else:
                    testMetrics[nameFull] = valueNorm

        return testMetrics

    def _compute_batch_metrics(self, prediction, predictionTarget) -> Dict[str, float]:

        metrics = {}
        for metricName in ['loss'] + self.model.metrics:
            if metricName == 'loss':
                if self.model.loss == 'mse':
                    metric = sklearn.metrics.mean_squared_error(predictionTarget.flatten(), prediction.flatten())
                elif self.model.loss == 'binary_crossentropy':
                    # We need a lower epsilon (1e-7, instead of 1e-14) because we're dealing with float32, not 64.
                    # Was getting NaNs here, but haven't 100% confirmed that this fixed the issue.
                    metric = sklearn.metrics.log_loss(predictionTarget, prediction, eps=1e-7)
                else:
                    raise ValueError("Unknown model loss: '{}'.".format(self.model.loss))
            elif metricName == 'accuracy':
                metricName = 'acc'  # Alias.
                metric = np.mean(np.round(prediction) == predictionTarget)
            else:
                raise ValueError("Unknown model metric: '{}'.".format(metricName))

            metrics[metricName] = metric

        return metrics

    @staticmethod
    def _to_list(value):
        return [value] if not isinstance(value, list) else value
