import os
import time
from typing import Callable, Any

import keras


class KerasCheckpointCallback(keras.callbacks.Callback):
    """
    Saves model checkpoints to disk based on the metric and/or time.
    Maintains three most recent/accurate models.
    """

    def __init__(self,
                 outputDir: str,
                 minIntervalMinutes: int = -1,
                 minIntervalEpochs: int = -1,
                 metric: str = None,
                 metricDelta: float = 0.0,
                 printFunc: Callable[[Any], None] = None):
        super().__init__()

        self.outputDir = outputDir
        self.minIntervalMinutes = minIntervalMinutes
        self.minIntervalEpochs = minIntervalEpochs
        self.metric = metric
        self.metricDelta = metricDelta
        self.printFunc = printFunc

        self.lastTime = time.time()
        self.lastEpoch = 0
        self.bestEpoch = 0
        self.bestMetricValue = float('inf')
        self.bestCheckpointPath = None

        self.savedPaths = []

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        minutesPassed   = (time.time() - self.lastTime) / 60
        timeCondition   = self.minIntervalMinutes != -1 and minutesPassed >= self.minIntervalMinutes
        epochCondition  = self.minIntervalEpochs != -1 and (epoch - self.lastEpoch) >= self.minIntervalEpochs
        metricCondition = self.metric is not None and logs[self.metric] + self.metricDelta < self.bestMetricValue

        if timeCondition or epochCondition or metricCondition:
            self.lastTime = time.time()
            self.lastEpoch = epoch

            checkpointPath = os.path.join(self.outputDir, 'model-e{}.hdf'.format(epoch))

            if metricCondition:
                # 'Metric' condition handled separately, since we need to remember
                # which model was the best one.
                self.bestEpoch = epoch
                self.bestMetricValue = logs[self.metric]
                self.bestCheckpointPath = checkpointPath

            t = time.time()
            self.model.save(checkpointPath)
            self.savedPaths.append(checkpointPath)
            timeWrite = time.time() - t

            # Clean up any old checkpoints, but don't delete the best model.
            if len(self.savedPaths) > 3 and self.savedPaths[0] != self.bestCheckpointPath:
                os.remove(self.savedPaths[0])
                self.savedPaths.pop(0)

            if self.printFunc:
                reason = 'metric' if metricCondition else 'time'
                self.printFunc("Saved a model checkpoint based on the rule '{}' in {:.2f}s.".format(reason, timeWrite))

    def load_best_model(self, customObjects=None) -> keras.models.Model:
        if self.metric is None:
            raise RuntimeError("Metric name has to be provided to determine the best model.")
        if self.bestCheckpointPath is None:
            raise RuntimeError("No checkpoints were saved.")

        if self.printFunc:
            self.printFunc("Loading the best model checkpoint from epoch {} with metric {:.2f}"
                           .format(self.bestEpoch, self.bestMetricValue))

        return keras.models.load_model(self.bestCheckpointPath, custom_objects=customObjects)
