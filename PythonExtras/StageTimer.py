import itertools
import time
from collections import OrderedDict
from typing import Callable, Tuple, List, Dict, Any, Union, Type


class StageTimer:
    """
    Helps with timing of long-running iterative multi-stage processes.
    Overall structure:

        Process (total)
            Pass
                Stage 1
                Stage 2
    """

    def __init__(self):
        self._durStagesPass  = OrderedDict()  # type Dict[str, float]
        self._durStagesTotal = OrderedDict()  # type Dict[str, float]
        self._timeProcessStart = 0
        self._timePassStart = 0
        self._timeStageLast = 0
        self._currentStage = None  # type: str

        self._durPass = 0
        self._durTotal = 0

        self._isPassRunning = False

    def start_pass(self):
        if self._isPassRunning:
            self.end_pass()

        if self._timeProcessStart == 0:
            self._timeProcessStart = time.time()

        self._timePassStart = time.time()
        self._currentStage = None
        self._durStagesPass.clear()
        self._isPassRunning = True

    def start_stage(self, stageName: str):
        if not self._isPassRunning:
            self.start_pass()
        if self._currentStage == stageName:
            return
        if self._currentStage is not None:
            self.end_stage()

        self._timeStageLast = time.time()
        self._currentStage = stageName

    def end_stage(self) -> float:
        if self._currentStage is None:
            raise RuntimeError("No stage was started, but 'end_stage' has been called.")

        if self._currentStage not in self._durStagesPass:
            self._durStagesPass[self._currentStage] = 0

        timeCurrent = time.time()
        duration = timeCurrent - self._timeStageLast
        self._durStagesPass[self._currentStage] += duration
        self._currentStage = None

        return duration

    def end_pass(self):
        if self._currentStage is not None:
            self.end_stage()

        for name, duration in self._durStagesPass.items():
            if name not in self._durStagesTotal:
                self._durStagesTotal[name] = 0
            self._durStagesTotal[name] += duration

        self._durPass = time.time() - self._timePassStart
        self._isPassRunning = False

    def end(self):
        if self._isPassRunning:
            self.end_pass()

        self._durTotal = time.time() - self._timeProcessStart

    def get_pass_report(self):
        if self._isPassRunning:
            raise RuntimeError("Can't fetch pass report while a pass is still running.")

        durAccounted = sum(self._durStagesPass.values())
        timings = itertools.chain(self._durStagesPass.items(),
                                  [('rest', self._durPass - durAccounted), ('total', self._durPass)])

        return ', '.join(['{}: {:.3f} s'.format(name, time) for name, time in timings])

    def get_pass_duration(self):
        if self._isPassRunning:
            raise RuntimeError("Can't fetch pass duration while a pass is still running.")

        return self._durPass

    def get_total_report(self):
        if self._isPassRunning:
            raise RuntimeError("Can't fetch process report while a pass is still running.")

        durAccounted = sum(self._durStagesTotal.values())
        timings = itertools.chain(self._durStagesTotal.items(),
                                  [('rest', self._durTotal - durAccounted),
                                   ('total', self._durTotal)])

        return ', '.join(['{}: {:.3f} s'.format(name, time) for name, time in timings])

    def get_total_duration(self):
        return self._durTotal
