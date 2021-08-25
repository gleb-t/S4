from logging import LogRecord

import dateutil.relativedelta
import logging
import os
import sys
import time

from typing import Union, Optional, Callable


def format_duration(seconds: float) -> str:
    dur = dateutil.relativedelta.relativedelta(seconds=seconds)
    bits = []
    for attr in ['days', 'hours', 'minutes']:
        attrVal = int(getattr(dur, attr))
        if attrVal > 0:
            bits.append("{:d}{}".format(attrVal, attr[0]))
    bits.append("{:.2f}s".format(dur.seconds))  # Always display seconds.
    return " ".join(bits)


def format_large_number(number: int):
    magnitude = 0
    while number > 1000:
        number /= 1000
        magnitude += 1

    return '{:.1f}{}'.format(number, ['', 'K', 'M', 'G', 'T'][magnitude])


def get_null_logger_if_none(anotherLogger: Union[logging.Logger, None]) -> logging.Logger:
    if anotherLogger is not None:
        return anotherLogger

    return get_null_logger()


def get_null_logger() -> logging.Logger:
    """
    Get a logger that does nothing.
    Useful for writing library functions where logging is optional.

    :return:
    """
    logger = logging.getLogger('_null')
    if len(logger.handlers) == 0:
        logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)

    return logger


def configure_logger(loggerName,
                     outPath: str = None,
                     runString: str = None,
                     logToStd: bool = False,
                     logLevel: int = logging.DEBUG,
                     warningCollector: 'LogWarningCollector' = None,
                     throttle: bool = True):

    logger = logging.getLogger(loggerName)
    logger.handlers = []
    logger.filters = []
    logger.setLevel(logLevel)

    if runString:
        # todo This is a deprecated tempflow feature, use context instead.
        formatString = '[%(asctime)s - {runString} - %(levelname)s] %(message)s'.format(runString=runString)
    else:
        formatString = '[%(asctime)s - %(context)s - %(levelname)s] %(message)s'

    formatter = AppendDefaultContextFormatter(formatString)

    if logToStd:
        stdoutHandler = logging.StreamHandler(sys.stdout)
        stdoutHandler.setFormatter(formatter)
        logger.addHandler(stdoutHandler)

    if outPath:
        if os.path.isdir(outPath):
            outPath = os.path.join(outPath, 'log.log')
        fileHandler = logging.FileHandler(outPath)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    if warningCollector:
        handler = warningCollector.get_handler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if throttle:
        logger.addFilter(LogThrottlingFilter())

    return logger


def configure_child(parent: logging.Logger, name: str, level: int) -> logging.Logger:
    """
    A convenience method to setup a child logging in a single line.
    """

    child = parent.getChild(name)
    child.setLevel(level)

    return child


def destruct_logger(loggerName: str):
    """
    Close and remove all handlers, used to stop all file handlers and close the log file.
    """
    logger = logging.getLogger(loggerName)
    for handler in logger.handlers.copy():
        handler.close()
        logger.removeHandler(handler)


class AppendDefaultContextFormatter(logging.Formatter):

    def format(self, record: LogRecord) -> str:
        if not hasattr(record, 'context'):
            record.context = ''
        return super().format(record)


class LogThrottlingFilter(logging.Filter):
    """
    Suppresses frequent log messages marked with the same throttling id.
    Meant for reducing noise from progress-reporting functions.
    """

    def __init__(self):
        super().__init__()

        self.timestamps = {}
        self.suppressedCount = {}

    def filter(self, record: logging.LogRecord):
        throttlingId = None

        if hasattr(record, 'throttlingId'):
            throttlingId = record.throttlingId
        elif hasattr(record, 'throttle'):
            throttlingId = record.filename + str(record.lineno)

        if throttlingId is not None:
            throttlingPeriod = 15
            if hasattr(record, 'throttlingPeriod'):
                throttlingPeriod = record.throttlingPeriod

            currentTimestamp = time.time()
            if throttlingId in self.timestamps:
                lastTimestamp = self.timestamps[throttlingId]

                if currentTimestamp - lastTimestamp < throttlingPeriod:
                    self.suppressedCount[throttlingId] += 1
                    return False
                else:
                    record.msg += " [{} suppressed]".format(self.suppressedCount[throttlingId])

                    self.suppressedCount[throttlingId] = 0
                    self.timestamps[throttlingId] = currentTimestamp
                    return True
            else:
                self.timestamps[throttlingId] = currentTimestamp
                self.suppressedCount[throttlingId] = 0

                return True

        return True


class LogWarningCollector:

    def __init__(self):
        self.collectedMessages = []
        self.isEnabled = True

    def get_handler(self):
        return LogWarningCollector.LogHandler(self)

    def pass_collected_messages(self, logger: logging.Logger):

        if len(self.collectedMessages) == 0:
            return

        self.isEnabled = False

        logger.warning("===================================================")
        logger.warning("============== Repeating messages =================")

        for i, message in enumerate(self.collectedMessages):
            logger.warning(message)

        logger.warning("===================================================")

        self.isEnabled = True

    class LogHandler(logging.Handler):
        """
        A custom logging handler for retaining and replaying important log messages.
        """

        def __init__(self, collector: 'LogWarningCollector'):
            super().__init__()

            self.collector = collector

        def emit(self, record):
            message = self.format(record)

            if record.levelno >= logging.WARNING and self.collector.isEnabled:
                self.collector.collectedMessages.append(message)


def setup_uncaught_exception_report(logger: logging.Logger, extraCallback: Optional[Callable] = None):

    def exception_handler(excType, excValue, excTraceback):
        if issubclass(excType, KeyboardInterrupt):
            sys.__excepthook__(excType, excValue, excTraceback)
            return

        logger.critical("Uncaught exception", exc_info=(excType, excValue, excTraceback))

        if extraCallback:
            extraCallback(excTraceback, excValue)

    sys.excepthook = exception_handler
