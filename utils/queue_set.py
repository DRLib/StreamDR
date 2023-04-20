from multiprocessing import Queue, Value
from threading import Semaphore


class StreamDataQueueSet:
    def __init__(self):
        self.data_queue = Queue()
        self.start_flag_queue = Queue()
        self.stop_flag_queue = Queue()

    def clear(self):
        self.data_queue.close()
        self.start_flag_queue.close()
        self.stop_flag_queue.close()


class ModelUpdateQueueSet:

    STOP = 0
    SAVE = 1
    DATA_STREAM_END = 2
    UPDATE = 3

    def __init__(self):
        self.training_data_queue = Queue()
        self.raw_data_queue = Queue()
        self.embedding_queue = Queue()
        self.flag_queue = Queue()
        self.WAITING_UPDATED_DATA = Value("b", False)
        self.INITIALIZING = Value("b", False)
        self.MODEL_UPDATING = Value("b", False)

    def clear(self):
        self.training_data_queue.close()
        self.embedding_queue.close()
        self.flag_queue.close()


class DataProcessorQueue:

    def __init__(self):
        self._embedding_data_queue = Queue()
        self._res_data_queue = Queue()
        self._PROCESSING = Value("b", False)
        self._STOP = Value("b", False)

    def get(self):
        return self._embedding_data_queue.get()

    def is_empty(self):
        return self._embedding_data_queue.empty()

    def put(self, data):
        self._embedding_data_queue.put(data)

    def clear(self):
        self._embedding_data_queue.close()

    def processing(self):
        self._PROCESSING.value = 1

    def processed(self):
        self._PROCESSING.value = 0

    def is_processing(self):
        return self._PROCESSING.value == 1

    def get_res(self):
        return self._res_data_queue.get()

    def put_res(self, res):
        self._res_data_queue.put(res)

    def is_stop(self):
        return self._STOP.value == 1

    def stop(self):
        self._STOP.value = 1

