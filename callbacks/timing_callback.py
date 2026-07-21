# đo lường và thu thập dữ liệu về thời gian chạy (benchmarking) của mô hình mạng nơ-ron trong quá trình huấn luyện (training)
# Tính toán Tổng thời gian huấn luyện, Độ trễ (Latency) của từng batch, và Thông lượng (Throughput) (tốc độ xử lý bao nhiêu dữ liệu/giây).
import sys
from timeit import default_timer as timer
import tensorflow as tf


class TimingCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to note training time, latency and throughput
    """

    def __init__(self, ):
        super(TimingCallback, self).__init__()
        self.train_start_time = None
        self.train_end_time = None
        self.batch_time = []
        self.batch_start_time = None

    def on_train_begin(self, logs: dict):
        tf.print("Training starting time noted.", output_stream=sys.stdout)
        self.train_start_time = timer()

    def on_train_end(self, logs: dict):
        tf.print("Training ending time noted.", output_stream=sys.stdout)
        self.train_end_time = timer()

    def on_train_batch_begin(self, batch: int, logs: dict):
        self.batch_start_time = timer()

    def on_train_batch_end(self, batch: int, logs: dict):
        self.batch_time.append(timer() - self.batch_start_time)
