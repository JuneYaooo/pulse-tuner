import pandas as pd
import json
import datetime,time
import shutil
import os
import re
import subprocess
from sklearn.metrics import classification_report
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)

def read_excel_file(file_path):
    df = pd.read_excel(file_path)
    return df

def save_to_excel(df, file_path):
    df.to_excel(file_path, index=False)

def get_available_gpu(threshold=20000):
    # Initialize NVML
    nvmlInit()

    # Get the number of GPU devices
    device_count = nvmlDeviceGetCount()

    # Find GPU devices with available memory greater than the threshold
    available_gpus = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        free_memory_mb = info.free / 1024 / 1024

        if free_memory_mb > threshold:
            available_gpus.append(i)

    # Shutdown NVML
    nvmlShutdown()

    return available_gpus


def is_numeric(value):
    try:
        float(value)  # 尝试将值转换为浮点数
        return True  # 如果转换成功，则表示值可以转换为数字
    except (ValueError, TypeError):
        return False  # 如果转换失败或者值的类型不是字符串或数字，则表示值不是数字

