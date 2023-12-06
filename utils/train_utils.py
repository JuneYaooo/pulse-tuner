import pandas as pd
import json
import datetime,time
import shutil
import os
import re
import subprocess
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)
from utils.data_utils import load_json, load_excel, detect_encoding
from config.common_config import *
from utils.gpu_utils import get_available_gpu
def on_train(model_name, select_lora, train_data_files,per_device_train_batch_size,num_train_epochs,learning_rate,lora_type):
    msg,msg1='',''
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    if not os.path.exists("workspace/data"):
        os.makedirs("workspace/data")
    log_file_path = f'workspace/data/logs/{now_str}.log'  # 定义log文件路径
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # 创建存储log的文件夹
    filelist = []
    for file in train_data_files:
        filename = os.path.basename(file.name)
        shutil.move(file.name, "workspace/data/" + filename)
        filedir = "workspace/data/" + filename
        filelist.append(filedir)
    train_data = []
    for file_path in filelist:
        if file_path.endswith('.json'):
            result = load_json(file_path)
            train_data+=result
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            result = load_excel(file_path)
            train_data+=result
        else:
            print(f'invalid file {os.path.basename(file_path)}')
            msg1 = f'invalid file {os.path.basename(file_path)}'
            result = None  # 未知文件类型，可以根据需要进行处理

    msg2 = train_model(model_name, select_lora, train_data, per_device_train_batch_size,num_train_epochs,learning_rate,lora_type)
    msg = msg1+msg2
    with open(log_file_path, 'w', encoding="utf-8") as f:
        f.write(msg)  # 将log内容写入文件
    return msg


def train_model(model_name, project_name, training_data, per_device_train_batch_size,num_train_epochs,learning_rate,lora_type):
    now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    
    current_directory = os.getcwd()
    model_file_name = llm_model_dict[model_name]['name']
    model_path = llm_model_dict[model_name]['model_path']
    template = llm_model_dict[model_name]['template']
    lora_target = llm_model_dict[model_name]['lora_target']

    lora_type_para = '--quantization_bit 4' if lora_type=='QLoRA' else ''


    folders_to_check = ["data", "checkpoints", "logs"]
    for folder in folders_to_check:
        folder_path = os.path.join(current_directory, "workspace","finetune", model_file_name,folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已创建文件夹：{folder_path}")
        else:
            print(f"文件夹已存在：{folder_path}")

    with open(f"{current_directory}/workspace/finetune/{ model_file_name}/data/{project_name}_dataset.json", "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=4,  ensure_ascii=False)
    
    available_gpus = get_available_gpu(threshold=10000) if model_file_name=='pulse' else get_available_gpu(threshold=20000) if model_file_name=='pulse_20b' else get_available_gpu(threshold=11000)
    print('available_gpus[0]',available_gpus[0])
    try:
        # 读取JSON文件
        with open(f'{current_directory}/workspace/finetune/{ model_file_name}/data/dataset_info.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {} 
    # 添加内容到JSON数据
    absolute_path = os.path.join(current_directory, f"workspace/finetune/{model_file_name}/data/{project_name}_dataset.json")
    print(' absolute_path', absolute_path)
    new_data = {
        project_name: {"file_name": absolute_path}
    }
    data.update(new_data)

    # 保存更新后的JSON文件
    with open(f'{current_directory}/workspace/finetune/{model_file_name}/data/dataset_info.json', 'w') as file:
        json.dump(data, file, indent=4)
    content = f'''CUDA_VISIBLE_DEVICES={available_gpus[0]} python {current_directory}/train_bash.py --stage sft   --model_name_or_path {model_path}   --do_train     --dataset {project_name}    --template {template}     --finetuning_type lora     --lora_target {lora_target}     --output_dir {current_directory}/workspace/finetune/{model_file_name}/checkpoints/{project_name}  --overwrite_output_dir  --overwrite_cache     --per_device_train_batch_size {per_device_train_batch_size}     --gradient_accumulation_steps 4     --lr_scheduler_type cosine     --logging_steps 10     --save_steps 1000     --learning_rate {learning_rate}     --num_train_epochs {num_train_epochs}     --plot_loss     --fp16  {lora_type_para}
    '''
    sh_file_name = f'workspace/finetune/{model_file_name}/train_{project_name}.sh'

    with open(sh_file_name , 'w') as file:
        file.write(content)

    # 设置文件可执行权限
    os.chmod(sh_file_name , 0o755)

    subprocess.Popen(f"""cd {current_directory}/workspace/finetune/{model_file_name} && . {conda_env_file} && conda activate pulse_tuner && nohup sh train_{project_name}.sh > ./logs/train_{now_str}.log 2>&1 &""", shell=True) # 换conda环境
    print('finish')
    
    # save parameters
    if not os.path.exists(f'{current_directory}/workspace/finetune/{model_file_name}/checkpoints/{project_name}'):
            os.makedirs(f'{current_directory}/workspace/finetune/{model_file_name}/checkpoints/{project_name}')
    params = {'lora_type':lora_type,'per_device_train_batch_size':per_device_train_batch_size,'num_train_epochs':num_train_epochs,'learning_rate':learning_rate}
    with open(f'{current_directory}/workspace/finetune/{model_file_name}/checkpoints/{project_name}/params.json', 'w') as file:
        json.dump(params, file, indent=4)

    return f'{model_name} on training'