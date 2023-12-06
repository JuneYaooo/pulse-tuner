import gradio as gr
import pandas as pd
import json
import datetime,time
import shutil
import os
import re
import subprocess
from pynvml import (nvmlInit, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex,
                    nvmlDeviceGetName, nvmlDeviceGetMemoryInfo, nvmlShutdown)
from config.common_config import *
from utils.gpu_utils import get_available_gpu
from utils.train_utils import on_train
from utils.data_utils import load_json, load_excel, detect_encoding

llm_model_dict_list = list(llm_model_dict.keys())
model_loaded = False
project_change = False
model_change = False
last_project_name = ''
last_model_name = ''


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = round(seconds % 60, 2)
    if hours > 0:
        return f"{hours} hours {minutes} minutes {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} minutes {seconds} seconds"
    else:
        return f"{seconds} seconds"


def get_file_modify_time(filename):
    try:
        return datetime.datetime.fromtimestamp(os.stat(filename).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print('Failed to get modification time for {}'.format(filename))
        print(e)
        return 'not available'


def get_model_update_time(model_name, lora_name):
    model_file_name = llm_model_dict[model_name]['name']
    print('get_model_update_time model_file_name', model_file_name)
    print('get_model_update_time lora_name', lora_name)
    model_lora_dir = os.path.join(f"workspace/finetune", model_file_name, 'checkpoints', lora_name, 'adapter_model.bin')
    print('model_lora_dir', model_lora_dir)
    update_time = get_file_modify_time(model_lora_dir)
    return update_time


def load_model(model_name,project_name,lora_type,temperature,top_p):
    global model_loaded, model, tokenizer, project_change, last_project_name, last_model_name, model_change
    current_directory = os.getcwd()
    model_file_name = llm_model_dict[model_name]['name']
    model_path = llm_model_dict[model_name]['model_path']
    template = llm_model_dict[model_name]['template']
    lora_target = llm_model_dict[model_name]['lora_target']
    quantization_bit = 4 if lora_type=='QLoRA' else None
    if project_name != last_project_name:
        project_change = True
    if model_name != last_model_name:
        model_change = True
    if not model_loaded or project_change or model_change:
        if model_loaded:
            model.model = model.model.to('cpu')
            del model
            import torch
            torch.cuda.empty_cache()
            model_loaded = False
        available_gpus = get_available_gpu(threshold=11000)
        if len(available_gpus)>0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(available_gpus[0])
            print('available_gpus[0]',available_gpus[0])
        else:
            return 'no enough GPU, please try it later!',''
        try:
            from llmtuner.chat.stream_chat import ChatModel
            args = {
                "model_name_or_path": model_path,
                "template": template,
                "finetuning_type": "lora",
                "lora_target": lora_target,
                "checkpoint_dir": f"{current_directory}/workspace/finetune/{model_file_name}/checkpoints/{project_name}",
                "max_length":max_length,
                "temperature":temperature,
                "top_p":top_p,
                "quantization_bit":quantization_bit
            }
            model = ChatModel(args)
            model_loaded = True
            last_project_name = project_name
            project_change = False
            last_model_name = model_name
            model_change = False
            # return model, tokenizer
        except Exception as e:
            print('error!! ', e)
            return e,''
    return model



def on_test(model_name, select_lora, lora_type, temperature,top_p,test_data_file):
    start_time = time.time()
    if not os.path.exists("workspace/data"):
        os.makedirs("workspace/data")
    if not os.path.exists("workspace/output"):
        os.makedirs("workspace/output")
    result_paths = []
    model = load_model(model_name,select_lora,lora_type,temperature,top_p)
    for file in test_data_file:
        filename = os.path.basename(file.name)
        shutil.move(file.name, "workspace/data/" + filename)
        file_path = "workspace/data/" + filename
        # filelist.append(filedir)
        if file_path.endswith('.json'):
            result = load_json(file_path)
            new_result = []
            for item in result:
                item['output'] , (prompt_length, response_length)=  model.chat(item['input'],item['history'],item['instruction']) 
                new_result.append(item)
            result_path = f'workspace/output/output_{filename}'
            # 检测文件编码
            detected_encoding, confidence = detect_encoding(file_path)
            print(f"Detected encoding: {detected_encoding} with confidence: {confidence}")
            with open(result_path , 'w', encoding=detected_encoding) as json_file:
                json.dump(new_result, json_file, ensure_ascii=False, indent=2)
            result_paths.append(result_path)
        elif file_path.endswith('.xls') or file_path.endswith('.xlsx'):
            result = load_excel(file_path)
            new_result = []
            for item in result:
                item['output'] , (prompt_length, response_length)=  model.chat(item['input'],item['history'],item['instruction']) 
                new_result.append(item)
            result_path = f'workspace/output/output_{filename}'
            # 检测文件编码
            detected_encoding, confidence = detect_encoding(file_path)
            print(f"Detected encoding: {detected_encoding} with confidence: {confidence}")
            # 将 DataFrame 写入 Excel 文件
            pd.DataFrame(new_result).to_excel(result_path, index=False, encoding=detected_encoding)
            result_paths.append(result_path)
            
        else:
            print(f'invalid file {os.path.basename(file_path)}')
            return None, f'invalid file {os.path.basename(file_path)}'
    end_time = time.time()
    cost_time = end_time - start_time

    info = 'Time taken: ' + format_duration(cost_time) + f" ({round(cost_time, 2)} seconds)"
    return result_paths, info


def get_chat_answer(query,model_name,select_lora,lora_type,temperature,top_p,instruction,chatbot):
    model = load_model(model_name,select_lora,lora_type,temperature,top_p)
    res , (prompt_length, response_length)=  model.chat(query,chatbot,instruction) 
    chatbot.append([query,res])
    print('chatbot',chatbot)
    return chatbot, ""

    
def on_stop(model_name, select_lora):
    process = subprocess.Popen('ps -ef | grep train_bash.py', shell=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    process.kill()    
    n = 0
    # 解析输出以获取进程ID
    print('output',output)
    try:
        lines = output.decode().split('\n')
        for line in lines:
            if 'train_bash.py' in line:
                parts = line.split()
                pid = parts[1]
                # 杀死进程
                subprocess.call(['kill', '-9', pid])
                n+=1
    except Exception as e:
        print('error!!',e)
    return f'停止了{n//2}个进程'

def get_lora_para(model_name, lora_name_en):
    current_directory = os.getcwd()
    model_file_name = llm_model_dict[model_name]['name']
    try:
        with open(f'{current_directory}/workspace/finetune/{model_file_name}/checkpoints/{lora_name_en}/params.json', 'r') as file:
            para_data = json.load(file)
    except Exception as e:
        print('error',e)
        para_data = {}
    return para_data

def change_lora_name_input(model_name, lora_name_en):
    if lora_name_en == "New":
        return gr.update(visible=True), gr.update(visible=True), 'not available',lora_type,per_device_train_batch_size,num_train_epochs,learning_rate
    else:
        file_status = f"Loaded {lora_name_en}"
        model_update_time = get_model_update_time(model_name, lora_name_en)
        para_data = get_lora_para(model_name, lora_name_en)
        return gr.update(visible=False), gr.update(visible=False), model_update_time, para_data['lora_type'] if 'lora_type' in para_data else lora_type, para_data['per_device_train_batch_size'] if 'per_device_train_batch_size' in para_data else per_device_train_batch_size,para_data['num_train_epochs'] if 'num_train_epochs' in para_data else num_train_epochs,para_data['learning_rate'] if 'learning_rate' in para_data else learning_rate

def add_lora(lora_name_en, lora_list):
    if lora_name_en in lora_list:
        print('Name conflict, not creating new')
        return gr.update(visible=True, value=lora_name_en), gr.update(visible=False), gr.update(visible=False), lora_list
    else:
        return gr.update(visible=True, choices=[lora_name_en] + lora_list, value=lora_name_en), gr.update(visible=False), gr.update(visible=False), [lora_name_en] + lora_list

def find_folders(directory):
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders

def get_lora_init_list(model_name):
    model_file_name = llm_model_dict[model_name]['name']
    model_dir = os.path.join(f"workspace/finetune", model_file_name, 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    lora_list = find_folders(model_dir)
    return lora_list

def get_lora_list(model_name):
    model_file_name = llm_model_dict[model_name]['name']
    model_dir = os.path.join(f"workspace/finetune", model_file_name, 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    lora_list = find_folders(model_dir)
    return gr.update(visible=True, choices=lora_list + ['New'], value=lora_list[0] if len(lora_list) > 0 else 'New'), lora_list + ['New']

lora_init_list = get_lora_init_list(llm_model_dict_list[0])


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# PULSE Tuner (Easy to use)
"""

model_status = 'Please manually load the model'

default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)



with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    set_lora_list = gr.State(lora_init_list + ['New'])

    gr.Markdown(webui_title)
    with gr.Row():
        with gr.Column():
            model_name = gr.Radio(llm_model_dict_list, 
                                  label="Select Model",
                                  value=llm_model_dict_list[0] if len(llm_model_dict_list) > 0 else 'No model available',
                                  interactive=True)
        with gr.Column():
            lora_type = gr.Dropdown(['QLoRA', 'LoRA'],
                                            label="lora_type",
                                            value= 'QLoRA',
                                            interactive=True)
        with gr.Column():
            select_lora = gr.Dropdown(set_lora_list.value,
                                      label="Choose or build a Lora",
                                      value=set_lora_list.value[0] if len(set_lora_list.value) > 0 else 'New', 
                                      interactive=True,
                                      visible=True)
            lora_name_en = gr.Textbox(label="Please enter the English name for Lora, no spaces, use lowercase letters, words can be separated by underscores",
                                      lines=1,
                                      interactive=True,
                                      visible=False)
            lora_add = gr.Button(value="Confirm add Lora", visible=False)
    with gr.Row():
        lastest_model = gr.Textbox(type="text", label='Model update time (Please switch models or projects to refresh)')

    with gr.Tab("Chat Test"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([],
                                     elem_id="chat-box",
                                     show_label=False)#.style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="Please enter your question and press enter to submit")#.style(container=False)
            with gr.Column(scale=5):
                gr.Markdown("## Parameters")
                chat_temperature = gr.Slider(0, 1,
                                        value=temperature,
                                        step=0.01,
                                        label="temperature",
                                        interactive=True)
                chat_top_p = gr.Slider(0, 1,
                                        value=top_p,
                                        step=0.01,
                                        label="top_p",
                                        interactive=True)
                instruction = gr.Textbox(show_label=False,
                                   placeholder="System Instruction (Optional)")

    with gr.Tab("Fine-tuning"):
        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown("## Training")
                with gr.Row():
                    with gr.Column(scale=8):
                        train_data_file = gr.File(label="Upload Training Data Files", file_types=['.xlsx, json'], file_count="multiple")
                    with gr.Column(scale=2):
                        gr.Markdown("""<font size="1" color="gray">You can use the example files or upload your own files, multiple files are supported, like .json,.xlsx \n\n\n</font>""")
                        gr.Examples(
                        [
                            os.path.join(os.path.dirname(__file__), "example/example_train.xlsx")

                        ],
                        inputs=[train_data_file]
                    )
                train_button = gr.Button("Start Training", label="Train")
                kill_train_button = gr.Button("Stop Training Processes", label="Train")
                train_res = gr.Textbox(type="text", label='')
            with gr.Column(scale=5):
                gr.Markdown("## Parameters")
                per_device_train_batch_size = gr.Slider(1, 32,
                                        value=per_device_train_batch_size,
                                        step=1,
                                        label="per_device_train_batch_size",
                                        interactive=True)
                num_train_epochs = gr.Slider(1, 15,
                                        value=num_train_epochs,
                                        step=1,
                                        label="num_train_epochs",
                                        interactive=True)
                learning_rate = gr.Number(value=learning_rate,label="learning_rate",
                                                    interactive=True)

    with gr.Tab("Batch Test"):
        with gr.Row():
            with gr.Column(scale=10):
                gr.Markdown("## Prediction")
                with gr.Row():
                    with gr.Column(scale=8):
                        test_data_file = gr.File(label="Upload Test Data File", file_types=['.xlsx, json'], file_count="multiple")
                    with gr.Column(scale=2):
                        gr.Markdown("""<font size="1" color="gray">You can use the example files or upload your own files, multiple files are supported, like .json,.xlsx \n\n\n</font>""")
                        gr.Examples(
                            [
                                os.path.join(os.path.dirname(__file__), "example/example_test.json"),
                                os.path.join(os.path.dirname(__file__), "example/example_test.xlsx")

                            ],
                            inputs=[test_data_file]
                        )
                test_button = gr.Button(label="Evaluate")
                test_res = gr.Textbox(type="text", label='')
                download_test = gr.File(label="Download Result File", file_count="multiple")
            with gr.Column(scale=5):
                gr.Markdown("## Parameters")
                test_temperature = gr.Slider(0, 1,
                                        value=temperature,
                                        step=0.01,
                                        label="temperature",
                                        interactive=True)
                test_top_p = gr.Slider(0, 1,
                                        value=top_p,
                                        step=0.01,
                                        label="top_p",
                                        interactive=True)

    select_lora.change(fn=change_lora_name_input,
                       inputs=[model_name, select_lora],
                       outputs=[lora_name_en, lora_add, lastest_model, lora_type,per_device_train_batch_size,num_train_epochs,learning_rate])
    lora_add.click(fn=add_lora,
                   inputs=[lora_name_en, set_lora_list],
                   outputs=[select_lora, lora_name_en, lora_add, set_lora_list])
    model_name.change(fn=get_lora_list, inputs=[model_name], outputs=[select_lora, set_lora_list])
    train_button.click(on_train, inputs=[model_name, select_lora, train_data_file,per_device_train_batch_size,num_train_epochs,learning_rate,lora_type], outputs=[train_res]) 
    kill_train_button.click(on_stop, inputs=[model_name, select_lora], outputs=[train_res]) 
    test_button.click(on_test, show_progress=True, inputs=[model_name, select_lora, lora_type, test_temperature,test_top_p,test_data_file], outputs=[download_test, test_res]) 
    query.submit(get_chat_answer,
                 [query,model_name,select_lora,lora_type,chat_temperature,chat_top_p,instruction,chatbot],
                 [chatbot, query])

(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=6688,
        #  show_api=False,
        #  share=True,
        #  debug= True,
         inbrowser=True))