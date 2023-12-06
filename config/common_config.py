
llm_model_dict = {
"PULSE": {"name": "pulse_7b",
        "model_path": "your-path/PULSE",
        "template":"pulse",
        "lora_target":"query_key_value"
    },
"PULSE-20B": {"name": "pulse_20b",
        "model_path": "your-path/PULSE-20b",
        "template":"pulse",
        "lora_target":"q_proj,v_proj"
    },
}

# 找到 profile.d/conda.sh 文件的绝对路径，填进来
conda_env_file = '/home/pai/etc/profile.d/conda.sh'

# 默认参数
lora_type = 'QLoRA'
per_device_train_batch_size = 4
num_train_epochs = 3
learning_rate = 5e-5
max_length=1500
temperature=0.7
top_p = 0.8