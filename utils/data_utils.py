import pandas as pd
import json
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        detector = chardet.universaldetector.UniversalDetector()

        for line in file:
            detector.feed(line)
            if detector.done:
                break

        detector.close()
        encoding = detector.result['encoding']
        confidence = detector.result['confidence']

        return encoding, confidence


def load_json(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if type(data) == list:
                new_data = []
                for item in data:
                    if type(item) == dict:
                        new_dict = {}
                        new_dict['instruction'] = item['instruction'] if 'instruction' in item else ''
                        new_dict['input'] = item['input'] if 'input' in item else ''
                        new_dict['output'] = item['output'] if 'output' in item else ''
                        new_dict['history'] = item['history'] if 'history' in item else []
                        new_data.append(new_dict)
                    else:
                        return []
                return new_data
            else:
                return []
    except UnicodeDecodeError:
        # If utf-8 decoding fails, try with other common encodings
        try_encodings = ['utf-16', 'latin-1']
        for encoding in try_encodings:
            try:
                with open(json_file_path, 'r', encoding=encoding) as file:
                    data = json.load(file)
                return data, f'Successfully loaded using encoding: {encoding}'
            except UnicodeDecodeError:
                print('Failed to load JSON file with all attempted encodings')
        if type(data) == list:
            new_data = []
            for item in data:
                if type(item) == dict:
                    new_dict = {}
                    new_dict['instruction'] = item['instruction'] if 'instruction' in item else ''
                    new_dict['input'] = item['input'] if 'input' in item else ''
                    new_dict['output'] = item['output'] if 'output' in item else ''
                    new_dict['history'] = item['history'] if 'history' in item else []
                    new_data.append(new_dict)
                else:
                    return []
            return new_data
        else:
            return []


def load_excel(file_path):
     # 读取 Excel 文件
    df = pd.read_excel(file_path)
    if 'input' not in df.columns:
        return [], 'no input or output field'
    log = []
    log.append(f'开始处理数据')
    
    all_data = []
    # 遍历每一行数据
    for index, row in df.iterrows():
        instruction = row['instruction'] if 'instruction' in row else ''
        question = row['input'] if 'input' in row else ''
        answer = row['output'] if 'output' in row else ''
        history = row['history'].split('|') if 'history' in row else []
        
        # 创建字典并将数据添加到列表中
        data = {"instruction": instruction, "input": question, "output": answer, "history": history}
        all_data.append(data)

    log = '\n'.join(log)  # 使用换行符拼接log内容
    return all_data