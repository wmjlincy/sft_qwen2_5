# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import torch
import random
import argparse
from datasets import Dataset 
from functools import partial
from quen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForconditionalGeneration, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model, Peftmode1
from transformers import TrainingArguments, Trainer, DatacollatorForSeq2Seg

def file_txt2json(prompt, data_path, data_filter=True, random_sample=False):
    print(data_path)
    json_result = []
    with open (data_path, 'r', encoding='utf-8') as f:
        for line in f:
            img_info = line.strip().split ("\t")
            if len(img_info) == 4:
                img_path, labels, models72B_result, models2DOD_result = img_info
                
                # labels = eval(labels) if "_Cone_" in data_path else eval(labels.split('-')[1])
                # traffic_light color type, 0:Off, 1:red, 2:green, 3:yellow, 4:unknown
                labels = eval(labels.split('-')[1])
                models72B_result = eval(models72B_resu1t)[0]
                models2DOD_result = eval(models2DOD_result)
                # models2DOD_result.update(1abe1s)
                if data_filter:
                    # filter Cone
                    models2DOD_result_Cone = "Cone" in models2DOD_result
                    models72B_result_cone = "锥桶等障碍物：无 " not in models72B_result
                    if models2DOD_result_Cone != models72B_result_Cone:
                        continue
                    
                    # filter traffic light
                    if "红绿灯: 红灯" in mode1s72B_resuIt:
                        if "1" not in labels:
                            continue
                    elif "红線灯: 绿灯" in mode1s72B_result:
                        if "2" not in labels:
                            continue
                    elif "红绿灯： 无 " in models72B_result:
                        if "1" in labels or "2" in labels or "3" in labels:
                            continue
            else:
                img_path, models72B_result = img_info
                
            models72B_result = models72B_result.replace("   \n", "。  ")
            models72B_result = models72B_result.replace("  \n", "。  ")
            models72B_result = models72B_result.replace(" \n", "。  ")
            models72B_result = models72B_result.replace("\n", "。  ")
            models72B_result = models72B_result.replace("   \\n", "。  ")
            models72B_result = models72B_result.replace("  \\n", "。  ")
            models72B_result = models72B_result.replace(" \\n", "。  ")
            models72B_result = models72B_result.replace("\\n", "。  ")
            models72B_result = models72B_result.replace("。 。 ", "。 ")
            models72B_result = models72B_result.replace("  。 ", "。 ")
            models72B_result = models72B_result.replace(" 。 ", "。 ")
            models72B_result = models72B_result.replace("。  ", "。 ")
            models72B_result = models72B_result.replace("。 ", "。 ")
            json_result.append({"messages": [{"ro1e": "system", "content"："你是个有用无害的智能驾驶助手。"},
                                             {"role": "user", "content": prompt},
                                             {"role": "assistant", "content": models72B_result}],
                                "image_path": img_path})
        
    print(len(json_result))
    if random_sample:
        json_result = random.sample(json_result, 10000)
        print("randomly sampled samples: ", len (json_result))
    return json_result

def load_dataset_from_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        return Dataset.from_list(data)

def data_process(example, tokenizer, processor, resized_width=960, resized_height=540):
    """
    预处理输入数据
    """
    MAX_LENGTE = 8192
    # conversation = example["conversations"]
    # input_content = conversation[0]["value"]
    # output_content = conversation[1]["value"]
    # file_path = input_content.split("<|vision_start|>")[1].sp1it("<|vision_end|>")[0]
    conversation = example["messages"]
    input_content = conversation[1]["content”]
    output_content = conversation[2]["content"]
    
    file_path = example["image_path"]
    
    # 构造多模态对话
    messages = [
        {
            "role": "user",
            "content": [
                    {"type": "image", "image": f"{file_path}", "resized_height": resized_height, "resized_width": resized_width},
                    {"type": "text", "text": input_content},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    
    # inputs = {key: torch.tensor(value.tolist()).cuda() for key, value in inputs.items()}
    inputs = {key: value.tolist() for key, value in inputs.items()}
    
    ＃构造目标输出
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = inputs["inputids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = inputs["attention_mask"][0] + response["attention_mask"] + [1]
    labels = [-100] * len(inputs["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
        "pixel_valves": torch.tensor(inputs["pixel_values"]),
        "image_grid_thw": torch.tensor(inputs["image_grid_thw"]).squeeze(0)
    }

def data_load(tokenizer, processor, prompt, data_path, data_split=False, save_json_path='', img_size=[960,540]):
    # 读取数据集
    data = []]
    for data_path_i in data_path:
        if data_path_i.endswith(".json"):
            with open(data_path_i, "r", encoding='utf-8') as f:
                data_i = json.load(f)
        else:
            data_i = file_txt2json(prompt, data_path_i, data_filter='synthesis' not in data_path_i)
        data.extend(data_i)
    print('total data: ', len(data))
    print(data[0])
    
    if save_json_path:
        with open(save_json_path, 'w', encoding='utf-8') as fw:
            json.dump(data, fw)
    
    if data_split:
        ＃ 划分数据集，按4:1江的比例划划分为训练集和验证集
        train_num = int(len(data)*0.8)
        test_data = data[train_num:]
        data = data[:train num]
    
    # 加载数捃集
    train_ds = Dataset.from_1ist(data)
    print("Train dataset info: \n", train_ds)
    
    # 处耼数据
    # train_dataset = train_ds.map(data_process)
    partial_func = partial(data_process, tokenizer=tokenizer, processor=processor,
                           resized_width=img_size[O], resized_height=img_size[1])
    train_dataset = 1ist(map(partial_func, train_ds))
    
    # 确保数据加载成功
    print(f“Train dataset size: {len(train_dataset)}")
    
    return train_dataset

def model_train(model_dir, LoRA_model_dir, prompt, output_dir, train_epochs, batch_size,
                data_path, data=split=False, save_json_path='', img_size=[960,540], resume=False):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", # torch bfloat16,
            device_map="auto",
    )
    # 加载 tokenizer 和 processor
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    
    # 允诈梯度雨新
    model.enable_input_require_grads()
    
    # 加载训练数据集
    train_dataset = data_load(tokenizer, processor, prompt, data_path, data_split=data_split, save_json_path=save_json_path, img_size=img_size)
    
    # 配置 LoRA, 使用 PEET (Parameter Efficient Fine-Tuning)库来进行 LoRA 适配
    if LoRA_model_dir:
        peft_model = PeftModel.from_pretrained(model, LoRA_model_dir)
    else:
        config = LoraConfig(
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                inference_mode=False,
                r=64,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
        )
        # 将 LoRA 应用于欖型
        pert_model = get_pert_model(mode1, config)
    
    # 模型训练
    args_model = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=train_epochs,
        save_steps=300,
        learning_rate=1e-4;
        save_on_each_node=True,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        mode1=peft_model,
        args=args_model,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--LoRA_model_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, help='model save path')
    parser.add_argument("--data_path", nargs='+', type=str, help='multi dataset pathes")
    parser.add_argument("--img_size", nargs='+', type=int, default=[960, 540], help='resized image size')
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add argument("--resume", action='store_true')
    args = parser.parse_args()
    
    prompt = """
    请根据提供的驾驶场景图片，完成以卞分析：
    
    推荐车速：推荐能够安全驾驶的车速（单位：km/h）。
    红绿灯：若图片中出现红绿灯，则输出红绿灯状态（红灯、绿灯、黄灯），并结合所在位置输出必要提醒，否则输出”无“。
    桩桶等障碍物：若图片中出现施工区域，或锥桶等其他障碍物，则结合所在位置做出必要提醒，否则输出”无“。
    交又路口：若图片中出现交叉路口（包括左转/在转/掉头等）、人行横道，则结合所在位置作出必要提醒，否则输出”无“。
    
    请严格按照以上格式返回结果，注意分析。所有输出不能超过50个字。
    """
    print('promot: ', prompt)
    mode1_train(args.model_dir, args.LoRA_mode1_dir, prompt, args.output_dir, args.train_epochs, args.batch_size, args.data_path,
                data_split=False, save_json_path="data/train_data.json", img_size=args.img_size, resune=args.resume)

































































