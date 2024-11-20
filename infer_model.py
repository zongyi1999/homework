import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift import Swift
import torch
from swift.utils import seed_everything
import torch
model_type = ModelType.qwen2_vl_7b_instruct
template_type = get_default_template_type(model_type)

print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})
model = Swift.from_pretrained(model, '/media/sdd/lzy/mllm/ms-swift/output/qwen2-vl-7b-instruct/v7-20241120-163331/checkpoint-17')
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
total_images = 0
matched_images = 0

total_images_1 = 0
matched_images_1 = 0

dict_map = {"10":"石头",
                "2":"剪刀",
                "5":"布"}
for img_path in os.listdir("/media/sdd/lzy/mllm/picture"):
        query = f"""<img>/media/sdd/lzy/mllm/picture/{img_path}</img>请问图片中表示的是0-10中的哪一个手势？"""
        response, history = inference(model, template, query)
        print(f'query: {query}')
        print(f'response: {response}')
        total_images += 1
        number_part = img_path.split('_')[0]
        if number_part == response:
            matched_images += 1

        if number_part in ["10","2","5"]:
            query = f"""<img>/media/sdd/lzy/mllm/picture/{img_path}</img>图请问图片中的手势表示的是“石头，剪刀，布”中的哪一种？"""
            response, history = inference(model, template, query)
            print(f'query: {query}')
            print(f'response: {response}')
            total_images_1 += 1
            number_part = img_path.split('_')[0]
            print(number_part, dict_map[number_part])
            if dict_map[number_part] == response:
                matched_images_1 += 1
if total_images > 0:
    accuracy = matched_images / total_images
else:
    accuracy = 0

if total_images > 0:
    accuracy_1 = total_images_1 / total_images_1
else:
    accuracy_1 = 0
print(f"数字总图片数量: {total_images}")
print(f"数字匹配的图片数量: {matched_images}")
print(f"数字准确率: {accuracy:.2%}")


print(f"划拳总图片数量: {total_images_1}")
print(f"划拳匹配的图片数量: {total_images_1}")
print(f"划拳准确率: {accuracy_1:.2%}")