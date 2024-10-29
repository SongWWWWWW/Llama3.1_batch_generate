from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch
import gc
# 设置设备参数
DEVICE = "cuda"  # 使用CUDA
DEVICE_ID = "0"  # CUDA设备ID，如果未设置则为空
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # 组合CUDA设备信息

# # 假设有4个GPU
num_gpus = torch.cuda.device_count()

# 遍历所有GPU
# for i in range(num_gpus):
#     # 设置每个GPU的显存使用上限比例，例如限制为50%
#     torch.cuda.set_per_process_memory_fraction(0.5, i)


# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():  # 检查是否可用CUDA
        with torch.cuda.device(CUDA_DEVICE):  # 指定CUDA设备
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片

# 创建FastAPI应用
app = FastAPI()

# 处理POST请求的端点
@app.post("/")
async def create_item(request: Request):
    global model, tokenizer  # 声明全局变量以便在函数内部使用模型和分词器
    json_post_raw = await request.json()  # 获取POST请求的JSON数据
    json_post = json.dumps(json_post_raw)  # 将JSON数据转换为字符串
    json_post_list = json.loads(json_post)  # 将字符串转换为Python对象
    prompts = json_post_list.get('prompts')  # 获取请求中的提示

    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "prompt"}
    ]
    messages = []
    for prompt in prompts:
        messages.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
    ])
    # 调用模型进行对话生成
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True) #新版本弃用
    # print("input_ids: ", input_ids)
    model_inputs = tokenizer(input_ids,  padding=True, truncation=True, return_tensors="pt").to('cuda')
    # print("model_inputs", model_inputs)
    # try:
    generated_ids = model.module.generate(model_inputs.input_ids,
                                        attention_mask=model_inputs['attention_mask'].to('cuda'), 
                                        max_length=512)
    # print("generated_ids: ",generated_ids)
    # except Exception as e:
        # print("error", e)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # generated_ids = generated_ids[0][len(input_ids["input_ids"][0]):]
    # response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)



    del model_inputs
    del generated_ids
    del input_ids
    # 执行显存清理和垃圾回收
    torch.cuda.empty_cache()  # 释放显存
    gc.collect()  # 强制进行垃圾回收
    now = datetime.datetime.now()  # 获取当前时间
    time = now.strftime("%Y-%m-%d %H:%M:%S")  # 格式化时间为字符串
    # 构建响应JSON
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    # 构建日志信息
    log = "[" + time + "] " + '", prompts:"' + str(prompts) + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

# 主函数入口
if __name__ == '__main__':
    # 加载预训练的分词器和模型
    model_name_or_path = '/root/autodl-tmp/LLM-Research/Meta-Llama-3___1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.float32)
    model.eval()
    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #     tokenizer.pad_token_id = 0
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # 如果有多个GPU可用，使用DataParallel来包装模型
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用