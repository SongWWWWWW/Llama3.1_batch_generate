import random
import requests
import pyarrow.parquet as pq
import os
import json
from typing import List
import time

def batch_format_prompt(demonstrations_list:List[List[int]],question:str) -> List[str]: # 注：这里的question是str，代表1000个train对一个vail
    prompt = []
    for i in demonstrations_list:
        prompt.append(format_prompt(i,question))
    return prompt

def format_prompt(demonstrations_list:List[int],question:str) -> str:
    global train_set
    global train_answer
    global n
    p_prompt = f"""
    You are an expert in sentiment analysis. You will be given {𝑛}
    examples to illustrate how to classify sentiment analysis, 
    and then you will be given a question to answer. 
    Please respond according to the format of the example provided below.
    """  # preprompt
    demonstrations = """
    Question {index}: {question}
    answer:
    {{
        "category": "{category}"
    }}
    """  # demonstrations_prompt
    d_prompt = ""
    for i in range(n):
        d_prompt += demonstrations.format(index=i+1,question=train_set[demonstrations_list[i]],category="positive" if train_answer[demonstrations_list[i]] == 1 else "negative")
    
    q_prompt = f"""
    Your question as follow. 
    Question: {question}, please answer this question.
    """  # question_prompt
    prompt = p_prompt + d_prompt + q_prompt
    return prompt

def vail_answer(answer:str,ground_truth:int) -> str:
    try:
        answer = json.loads(answer)
        if answer["category"] == "positive" and ground_truth == 1:
            return "True"
        elif answer["category"] == "negative" and ground_truth == 0:
            return "True"
        return "False"
    except Exception as e:
        return "error"
    

def batch_vail_answer(answers:List[str],ground_truth:int) -> List[str]: 
    vail = []
    for i in answers:
        vail.append(vail_answer(i,ground_truth))
    return vail


def sample(m,n,train_set):
    # 裁减 m*len(train_set)//n
    if m*len(train_set)%n != 0 :
        train_set = train_set[:len(train_set)-(m*len(train_set))%n]
    length = len(train_set)
    num_list = [m]*length # 计数
    count_list = [i for i in range(length)] # 采样用
    epoch = m*len(train_set)//n

    result = []
    
    for _ in range(epoch):
        ls = []
        for _ in range(n):

            random_num = random.randint(0,len(count_list)-1) # 随机采样，采样下标

            # print(random_num,len(count_list))

            while count_list[random_num] in ls:  # 取消重复
                random_num = random.randint(0,len(count_list)-1)
                # print(random_num,len(count_list))
            ls.append(count_list[random_num]) # 添加内容
            num_list[random_num] -= 1
            # print(num_list)
            if num_list[random_num] <= 0 : # 次数够就删掉
                num_list.pop(random_num)
                count_list.pop(random_num)
        result.append(ls)
        # print(ls)
    return result

def get_completion(prompts):
    headers = {'Content-Type': 'application/json'}
    data = {"prompts": prompts}
    response = requests.post(url='http://127.0.0.1:6006', headers=headers, data=json.dumps(data))
    return response.json()['response']

if __name__ == "__main__":
    random.seed(32)
    m = 5 
    n = 5
    batch = 64
    data_path = "./sst-2/"
    data_list = [
        "test-00000-of-00001.parquet",
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet"
    ]
    train_table = pq.read_table(os.path.join(data_path,data_list[1]))
    train_df = train_table.to_pandas()
    train_set = train_df["sentence"].tolist()
    # print(type(train_set))
    train_answer = train_df["label"].tolist()
    # print(type(train_answer))

    vail_table = pq.read_table(os.path.join(data_path,data_list[2]))
    vail_df = vail_table.to_pandas()
    vail_set = vail_df["sentence"].tolist()
    vail_answers = vail_df["label"].tolist()
    json_log_process = []
    answer = []
    sample_data = sample(m,n,train_set)
    for i in range(0,len(train_set),batch):
        # print(i,batch)
        epoch_train_set = sample_data[i:i+batch]
        
        for j in range(len(vail_set)):
            start = time.time()
            batch_prompt = batch_format_prompt(epoch_train_set,vail_set[i])
            batch_answer = get_completion(batch_prompt)
            batch_vail = batch_vail_answer(batch_answer,vail_answers[i])
            json_log_process.append(
                {
                    "demonstrations": epoch_train_set, # some index of demonstrations
                    "questions": vail_set[i] , # single question
                    "answers":  batch_vail # some true or false or error
                }
            )
            end = time.time()
            print(f"[{i}/{len(train_set)//batch}]-[{j}/{len(vail_set)}] spent-time: {end-start}s")
    with open("log_result.json","w") as f:
        json.dump(json_log_process,f,indent=4)






    # print(format_prompt(sample_data[1],question="keeps the film grounded in an undeniable social realism "))
    # print(get_completion([format_prompt(sample_data[1],question="keeps the film grounded in an undeniable social realism ")]))

    # pass