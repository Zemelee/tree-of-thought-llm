import itertools
import pdb
import numpy as np
from functools import partial
from tot.models import gpt
from tot.tasks.game24 import Game24Task
from tot.tasks.text import TextTask


# 评估单个候选路径的价值
def get_value(task:Game24Task, x, y, n_evaluate_sample, cache_value=True):
    # x:4个源数字 y:某一步的具体操作
    # 构造提示词: 评估当前操作的sure/likely/impossible(当前价值) or 评估答案正误(检测到答案)
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    # 重复生成3次评估结果
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    # 根据评估关键词生成价值(计数)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    # task 原始输入x 生成的所有候选ys
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            # 根据评估关键单词生成价值
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task:TextTask, x, ys, n_evaluate_sample):
    # 构造提示词: 选择最有希望的项 id+ys
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample)
    # 返回每个思路的被选择次数列表
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

# 根据当前状态生成下一步的可能操作
def get_proposals(task:Game24Task, x, y): # 第一次y是空字符串
    # 构造提示词: "下一步可能的操作有哪些?" or 5shot+x+ys总结
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt)[0].split('\n') # ["#1","#2"...]
    # 拼接之前的推理过程
    return [y + _ + '\n' for _ in proposals] # [steps1+#1,steps1+#2...]

def get_samples(task:Game24Task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        # 构造提示词: 一次性生成最终24的答案
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        # 构造提示词: 提供完整的阶梯过程
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def solve(args, task:Game24Task, idx, to_print=True):
    global gpt
    gpt = partial(gpt)
    print(gpt)
    x = task.get_input(idx)  # 读取第idx个问题
    # 1 1 4 6
    ys = ['1 + 1 = 2 (left: 2 10 14)','4 - 1 = 3 (left: 1 3 14)',]  # 当前所有输出候选,第二轮有5个
    infos = []
    for step in range(task.steps):# step=4
        print(f'\nidx:{idx} --- Step:{step}')

        # 生成：从当前状态生成多个可能的下一步操作
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            # 根据当前的所有状态，依次生成下一步可能的操作（一个step）
            new_ys = [get_proposals(task, x, y) for y in ys] # 二维数组
        new_ys = list(itertools.chain(*new_ys)) # 展平:二维-->一维
        ids = list(range(len(new_ys)))

        # 评估:对每个候选操作路径进行价值评估
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            # 任务对象 原始数字 生成的所有候选 评估的样本数量3
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # 选择：根据评估结果选择最有希望的路径继续探索
        if args.method_select == 'sample':
            # 根据评估得分进行概率采样，得分高的路径被选中的概率更大
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            # 贪心选择得分最高的前 N 个路径
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample] #5
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
    
    if to_print: 
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}