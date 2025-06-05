import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import * 

# (left: 4 6)中提取数字
def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles']) # 读取csv的4个数字
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    # 检查最终表达式是否使用了原始的 4 个数字
    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        # 如果当前数字24，则构造完整的解题步骤提示词;否则继续下一步生成
        # input+steps+answer
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24': # 5-shot + x + ys
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else: # "下一步的多个可能有哪些?"
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        # x原始数字 y操作历史
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # 最后一步
            ans = last_line.lower().replace('answer: ', '')
            # 构造提示词: 根据源数字和最终答案,让llm判断是否能构成24，sure/impossble
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        # 构造提示词: 根据源数字和当前状态,让llm判断是否能构成24，sure/likely/impossble
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # x原始数字 y操作历史 value_outputs评估结果
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0 # 最后一步没有答案直接返回0
        # 从每个评估结果中提取最后一行的关键词
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        # 计算最终得分：每个关键词的分数乘以出现次数，然后求和
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value # 根据关键词计算得分