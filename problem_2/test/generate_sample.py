import json
import random
from pathlib import Path


# 用于生成一个情况
# 列表[0]代表0号位置上的元素,[1]代表1号位置上的元素
def generate_one_permutation(n):
    numbers = list(range(n * n))
    random.shuffle(numbers)
    return numbers


# 计算逆序数(不考虑空格0)
def calculate_inversion_number(f_input):
    result = 0
    for i in range(len(f_input)):
        if f_input[i] == 0:
            continue
        for j in range(i):
            if f_input[j] > f_input[i]:
                result += 1
            else:
                pass
    return result


# 基于逆序数判断是否可解
# n是解决n*n问题需要参数，如求解8数码问题则需要传入n=3
def determine_solvability(f_input, f_goal, n):
    input_inversion_number = calculate_inversion_number(f_input)
    goal_inversion_number = calculate_inversion_number(f_goal)
    # evidence是用于判断是否可解的关键数值
    evidence_1 = input_inversion_number % 2
    evidence_2 = goal_inversion_number % 2
    if n % 2 == 0:
        row_1 = n - (f_input.index(0) // n)
        row_2 = n - (f_goal.index(0) // n)
        evidence_1 += row_1
        evidence_2 += row_2
    if (evidence_1 % 2) == (evidence_2 % 2):
        return True
    else:
        return False


current_file = Path(__file__).resolve()
current_dir = current_file.parent
parent_dir = current_dir.parent
test_dir = parent_dir / "test"

num = 0
goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
my_list = []
while num < 50:
    state = generate_one_permutation(4)
    if determine_solvability(state, goal, 4):
        print(f"{state},")
        my_list.append(state)
        num += 1
with open(str(test_dir / "data.json"), "w") as f:
    json.dump(my_list, f)
