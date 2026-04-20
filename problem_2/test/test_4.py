import json
import pickle
import sys
import time
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
parent_dir = current_dir.parent
src_dir = parent_dir / "src"
test_dir = parent_dir / "test"
sys.path.insert(0, str(src_dir))

from Linear import function, main
from Pattern import function2, main2


def load_pattern_dbs(pattern_file_list):
    loaded = []
    for tiles, db_path in pattern_file_list:
        with open(db_path, "rb") as f:
            loaded.append((tiles, pickle.load(f)))
    return loaded


def compare_two_method(sample_list, pattern_dbs):
    Manhattan_time = 0
    Linear_time = 0
    Db_time = 0
    test_times = 0
    for test_input in sample_list:
        test_times += 1
        print(f"进行第{test_times}次测试")
        print(f"初始状态: {test_input}")
        print()

        print("曼哈顿距离:")
        start = time.perf_counter_ns()
        node_1, node_2 = main.solve_8_digital_problem(
            f_input=test_input, f_goal=goal, f_n=4, use_linear_conflict=False
        )
        end = time.perf_counter_ns()
        if node_1 and node_2:
            path_f = function.get_path_forward(node_1)
            path_b = function.get_path_backward(node_2)
            print(f"  总步数: {len(path_f) + len(path_b) - 2}")
            print(f"  用时: {(end - start) / 1e6:.2f}ms")
            Manhattan_time += (end - start) / 1e6
        else:
            print(" 未找到解")

        print("线性冲突增强:")
        start = time.perf_counter_ns()
        node_1, node_2 = main.solve_8_digital_problem(
            f_input=test_input, f_goal=goal, f_n=4, use_linear_conflict=True
        )
        end = time.perf_counter_ns()
        if node_1 and node_2:
            path_f = function.get_path_forward(node_1)
            path_b = function.get_path_backward(node_2)
            print(f"  总步数: {len(path_f) + len(path_b) - 2}")
            print(f"  用时: {(end - start) / 1e6:.2f}ms")
            Linear_time += (end - start) / 1e6
        else:
            print(" 未找到解")

        print("模式数据库:")
        start = time.perf_counter_ns()
        node_1, node_2 = main2.solve_8_digital_problem(
            f_input=test_input, f_goal=goal, f_n=4, pattern_db_file=pattern_dbs
        )
        end = time.perf_counter_ns()
        if node_1 and node_2:
            path_f = function.get_path_forward(node_1)
            path_b = function.get_path_backward(node_2)
            print(f"  总步数: {len(path_f) + len(path_b) - 2}")
            print(f"  用时: {(end - start) / 1e6:.2f}ms")
            Db_time += (end - start) / 1e6
        else:
            print(" 未找到解")
        print()
    average_Manhattan_time = Manhattan_time / test_times
    average_Linear_time = Linear_time / test_times
    average_Db_time = Db_time / test_times
    print(f"""完成{test_times}轮对比测试,
曼哈顿距离平均时间:{average_Manhattan_time}
线性冲突平均时间:{average_Linear_time}
模式数据库平均时间:{average_Db_time}
""")


# 读取数据
with open(str(test_dir / "data.json"), "r") as f:
    sample_list = json.load(f)

goal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
pattern_files = [
    ([1, 2, 3, 4, 5], str(src_dir / "Pattern//pattern_db_1_5.pkl")),
    ([6, 7, 8, 9, 10], str(src_dir / "Pattern//pattern_db_6_10.pkl")),
    ([11, 12, 13, 14, 15], str(src_dir / "Pattern//pattern_db_11_15.pkl")),
]
pattern_dbs = load_pattern_dbs(pattern_files)
compare_two_method(sample_list, pattern_dbs)
