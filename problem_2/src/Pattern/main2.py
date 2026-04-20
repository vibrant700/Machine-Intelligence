import heapq
import pickle
import time

import function2
from function2 import backward_Node, forward_Node


# 清理堆顶过期节点，直到堆顶为当前最优版本或堆为空
def clean_invalid_top(heap, g_values):
    while heap:
        node = heap[0]
        if g_values.get(node.num2pos) == node.g:
            return
        heapq.heappop(heap)


# 返回开放列表中最小有效f值
def peek_min_valid_f(heap, g_values):
    clean_invalid_top(heap, g_values)
    if heap:
        return heap[0].f
    return float("inf")


# 弹出并返回一个有效节点，若无有效节点则返回None
def pop_valid_node(heap, g_values):
    clean_invalid_top(heap, g_values)
    if heap:
        return heapq.heappop(heap)
    return None


def solve_8_digital_problem(f_input, f_goal, f_n, pattern_db_file):
    if function2.determine_solvability(f_input, f_goal, f_n) is False:
        print("无解")
        return None, None
    else:
        # 获得每个数字比特数
        bits_per_number = function2.get_bits_per_number(f_n)
        # 获取掩码
        mask = function2.get_mask(bits_per_number)
        forward_hamilton_distance_table = function2.precalculate_hamilton_table(
            f_goal, f_n
        )
        backward_hamilton_distance_table = function2.precalculate_hamilton_table(
            f_input, f_n
        )
        # 调用类方法，定义类参数
        forward_Node.set_puzzle_params(
            f_n, forward_hamilton_distance_table, pattern_db_file
        )
        backward_Node.set_puzzle_params(f_n, backward_hamilton_distance_table)

        # 定义开放列表、关闭列表、状态-g字典
        open_heap_1 = []
        close_set_1 = set()  # 储存num2pos
        g_values_1 = {}  # 用于记录num2pos-g(min)
        best_node_1 = {}  # 用于记录num2pos-node

        open_heap_2 = []
        close_set_2 = set()  # 储存num2pos
        g_values_2 = {}  # 用于记录num2pos-g(min)
        best_node_2 = {}  # 用于记录num2pos-node

        # 定义mu,结果节点
        mu = float("inf")
        forward = None
        backward = None

        # 将列表转化为二进制数
        init_num2pos = function2.from_list_to_num2pos(f_input, bits_per_number)
        init_pos2num = function2.from_list_to_pos2num(f_input, bits_per_number)
        goal_num2pos = function2.from_list_to_num2pos(f_goal, bits_per_number)
        goal_pos2num = function2.from_list_to_pos2num(f_goal, bits_per_number)

        # 计算初始节点的哈密顿距离(无法增量更新，需要遍历)
        init_hamilton_distance = 0
        for i in range(1, f_n * f_n):
            pos = function2.get_pos(init_num2pos, i, bits_per_number, mask)
            init_hamilton_distance += forward_hamilton_distance_table[i - 1][pos]
        goal_hamilton_distance = 0
        for i in range(1, f_n * f_n):
            pos = function2.get_pos(goal_num2pos, i, bits_per_number, mask)
            goal_hamilton_distance += backward_hamilton_distance_table[i - 1][pos]
        # 生成初始节点并添加至开放列表
        init_node = forward_Node(
            f_num2pos=init_num2pos,
            f_pos2num=init_pos2num,
            f_parent=None,
            f_g=0,
            f_h=0,
        )
        goal_node = backward_Node(
            f_num2pos=goal_num2pos,
            f_pos2num=goal_pos2num,
            f_parent=None,
            f_g=0,
            f_h=goal_hamilton_distance,
        )
        heapq.heappush(open_heap_1, init_node)
        heapq.heappush(open_heap_2, goal_node)
        g_values_1[init_num2pos] = 0
        best_node_1[init_num2pos] = init_node
        g_values_2[goal_num2pos] = 0
        best_node_2[goal_num2pos] = goal_node
        # 开始循环
        while True:
            # 从开放列表中弹出f最小者，作为工作节点
            working_node_1 = pop_valid_node(open_heap_1, g_values_1)
            working_node_2 = pop_valid_node(open_heap_2, g_values_2)
            # 如果有无法获得有效的工作节点了，则直接返回已有的forward和backward
            if working_node_1 is None or working_node_2 is None:
                return forward, backward

            # 至此成功获得工作节点，可以开始工作

            if working_node_1.num2pos == working_node_2.num2pos:
                if working_node_1.g + working_node_2.g < mu:
                    mu = working_node_1.g + working_node_2.g
                    forward = working_node_1
                    backward = working_node_2

            close_set_1.add(working_node_1.num2pos)
            close_set_2.add(working_node_2.num2pos)
            # 遍历工作节点1的所有子节点
            for item in working_node_1.generate_sub_nodes():
                # 子节点在对方关闭列表中
                if item.num2pos in close_set_2:
                    node = best_node_2[item.num2pos]
                    if node.g + item.g < mu:
                        mu = node.g + item.g
                        forward = item
                        backward = node
                # 子节点在对方的g_values中
                if item.num2pos in g_values_2:
                    node = best_node_2[item.num2pos]
                    if node.g + item.g < mu:
                        mu = node.g + item.g
                        forward = item
                        backward = node
                # 如果是第一次遇到或者找到更小g值
                if item.num2pos not in g_values_1 or item.g < g_values_1[item.num2pos]:
                    g_values_1[item.num2pos] = item.g  # 更新最小距离
                    best_node_1[item.num2pos] = item
                    heapq.heappush(open_heap_1, item)  # 将该节点添加至优先队列中
                    # 若当前状态属于关闭列表,将该状态从关闭集合中移除
                    if item.num2pos in close_set_1:
                        close_set_1.remove(item.num2pos)

            # 遍历工作节点2的所有子节点
            for item in working_node_2.generate_sub_nodes():
                # 子节点在对方关闭列表中
                if item.num2pos in close_set_1:
                    node = best_node_1[item.num2pos]
                    if node.g + item.g < mu:
                        mu = node.g + item.g
                        forward = node
                        backward = item
                # 子节点在对方的g_values中
                if item.num2pos in g_values_1:
                    node = best_node_1[item.num2pos]
                    if node.g + item.g < mu:
                        mu = node.g + item.g
                        forward = node
                        backward = item
                # 如果是第一次遇到或者找到更小g值
                if item.num2pos not in g_values_2 or item.g < g_values_2[item.num2pos]:
                    g_values_2[item.num2pos] = item.g  # 更新最小距离
                    best_node_2[item.num2pos] = item
                    heapq.heappush(open_heap_2, item)  # 将该节点添加至优先队列中
                    # 若当前状态属于关闭列表,将该状态从关闭集合中移除
                    if item.num2pos in close_set_2:
                        close_set_2.remove(item.num2pos)

            # 终止条件：比较两个开放列表当前最小有效f值与mu
            min_f_1 = peek_min_valid_f(open_heap_1, g_values_1)
            min_f_2 = peek_min_valid_f(open_heap_2, g_values_2)
            if max(min_f_1, min_f_2) >= mu:
                return forward, backward


def solve_with_custom_goal(f_input, f_goal, f_n, pattern_db_file):
    """
    支持标准状态的求解函数

    注意：模式数据库算法仅支持标准目标状态（0在最后）
    对于非标准目标状态，请使用线性冲突算法

    Args:
        f_input: 初始状态
        f_goal: 目标状态（必须是标准目标状态）
        f_n: 棋盘大小
        pattern_db_file: 模式数据库文件

    Returns:
        正向和反向路径节点（已映射回原始编号）
    """
    # 标准目标状态
    standard_goal = list(range(1, f_n * f_n)) + [0]

    # 验证目标状态必须是标准目标
    if f_goal != standard_goal:
        print(f"[ERROR] 模式数据库算法仅支持标准目标状态（0在最后）")
        print(f"[ERROR] 当前目标: {f_goal}")
        print(f"[ERROR] 请使用线性冲突算法")
        return None, None

    # 直接使用原求解函数
    node_1, node_2 = solve_8_digital_problem(
        f_input, f_goal, f_n, pattern_db_file=pattern_db_file
    )
    if node_1 is None or node_2 is None:
        return None, None

    # 提取路径
    forward_path = function2.get_path_forward(node_1)
    backward_path = function2.get_path_backward(node_2)

    return forward_path, backward_path


def test(test_times, n, test_input=None, pattern_db_files=None):
    total_time = 0
    pattern_dbs = []
    for tiles, filepath in pattern_db_files:
        with open(filepath, "rb") as f:
            db = pickle.load(f)
            pattern_dbs.append((tiles, db))
    print(f"Loaded {len(pattern_dbs)} pattern databases")

    for i in range(test_times):
        if test_input is None:
            input_state, goal_state = function.generate_one_permutation(size)
        else:
            input_state = test_input[i]
            goal_state = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
        print(
            f"input_state:{input_state},goal_state:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]"
        )
        start_time = time.perf_counter_ns()
        node_1, node_2 = solve_8_digital_problem(
            f_input=input_state,
            f_goal=goal_state,
            f_n=n,
            pattern_db_file=pattern_dbs,
        )
        end_time = time.perf_counter_ns()
        used_time = end_time - start_time
        print(f"用时:{used_time/1e6}ms")
        total_time += used_time
        if node_1 and node_2:
            print("正向步数：", len(function2.get_path_forward(node_1)) - 1)
            print("反向步数：", len(function2.get_path_backward(node_2)) - 1)
            print(
                "总步数：",
                len(function2.get_path_forward(node_1))
                + len(function2.get_path_forward(node_2))
                - 1,
            )
    average_time = total_time / test_times
    print(f"平均用时:{average_time / 1e6}ms")


if __name__ == "__main__":
    pattern_files = [
        ([1, 2, 3, 4, 5], r"pattern_db_1_5.pkl"),
        ([6, 7, 8, 9, 10], r"pattern_db_6_10.pkl"),
        (
            [11, 12, 13, 14, 15],
            r"pattern_db_11_15.pkl",
        ),
    ]
    test_inputs = pickle.load(open("test.pkl", "rb"))
    test(50, 4, test_inputs, pattern_files)
