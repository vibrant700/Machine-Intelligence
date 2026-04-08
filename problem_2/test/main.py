import heapq
import time

import function
from function import backward_Node, forward_Node


def solve_8_digital_problem(f_input, f_goal, f_n, use_linear_conflict=False):
    if function.determine_solvability(f_input, f_goal, f_n) is False:
        print("无解")
        return None, None
    else:
        # 获得每个数字比特数
        bits_per_number = function.get_bits_per_number(f_n)
        # 获取掩码
        mask = function.get_mask(bits_per_number)

        # 预计算哈密顿距离表
        forward_hamilton_distance_table = function.precalculate_hamilton_table(
            f_goal, f_n
        )
        backward_hamilton_distance_table = function.precalculate_hamilton_table(
            f_input, f_n
        )

        # 默认不启用线性冲突
        forward_goal_rows = forward_goal_cols = None
        forward_row_pairs = forward_col_pairs = None
        backward_goal_rows = backward_goal_cols = None
        backward_row_pairs = backward_col_pairs = None

        if use_linear_conflict:
            (
                forward_goal_rows,
                forward_goal_cols,
                forward_row_pairs,
                forward_col_pairs,
            ) = function.precalculate_linear_conflict_info(f_goal, f_n)
            (
                backward_goal_rows,
                backward_goal_cols,
                backward_row_pairs,
                backward_col_pairs,
            ) = function.precalculate_linear_conflict_info(f_input, f_n)

        # 调用类方法，定义类参数
        forward_Node.set_puzzle_params(
            f_n,
            forward_hamilton_distance_table,
            forward_goal_rows,
            forward_goal_cols,
            forward_row_pairs,
            forward_col_pairs,
            use_linear_conflict,
        )
        backward_Node.set_puzzle_params(
            f_n,
            backward_hamilton_distance_table,
            backward_goal_rows,
            backward_goal_cols,
            backward_row_pairs,
            backward_col_pairs,
            use_linear_conflict,
        )

        # 定义开放列表、关闭列表、状态-g字典
        open_heap_1 = []
        close_set_1 = set()  # 储存num2pos
        g_values_1 = {}  # 用于记录num2pos-g(min)

        open_heap_2 = []
        close_set_2 = set()  # 储存num2pos
        g_values_2 = {}  # 用于记录num2pos-g(min)

        # 将列表转化为二进制数
        init_num2pos = function.from_list_to_num2pos(f_input, bits_per_number)
        init_pos2num = function.from_list_to_pos2num(f_input, bits_per_number)
        goal_num2pos = function.from_list_to_num2pos(f_goal, bits_per_number)
        goal_pos2num = function.from_list_to_pos2num(f_goal, bits_per_number)

        # 计算初始节点的哈密顿距离(无法增量更新，需要遍历)
        init_hamilton_distance = 0
        for i in range(1, f_n * f_n):
            pos = function.get_pos(init_num2pos, i, bits_per_number, mask)
            init_hamilton_distance += forward_hamilton_distance_table[i - 1][pos]

        goal_hamilton_distance = 0
        for i in range(1, f_n * f_n):
            pos = function.get_pos(goal_num2pos, i, bits_per_number, mask)
            goal_hamilton_distance += backward_hamilton_distance_table[i - 1][pos]

        # 计算初始节点的线性冲突（如果启用）
        init_conflict = 0
        goal_conflict = 0
        if use_linear_conflict:
            init_conflict = function.calculate_linear_conflict(
                init_pos2num,
                f_n,
                bits_per_number,
                mask,
                forward_goal_rows,
                forward_goal_cols,
                forward_row_pairs,
                forward_col_pairs,
            )
            goal_conflict = function.calculate_linear_conflict(
                goal_pos2num,
                f_n,
                bits_per_number,
                mask,
                backward_goal_rows,
                backward_goal_cols,
                backward_row_pairs,
                backward_col_pairs,
            )

        # 生成初始节点并添加至开放列表
        init_node = forward_Node(
            f_num2pos=init_num2pos,
            f_pos2num=init_pos2num,
            f_parent=None,
            f_g=0,
            f_h=init_hamilton_distance,
            f_conflict=init_conflict,
        )
        goal_node = backward_Node(
            f_num2pos=goal_num2pos,
            f_pos2num=goal_pos2num,
            f_parent=None,
            f_g=0,
            f_h=goal_hamilton_distance,
            f_conflict=goal_conflict,
        )
        heapq.heappush(open_heap_1, init_node)
        heapq.heappush(open_heap_2, goal_node)
        g_values_1[init_num2pos] = 0
        g_values_2[goal_num2pos] = 0
        # 开始循环
        while True:
            # 从开放列表中弹出f最小者，作为工作节点
            working_node_1 = heapq.heappop(open_heap_1)
            working_node_2 = heapq.heappop(open_heap_2)
            if working_node_1.num2pos == working_node_2.num2pos:
                return working_node_1, working_node_2
            close_set_1.add(working_node_1.num2pos)
            close_set_2.add(working_node_2.num2pos)
            # 遍历工作节点1的所有子节点
            for item in working_node_1.generate_sub_nodes():
                # 子节点在对方关闭列表中
                if item.num2pos in close_set_2:
                    for node in open_heap_2:
                        if node.num2pos == item.num2pos:
                            return item, node
                # 子节点在对方的g_values中
                if item.num2pos in g_values_2:
                    for node in open_heap_2:
                        if node.num2pos == item.num2pos:
                            return item, node
                # 如果是第一次遇到或者找到更小g值
                if item.num2pos not in g_values_1 or item.g < g_values_1[item.num2pos]:
                    g_values_1[item.num2pos] = item.g  # 更新最小距离
                    heapq.heappush(open_heap_1, item)  # 将该节点添加至优先队列中
                    # 若当前状态属于关闭列表,将该状态从关闭集合中移除
                    if item.num2pos in close_set_1:
                        close_set_1.remove(item.num2pos)

            # 遍历工作节点2的所有子节点
            for item in working_node_2.generate_sub_nodes():
                # 子节点在对方关闭列表中
                if item.num2pos in close_set_1:
                    for node in open_heap_1:
                        if node.num2pos == item.num2pos:
                            return node, item
                # 子节点在对方的g_values中
                if item.num2pos in g_values_1:
                    for node in open_heap_1:
                        if node.num2pos == item.num2pos:
                            return node, item
                # 如果是第一次遇到或者找到更小g值
                if item.num2pos not in g_values_2 or item.g < g_values_2[item.num2pos]:
                    g_values_2[item.num2pos] = item.g  # 更新最小距离
                    heapq.heappush(open_heap_2, item)  # 将该节点添加至优先队列中
                    # 若当前状态属于关闭列表,将该状态从关闭集合中移除
                    if item.num2pos in close_set_2:
                        close_set_2.remove(item.num2pos)


def test(test_times, n, use_linear_conflict=False):
    total_time = 0
    for i in range(test_times):
        input_state = function.generate_one_permutation(n)
        goal_state = function.generate_one_permutation(n)
        print(f"input_state:{input_state},goal_state:{goal_state}")
        if use_linear_conflict:
            print("使用线性冲突启发式")
        else:
            print("使用纯曼哈顿距离启发式")
        start_time = time.perf_counter_ns()
        node_1, node_2 = solve_8_digital_problem(
            f_input=input_state,
            f_goal=goal_state,
            f_n=n,
            use_linear_conflict=use_linear_conflict,
        )
        end_time = time.perf_counter_ns()
        used_time = end_time - start_time
        total_time += used_time
        if node_1 and node_2:
            path_forward = function.get_path_forward(node_1)
            path_backward = function.get_path_backward(node_2)
            print(
                f"正向步数: {len(path_forward) - 1}, 反向步数: {len(path_backward) - 1}"
            )
            print(f"总步数: {len(path_forward) + len(path_backward) - 2}")
        else:
            print("未找到解")
    average_time = total_time / test_times
    print(f"平均用时:{average_time / 1e6}ms")


# 对比测试：纯曼哈顿距离 vs 线性冲突
def compare_two_method(times, size):
    for i in range(times):
        # 测试用例
        test_input = function.generate_one_permutation(size)
        test_goal = function.generate_one_permutation(size)

        print(f"\n初始状态: {test_input}")
        print(f"目标状态: {test_goal}")
        print()

        print("曼哈顿距离")
        start = time.perf_counter_ns()
        node_1, node_2 = solve_8_digital_problem(
            f_input=test_input, f_goal=test_goal, f_n=size, use_linear_conflict=False
        )
        end = time.perf_counter_ns()
        if node_1 and node_2:
            path_f = function.get_path_forward(node_1)
            path_b = function.get_path_backward(node_2)
            print(" 找到解")
            print(f"  正向步数: {len(path_f) - 1}")
            print(f"  反向步数: {len(path_b) - 1}")
            print(f"  总步数: {len(path_f) + len(path_b) - 2}")
            print(f"  用时: {(end - start) / 1e6:.2f}ms")
        else:
            print("✗ 未找到解")
        print()

        print("线性冲突增强")
        start = time.perf_counter_ns()
        node_1, node_2 = solve_8_digital_problem(
            f_input=test_input, f_goal=test_goal, f_n=size, use_linear_conflict=True
        )
        end = time.perf_counter_ns()
        if node_1 and node_2:
            path_f = function.get_path_forward(node_1)
            path_b = function.get_path_backward(node_2)
            print(" 找到解")
            print(f"  正向步数: {len(path_f) - 1}")
            print(f"  反向步数: {len(path_b) - 1}")
            print(f"  总步数: {len(path_f) + len(path_b) - 2}")
            print(f"  用时: {(end - start) / 1e6:.2f}ms")
        else:
            print("✗ 未找到解")


compare_two_method(10, 4)
