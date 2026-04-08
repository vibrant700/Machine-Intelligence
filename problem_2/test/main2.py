import heapq
import time
import pickle
import function2
from function2 import backward_Node, forward_Node


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
        forward_Node.set_puzzle_params(f_n, forward_hamilton_distance_table,pattern_db_file)
        backward_Node.set_puzzle_params(f_n, backward_hamilton_distance_table)

        # 定义开放列表、关闭列表、状态-g字典
        open_heap_1 = []
        close_set_1 = set()  # 储存num2pos
        g_values_1 = {}  # 用于记录num2pos-g(min)

        open_heap_2 = []
        close_set_2 = set()  # 储存num2pos
        g_values_2 = {}  # 用于记录num2pos-g(min)
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


def test(test_times, n, pattern_db_files=None):
    total_time = 0
    pattern_dbs = []
    for tiles, filepath in pattern_db_files:
        with open(filepath, 'rb') as f:
            db = pickle.load(f)
            pattern_dbs.append((tiles, db))
    print(f"Loaded {len(pattern_dbs)} pattern databases")

    for i in range(test_times):
        input_state = function2.generate_one_permutation(n)
        goal_state = function2.generate_one_permutation(n)
        print(f"input_state:{input_state},goal_state:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]")
        start_time = time.perf_counter_ns()
        node_1, node_2 = solve_8_digital_problem(
            f_input=input_state, f_goal=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0], f_n=n,
        pattern_db_file = pattern_dbs)
        end_time = time.perf_counter_ns()
        used_time = end_time - start_time
        total_time += used_time
        if node_1 and node_2:
            print(function2.get_path_forward(node_1), end="")
            print(function2.get_path_backward(node_2))
    average_time = total_time / test_times
    print(f"平均用时:{average_time / 1e6}ms")

pattern_files = [
        ([1,2,3,4,5], "pattern_db_1_5.pkl"),
        ([6,7,8,9,10], "pattern_db_6_10.pkl"),
        ([11,12,13,14,15], "pattern_db_11_15.pkl")
    ]
test(20, 4,pattern_db_files=pattern_files)