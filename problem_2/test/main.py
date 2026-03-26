import heapq
import random
import time

import function
from function import single_Node


def solve_8_digital_problem(f_input, f_goal):
    if function.determine_solvability(f_input, f_goal) is False:
        print("无解")
        return None
    else:
        # 预计算哈密顿距离表,并为类属性赋值
        hamilton_distance_table = function.precalculate_hamilton_table(f_goal)
        single_Node._hamilton_table = hamilton_distance_table
        # 定义开放列表、关闭列表、状态-g字典
        open_heap = []
        close_set = set()  # 储存num2pos
        g_values = {}  # 用于记录num2pos-g(min)
        # 将列表转化为二进制数
        init_num2pos = function.from_list_to_num2pos(f_input)
        init_pos2num = function.from_list_to_pos2num(f_input)
        goal_num2pos = function.from_list_to_num2pos(f_goal)

        # 计算初始节点的哈密顿距离(无法增量更新，需要遍历)
        init_hamilton_distance = 0
        for i in range(1, 9):
            pos = function.get_pos(init_num2pos, i)
            init_hamilton_distance += hamilton_distance_table[i - 1][pos]
        # 生成初始节点并添加至开放列表
        init_node = single_Node(
            f_num2pos=init_num2pos,
            f_pos2num=init_pos2num,
            f_parent=None,
            f_g=0,
            f_h=init_hamilton_distance,
        )
        heapq.heappush(open_heap, init_node)
        g_values[init_num2pos] = 0
        # 开始循环
        while open_heap:
            # 从开放列表中弹出f最小者，作为工作节点
            working_node = heapq.heappop(open_heap)
            # 若工作节点的g不是工作节点num2pos数对应的最小的g(说明不是最新的)，则跳过
            if working_node.g != g_values[working_node.num2pos]:
                continue
            if working_node.num2pos == goal_num2pos:
                return working_node
            close_set.add(working_node.num2pos)
            # 遍历工作节点的所有子节点
            for item in working_node.generate_sub_nodes():
                if item.num2pos == goal_num2pos:
                    return item
                # 如果是第一次遇到或者找到更小g值
                if (
                    item.num2pos not in g_values
                    or item.g < g_values[item.num2pos]
                ):
                    g_values[item.num2pos] = item.g  # 更新最小距离
                    heapq.heappush(open_heap, item)  # 将该节点添加至优先队列中
                    # 若当前状态属于关闭列表,将该状态从关闭集合中移除
                    if item.num2pos in close_set:
                        close_set.remove(item.num2pos)


def test(test_times):
    all_condition = function.generate_full_permutation(8)
    total_time = 0
    for i in range(test_times):
        input_state = all_condition[random.randint(0, len(all_condition) - 1)]
        goal_state = all_condition[random.randint(0, len(all_condition) - 1)]
        print(f"input_state:{input_state},goal_state:{goal_state}")
        start_time = time.perf_counter_ns()
        result = solve_8_digital_problem(
            f_input=input_state, f_goal=goal_state
        )
        end_time = time.perf_counter_ns()
        used_time = end_time - start_time
        total_time += used_time
        if result:
            print(function.get_path(result))
    average_time = total_time / test_times
    print(f"平均用时:{average_time}ns")


test(5000)
