"""
解决N数码难题
以15数码难题为例
运行结果如下：
实例1：
不使用模式数据库运行时间：43.95s
使用1-4瓷砖运行时间：37.87s
使用1-7瓷砖运行时间：20.53s
实例2：
不使用模式数据库运行时间：5.12s
使用1-4瓷砖运行时间：3.41s
使用1-7瓷砖运行时间：11.73s
实例3：
不使用模式数据库运行时间：0.002s
使用1-4瓷砖运行时间：0.004s
使用1-7瓷砖运行时间：11.03s
较大模式数据库占用空间大，加载时间长,在解决简单问题时没有优势,数字越乱越没有规律，1-7瓷砖越占优势
"""
import heapq
import pickle
from typing import List, Optional, Tuple
import time
class NPuzzleSolver:
    """
    通用 N×N 数码问题求解器 (A* + 曼哈顿距离 + 模式数据库)
    """
    def __init__(self, n: int, pattern_db_files: Optional[List[Tuple[List[int], str]]] = None):
        self.n = n
        self.size = n * n
        self.bits = (self.size - 1).bit_length()
        self.mask = (1 << self.bits) - 1

        # 目标状态（顺序排列，0在最后）
        self.goal_state = list(range(1, self.size)) + [0]
        self.goal_num2pos = self.list_to_num2pos(self.goal_state)

        # 预计算曼哈顿距离表（仅用于未覆盖的瓷砖）
        self.h_table = self._precompute_h_table()

        # 预计算移动偏移表
        self.motion_table = self._precompute_motion_table()

        # ---------- 加载模式数据库 ----------
        self.pattern_dbs = []          # 每个元素为 (pattern_tiles, db_dict)
        self.covered_tiles = set()     # 记录被模式数据库覆盖的瓷砖编号

        if pattern_db_files:
            for pattern_tiles, db_file in pattern_db_files:
                try:
                    with open(db_file, 'rb') as f:
                        db = pickle.load(f)
                    self.pattern_dbs.append((pattern_tiles, db))
                    self.covered_tiles.update(pattern_tiles)
                    print(f"Loaded pattern DB for {pattern_tiles}, size={len(db)}")
                except FileNotFoundError:
                    print(f"Warning: {db_file} not found, skipping.")
                except Exception as e:
                    print(f"Error loading {db_file}: {e}")

    # ======================== 状态编码与解码 ========================
    def list_to_num2pos(self, state: List[int]) -> int:
        num2pos = 0
        for pos, num in enumerate(state):
            num2pos |= pos << (self.bits * num)
        return num2pos

    def list_to_pos2num(self, state: List[int]) -> int:
        pos2num = 0
        for pos, num in enumerate(state):
            pos2num |= num << (self.bits * pos)
        return pos2num

    def pos2num_to_list(self, pos2num: int) -> List[int]:
        lst = [0] * self.size
        for p in range(self.size):
            lst[p] = (pos2num >> (self.bits * p)) & self.mask
        return lst

    def get_digit(self, pos2num: int, p: int) -> int:
        return (pos2num >> (self.bits * p)) & self.mask

    def get_pos(self, num2pos: int, d: int) -> int:
        return (num2pos >> (self.bits * d)) & self.mask

    # ======================== 移动操作 ========================
    def move(self, num2pos: int, pos2num: int, p0: int, p1: int) -> Tuple[int, int]:
        p1_num = self.get_digit(pos2num, p1)
        new_num2pos = num2pos
        new_num2pos &= ~self.mask
        new_num2pos |= p1
        new_num2pos &= ~(self.mask << (self.bits * p1_num))
        new_num2pos |= p0 << (self.bits * p1_num)

        new_pos2num = pos2num
        new_pos2num &= ~(self.mask << (self.bits * p0))
        new_pos2num |= p1_num << (self.bits * p0)
        new_pos2num &= ~(self.mask << (self.bits * p1))
        return new_num2pos, new_pos2num

    # ======================== 启发式预计算 ========================
    def _precompute_h_table(self) -> List[List[int]]:
        table = []
        for num in range(1, self.size):
            goal_pos = self.goal_state.index(num)
            goal_row, goal_col = divmod(goal_pos, self.n)
            row_dist = []
            for pos in range(self.size):
                row, col = divmod(pos, self.n)
                row_dist.append(abs(goal_row - row) + abs(goal_col - col))
            table.append(row_dist)
        return table

    def _precompute_motion_table(self) -> List[List[int]]:
        motion = []
        for pos in range(self.size):
            row, col = divmod(pos, self.n)
            offsets = []
            if col > 0:          offsets.append(-1)
            if col < self.n-1:   offsets.append(1)
            if row > 0:          offsets.append(-self.n)
            if row < self.n-1:   offsets.append(self.n)
            motion.append(offsets)
        return motion

    # ======================== 模式数据库相关 ========================
    def _encode_pattern(self, positions: List[int]) -> int:
        """将位置列表编码为整数（每个位置占4位）与构建脚本一致"""
        code = 0
        for i, pos in enumerate(positions):
            code |= pos << (4 * i)
        return code

    def _get_pattern_heuristic(self, state_list: List[int]) -> int:
        """查询所有模式数据库，返回累加距离（未命中返回0）"""
        total = 0
        for pattern_tiles, db in self.pattern_dbs:
            positions = [state_list.index(tile) for tile in pattern_tiles]
            code = self._encode_pattern(positions)
            total += db.get(code, 0)
        return total

    def _compute_manhattan_for_uncovered(self, num2pos: int) -> int:
        """计算未被模式数据库覆盖的瓷砖的曼哈顿距离之和"""
        h = 0
        for num in range(1, self.size):
            if num not in self.covered_tiles:
                pos = self.get_pos(num2pos, num)
                h += self.h_table[num-1][pos]
        return h

    # ======================== 节点类 ========================
    class Node:
        __slots__ = ('num2pos', 'pos2num', 'parent', 'g', 'h', 'f')
        def __init__(self, num2pos, pos2num, parent, g, h):
            self.num2pos = num2pos
            self.pos2num = pos2num
            self.parent = parent
            self.g = g
            self.h = h
            self.f = g + h

        def __lt__(self, other):
            return self.f < other.f or (self.f == other.f and self.h < other.h)

    # ======================== 可解性判断 ========================
    def is_solvable(self, state: List[int]) -> bool:
        inv_count = 0
        arr = [x for x in state if x != 0]
        for i in range(len(arr)):
            for j in range(i+1, len(arr)):
                if arr[i] > arr[j]:
                    inv_count += 1
        if self.n % 2 == 1:
            return inv_count % 2 == 0
        else:
            zero_row = state.index(0) // self.n
            row_from_bottom = self.n - zero_row
            return (inv_count + row_from_bottom) % 2 == 1

    # ======================== A* 搜索 ========================
    def solve(self, init_state: List[int]) -> Optional[List[List[int]]]:
        if not self.is_solvable(init_state):
            print("无解")
            return None

        init_num2pos = self.list_to_num2pos(init_state)
        init_pos2num = self.list_to_pos2num(init_state)
        goal_num2pos = self.goal_num2pos

        # 初始启发式 = 未覆盖瓷砖曼哈顿距离 + 模式距离
        init_h = self._compute_manhattan_for_uncovered(init_num2pos) + \
                 self._get_pattern_heuristic(init_state)

        open_heap = []
        close_set = set()
        g_values = {init_num2pos: 0}

        init_node = self.Node(init_num2pos, init_pos2num, None, 0, init_h)
        heapq.heappush(open_heap, init_node)

        while open_heap:
            node = heapq.heappop(open_heap)
            if node.g != g_values.get(node.num2pos, float('inf')):
                continue
            if node.num2pos == goal_num2pos:
                return self._reconstruct_path(node)
            close_set.add(node.num2pos)

            p0 = self.get_pos(node.num2pos, 0)   # 空格位置
            for offset in self.motion_table[p0]:
                p1 = p0 + offset
                new_num2pos, new_pos2num = self.move(node.num2pos, node.pos2num, p0, p1)
                if new_num2pos in close_set:
                    continue
                new_g = node.g + 1

                # 全量计算新状态的启发式（避免重复计算和增量错误）
                new_state_list = self.pos2num_to_list(new_pos2num)
                new_h = self._compute_manhattan_for_uncovered(new_num2pos) + \
                        self._get_pattern_heuristic(new_state_list)

                new_node = self.Node(new_num2pos, new_pos2num, node, new_g, new_h)

                if new_num2pos not in g_values or new_g < g_values[new_num2pos]:
                    g_values[new_num2pos] = new_g
                    heapq.heappush(open_heap, new_node)
                    if new_num2pos in close_set:
                        close_set.remove(new_num2pos)

        return None

    def _reconstruct_path(self, node: 'NPuzzleSolver.Node') -> List[List[int]]:
        path = []
        while node:
            path.append(self.pos2num_to_list(node.pos2num))
            node = node.parent
        return path[::-1]
import random

def generate_random_state(n=4):
    """生成随机可解的初始状态"""
    size = n * n
    state = list(range(1, size)) + [0]
    random.shuffle(state)
    while not NPuzzleSolver(n).is_solvable(state):   # 临时求解器判断可解性
        random.shuffle(state)
    return state

if __name__ == "__main__":
    # 使用 7‑瓷砖数据库
    start_time = time.perf_counter()
    solver = NPuzzleSolver(
        n=4,
        pattern_db_files=[([1,2,3,4], "pattern_db_1_4.pkl")]
        # pattern_db_files = [([1, 2, 3, 4,5,6,7], "pattern_db_1_7.pkl")]
    )

    # init_state = [7,6,1,8,5,2,0,9,12,4,13,14,15,10,11,3]
    # init_state = [7,1,2,3,4,5,6,8,9,10,11,0,12,14,13,15]
    init_state = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,15]
    print("初始状态：")
    for i in range(0, 16, 4):
        print(init_state[i:i+4])

    path = solver.solve(init_state)
    if path:
        for step in path:
            for i in range(0, 16, 4):
                print(step[i:i+4])
            print()
        print(f"\n找到解，步数: {len(path) - 1}")
    else:
        print("无解")
    end_time = time.perf_counter()
    print(end_time - start_time)