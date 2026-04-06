import math
import random


# 生成所有情况
# 使用递归以生成0-n的全排列
# 列表[0]代表0号位置上的元素,[1]代表1号位置上的元素
def generate_full_permutation(n):
    # 递归停止条件
    if n == 0:
        return [[0]]
    result = []
    # 递归调用
    smaller_perm = generate_full_permutation(n - 1)
    for item in smaller_perm:
        for index in range(len(item) + 1):
            new_perm = item[:index] + [n] + item[index:]
            result.append(new_perm)
    return result


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


# 预计算哈密顿距离表
# 返回值为一个2维数组,list[数字][数字位置编号] = 该数字对应的哈密顿距离
# n是解决n*n问题需要参数，如求解8数码问题则需要传入n=3
def precalculate_hamilton_table(f_goal, n):
    hamilton_table = []
    for i in range(1, pow(n, 2)):
        temp = f_goal.index(i)
        goal_row_index = temp // n
        goal_col_index = temp % n
        temp_list = []
        for j in range(pow(n, 2)):
            current_row_index = j // n
            current_col_index = j % n
            temp_list.append(
                abs(goal_row_index - current_row_index)
                + abs(goal_col_index - current_col_index)
            )
        hamilton_table.append(temp_list)
    return hamilton_table


# 线性冲突相关函数


# 预计算线性冲突所需的辅助信息
def precalculate_linear_conflict_info(f_goal, n):
    """
    返回:
        goal_rows: list, goal_rows[num] = 数字num的目标行号
        goal_cols: list, goal_cols[num] = 数字num的目标列号
        row_conflict_pairs: set, 可能产生行冲突的数字对 (num1, num2)
        col_conflict_pairs: set, 可能产生列冲突的数字对 (num1, num2)
    """
    size = n * n

    # 1. 计算每个数字的目标行列
    goal_rows = [0] * size
    goal_cols = [0] * size
    for num in range(1, size):
        pos = f_goal.index(num)
        goal_rows[num] = pos // n
        goal_cols[num] = pos % n

    # 2. 预计算可能产生行冲突的数字对
    row_conflict_pairs = set()
    for row in range(n):
        # 找出目标在这一行的所有数字
        nums_in_row = [num for num in range(1, size) if goal_rows[num] == row]
        # 对于这些数字，如果它们的目标列不同，则可能产生冲突
        for i in range(len(nums_in_row)):
            for j in range(i + 1, len(nums_in_row)):
                num1, num2 = nums_in_row[i], nums_in_row[j]
                if goal_cols[num1] != goal_cols[num2]:
                    row_conflict_pairs.add((num1, num2))
                    row_conflict_pairs.add((num2, num1))

    # 3. 预计算可能产生列冲突的数字对
    col_conflict_pairs = set()
    for col in range(n):
        # 找出目标在这一列的所有数字
        nums_in_col = [num for num in range(1, size) if goal_cols[num] == col]
        # 对于这些数字，如果它们的目标行不同，则可能产生冲突
        for i in range(len(nums_in_col)):
            for j in range(i + 1, len(nums_in_col)):
                num1, num2 = nums_in_col[i], nums_in_col[j]
                if goal_rows[num1] != goal_rows[num2]:
                    col_conflict_pairs.add((num1, num2))
                    col_conflict_pairs.add((num2, num1))

    return goal_rows, goal_cols, row_conflict_pairs, col_conflict_pairs


# 计算线性冲突值
def calculate_linear_conflict(
    pos2num,
    n,
    bits_per_number,
    mask,
    goal_rows,
    goal_cols,
    row_conflict_pairs,
    col_conflict_pairs,
):
    """
    计算当前状态的线性冲突惩罚值

    参数:
        pos2num: 位置-数字编码
        n: 拼图大小
        bits_per_number: 每个数字占用的位数
        mask: 位掩码
        goal_rows, goal_cols: 目标位置信息
        row_conflict_pairs, col_conflict_pairs: 可能冲突的数字对

    返回:
        线性冲突惩罚值（每个冲突对+2）
    """
    conflict = 0

    # 1. 检测行冲突
    for row in range(n):
        # 收集当前在这一行的所有数字（且目标也在这一行）
        tiles_in_row = []
        for col in range(n):
            pos = row * n + col
            num = get_digit(pos2num, pos, bits_per_number, mask)
            if num != 0 and goal_rows[num] == row:
                tiles_in_row.append((num, col))

        # 检查这些数字对是否产生冲突
        for i in range(len(tiles_in_row)):
            for j in range(i + 1, len(tiles_in_row)):
                num1, col1 = tiles_in_row[i]
                num2, col2 = tiles_in_row[j]

                # 只检查预计算出的可能冲突对
                if (num1, num2) in row_conflict_pairs:
                    # 目标位置相反，当前位置也相反
                    if (goal_cols[num1] > goal_cols[num2] and col1 < col2) or (
                        goal_cols[num1] < goal_cols[num2] and col1 > col2
                    ):
                        conflict += 2

    # 2. 检测列冲突
    for col in range(n):
        # 收集当前在这一列的所有数字（且目标也在这一列）
        tiles_in_col = []
        for row in range(n):
            pos = row * n + col
            num = get_digit(pos2num, pos, bits_per_number, mask)
            if num != 0 and goal_cols[num] == col:
                tiles_in_col.append((num, row))

        # 检查这些数字对是否产生冲突
        for i in range(len(tiles_in_col)):
            for j in range(i + 1, len(tiles_in_col)):
                num1, row1 = tiles_in_col[i]
                num2, row2 = tiles_in_col[j]

                # 只检查预计算出的可能冲突对
                if (num1, num2) in col_conflict_pairs:
                    # 目标位置相反，当前位置也相反
                    if (goal_rows[num1] > goal_rows[num2] and row1 < row2) or (
                        goal_rows[num1] < goal_rows[num2] and row1 > row2
                    ):
                        conflict += 2

    return conflict


# 计算表示 n*n 拼图所需每个数字的位数
def get_bits_per_number(n):
    return math.ceil(math.log2(n * n))


# 计算n*n拼图的mask，避免每次在函数内计算
def get_mask(bits_per_number):
    return (1 << bits_per_number) - 1


# 用于将列表转换为数字-位置2进制数
def from_list_to_num2pos(f_list, bits_per_number):
    num2pos = 0
    for pos, num in enumerate(f_list):
        num2pos |= pos << (bits_per_number * num)
    return num2pos


# 用于将列表转换为位置-数字2进制数
def from_list_to_pos2num(f_list, bits_per_number):
    pos2num = 0
    for pos, num in enumerate(f_list):
        pos2num |= num << (bits_per_number * pos)
    return pos2num


def from_pos2num_to_list(f_pos2num, n, bits_per_number, mask):
    size = pow(n, 2)
    result = [0] * size
    for i in range(size):
        num = get_digit(f_pos2num, i, bits_per_number, mask)
        result[i] = num
    return result


# 一些经过封装的二进制操作(目的是代码可读性)
# 从位置-数字数中获取位置p的数字
def get_digit(f_pos2num, position, bits_per_number, mask):
    return (f_pos2num >> (bits_per_number * position)) & mask


# 从数字-位置数中获取数字d的位置
def get_pos(f_num2pos, digit, bits_per_number, mask):
    return (f_num2pos >> (bits_per_number * digit)) & mask


# 给定原数以及与0交换的位置编号，返回变化后的数
# f_position0为0所在的位置的编号
# f_position1为即将于p0交换的位置的编号
def move(f_num2pos, f_pos2num, f_position0, f_position1, bits_per_number, mask):
    p1_num = get_digit(f_pos2num, f_position1, bits_per_number, mask)

    # 更新数字-位置数
    new_num2pos = f_num2pos
    # 将数字0对应的位置清空(在数字-位置数中，第1-4就对应着数字0)
    new_num2pos &= ~mask
    # 将数字0对应的位置置为p1
    new_num2pos |= f_position1
    # 将p1_num对应的位置清空
    new_num2pos &= ~(mask << (bits_per_number * p1_num))
    # 将p1_num对应的位置置为p0
    new_num2pos |= f_position0 << (bits_per_number * p1_num)

    # 更新位置-数字数
    new_pos2num = f_pos2num
    # 将p0对应的数字清空
    new_pos2num &= ~(mask << (bits_per_number * f_position0))
    # 将p0处的位置置为p1_num(即p1处的数字)
    new_pos2num |= p1_num << (bits_per_number * f_position0)
    # 将p1对应的数字变为0(即p0处的数字0)
    new_pos2num &= ~(mask << (bits_per_number * f_position1))

    # 返回结果
    return new_num2pos, new_pos2num


# 生成n*n拼图的移动表
def generate_motion_table(n):
    size = n * n
    motion_table = []
    for i in range(size):
        row = i // n
        col = i % n
        temp = []
        if col < n - 1:  # 可以右移
            temp.append(1)
        if col > 0:  # 可以左移
            temp.append(-1)
        if row < n - 1:  # 可以下移
            temp.append(n)
        if row > 0:  # 可以上移
            temp.append(-n)
        motion_table.append(tuple(temp))
    return motion_table


# 一个正向搜索节点(即一个状态)对象，应该记录其num2pos、pos2num、父节点、g、h、f
class forward_Node:
    # 类属性，所有实例共享
    _n = None
    _bits_per_number = None
    _mask = None
    _motion_table = None
    _hamilton_table = None
    # 线性冲突相关类属性
    _goal_rows = None
    _goal_cols = None
    _row_conflict_pairs = None
    _col_conflict_pairs = None
    _use_linear_conflict = False  # 是否使用线性冲突

    # 设置类属性，需要在创建第一个节点前调用
    @classmethod
    def set_puzzle_params(
        cls,
        n,
        hamilton_table,
        goal_rows=None,
        goal_cols=None,
        row_conflict_pairs=None,
        col_conflict_pairs=None,
        use_linear_conflict=False,
    ):
        cls._n = n
        cls._bits_per_number = get_bits_per_number(n)
        cls._mask = get_mask(cls._bits_per_number)
        cls._motion_table = generate_motion_table(n)
        cls._hamilton_table = hamilton_table

        # 线性冲突相关参数
        cls._goal_rows = goal_rows
        cls._goal_cols = goal_cols
        cls._row_conflict_pairs = row_conflict_pairs if row_conflict_pairs else set()
        cls._col_conflict_pairs = col_conflict_pairs if col_conflict_pairs else set()
        cls._use_linear_conflict = use_linear_conflict

    # 初始化一个节点
    def __init__(self, f_num2pos, f_pos2num, f_parent, f_g=0, f_h=0, f_conflict=0):
        self.num2pos = f_num2pos
        self.pos2num = f_pos2num
        self.parent = f_parent
        self.g = f_g
        self.h_manhattan = f_h  # 曼哈顿距离
        self.conflict = f_conflict  # 线性冲突值
        self.h = f_h + f_conflict  # 总启发式值
        self.f = self.g + self.h

    def generate_one_sub_node(self, f_position0, f_position1):
        new_num2pos, new_pos2num = move(
            self.num2pos,
            self.pos2num,
            f_position0,
            f_position1,
            self._bits_per_number,
            self._mask,
        )
        new_parent = self
        new_g = self.g + 1
        p1_num = get_digit(self.pos2num, f_position1, self._bits_per_number, self._mask)

        # 增量更新曼哈顿距离
        new_h_manhattan = (
            self.h_manhattan
            - self._hamilton_table[p1_num - 1][f_position1]
            # 减去该数字原先的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
            + self._hamilton_table[p1_num - 1][f_position0]
            # 加上给数值后来的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
        )

        # 计算线性冲突（如果启用）
        new_conflict = 0
        if self._use_linear_conflict:
            new_conflict = calculate_linear_conflict(
                new_pos2num,
                self._n,
                self._bits_per_number,
                self._mask,
                self._goal_rows,
                self._goal_cols,
                self._row_conflict_pairs,
                self._col_conflict_pairs,
            )

        new_node = forward_Node(
            f_num2pos=new_num2pos,
            f_pos2num=new_pos2num,
            f_parent=new_parent,
            f_g=new_g,
            f_h=new_h_manhattan,
            f_conflict=new_conflict,
        )
        return new_node

    # 用于生成某一节点的所有子节点，返回子节点的列表
    def generate_sub_nodes(self):
        result = []
        # 找到数字0所在的位置编号
        index_0 = get_pos(self.num2pos, 0, self._bits_per_number, self._mask)
        for motion in self._motion_table[index_0]:
            result.append(self.generate_one_sub_node(index_0, index_0 + motion))
        return result

    # 更改父节点函数
    def change_parent(self, parent):
        self.parent = parent
        self.g = parent.g + 1
        self.f = self.g + self.h

    # 用于排序
    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h


# 一个反向搜索节点(即一个状态)对象，应该记录其num2pos、pos2num、父节点、g、h、f
class backward_Node:
    # 类属性，所有实例共享
    _n = None
    _bits_per_number = None
    _mask = None
    _motion_table = None
    _hamilton_table = None
    # 线性冲突相关类属性
    _goal_rows = None
    _goal_cols = None
    _row_conflict_pairs = None
    _col_conflict_pairs = None
    _use_linear_conflict = False  # 是否使用线性冲突

    # 设置类属性，需要在创建第一个节点前调用
    @classmethod
    def set_puzzle_params(
        cls,
        n,
        hamilton_table,
        goal_rows=None,
        goal_cols=None,
        row_conflict_pairs=None,
        col_conflict_pairs=None,
        use_linear_conflict=False,
    ):
        cls._n = n
        cls._bits_per_number = get_bits_per_number(n)
        cls._mask = get_mask(cls._bits_per_number)
        cls._motion_table = generate_motion_table(n)
        cls._hamilton_table = hamilton_table

        # 线性冲突相关参数
        cls._goal_rows = goal_rows
        cls._goal_cols = goal_cols
        cls._row_conflict_pairs = row_conflict_pairs if row_conflict_pairs else set()
        cls._col_conflict_pairs = col_conflict_pairs if col_conflict_pairs else set()
        cls._use_linear_conflict = use_linear_conflict

    # 初始化一个节点
    def __init__(self, f_num2pos, f_pos2num, f_parent, f_g=0, f_h=0, f_conflict=0):
        self.num2pos = f_num2pos
        self.pos2num = f_pos2num
        self.parent = f_parent
        self.g = f_g
        self.h_manhattan = f_h  # 曼哈顿距离
        self.conflict = f_conflict  # 线性冲突值
        self.h = f_h + f_conflict  # 总启发式值
        self.f = self.g + self.h

    def generate_one_sub_node(self, f_position0, f_position1):
        new_num2pos, new_pos2num = move(
            self.num2pos,
            self.pos2num,
            f_position0,
            f_position1,
            self._bits_per_number,
            self._mask,
        )
        new_parent = self
        new_g = self.g + 1
        p1_num = get_digit(self.pos2num, f_position1, self._bits_per_number, self._mask)

        # 增量更新曼哈顿距离
        new_h_manhattan = (
            self.h_manhattan
            - self._hamilton_table[p1_num - 1][f_position1]
            # 减去该数字原先的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
            + self._hamilton_table[p1_num - 1][f_position0]
            # 加上给数值后来的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
        )

        # 计算线性冲突（如果启用）
        new_conflict = 0
        if self._use_linear_conflict:
            new_conflict = calculate_linear_conflict(
                new_pos2num,
                self._n,
                self._bits_per_number,
                self._mask,
                self._goal_rows,
                self._goal_cols,
                self._row_conflict_pairs,
                self._col_conflict_pairs,
            )

        new_node = backward_Node(
            f_num2pos=new_num2pos,
            f_pos2num=new_pos2num,
            f_parent=new_parent,
            f_g=new_g,
            f_h=new_h_manhattan,
            f_conflict=new_conflict,
        )
        return new_node

    # 用于生成某一节点的所有子节点，返回子节点的列表
    def generate_sub_nodes(self):
        result = []
        # 找到数字0所在的位置编号
        index_0 = get_pos(self.num2pos, 0, self._bits_per_number, self._mask)
        for motion in self._motion_table[index_0]:
            result.append(self.generate_one_sub_node(index_0, index_0 + motion))
        return result

    # 更改父节点函数
    def change_parent(self, parent):
        self.parent = parent
        self.g = parent.g + 1
        self.f = self.g + self.h

    # 用于排序
    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h


def get_path_forward(node: forward_Node):
    result = []
    # 添加自身(即终点)
    result.append(
        from_pos2num_to_list(node.pos2num, node._n, node._bits_per_number, node._mask)
    )
    while node.parent:
        # 添加至尾部以加快速度
        result.append(
            from_pos2num_to_list(
                node.parent.pos2num, node._n, node._bits_per_number, node._mask
            )
        )
        node = node.parent
    # 逆序输出
    return result[::-1]


def get_path_backward(node: backward_Node):
    result = []
    # 添加自身(即终点)
    result.append(
        from_pos2num_to_list(node.pos2num, node._n, node._bits_per_number, node._mask)
    )
    while node.parent:
        # 添加至尾部以加快速度
        result.append(
            from_pos2num_to_list(
                node.parent.pos2num, node._n, node._bits_per_number, node._mask
            )
        )
        node = node.parent
    # 正序输出
    return result[1::1]
