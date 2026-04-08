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
    _pattern_dbs = []

    # 设置类属性，需要在创建第一个节点前调用
    @classmethod
    def set_puzzle_params(cls, n, hamilton_table,pattern_dbs=None):
        cls._n = n
        cls._bits_per_number = get_bits_per_number(n)
        cls._mask = get_mask(cls._bits_per_number)
        cls._motion_table = generate_motion_table(n)
        cls._hamilton_table = hamilton_table
        cls._pattern_dbs = pattern_dbs if pattern_dbs else []

    # 初始化一个节点
    def __init__(self, f_num2pos, f_pos2num, f_parent, f_g=0, f_h=0):
        self.num2pos = f_num2pos
        self.pos2num = f_pos2num
        self.parent = f_parent
        self.g = f_g
        if f_h == 0:
            # 自动计算模式距离（三个数据库之和）
            state_list = from_pos2num_to_list(f_pos2num, self.__class__._n,
                                              self.__class__._bits_per_number, self.__class__._mask)
            h = 0
            bits_per_pos = (self.__class__._n * self.__class__._n - 1).bit_length()
            for tiles, db in self.__class__._pattern_dbs:
                # 提取这些瓷砖在当前状态中的位置
                positions = [state_list.index(tile) for tile in tiles]
                # 编码
                code = 0
                for i, p in enumerate(positions):
                    code |= p << (bits_per_pos * i)
                h += db.get(code, 0)
            self.h = h
        else:
            self.h = f_h
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
        new_g = self.g + 1
        # 新节点的 h 全量计算（三个数据库之和）
        new_state_list = from_pos2num_to_list(new_pos2num, self.__class__._n,
                                              self.__class__._bits_per_number, self.__class__._mask)
        h = 0
        bits_per_pos = (self.__class__._n * self.__class__._n - 1).bit_length()
        for tiles, db in self.__class__._pattern_dbs:
            positions = [new_state_list.index(tile) for tile in tiles]
            code = 0
            for i, p in enumerate(positions):
                code |= p << (bits_per_pos * i)
            h += db.get(code, 0)
        return forward_Node(new_num2pos, new_pos2num, self, new_g, h)
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

    # 设置类属性，需要在创建第一个节点前调用
    @classmethod
    def set_puzzle_params(cls, n, hamilton_table):
        cls._n = n
        cls._bits_per_number = get_bits_per_number(n)
        cls._mask = get_mask(cls._bits_per_number)
        cls._motion_table = generate_motion_table(n)
        cls._hamilton_table = hamilton_table

    # 初始化一个节点
    def __init__(self, f_num2pos, f_pos2num, f_parent, f_g=0, f_h=0):
        self.num2pos = f_num2pos
        self.pos2num = f_pos2num
        self.parent = f_parent
        self.g = f_g
        self.h = f_h
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
        new_h = (
            self.h
            - self._hamilton_table[p1_num - 1][f_position1]
            # 减去该数字原先的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
            + self._hamilton_table[p1_num - 1][f_position0]
            # 加上给数值后来的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
        )
        new_node = backward_Node(
            f_num2pos=new_num2pos,
            f_pos2num=new_pos2num,
            f_parent=new_parent,
            f_g=new_g,
            f_h=new_h,
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