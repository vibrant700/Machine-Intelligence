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
# 对于3*3的情况,由于不管是竖直方向还是水平方向移动空格，逆序数的奇偶性都不会发生改变，因此可以这么做
def determine_solvability(f_input, f_goal):
    input_inversion_number = calculate_inversion_number(f_input)
    goal_inversion_number = calculate_inversion_number(f_goal)
    if input_inversion_number % 2 == goal_inversion_number % 2:
        return True
    else:
        return False


# 预计算哈密顿距离表
# 返回值为一个2维数组,list[数字][数字位置编号] = 该数字对应的哈密顿距离
def precalculate_hamilton_table(f_goal):
    hamilton_table = []
    for i in range(1, 9):
        temp = f_goal.index(i)
        goal_row_index = temp // 3
        goal_col_index = temp % 3
        temp_list = []
        for j in range(9):
            current_row_index = j // 3
            current_col_index = j % 3
            temp_list.append(
                abs(goal_row_index - current_row_index)
                + abs(goal_col_index - current_col_index)
            )
        hamilton_table.append(temp_list)
    return hamilton_table


# 用于将列表转换为数字-位置2进制数
def from_list_to_num2pos(f_list):
    num2pos = 0
    for pos, num in enumerate(f_list):
        num2pos |= pos << (4 * num)
    return num2pos


# 用于将列表转换为位置-数字2进制数
def from_list_to_pos2num(f_list):
    pos2num = 0
    for pos, num in enumerate(f_list):
        pos2num |= num << (4 * pos)
    return pos2num


def from_pos2num_to_list(f_pos2num):
    result = [0] * 9
    for i in range(9):
        num = get_digit(f_pos2num, i)
        result[i] = num
    return result


# 一些经过封装的二进制操作(目的是代码可读性)
# 从位置-数字数中获取位置p的数字
def get_digit(f_pos2num, p):
    return (f_pos2num >> (4 * p)) & 0xF


# 从数字-位置数中获取数字d的位置
def get_pos(f_num2pos, d):
    return (f_num2pos >> (4 * d)) & 0xF


# 给定原数以及与0交换的位置编号，返回变化后的数
# p0为0所在的位置的编号
# p1为即将于p0交换的位置的编号
def move(f_num2pos, f_pos2num, f_p0, f_p1):
    MASK = 0xF  # 即0b1111
    p1_num = get_digit(f_pos2num, f_p1)

    # 更新数字-位置数
    new_num2pos = f_num2pos
    # 将数字0对应的位置清空(在数字-位置数中，第1-4就对应着数字0)
    new_num2pos &= ~MASK
    # 将数字0对应的位置置为p1
    new_num2pos |= f_p1
    # 将p1_num对应的位置清空
    new_num2pos &= ~(MASK << (4 * p1_num))
    # 将p1_num对应的位置置为p0
    new_num2pos |= f_p0 << (4 * p1_num)

    # 更新位置-数字数
    new_pos2num = f_pos2num
    # 将p0对应的数字清空
    new_pos2num &= ~(MASK << (4 * f_p0))
    # 将p0处的位置置为p1_num(即p1处的数字)
    new_pos2num |= p1_num << (4 * f_p0)
    # 将p1对应的数字变为0(即p0处的数字0)
    new_pos2num &= ~(MASK << (4 * f_p1))

    # 返回结果
    return new_num2pos, new_pos2num


# 一个节点(即一个状态)对象，应该记录其num2pos、pos2num、父节点、g、h、f
class single_Node:
    # 设置类属性，所有实例共享
    # 使用前赋值即可
    # 预计算的哈密顿距离表
    _hamilton_table = None
    # 预计算(手动计算，无需每次运行时计算)的可行移动表
    _motion_table = (
        (1, 3),  # 位置0,可以右移、下移
        (1, -1, 3),  # 位置1,可以右移、左移、下移
        (-1, 3),  # 位置2,可以左移、下移
        (1, 3, -3),  # 位置3,可以右移、下移、上移
        (1, -1, 3, -3),  # 位置4,可以右移、左移、下移、上移
        (-1, 3, -3),  # 位置5,可以左移、下移、上移
        (1, -3),  # 位置6,可以右移、上移
        (1, -1, -3),  # 位置7,可以右移、左移、上移
        (-1, -3),  # 位置8,可以左移、上移
    )

    # 初始化一个节点
    def __init__(self, f_num2pos, f_pos2num, f_parent, f_g=0, f_h=0):
        self.num2pos = f_num2pos
        self.pos2num = f_pos2num
        self.parent = f_parent
        self.g = f_g
        self.h = f_h
        self.f = self.g + self.h

    def generate_one_sub_node(self, f_p0, f_p1):
        new_num2pos, new_pos2num = move(self.num2pos, self.pos2num, f_p0, f_p1)
        new_parent = self
        new_g = self.g + 1
        p1_num = get_digit(self.pos2num, f_p1)
        new_h = (
            self.h
            - self._hamilton_table[p1_num - 1][f_p1]
            # 减去该数字原先的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
            + self._hamilton_table[p1_num - 1][f_p0]
            # 加上给数值后来的哈密顿距离。索引-1的原因是哈密顿距离表中未存储0相关距离，0位置记录的是1
        )
        new_node = single_Node(
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
        index_0 = get_digit(self.num2pos, 0)
        for motion in self._motion_table[index_0]:
            result.append(
                self.generate_one_sub_node(index_0, index_0 + motion)
            )
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


def get_path(node: single_Node):
    result = []
    # 添加自身(即终点)
    result.append(from_pos2num_to_list(node.pos2num))
    while node.parent:
        # 添加至尾部以加快速度
        result.append(from_pos2num_to_list(node.parent.pos2num))
        node = node.parent
    # 逆序输出
    return result[::-1]
