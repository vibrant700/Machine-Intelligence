import pickle
from collections import deque

# 参数
N = 4
SIZE = N * N                     # 16
GOAL_STATE = list(range(1, SIZE)) + [0]  # 目标状态 [1,2,...,15,0]

def encode_positions(positions):
    """将位置列表编码为整数（每个位置占4位）"""
    code = 0
    for i, pos in enumerate(positions):
        code |= pos << (4 * i)
    return code

def get_neighbors(pos):
    row, col = divmod(pos, N)
    neighbors = []
    if row > 0:   neighbors.append(pos - N)
    if row < N-1: neighbors.append(pos + N)
    if col > 0:   neighbors.append(pos - 1)
    if col < N-1: neighbors.append(pos + 1)
    return neighbors

def build_pattern_db(pattern_tiles):
    """
    构建独立模式数据库（忽略组外瓷砖）
    pattern_tiles: 列表，例如 [1,2,3,4,5,6,7]
    返回字典 {编码: 最小步数}
    """
    k = len(pattern_tiles)
    # 目标状态中模式瓷砖的位置
    goal_positions = [GOAL_STATE.index(tile) for tile in pattern_tiles]
    goal_code = encode_positions(goal_positions)

    db = {goal_code: 0}
    q = deque([goal_positions])

    while q:
        positions = q.popleft()
        cur_code = encode_positions(positions)
        cur_dist = db[cur_code]

        # 所有未被模式瓷砖占用的位置（空格可能出现的位置）
        occupied = set(positions)
        empty_spaces = [p for p in range(SIZE) if p not in occupied]

        for empty in empty_spaces:
            for nb in get_neighbors(empty):
                if nb in occupied:
                    idx = positions.index(nb)
                    new_positions = list(positions)
                    new_positions[idx] = empty
                    new_code = encode_positions(new_positions)
                    if new_code not in db:
                        db[new_code] = cur_dist + 1
                        q.append(new_positions)
    return db

if __name__ == "__main__":
    pattern = [1, 2, 3, 4, 5, 6, 7]   # 7个瓷砖
    print(f"Building pattern database for {pattern} ...")
    print("This may take a while (several minutes) and use ~200MB memory.")
    db = build_pattern_db(pattern)
    print(f"Database size: {len(db)} states")

    filename = f"pattern_db_{pattern[0]}_{pattern[-1]}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(db, f)
    print(f"Saved to {filename}")