import pickle
from collections import deque
from typing import List, Dict

class PatternDatabase:
    def __init__(self, n: int, pattern_tiles: List[int], goal_state: List[int]):
        self.n = n
        self.size = n * n
        self.pattern_tiles = pattern_tiles
        self.goal_state = goal_state
        self.bits_per_pos = (self.size - 1).bit_length()
        self.db: Dict[int, int] = {}

    def _encode(self, positions: List[int]) -> int:
        code = 0
        for i, pos in enumerate(positions):
            code |= pos << (self.bits_per_pos * i)
        return code

    def _neighbors(self, pos: int) -> List[int]:
        row, col = divmod(pos, self.n)
        nb = []
        if row > 0:   nb.append(pos - self.n)
        if row < self.n-1: nb.append(pos + self.n)
        if col > 0:   nb.append(pos - 1)
        if col < self.n-1: nb.append(pos + 1)
        return nb

    def build(self):
        goal_positions = [self.goal_state.index(tile) for tile in self.pattern_tiles]
        goal_code = self._encode(goal_positions)
        self.db = {goal_code: 0}
        q = deque([goal_positions])

        while q:
            positions = q.popleft()
            cur_code = self._encode(positions)
            cur_dist = self.db[cur_code]

            occupied = set(positions)
            empty_spaces = [p for p in range(self.size) if p not in occupied]

            for empty in empty_spaces:
                for nb in self._neighbors(empty):
                    if nb in occupied:
                        idx = positions.index(nb)
                        new_positions = list(positions)
                        new_positions[idx] = empty
                        new_code = self._encode(new_positions)
                        if new_code not in self.db:
                            self.db[new_code] = cur_dist + 1
                            q.append(new_positions)

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self.db, f)


if __name__ == "__main__":
    N = 4
    GOAL = list(range(1, N*N)) + [0]   # [1,2,...,15,0]

    groups = [
        ([1,2,3,4,5], "pattern_db_1_5.pkl"),
        ([6,7,8,9,10], "pattern_db_6_10.pkl"),
        ([11,12,13,14,15], "pattern_db_11_15.pkl")
    ]

    for tiles, filename in groups:
        print(f"Building database for {tiles} ...")
        pdb = PatternDatabase(N, tiles, GOAL)
        pdb.build()
        print(f"  Size: {len(pdb.db)} states")
        pdb.save(filename)
        print(f"  Saved to {filename}")