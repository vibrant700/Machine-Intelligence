"""
Flask 后端服务器，为前端提供 8 数码问题求解 API
"""

import os
import pickle
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
parent_dir = current_dir.parent
src_dir = parent_dir / "src"
sys.path.insert(0, str(src_dir))

from flask import Flask, jsonify, request
from flask_cors import CORS
from Linear import function, main
from Pattern import main2

app = Flask(__name__)
CORS(app)


@app.route("/api/puzzle/solve", methods=["POST"])
def solve_puzzle():
    try:
        data = request.get_json()

        size = data.get("size")
        algorithm = data.get("algorithm", "linear_conflict")
        start = data.get("start")
        goal = data.get("goal")

        if algorithm == "pattern_db" and size != 4:
            algorithm = "linear_conflict"

        # 检查目标状态是否为标准目标（0在最后）
        # 模式数据库只适用于标准目标状态，因为模式数据库是基于标准目标构建的
        standard_goal = list(range(1, size * size)) + [0]
        if algorithm == "pattern_db" and goal != standard_goal:
            print(
                "[INFO] 目标状态非标准（模式数据库仅支持标准目标），切换到线性冲突算法"
            )
            algorithm = "linear_conflict"

        if not function.determine_solvability(start, goal, size):
            return jsonify({"error": "无解：初始状态无法到达目标状态"}), 400

        states = []

        # 如果初始状态就是目标状态，直接返回
        if start == goal:
            return jsonify({"states": [start]})

        if algorithm == "pattern_db" and size == 4:

            pattern_files = [
                ([1, 2, 3, 4, 5], str(src_dir / "Pattern\\pattern_db_1_5.pkl")),
                ([6, 7, 8, 9, 10], str(src_dir / "Pattern\\pattern_db_6_10.pkl")),
                ([11, 12, 13, 14, 15], str(src_dir / "Pattern\\pattern_db_11_15.pkl")),
            ]

            pattern_dbs = []
            for tiles, filepath in pattern_files:
                if os.path.exists(filepath):
                    with open(filepath, "rb") as f:
                        db = pickle.load(f)
                        pattern_dbs.append((tiles, db))

            if len(pattern_dbs) == 3:
                try:
                    forward_path, backward_path = main2.solve_with_custom_goal(
                        start, goal, size, pattern_dbs
                    )

                    if forward_path is None or backward_path is None:
                        return jsonify({"error": "求解失败"}), 500

                    if not forward_path or not backward_path:
                        return jsonify({"error": "求解失败：路径为空"}), 500

                    # 使用与线性冲突相同的路径合并方式
                    full_path = forward_path + backward_path

                    for state_list in full_path:
                        states.append(list(state_list))
                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    return jsonify({"error": f"模式数据库求解错误: {str(e)}"}), 500
            else:
                return jsonify({"error": "模式数据库文件缺失，需要3个.pkl文件"}), 500
        else:
            node_1, node_2 = main.solve_8_digital_problem(
                f_input=start, f_goal=goal, f_n=size, use_linear_conflict=True
            )

            if node_1 is None or node_2 is None:
                return jsonify({"error": "求解失败"}), 500

            path_forward = function.get_path_forward(node_1)
            path_backward = function.get_path_backward(node_2)
            full_path = path_forward + path_backward

            for state_list in full_path:
                states.append(list(state_list))

        return jsonify({"states": states})

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": f"服务器错误: {str(e)}"}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "8-Puzzle Solver API is running"})


if __name__ == "__main__":
    # 只在第一次启动时打印信息（避免Flask debug模式重复重启导致重复输出）

    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        print("启动 8 数码问题求解服务器...")
        print("前端页面: http://localhost:8080")
        print("API 接口: http://localhost:5000/api/puzzle/solve")
        print("健康检查: http://localhost:5000/api/health")
        print()

    app.run(host="0.0.0.0", port=5000, debug=True)
