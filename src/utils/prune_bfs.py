import chess
import time
from typing import List, Optional, Tuple

BRANCHING_BFS: List[int] = [
    5,  # a1
    10,  # a2
    15,  # a3
    20,  # a4
    20,  # a5
    20,  # a6
    20,  # a7
    20,  # a8
    20,  # a9
    20,  # a10
]
MAX_DEPTH = 2

def _per_move_time_cap(time_left: float) -> float:
    if time_left >= 300.0:
        return 2.0
    if time_left >= 60.0:
        return 1.5
    if time_left >= 30.0:
        return 1.0
    if time_left >= 10.0:
        return 0.5
    if time_left >= 2.0:
        return 0.2
    return 0.05

def bfs(rem, depth, color, net, eval_fun):
    if depth == MAX_DEPTH:
        best_ret = -1e9
        for _, _, bd in rem[-1]:
            for mv in bd.legal_moves:
                bd.push(mv)
                gb = eval_fun(bd.fen(), net)
                bd.pop()
                best_ret = max(best_ret, color * gb)
        return best_ret
    
    srt = []
    for _, _, bd in rem[-1]:
        for mv in bd.legal_moves:
            bd.push(mv)
            gb = eval_fun(bd.fen(), net)
            srt.append((mv, gb*color, bd.copy()))
            bd.pop()
    srt.sort(key=lambda x: x[1], reverse=True)
    rem.append(srt[:BRANCHING_BFS[depth]])
    return bfs(rem, depth+1, -color, net, eval_fun)

# color is +1 for white, -1 for black
def bfs_next_move(board: chess.Board, time_left: float, color, net, eval_fun) -> Optional[chess.Move]:
    # per_move_cap = _per_move_time_cap(time_left)
    # time_budget = min(2.0, per_move_cap)
    # start_time = time.perf_counter()
    # node_limit = 20000
    # nodes = {"count": 0}

    moves = list(board.legal_moves) # try rm list wrap

    best_move, best_ret = "", -1e9
    for mv in moves:
        board.push(mv)
        srt = []
        for mv2 in board.legal_moves:
            board.push(mv2)
            gb = eval_fun(board.fen(), net)
            srt.append((mv2, gb*color, board.copy()))
            board.pop()
        srt.sort(key=lambda x: x[1], reverse=True)
        gb = bfs([srt[:BRANCHING_BFS[0]]], 1, -color, net, eval_fun)
        print(f"Move: {mv}, Eval: {gb*color}")
        if gb*color > best_ret:
            best_ret = gb*color
            best_move = mv
        board.pop()
    
    return best_move