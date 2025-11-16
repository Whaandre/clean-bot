import time
import chess

# Constants for extreme terminal conditions
STALEMATE_REWARD = 1e9
MATE_PENALTY = -1e9

start_time = 0
cutoff = 0


def evaluate_with_zero_goal(fen: str, net, board: chess.Board, eval_fun) -> float:
    """
    Returns evaluation with goal to minimize abs(eval).
    Overrides for terminal states:
      - stalemate: very high positive (best)
      - checkmate: very high negative (worst)
    Otherwise uses eval_fun output and returns -abs(eval).
    """
    if board.is_stalemate():
        return STALEMATE_REWARD
    if board.is_checkmate():
        return MATE_PENALTY

    val = eval_fun(fen, net)
    return -abs(val)


def negamax(board: chess.Board, depth: int, alpha: float, beta: float,
            net, eval_fun, should_print=False) -> tuple:
    """
    Negamax search with nanosecond cutoff logic.
    Returns: (best_move, score)
    """
    global start_time
    global cutoff

    # Time cutoff check using nanosecond cutoff logic
    if (start_time + 1_000_000_000 - time.time_ns()) // 1_000_000 < cutoff:
        return None, evaluate_with_zero_goal(board.fen(), net, board, eval_fun)

    if depth == 0 or board.is_game_over():
        return None, evaluate_with_zero_goal(board.fen(), net, board, eval_fun)

    best_move = None
    best_score = -float('inf')

    for move in board.legal_moves:
        board.push(move)
        _, score = negamax(board, depth - 1, -beta, -alpha, net, eval_fun, False)
        score = -score
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)
        if alpha >= beta:
            break

        # Extra check inside loop for time cutoff (redundant but safe)
        if (start_time + 1_000_000_000 - time.time_ns()) // 1_000_000 < cutoff:
            break

    return best_move, best_score


def next_move(board: chess.Board, time_left: float, color: int, net, eval_fun) -> chess.Move:
    """
    Returns the next move to play, trying to minimize absolute eval to approach stalemate.
    time_left: time remaining in seconds.
    color: side to move (chess.WHITE or chess.BLACK).
    net: your neural net model, passed to eval_fun.
    eval_fun: function taking (fen:str, net) and returning centipawn eval.

    Uses nanosecond cutoff logic with cutoff values based on time_left.
    """

    global start_time
    global cutoff

    if time_left > 5.0:
        cutoff = 70
    elif time_left > 2.0:
        cutoff = 500
    else:
        cutoff = 1000

    start_time = time.time_ns()

    best_move, best_score = negamax(
        board,
        depth=2,
        alpha=-float('inf'),
        beta=float('inf'),
        net=net,
        eval_fun=eval_fun,
        should_print=True
    )

    if best_move is None:
        # fallback if no move found or timeout
        moves = list(board.legal_moves)
        if moves:
            return moves[0]
        else:
            return None

    return best_move
