import time
import chess
import math

ops_left = 0

# Piece values in centipawns
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0  # King value irrelevant for evaluation (infinite value)
}

# Simple piece-square tables to add positional value (only pawns, knights, bishops for demo)

PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 27, 27, 10, 5, 5,
    0, 0, 0, 25, 25, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -25, -25, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0
]

KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50
]

BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20
]

# Helper to get piece-square value with perspective
def piece_square_value(piece, square):
    if piece.piece_type == chess.PAWN:
        table = PAWN_TABLE
    elif piece.piece_type == chess.KNIGHT:
        table = KNIGHT_TABLE
    elif piece.piece_type == chess.BISHOP:
        table = BISHOP_TABLE
    else:
        return 0
    # Flip table for black
    if piece.color == chess.WHITE:
        return table[square]
    else:
        return table[chess.square_mirror(square)]

def is_isolated_pawn(board, square):
    # Check if pawn at square is isolated (no friendly pawns on adjacent files)
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    color = board.piece_at(square).color
    for adj_file in [file - 1, file + 1]:
        if 0 <= adj_file <= 7:
            for r in range(8):
                sq = chess.square(adj_file, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    return False
    return True

def count_attacks_on_square(board, square, color):
    # Count number of opponent pieces attacking square
    attackers = board.attackers(not color, square)
    return len(attackers)

def primitive_fast_heuristic(board: chess.Board):
    """
    Returns an approximate evaluation of the position in centipawns.
    Positive means advantage to White, negative to Black.
    Combines material, piece-square-related positional bonuses,
    mobility, pawn structure, and basic king safety heuristics.
    """
    material_score = 0
    pos_score = 0
    mobility_score = 0
    pawn_structure_score = 0
    king_safety_score = 0

    # Material and positional
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            val = PIECE_VALUES[piece.piece_type]
            sign = 1 if piece.color == chess.WHITE else -1
            material_score += sign * val

            # Add simple piece-square bonus for pawns, knights, bishops
            pos_score += sign * piece_square_value(piece, square)

    # Mobility: number of legal moves
    mobility_score = 10 * (board.legal_moves.count() if hasattr(board.legal_moves, 'count') else len(list(board.legal_moves)))
    if board.turn == chess.BLACK:
        mobility_score = -mobility_score

    # Pawn structure: penalize isolated pawns (-15 cp each)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN:
            if is_isolated_pawn(board, square):
                pawn_structure_score += (-15 if piece.color == chess.WHITE else 15)

    # King safety: penalize if king is attacked or open file near king (-20 cp)
    for color in [chess.WHITE, chess.BLACK]:
        king_square = board.king(color)
        if king_square is not None:
            attacks = count_attacks_on_square(board, king_square, color)
            penalty = attacks * 20
            if color == chess.WHITE:
                king_safety_score -= penalty
            else:
                king_safety_score += penalty

    total = material_score + pos_score + mobility_score + pawn_structure_score + king_safety_score
    return total


start_time = 0
cutoff = 0

def minimax(board, depth, alpha, beta, maximizing_player, net, eval_fun, should_print=False):
    global ops_left
    global start_time
    global cutoff
    # print(ops_left)

     # if ran out of operations or short on time, just use a cheap static evaluation
    if (start_time + 1000000000 - time.time_ns()) // 1000000 < cutoff:
        if should_print:
            best_eval, best_move = -math.inf, None
            for mv in board.legal_moves:
                board.push(mv)
                curr_eval = primitive_fast_heuristic(board)
                if curr_eval > best_eval:
                    best_eval, best_move = curr_eval, mv
                board.pop()
            return best_eval, best_move
        return primitive_fast_heuristic(board)
    

    if depth == 0 or board.is_game_over():
        fen = board.fen()
        eval_score = eval_fun(fen, net)
        return eval_score

    if maximizing_player:
        max_eval = -float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, net, eval_fun, False)
            board.pop()

            if eval > max_eval:
                best_move = move
                max_eval = eval

            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off because Black will never allow this position
        if should_print:
            return max_eval, best_move
        return max_eval
    else:
        min_eval = float('inf')
        best_move = None
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, net, eval_fun, False)
            board.pop()

            if eval < min_eval:
                best_move = move
                min_eval = eval

            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off because White will never allow this position
        if should_print:
            return min_eval, best_move
        return min_eval

# takes in a board, time left, and side to play - returns a chess.move
def next_move(board, time_left, color, net, eval_fun):
    global ops_left
    global start_time
    global cutoff

    # hard code logic for # of operations left
    if time_left > 5.0:
        cutoff = 70
    elif time_left > 2.0:
        cutoff = 500
    else: 
        cutoff = 1000

    # 7 seconds for 1000 static evals

    start_time = time.time_ns()
    best_eval, best_move = minimax(board, 2, -float('inf'), float('inf'), color == 1, net, eval_fun, should_print=True)
    return best_move