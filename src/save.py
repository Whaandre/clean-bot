# alphabeta_engine_debug.py
import chess
import time
from typing import Optional, List

class TTEntry:
    EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2
    __slots__ = ['key', 'depth', 'score', 'flag', 'best_move', 'age']
    def __init__(self, key=0, depth=0, score=0, flag=0, best_move=None, age=0):
        self.key, self.depth, self.score, self.flag, self.best_move, self.age = key, depth, score, flag, best_move, age

class AlphaBetaEngine:
    INF = 999_999
    MATE_SCORE = 100_000
    MAX_DEPTH = 20
    MAX_KILLER = 2

    def __init__(self, evaluate_fn, net, tt_size_mb: float = 64.0):
        self.evaluate_fn = evaluate_fn
        self.net = net

        # Transposition Table
        self.tt_size = int((tt_size_mb * 1024 * 1024) / 64)
        self.tt = [None] * self.tt_size
        self.tt_age = 0

        self.killer_moves = [[None]*self.MAX_KILLER for _ in range(self.MAX_DEPTH)]
        self.nodes = 0
        self._alphabeta_best_move = None

    # ---------------- TT Methods ----------------
    def _tt_probe(self, board, depth, alpha, beta):
        key = board._transposition_key()
        idx = hash(key) % self.tt_size
        entry = self.tt[idx]
        if entry and entry.key == key:
            if entry.depth >= depth:
                score = entry.score
                if entry.flag == TTEntry.EXACT:
                    return score, entry.best_move
                if entry.flag == TTEntry.LOWERBOUND and score >= beta:
                    return score, entry.best_move
                if entry.flag == TTEntry.UPPERBOUND and score <= alpha:
                    return score, entry.best_move
        return None, entry.best_move if entry else None

    def _tt_store(self, board, depth, score, flag, best_move):
        key = board._transposition_key()
        idx = hash(key) % self.tt_size
        entry = self.tt[idx]
        if not entry or entry.age < self.tt_age or (entry.key == key and entry.depth <= depth):
            self.tt[idx] = TTEntry(key, depth, score, flag, best_move, self.tt_age)

    def _order_moves(self, board, moves: list[chess.Move], tt_move, ply):
        """Order moves for alpha-beta search with TT move, captures, promotions, and killer heuristics"""
        def score(m):
            s = 0
            # Prioritize TT move
            if tt_move and m == tt_move:
                return 10_000_000
            # Capture scoring (MVV-LVA)
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                if victim and attacker:
                    s += 1_000_000 + victim.piece_type*10 - attacker.piece_type
            # Promotion
            if m.promotion:
                s += 900_000
            # Killer moves (only if ply within table)
            if ply < self.MAX_DEPTH:
                for i, k in enumerate(self.killer_moves[ply]):
                    if k and m == k:
                        s += 500_000 - i*1000
            return s

        return sorted(moves, key=score, reverse=True)

    # ---------------- Evaluation ----------------
    def _eval_cp(self, board):
        if board.is_checkmate(): return -self.MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material(): return 0
        val = self.evaluate_fn(board.fen(), self.net)
        return int(val) if isinstance(val, (int, float)) else 0

    # ---------------- Quiescence Search ----------------
    def _quiescence(self, board, alpha, beta, ply):
        """Quiescence search: only consider captures, checks, promotions"""
        self.nodes += 1
        stand_pat = self._eval_cp(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Only captures, promotions, or moves giving check
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion or board.gives_check(m)]
        for m in self._order_moves(board, moves, None, ply):
            board.push(m)
            score = -self._quiescence(board, -beta, -alpha, ply+1)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # ---------------- Alpha-Beta ----------------
    def _alphabeta(self, board, depth, alpha, beta, ply):
        self.nodes += 1
        if depth <= 0: 
            return self._quiescence(board, alpha, beta, ply)

        original_alpha = alpha
        tt_score, tt_move = self._tt_probe(board, depth, alpha, beta)
        if tt_score is not None: 
            return tt_score
        if tt_move not in board.legal_moves:
            tt_move = None

        best_score, best_move = -self.INF, None
        moves = self._order_moves(board, list(board.legal_moves), tt_move, ply)

        for move in moves:
            board.push(move)
            score = -self._alphabeta(board, depth-1, -beta, -alpha, ply+1)
            board.pop()

            if score > best_score:
                best_score, best_move = score, move
                if ply == 0:
                    self._alphabeta_best_move = move

            if score > alpha: alpha = score
            if alpha >= beta:
                if not board.is_capture(move) and not move.promotion:
                    self.killer_moves[ply][1] = self.killer_moves[ply][0]
                    self.killer_moves[ply][0] = move
                break

        # TT flag
        if best_score <= original_alpha:
            flag = TTEntry.UPPERBOUND
        elif best_score >= beta:
            flag = TTEntry.LOWERBOUND
        else:
            flag = TTEntry.EXACT
        if best_move:
            self._tt_store(board, depth, best_score, flag, best_move)

        return best_score

    # ---------------- Iterative Deepening ----------------
    def _iterative_deepening(self, board, max_depth, time_limit):
        self._alphabeta_best_move = None
        start_time = time.time()
        for depth in range(1, max_depth+1):
            if time.time() - start_time > time_limit:
                break
            score = self._alphabeta(board, depth, -self.INF, self.INF, 0)
            print(f"[Depth {depth}] Score: {score}, BestMove: {self._alphabeta_best_move}")
            if abs(score) >= self.MATE_SCORE - 10:
                break
            # Assign a move even if time runs out
            if self._alphabeta_best_move is None and board.legal_moves:
                self._alphabeta_best_move = next(iter(board.legal_moves))
        return self._alphabeta_best_move

    # ---------------- Time Allocation ----------------
    def _allocate_time(self, board, remaining_time):
        num_pieces = len(board.piece_map())
        if num_pieces > 20:        # opening
            return remaining_time * 0.03
        elif num_pieces > 10:      # middlegame
            return remaining_time * 0.07
        else:                      # endgame
            return remaining_time * 0.10

    # ---------------- Public Method ----------------
    def search_move(self, board, remaining_time, verbose=False) -> Optional[chess.Move]:
        if not board.legal_moves:
            return None

        self.nodes = 0
        self.tt_age += 1
        time_allowed = self._allocate_time(board, remaining_time)
        print(f"Time allocated for this move: {time_allowed:.2f} seconds")
        move = self._iterative_deepening(board, self.MAX_DEPTH, time_allowed)

        if move not in board.legal_moves:
            move = next(iter(board.legal_moves), None)

        if verbose:
            print(f"Selected Move: {move}, Nodes Searched: {self.nodes}")
        return move
