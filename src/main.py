from .utils import chess_manager, GameContext
import chess
import math
import torch
import numpy as np
from .utils.model import EvalNet
from .utils.sir_stalemate import next_move as sm_next_move

PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path="src/utils/most_recent2_slightly_better.pt"):
    net = EvalNet().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

def fen_to_tensor(fen):
    board = chess.Board(fen)

    planes = torch.zeros((14, 8, 8), dtype=torch.float32)  # avoid numpy allocation + conversion

    # White = 0–5, Black = 6–11
    for sq, pc in board.piece_map().items():
        p = PIECE_MAP[pc.piece_type] + (0 if pc.color else 6)
        r, c = divmod(sq, 8)
        planes[p, 7 - r, c] = 1

    # Side to move plane
    planes[12].fill_(1 if board.turn == chess.WHITE else 0)

    # Castling rights plane
    # (could be simplified by writing directly)
    planes[13].zero_()
    if board.has_kingside_castling_rights(True):  planes[13, 0, 7] = 1
    if board.has_queenside_castling_rights(True): planes[13, 0, 0] = 1
    if board.has_kingside_castling_rights(False): planes[13, 7, 7] = 1
    if board.has_queenside_castling_rights(False): planes[13, 7, 0] = 1

    return planes

def evaluate_position(fen, net=None):
    if net is None:
       net = load_model("src/utils/most_recent2_slightly_better.pt")

    t = fen_to_tensor(fen).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = net(t).item()
    return score

net = load_model("src/utils/most_recent2_slightly_better.pt")

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    return sm_next_move(ctx.board, min(8, ctx.timeLeft), 1 if ctx.board.turn else -1, net, evaluate_position)

@chess_manager.reset
def reset_func(ctx: GameContext):
    pass