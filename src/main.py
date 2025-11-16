from .utils import chess_manager, GameContext
import chess
import math
import torch
import numpy as np
from .utils.model import EvalNet
from .utils.zetalambda import next_move as zl_next_move
from .utils.chess_game import fen_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(path="src/utils/most_recent3.pt"):
    net = EvalNet().to(DEVICE)
    checkpoint = torch.load(path, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net

def evaluate_position(fen, net=None):
    if net is None:
       net = load_model()

    t = fen_to_tensor(fen).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = net(t).item()
    return score

net = load_model()

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    return zl_next_move(ctx.board, min(8.0, ctx.timeLeft), 1 if ctx.board.turn else -1, net, evaluate_position)

@chess_manager.reset
def reset_func(ctx: GameContext):
    pass