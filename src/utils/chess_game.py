# chess_game.py
import chess
import chess.svg
import torch
import numpy as np
import torch
import chess

# Predefine piece_map once globally
PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def attacked_squares(board, color):
    attacked = 0
    for square, piece in board.piece_map().items():
        if piece.color == color:
            attacked |= board.attacks_mask(square)
    return attacked
import chess
import torch

PIECE_MAP = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

def attacked_squares(board, color):
    attacked_bb = 0
    for square, piece in board.piece_map().items():
        if piece.color == color:
            attacked_bb |= board.attacks_mask(square)
    return attacked_bb

def bitboard_to_tensor(bitboard):
    bits = bin(bitboard)[2:].zfill(64)
    bits = bits[::-1]  # a1 is bit 0
    arr = [int(b) for b in bits]
    matrix = torch.tensor(arr, dtype=torch.float32).view(8, 8)
    return torch.flip(matrix, [0])  # rank 8 at top (row 0)


def fen_to_tensor(fen):
    board = chess.Board(fen)
    planes = torch.zeros((18, 8, 8), dtype=torch.float32)  # updated to 18 planes

    # PIECES planes 0-11
    for sq, pc in board.piece_map().items():
        base = PIECE_MAP[pc.piece_type]
        if not pc.color:
            base += 6
        r, c = divmod(sq, 8)
        planes[base, 7 - r, c] = 1.0

    # Side to move plane 12
    planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)

    # Castling rights plane 13
    p13 = planes[13]
    if board.has_kingside_castling_rights(chess.WHITE):
        p13[7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        p13[7, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        p13[0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        p13[0, 0] = 1.0

    # Attack planes 14 (white attacks) and 15 (black attacks)
    w_attacks = attacked_squares(board, chess.WHITE)
    b_attacks = attacked_squares(board, chess.BLACK)

    planes[14] = bitboard_to_tensor(w_attacks)
    planes[15] = bitboard_to_tensor(b_attacks)

    # Occupied squares bitboards
    white_occupied = 0
    black_occupied = 0
    for sq, pc in board.piece_map().items():
        if pc.color == chess.WHITE:
            white_occupied |= 1 << sq
        else:
            black_occupied |= 1 << sq

    # Plane 16: white occupied squares attacked by black
    white_attacked_by_black = white_occupied & b_attacks
    planes[16] = bitboard_to_tensor(white_attacked_by_black)

    # Plane 17: black occupied squares attacked by white
    black_attacked_by_white = black_occupied & w_attacks
    planes[17] = bitboard_to_tensor(black_attacked_by_white)

    return planes

def bitboard_to_tensor(bitboard):
    bits = bin(bitboard)[2:].zfill(64)   # 64-bit binary string
    bits = bits[::-1]                     # Reverse to align a1 = bit 0
    arr = [int(b) for b in bits]
    matrix = torch.tensor(arr, dtype=torch.float32).view(8, 8)
    return torch.flip(matrix, [0])

def board_to_tensor(board: chess.Board):
    """
    Converts a python-chess board to a (18,8,8) tensor.
    """

    planes = np.zeros((14, 8, 8), dtype=np.float32)

    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # White pieces 0–5, black pieces 6–11
    for square, piece in board.piece_map().items():
        plane = piece_map[piece.piece_type] + (0 if piece.color else 6)
        r, c = divmod(square, 8)
        planes[plane, 7 - r, c] = 1

    board.attackers_mask

    # Side to move plane
    planes[12] = np.ones((8, 8)) if board.turn == chess.WHITE else np.zeros((8, 8))

    # Repetition or castling rights plane (optional)
    planes[13] = np.zeros((8, 8))
    if board.has_kingside_castling_rights(True): planes[13][0][7] = 1
    if board.has_queenside_castling_rights(True): planes[13][0][0] = 1
    if board.has_kingside_castling_rights(False): planes[13][7][7] = 1
    if board.has_queenside_castling_rights(False): planes[13][7][0] = 1

    return torch.tensor(planes)


ALL_MOVES = list(range(4672))  # fixed move encoding


def move_to_index(move: chess.Move):
    """
    Converts a move to an index 0..4671
    Format: from_square * 73 + move_type
    You can pick any encoding as long as it's consistent.
    """
    return move.from_square * 64 + move.to_square  # 4096 moves + promotions etc.


def index_to_move(idx: int, board: chess.Board):
    """
    Reverse of move_to_index.
    """
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)


# just saving this fen as an svg
def fen_to_svg(fen, filename="board.svg"):
    board = chess.Board(fen)
    svg_content = chess.svg.board(board=board)

    with open("./images/" + filename, "w") as f:
        f.write(svg_content)
    
    print(f"saved fen ${fen} into ${filename}")

import torch
import chess

def bitboard_to_matrix(bitboard):
    # bitboard: int representing 64-bit bitboard
    
    # Convert to 64-length binary string, padded with zeros
    bits = bin(bitboard)[2:].zfill(64)
    
    # Convert string to list of ints (0/1)
    bits_list = [int(b) for b in bits]
    
    # bits_list[0] is the highest bit → corresponds to square h8
    # But bit 0 in bitboard is a1, so bits_list is reversed
    bits_list.reverse()  # Now bits_list[0] is a1, bits_list[63] is h8
    
    # Reshape into 8x8 matrix
    matrix = [bits_list[i*8:(i+1)*8] for i in range(8)]  # rank 1 at index 0
    
    # Flip vertically so rank 8 is row 0, rank 1 is row 7 (if you want that)
    matrix.reverse()
    
    return matrix

def tensor_to_board(tensor):
    """
    Convert a tensor (14x8x8) representing a chess board to a `chess.Board` object.
    """
    board = chess.Board(None)

    # Extract the piece positions from the tensor
    piece_map = {}

    # 0–5 for white pieces, 6–11 for black pieces
    for p in range(12):  # There are 12 piece types (6 for white, 6 for black)
        piece_positions = tensor[p]
        for r in range(8):
            for c in range(8):
                if piece_positions[r, c] == 1:
                    square = chess.square(c, 7 - r)  # Convert row, column to square index
                    piece_type = p % 6  # 0-5 for piece types
                    color = chess.WHITE if p < 6 else chess.BLACK
                    piece = chess.Piece(piece_type + 1, color)  # Create piece object
                    piece_map[square] = piece

    # Set the pieces on the board
    for square, piece in piece_map.items():
        board.set_piece_at(square, piece)

    # Handle side-to-move (plane 12)
    if tensor[12, 0, 0].item() == 1:
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK

    # Handle castling rights (plane 13)
    # Castling rights encoded as:
    #  [white_kingside, white_queenside, black_kingside, black_queenside]
    castling_rights = tensor[13]
    if castling_rights[0, 7] == 1:
        board.set_kingside_castling_rights(chess.WHITE, True)
    if castling_rights[0, 0] == 1:
        board.set_queenside_castling_rights(chess.WHITE, True)
    if castling_rights[7, 7] == 1:
        board.set_kingside_castling_rights(chess.BLACK, True)
    if castling_rights[7, 0] == 1:
        board.set_queenside_castling_rights(chess.BLACK, True)

    return board


def update_tensor_for_move(tensor, move: chess.Move):
    """
    Update the tensor representing the chessboard based on a move.
    Takes a tensor and a chess.Move object and updates the tensor in O(1) time.
    """
    print(move)

    from_square = move.from_square
    from_r, from_c = divmod(from_square, 8)
    to_square = move.to_square
    to_r, to_c = divmod(to_square, 8)

    piece_index = -1
    taken_piece = -1

    for i in range(12):
        if tensor[i, 7-from_r, from_c] == 1:
            piece_index = i
            break
    for i in range(12):
        if tensor[i, 7-to_r, to_c] == 1:
            taken_piece = i
            break

    # Piece Type: Get piece from the tensor (8x8 plane)
    piece_plane = tensor[piece_index]
    
    # Remove the piece from the "from" square
    piece_plane[7 - from_r, from_c] = 0
    
    # Place the piece on the "to" square
    piece_plane[7 - to_r, to_c] = 1

    if taken_piece != -1:
        # Remove the taken piece from the "to" square
        taken_piece_plane = tensor[taken_piece]
        taken_piece_plane[7 - to_r, to_c] = 0

    # Update the side-to-move plane
    board_side = tensor[12, 0, 0].item()
    tensor[12].fill_(1 - board_side)
    
    # Check and update Castling rights if the move involves castling
    if move == chess.Move.from_uci("e1g1"):  # White kingside
        tensor[13, 0, 7] = 0
        tensor[13, 0, 0] = 0
    elif move == chess.Move.from_uci("e8g8"):  # Black kingside
        tensor[13, 7, 7] = 0
        tensor[13, 7, 0] = 0

    return tensor


def undo_move_from_tensor(tensor, move: chess.Move, taken_piece: int, prev_castling_plane):
    from_square = move.from_square
    to_square = move.to_square
    from_r, from_c = divmod(from_square, 8)
    to_r, to_c = divmod(to_square, 8)

    # Find which piece moved by scanning planes at 'to' square
    piece_index = -1
    for i in range(12):
        if tensor[i, 7 - to_r, to_c] == 1:
            piece_index = i
            break
    assert piece_index != -1, "Moved piece not found on 'to' square for undo."

    piece_plane = tensor[piece_index]

    # Remove moved piece from 'to' square
    piece_plane[7 - to_r, to_c] = 0

    # Place moved piece back on 'from' square
    piece_plane[7 - from_r, from_c] = 1

    # Restore captured piece if any
    if taken_piece != -1:
        taken_piece_plane = tensor[taken_piece]
        taken_piece_plane[7 - to_r, to_c] = 1

    # Restore side to move (flip it back)
    board_side = tensor[12, 0, 0].item()
    tensor[12].fill_(1 - board_side)

    # Restore castling rights plane exactly to previous state
    tensor[13] = prev_castling_plane.clone()

    return tensor