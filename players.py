import random
import pygame
import math
from connect4 import connect4
import sys
import numpy as np

SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

WIN_SCORE = 100000
LOSS_SCORE = -100000

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)

class connect4Player(object):
	def __init__(self, position, seed=0, CVDMode=False):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)
		if CVDMode:
			global P1COLOR
			global P2COLOR
			P1COLOR = (227, 60, 239)
			P2COLOR = (0, 255, 0)

	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict["move"] = -1

class humanConsole(connect4Player):
	'''
	Human player where input is collected from the console
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict['move'] = int(input('Select next move: '))
		while True:
			if int(move_dict['move']) >= 0 and int(move_dict['move']) <= 6 and env.topPosition[int(move_dict['move'])] >= 0:
				break
			move_dict['move'] = int(input('Index invalid. Select next move: '))

class humanGUI(connect4Player):
	'''
	Human player where input is collected from the GUI
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move_dict['move'] = col
					done = True

class randomAI(connect4Player):
	'''
	connect4Player that elects a random playable column as its move
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move_dict['move'] = random.choice(indices)

class stupidAI(connect4Player):
	'''
	connect4Player that will play the same strategy every time
	Tries to fill specific columns in a specific order 
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move_dict['move'] = 3
		elif 2 in indices:
			move_dict['move'] = 2
		elif 1 in indices:
			move_dict['move'] = 1
		elif 5 in indices:
			move_dict['move'] = 5
		elif 6 in indices:
			move_dict['move'] = 6
		else:
			move_dict['move'] = 0

class minimaxAI(connect4Player):
	"""
	Minimax AI to play connect 4
	"""
	def play(self, env: connect4, move_dict: dict) -> None:
		DEPTH = 4 

		board = env.board.copy()         
		topPositions = env.topPosition.copy() 

		if np.count_nonzero(board) == 0:
			move_dict['move'] = 3
			return

		def minimax(board, topPositions, depth, maximizingPlayer):
			valid_locations = get_valid_locations(topPositions)
			is_terminal = is_terminal_node(board, topPositions)

			if depth == 0 or is_terminal:
				if is_terminal:
					if winning_move(board, self.position):
						return (None, WIN_SCORE)
					elif winning_move(board, self.opponent.position):
						return (None, LOSS_SCORE)
					else:  
						return (None, 0)
				else:
					return (None, score_position(board, self.position))
				
			if maximizingPlayer:
				value = -math.inf
				best_col = random.choice(valid_locations)
				for col in valid_locations:
					new_board, new_top = make_move(board, topPositions, col, self.position)
					new_score = minimax(new_board, new_top, depth-1, False)[1]
					if new_score > value:
						value = new_score
						best_col = col
				return best_col, value
			else:
				value = math.inf
				best_col = random.choice(valid_locations)
				for col in valid_locations:
					new_board, new_top = make_move(board, topPositions, col, self.opponent.position)
					new_score = minimax(new_board, new_top, depth-1, True)[1]
					if new_score < value:
						value = new_score
						best_col = col
				return best_col, value

		best_move, _ = minimax(board, topPositions, DEPTH, True)
		move_dict['move'] = best_move

	
class alphaBetaAI(connect4Player):
	"""
	Alpha beta AI to play connect 4
	"""
	def play(self, env: connect4, move_dict: dict) -> None:
		DEPTH = 4 
		board = env.board.copy()
		topPositions = env.topPosition.copy() 

		if np.count_nonzero(board) == 0:
			move_dict['move'] = 3
			return

		def alphabeta(board, topPositions, depth, alpha, beta, maximizingPlayer):
			valid_locations = get_valid_locations(topPositions)
			is_terminal = is_terminal_node(board, topPositions)
			if depth == 0 or is_terminal:
				if is_terminal:
					if winning_move(board, self.position):
						return (None, WIN_SCORE)
					elif winning_move(board, self.opponent.position):
						return (None, LOSS_SCORE)
					else:
						return (None, 0)
				else:
					return (None, score_position(board, self.position))
			if maximizingPlayer:
				value = -math.inf
				best_col = random.choice(valid_locations)
				for col in valid_locations:
					new_board, new_top = make_move(board, topPositions, col, self.position)
					new_score = alphabeta(new_board, new_top, depth-1, alpha, beta, False)[1]
					if new_score > value:
						value = new_score
						best_col = col
					alpha = max(alpha, value)
					if alpha >= beta:
						break  
				return best_col, value
			else:
				value = math.inf
				best_col = random.choice(valid_locations)
				for col in valid_locations:
					new_board, new_top = make_move(board, topPositions, col, self.opponent.position)
					new_score = alphabeta(new_board, new_top, depth-1, alpha, beta, True)[1]
					if new_score < value:
						value = new_score
						best_col = col
					beta = min(beta, value)
					if alpha >= beta:
						break 
				return best_col, value

		best_move, _ = alphabeta(board, topPositions, DEPTH, -math.inf, math.inf, True)
		move_dict['move'] = best_move

def get_valid_locations(topPositions):
    """
	Columns that a move can be made
	"""
    valid_locations = []
    for col in range(len(topPositions)):
        if topPositions[col] >= 0:
            valid_locations.append(col)
    return valid_locations

def make_move(board, topPositions, col, piece):
	"""
	Makes the move given
	"""
	new_board = board.copy()
	new_top = topPositions.copy()
	row = new_top[col]
	new_board[row, col] = piece
	new_top[col] = row - 1  
	return new_board, new_top

def get_winning_lines():
	"""
	All vertical, horizontal, diagonal sets of 4
	""" 
	winning_lines = []
	
	for r in range(ROW_COUNT):
		for c in range(COLUMN_COUNT - 3):
			winning_lines.append([(r, c+i) for i in range(4)])

	for c in range(COLUMN_COUNT):
		for r in range(ROW_COUNT - 3):
			winning_lines.append([(r+i, c) for i in range(4)])

	for r in range(ROW_COUNT - 3):
		for c in range(COLUMN_COUNT - 3):
			winning_lines.append([(r+i, c+i) for i in range(4)])

	for r in range(3, ROW_COUNT):
		for c in range(COLUMN_COUNT - 3):
			winning_lines.append([(r-i, c+i) for i in range(4)])
    
	return winning_lines

WINNING_LINES = get_winning_lines()

def winning_move(board, piece):
    """
	Checks for a win
	"""
    for line in WINNING_LINES:
        if all(board[r, c] == piece for r, c in line):
            return True
    return False


def is_terminal_node(board, topPositions):
    """
	Return True if the board is in a terminal state (win for either or no valid moves)
	"""
    return (winning_move(board, 1) or winning_move(board, 2) or 
            len(get_valid_locations(topPositions)) == 0)

def score_position(board, piece):
    """
    Evaluation function for connect 4 (explained further in comments + report)
    """
    opponent = 1 if piece == 2 else 2
    score = 0
    
    # Give more points for pieces in center column
    center_col = COLUMN_COUNT // 2
    center_array = [board[r][center_col] for r in range(ROW_COUNT)]
    center_count = center_array.count(piece)
    score += center_count * 4
    
    # Lesser points to pieces closer to but not center
    for c in [center_col-1, center_col+1]:
        if 0 <= c < COLUMN_COUNT:
            col_array = [board[r][c] for r in range(ROW_COUNT)]
            col_count = col_array.count(piece)
            score += col_count * 2
    
    # Give weights to rows (bottom rows are more strategic so give more points to them)
    row_weights = [1, 2, 3, 5, 7, 8]  
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT):
            if board[r][c] == piece:
                score += row_weights[r] // 2
    
    # Use WINNING_LINES to find score for each of the possible winning line
    for line in WINNING_LINES:
        line_values = [board[r][c] for r, c in line]
        score += evaluate_line(line_values, piece, opponent)
    
    return score

def evaluate_line(line, piece, opponent):
    """
    Gives points based on how many pieces there are and their positions relative to each other
    """
	# Count pieces
    piece_count = line.count(piece)
    opp_count = line.count(opponent)
    empty_count = line.count(0)
    
    # No win possible
    if piece_count > 0 and opp_count > 0:
        return 0
    
    # Check for player's pieces
    if piece_count == 4:
        return 1000
    elif piece_count == 3 and empty_count == 1:
        return 50
    elif piece_count == 2 and empty_count == 2:
        # Pieces next to each other are better
        if line[0] == piece and line[1] == piece:
            return 10
        elif line[1] == piece and line[2] == piece:
            return 10
        elif line[2] == piece and line[3] == piece:
            return 10
        else:
            return 5
    elif piece_count == 1 and empty_count == 3:
        return 1
    
    # Same thing but remove points for opponents pieces
    if opp_count == 3 and empty_count == 1:
        return -80 
    elif opp_count == 2 and empty_count == 2:
        if line[0] == opponent and line[1] == opponent:
            return -8
        elif line[1] == opponent and line[2] == opponent:
            return -8
        elif line[2] == opponent and line[3] == opponent:
            return -8
        else:
            return -4
    
    return 0




