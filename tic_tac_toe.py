import random
import pygame
import numpy as np
import model as ann
from sklearn.utils.extmath import cartesian
from config import *
import time

pygame.init()
small_font  = pygame.font.Font(None, 25)
medium_font = pygame.font.Font(None, 30)
large_font  = pygame.font.Font(None, 50)

NB_EPISODES = 10000
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
GRID_SIZE = 300
GRID_DIM = 3
BORDER_THICKNESS = 7
BLOCK_WIDTH, BLOCK_HEIGHT = (GRID_SIZE / GRID_DIM, GRID_SIZE / GRID_DIM)

rewards = {
    'win': 1,
    'lose': 0,
    'draw': 0,
    'other': 0.5
}
if RENDER_ENV:
    display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))

def start_game(p1, p2):
    players = [p1, p2]
    random.shuffle(players)

    grid, center_coords = init_board()
    env = Env(center_coords, grid)
    env.board_tokens = env.board_tokens
    env.p1 = p1
    env.p2 = p2

    for episode in range(NB_EPISODES):
        env.reset()
        if RENDER_ENV:
            display.fill(pygame.Color("white"))
            display_message("TIC TAC TOE", pygame.Color("orange"), large_font_on=True,
                            y_displace=int(-(GRID_SIZE / 2 + GRID_SIZE / 4)))
            display_score(env)
            display_info(env)
            pygame.display.update()
            draw_board(env)

        while not env.done:
            if RENDER_ENV:
                display_message("TIC TAC TOE", pygame.Color("orange"), large_font_on=True, y_displace=int(-(GRID_SIZE / 2 + GRID_SIZE / 4)))
                display_score(env)
                display_info(env)
            players[0], players[1] = players[1], players[0]
            current_player = players[0]
            other_player = players[1]

            state = np.expand_dims(env.board_tokens, 0)
            action = current_player.act(env, state)
            state, reward, winner, done = env.update_env(action, current_player)
            if winner == 3:
                state = np.expand_dims(env.board_tokens, 0)
                action = current_player.act(env, state)

            state_ind = env.get_state_index(state)

            if RENDER_ENV:
                draw_board(env)

            if not done:
                current_player.log_memory(state, action, reward, winner, done, state_ind)
                other_player.log_memory(state, action, reward, winner, done, state_ind)

        current_player.learn(state, action, reward, winner, done, state_ind)
        other_player.learn(state, action, reward, winner, done, state_ind)
        env.games_played += 1
        print(env.games_played)

    np.savetxt('p1_q.tbl.txt', current_player.q_table, fmt='%d')
    np.savetxt('p2_q_tbl.txt', other_player.q_table, fmt='%d')


def reset_game(env):
    env.board_tokens = np.full(shape=env.board_tokens.shape, fill_value=0, dtype=int)


def init_board():
    center_point = (DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 2)
    centre_block_x = center_point[0] - (BLOCK_WIDTH / 2)
    centre_block_y = center_point[1] - (BLOCK_HEIGHT / 2)

    starting_x = centre_block_x - BLOCK_WIDTH
    starting_y = centre_block_y - BLOCK_WIDTH
    centers = np.zeros(shape=(3, 3, 2), dtype=int)
    grid = list()
    for row in range(GRID_DIM):
        x = starting_x
        for col in range(GRID_DIM):
            grid.append(pygame.Rect(x, starting_y, BLOCK_WIDTH, BLOCK_HEIGHT))
            centers[row][col] = (x + (BLOCK_WIDTH / 2), starting_y + (BLOCK_HEIGHT / 2))
            x += BLOCK_WIDTH
        starting_y += BLOCK_HEIGHT

    return grid, centers


def draw_board(env):
    if not RENDER_ENV:
        return

    for rect in env.grid_display:
        pygame.draw.rect(display, pygame.Color("black"), rect, BORDER_THICKNESS)

    for i in range(GRID_DIM):
        for j in range(GRID_DIM):
            if env.board_tokens[i][j] == -1:
                center_x, center_y = env.center_coords[i][j]
                pygame.draw.circle(display, pygame.Color("blue"),
                                   (int(center_x), int(center_y)), int((BLOCK_WIDTH / 2) - BORDER_THICKNESS), BORDER_THICKNESS)

            elif env.board_tokens[i][j] == 1:
                center_x, center_y = env.center_coords[i][j]
                pygame.draw.line(display, pygame.Color("red"),
                                 (center_x - (BLOCK_WIDTH / 2) + BORDER_THICKNESS, center_y - (BLOCK_HEIGHT / 2) + BORDER_THICKNESS),
                                 (center_x + (BLOCK_WIDTH / 2) - BORDER_THICKNESS, center_y + (BLOCK_HEIGHT / 2) - BORDER_THICKNESS),
                                  BORDER_THICKNESS)
                pygame.draw.line(display, pygame.Color("red"),
                                 (center_x - (BLOCK_WIDTH / 2) + BORDER_THICKNESS, center_y + (BLOCK_HEIGHT / 2) - BORDER_THICKNESS),
                                 (center_x + (BLOCK_WIDTH / 2) - BORDER_THICKNESS, center_y - (BLOCK_HEIGHT / 2) + BORDER_THICKNESS),
                                  BORDER_THICKNESS)
    pygame.display.update()


class Human:
    def __init__(self, name, token):
        self.name = name
        self.token = token

    def act(self, env, state):
        validated_action = False
        legal_positions = ['1', '2', '3']
        while not validated_action:
            row = input("\n-------------------------\n"
                        "Your turn!\n"
                        "\tSelect row: ")
            if not row in legal_positions:
                print("Row {} is invalid. Select 1, 2 or 3.".format(row))
                continue

            col = input("\n\tSelect column: ")
            if not col in legal_positions:
                print("Column {} is invalid. Select 1, 2 or 3.".format(col))
                continue

            print("You selected coordinates ({}, {})".format(row, col))
            action = (int(col) - 1, int(row) - 1)
            if env.is_action_valid(state[0], action):
                validated_action = True
            else:
                print("Invalid action. Please try again.")
        return action

    def log_memory(self, state, action, reward, winner, done, state_ind):
        pass

    def learn(self, state, action, reward, winner, done, state_ind):
        pass

    def forget(self):
        pass


class Agent:
    def __init__(self, name, token, gamma=0.9, alpha=0.01, epsilon=0.9, decay=0.9, epsilon_min=0.1):
        self.name = name
        self.token = token
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.rewards = []
        self.memory = []
        self.q_table = np.full([3**9], 0.5, dtype=float)


    def act(self, env, state):
        if self.epsilon > np.random.rand():
            validated_action = False
            while not validated_action:
                action = np.random.randint(3, size=(2))
                if env.is_action_valid(state[0], action):
                    validated_action = True
            if AI_DELAY:
                time.sleep(AI_DELAY)
            return action
        else:
            valid_actions = []
            for row in range(3):
                for col in range(3):
                    if env.is_action_valid(state[0], [row, col]):
                        temp_state = state[0].copy()
                        temp_state[row][col] = self.token
                        valid_actions.append([(row, col), temp_state])

            best_action = valid_actions[0]
            for action in valid_actions:
                state_ind = env.get_state_index(action[1])
                if self.q_table[state_ind] > env.get_state_index(best_action[1]):
                    best_action = action


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay
            if AI_DELAY:
                time.sleep(AI_DELAY)
        return best_action[0]


    def log_memory(self, state, action, reward, winner, done, state_ind):
        self.memory.append([state, action, reward, winner, done, state_ind])

    def learn(self, state, action, reward, winner, done, state_ind):
        if winner != self.token:
            reward = 0
        self.q_table[state_ind] = reward
        target = reward
        for state, action, reward, winner, done, state_ind in reversed(self.memory):
            self.q_table[state_ind] = self.q_table[state_ind] + self.alpha * (target - self.q_table[state_ind])
            target = self.q_table[state_ind]

    def forget(self):
        self.memory = []


class Env:

    def __init__(self, center_coords, grid_display, board=(3, 3)):
        self.board_tokens = np.full(shape=board, fill_value=0, dtype=int)
        self.reward = 0
        self.winner = None
        self.center_coords = center_coords
        self.grid_display = grid_display
        self.possible_states = self.generate_all_states()
        self.terminal_states = self.determine_terminal_states()
        self.done = False

        self.p1 = None
        self.p1_score = 0
        self.p2 = None
        self.p2_score = 0

        self.draw_count = 0
        self.games_played = 0

    def reset(self):
        self.board_tokens = np.full(shape=self.board_tokens.shape, fill_value=0, dtype=int)
        self.reward = 0
        self.winner = None
        self.done = False

    def update_env(self, action, agent):
        try:
            self.board_tokens[action[1]][action[0]] = agent.token
        except TypeError:
            return self.board_tokens, 2, 3, 4
        state = self.board_tokens.copy()
        state_result = self.terminal_states[self.get_state_index(state)]
        if state_result == 1:
            self.reward = 1
            self.winner = 1
            self.done = True
            if self.p1.token == 1:
                self.p1_score += 1
            else:
                self.p2_score += 1

        elif state_result == -1:
            self.reward = 1
            self.winner = -1
            self.done = True
            if self.p1.token == -1:
                self.p1_score += 1
            else:
                self.p2_score += 1

        elif state_result == 0:
            self.reward = 0
            self.winner = 0
            self.done = True
            self.draw_count += 1
        else:
            self.reward = 0.5
            self.winner = 0
            self.done = False

        return state, self.reward, self.winner, self.done


    def get_state_index(self, state):
        collapsed = state.reshape((9))
        tern_arr = collapsed + 1
        state_ind = 0
        column = 0
        for i in range(len(tern_arr) - 1, -1, -1):
            state_ind += tern_arr[i] * (3**column)
            column += 1

        return state_ind

    def generate_all_states(self):
        states = cartesian(([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]] * GRID_DIM))
        return states.reshape((GRID_DIM**9, GRID_DIM, GRID_DIM))

    def determine_terminal_states(self):
        terminal_states = np.full([3**9], None)
        for ind, state in enumerate(self.possible_states):
            results = self.is_winner(state)
            if results == 'x_win':
                terminal_states[ind] = 1
            elif results == 'o_win':
                terminal_states[ind] = -1
            elif self.is_draw(state):
                    terminal_states[ind] = 0

        return terminal_states

    def is_winner(self, state):
        rows = [np.sum(state[i]) for i in range(GRID_DIM)]
        cols = [np.sum(np.transpose(state)[i]) for i in range(GRID_DIM)]
        for row, col in zip(rows, cols):
            if row == 3 or col == 3:
                return 'x_win'
            elif row == -3 or col == -3:
                return 'o_win'

        diagonals = [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
        for diagonal in diagonals:
            total = np.sum([state[i][j] for i, j in diagonal])
            if total == 3:
                return 'x_win'
            elif total == -3:
                return 'o_win'
        return False

    def is_draw(self, state):
        if np.count_nonzero(state) == GRID_DIM * GRID_DIM:
            return True
        return False

    def is_action_valid(self, state, action):
        return state[action[1]][action[0]] == 0


def get_player(player, token):
    player = None
    while not player:
        selection = input("\n-------------------------\n"
                          "Who will be player {}?"
                          "\n\t1 - Human"
                          "\n\t2 - AI\n".format(player))
        if selection == '1':
            name = input("\n\nPlease enter a name for player: ")
            return Human(name=name, token=token)

        elif selection == '2':
            name = input("\n\nPlease enter a name for agent: ")
            return Agent(name=name, token=token)
        else:
            print("\n\nInvalid selection! Try again.\n")
        print("\n-------------------------\n\n")


def get_text_object(text, color, large_font_on=False):
    if large_font_on:
        text_surf = large_font.render(text, True, color)
    else:
        text_surf = small_font.render(text, True, color)
    return text_surf, text_surf.get_rect()


def display_message(message, color, large_font_on=False, y_displace=0):
    text_surf, text_rect = get_text_object(message, color, large_font_on)
    text_rect.center = (DISPLAY_WIDTH / 2, DISPLAY_HEIGHT / 2 + y_displace)
    display.blit(text_surf, text_rect)


def display_score(env):
    text_surf, text_rect = get_text_object(env.p1.name + " score: " + str(env.p1_score), pygame.Color("blue"))
    text_rect.center = (80, 20)
    display.blit(text_surf, text_rect)

    text_surf, text_rect = get_text_object(env.p2.name + " score: " + str(env.p2_score), pygame.Color("red"))
    text_rect.center = (DISPLAY_WIDTH - 80, 20)
    display.blit(text_surf, text_rect)

    text_surf, text_rect = get_text_object("Draws: " + str(env.draw_count), pygame.Color("black"))
    text_rect.center = (DISPLAY_WIDTH / 2, 20)
    display.blit(text_surf, text_rect)


def display_info(env):
    text_surf, text_rect = get_text_object("Games played: " + str(env.games_played), pygame.Color("black"))
    text_rect.center = (80, DISPLAY_HEIGHT - 20)
    display.blit(text_surf, text_rect)



if __name__ == "__main__":

    p1 = get_player("1", -1)
    p2 = get_player("2", 1)
    start_game(p1, p2)
