import random
import pygame
import numpy as np
import model as ann

pygame.init()
NB_EPISODES = 5
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600
GRID_SIZE = (300, 300)
GRID_DIM = (3, 3)
BORDER_THICKNESS = 7
BLOCK_WIDTH, BLOCK_HEIGHT = (GRID_SIZE[0] / GRID_DIM[0], GRID_SIZE[1] / GRID_DIM[1])

rewards = {
    'win': 1,
    'lose': 0,
    'draw': 0,
    'other': 0.5
}
display = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT))

def start_game(p1, p2, env):
    players = [p1, p2]
    random.shuffle(players)
    draw_board(env)


    for episode in range(NB_EPISODES):
        done = False
        display.fill(pygame.Color("white"))
        grid, center_coords = init_board()
        env = Env(center_coords, grid)
        env.board_tokens = env.board_tokens
        draw_board(env)

        while not done:
            players[0], players[1] = players[1], players[0]
            current_player = players[0]
            other_player = players[1]

            state = np.expand_dims(env.board_tokens, 0)
            action = current_player.act(state)
            update_env(env, action, current_player)
            draw_board(env)
            outcome = env.determine_state(state[0])
            if outcome == 'win':
                current_player_reward = rewards['win']
                other_player_reward = rewards['lose']
                done = True
            elif outcome == 'draw':
                current_player_reward = rewards['draw']
                other_player_reward = rewards['draw']
                done = True
            else:
                current_player_reward = rewards['other']
                other_player_reward = rewards['other']

            if not done:
                current_player.log_memory(state, action, current_player_reward)


        current_player.learn(state, action, current_player_reward)
        other_player.learn(state, action, other_player_reward)

    current_player.model.save('p1_model')
    other_player.model.save('p2_model')
    current_player.model.save_weights('p1_model_weights')
    other_player.model.save_weights('p2_model_weights')





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
    for row in range(GRID_DIM[0]):
        x = starting_x
        for col in range(GRID_DIM[1]):
            grid.append(pygame.Rect(x, starting_y, BLOCK_WIDTH, BLOCK_HEIGHT))
            centers[row][col] = (x + (BLOCK_WIDTH / 2), starting_y + (BLOCK_HEIGHT / 2))
            x += BLOCK_WIDTH
        starting_y += BLOCK_HEIGHT

    return grid, centers


def draw_board(env):
    for rect in env.grid_display:
        pygame.draw.rect(display, pygame.Color("black"), rect, BORDER_THICKNESS)

    for i in range(GRID_DIM[0]):
        for j in range(GRID_DIM[1]):
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


def update_env(env, action, agent):
    env.board_tokens[action[1]][action[0]] = agent.token
    return env


def is_winner(state):
    rows = [np.sum(state[i]) for i in range(GRID_DIM[0])]
    cols = [np.sum(np.transpose(state)[i]) for i in range(GRID_DIM[0])]
    for row, col in zip(rows, cols):
        if row == 3 or col == 3 or row == -3 or col == -3:
            return True

    diagonals = [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
    for diagonal in diagonals:
        total = np.sum([state[i][j] for i, j in diagonal])
        if total == 3 or total == -3:
            return True
    return False



def is_draw(state):
    if np.count_nonzero(state) == GRID_DIM[0] * GRID_DIM[1]:
        return True
    return False


def is_action_valid(state, action):
    return state[action[1]][action[0]] == 0


class Agent():
    def __init__(self, name, model, token, gamma=0.9, alpha=0.01, epsilon=0.9, decay=0.9, epsilon_min=0.02):
        self.name = name
        self.model = model
        self.token = token
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        self.rewards = []
        self.memory = []

    def act(self, state):
        if self.epsilon > np.random.rand():
            validated_action = False
            while not validated_action:
                action = np.random.randint(3, size=(2))
                validated_action = is_action_valid(state[0], action)
            return action
        else:
            pred = self.model.predict(state)
            validated_action = False
            index = 0
            while not validated_action:
                # action = np.array(np.unravel_index(np.argmax(pred), pred[0].shape))
                values = np.unravel_index(np.argsort(-pred[0], axis=None), pred[0].shape)
                x_indices = values[1]
                y_indices = values[0]
                action = [x_indices[index], y_indices[index]]
                validated_action = is_action_valid(state[0], action)
                index += 1
            return action


    def log_memory(self, state, action, reward):
        self.memory.append([state, action, reward])

    def learn(self, state, action, reward):

        current_pred = self.model.predict(state)
        target = current_pred
        r_1 = reward
        target[0][action[0]][action[1]] = r_1
        self.model.fit(current_pred, target)
        for i, memory in enumerate(reversed(self.memory)):
            current_pred = self.model.predict(memory[0])
            a = memory[1]
            r = memory[2]
            q_reward = r + self.gamma * (r_1 - r)
            target[0][a[0]][a[1]] = q_reward
            self.model.fit(current_pred, target)
            r_1 = r

        self.forget()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.decay

    def forget(self):
        self.memory = []


class Env():
    def __init__(self, center_coords, grid_display, board=(3, 3)):
        self.board_tokens = np.full(shape=board, fill_value=0, dtype=int)
        self.center_coords = center_coords
        self.grid_display = grid_display


    def determine_state(self, state):
        if is_winner(state):
            return 'win'
        elif is_draw(state):
            return 'draw'
        else:
            return 'other'


def two_dim_to_one(node):
    return node[0] + (node[1] * GRID_DIM[1])


def one_dim_to_two(num):
    node_col =GRID_DIM[0] % 3
    node_row = int(num / GRID_DIM[1])
    return (node_col, node_row)


if __name__ == "__main__":
    grid, center_coords = init_board()
    env = Env(center_coords, grid)
    env.board_tokens = env.board_tokens
    model_1 = ann.create_model(env.board_tokens.shape)
    model_2 = ann.create_model(env.board_tokens.shape)
    p1 = Agent("p1", model=model_1, token=-1)
    p2 = Agent("p2", model=model_2, token=1)
    start_game(p1, p2, env)
