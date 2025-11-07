import pygame
import random
import math
import numpy as np
import os

class GameBoard():
    def __init__(self, width, height, radius, 
                 pre_trained_states, plot_obstacles = True,
                 speed = 1):
        self.WIDTH = width
        self.HEIGHT = height
        self.RADIUS = radius
        self.player_color = (0,0,255)
        self.obstacle_color = (255,0,0)
        self.goal_color = (255,255,0)
        self.bg_color = (0,0,0)
        self.obstacles = []
        self.plot_obstacles = plot_obstacles
        self.circles = []
        self.offset = self.RADIUS
        self.player_x = None
        self.player_y = None
        self.player_deployed = False
        self.goal_x = self.WIDTH  #jugaad # since the model was trained on a grid 100x smaller
        self.goal_y = self.HEIGHT #jugaad
        self.pre_trained_states = pre_trained_states
        self.speed = speed
        self.game_terminated = False

        pygame.font.init()

    def render_board(self):
        self.screen = pygame.display.set_mode((self.WIDTH + self.offset, self.HEIGHT + self.offset))
        pygame.display.set_caption("Game Board")
        self.clock = pygame.time.Clock()

    def create_circle(self, x, y):
        dx = random.choice([-1, 1])
        dy = random.choice([-1, 1])
        return {"x": x, "y": y, "r": self.RADIUS, "color": self.player_color, "dx": dx, "dy": dy}
    
    def create_random_obstacles(self, num=5):
        obstacles = []
        for _ in range(num):
            x = random.randint(self.RADIUS, self.WIDTH - self.RADIUS)
            y = random.randint(self.RADIUS, self.HEIGHT - self.RADIUS)
            obstacles.append((x, y, self.RADIUS))
        return obstacles
    
    def create_obstacles(self, walls):
        obstacles = []
        for wall in walls:
            x = wall[0]
            y = wall[1]
            obstacles.append((x, y, self.RADIUS))
        return obstacles

    def check_collision(self, curr_x, curr_y, curr_r):
        if not self.plot_obstacles:
            return False
        
        for ob in self.obstacles:
            ob_x, ob_y, ob_r = ob
            distance = math.sqrt((curr_x - ob_x) ** 2 + (curr_y - ob_y) ** 2) # euclidean distance
            #if they collide then the sum of their radii (or less) is the distnace between their centers
            if distance <= curr_r + ob_r:
                return True
        return False
    
    def player_won(self, curr_x, curr_y, curr_r):
        distance = math.sqrt((curr_x - self.goal_x) ** 2 + (curr_y - self.goal_y) ** 2)
        if distance <= curr_r + self.RADIUS:
            return True

        return False
    
    def step(self, action):
        if action == 0: #up
            return 0, -1
        if action == 1: #down
            return 0,1
        if action == 2: #left
            return -1, 0
        if action == 3: #right
            return 1,0
        
    def display_text(self, text):
        font = pygame.font.Font(None, 30)
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect()
        # text_rect.center = (self.WIDTH // 2, self.HEIGHT // 2)

        return text_surface, text_rect
    
    def play(self):
        self.render_board()
        # self.obstacles = self.create_obstacles()
        running = True

        while running:
            self.screen.fill(self.bg_color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Create goal on mouse click
                if event.type == pygame.MOUSEBUTTONDOWN and not self.player_deployed:
                    self.player_x, self.player_y = pygame.mouse.get_pos()
                    self.circles.append(self.create_circle(self.player_x, self.player_y))
                    self.player_deployed = True

            if self.plot_obstacles:
                for x, y, r in self.obstacles:
                    pygame.draw.circle(self.screen, self.obstacle_color, (x, y), r)

            # pygame.draw.circle(self.screen, self.goal_color, (self.WIDTH - self.offset, self.offset), r)
            pygame.draw.circle(self.screen, self.goal_color, (self.goal_x, self.goal_y), self.RADIUS)

            board_width, board_height = 1000, 500
            rows, cols = 5, 10

            cell_w = board_width / cols   # 100
            cell_h = board_height / rows  # 100

            if self.game_terminated:
                continue

            for c in self.circles:
                curr_x, curr_y = c["x"], c["y"]

                # map pixel coordinates → grid indices
                grid_x = int(curr_y // cell_h)
                grid_y = int(curr_x // cell_w)

                # clamp inside range
                grid_x = np.clip(grid_x, 0, rows - 1)
                grid_y = np.clip(grid_y, 0, cols - 1)

                action = np.argmax(self.pre_trained_states[grid_x, grid_y])
                next_x, next_y = self.step(action)

                c["x"] += next_x * self.speed
                c["y"] += next_y * self.speed

                # Bounce on walls
                if c["x"] - c["r"] < 0 or c["x"] + c["r"] > self.WIDTH:
                    c["dx"] *= -1 * self.speed
                if c["y"] - c["r"] < 0 or c["y"] + c["r"] > self.HEIGHT:
                    c["dy"] *= -1 * self.speed

                # Draw circle
                pygame.draw.circle(self.screen, c["color"], (int(c["x"]), int(c["y"])), c["r"])

                if self.check_collision(c["x"], c["y"], c["r"]):
                    text_surface, text_rect = self.display_text("GAME OVER!")
                    self.screen.blit(text_surface, text_rect)
                    # running = False
                    self.game_terminated = True

                if self.player_won(c["x"], c["y"], c["r"]):
                    text_surface, text_rect = self.display_text("Player won!")
                    self.screen.blit(text_surface, text_rect)
                    self.game_terminated = True

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def stop(self):
        pygame.quit()

if __name__ == "__main__":
    pre_trained_states = np.load(os.path.join("pt_states","sarsa_q.npy"))

    game = GameBoard(900, 400, 10, 
                     pre_trained_states, plot_obstacles = True,
                     speed = 3)
    
    walls = [(20,30),(56,79),(456,341)]
    game.obstacles = game.create_obstacles(walls)
    game.play()

