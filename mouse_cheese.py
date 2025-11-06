import pygame
import random
import math

class GameBoard():
    def __init__(self, width, height, radius):
        self.WIDTH = width
        self.HEIGHT = height
        self.RADIUS = radius
        self.player_color = (0,0,255)
        self.obstacle_color = (255,0,0)
        self.goal_color = (255,255,0)
        self.bg_color = (0,0,0)
        self.obstacles = []
        self.circles = []
        self.offset = 30

        # self.render_board()

    def render_board(self):
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Game Board")
        self.clock = pygame.time.Clock()

    def create_circle(self, x, y):
        # Random velocity
        dx = random.choice([-3, -2, -1, 1, 2, 3])
        dy = random.choice([-3, -2, -1, 1, 2, 3])
        return {"x": x, "y": y, "r": self.RADIUS, "color": self.player_color, "dx": dx, "dy": dy}
    
    def create_obstacles(self, num=5):
        obstacles = []
        for _ in range(num):
            x = random.randint(self.RADIUS, self.WIDTH - self.RADIUS)
            y = random.randint(self.RADIUS, self.HEIGHT - self.RADIUS)
            obstacles.append((x, y, self.RADIUS))
        return obstacles

    def check_collision(self, curr_x, curr_y, curr_r):
        for ob in self.obstacles:
            ob_x, ob_y, ob_r = ob
            distance = math.sqrt((curr_x - ob_x) ** 2 + (curr_y - ob_y) ** 2) # euclidean distance
            #if they collide then the sum of their radii (or less) is the distnace between their centers
            if distance <= curr_r + ob_r:
                return True
        return False
    
    def player_won(self, curr_x, curr_y, curr_r, offset):
        goal_x = self.WIDTH - self.offset
        goal_y = self.offset
        goal_r = 20

        distance = math.sqrt((curr_x - goal_x) ** 2 + (curr_y - goal_y) ** 2)
        if distance <= curr_r + goal_r:
            return True

        return False
    
    def play(self):
        self.render_board()
        self.obstacles = self.create_obstacles()
        running = True

        while running:
            self.screen.fill(self.bg_color)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                # Create circle on mouse click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    self.circles.append(self.create_circle(x, y))  

            for x, y, r in self.obstacles:
                pygame.draw.circle(self.screen, self.obstacle_color, (x, y), r)

            pygame.draw.circle(self.screen, self.goal_color, (self.WIDTH - self.offset, 0 + self.offset), 20)

            for c in self.circles:
                c["x"] += c["dx"]
                c["y"] += c["dy"]

                # Bounce on walls
                if c["x"] - c["r"] < 0 or c["x"] + c["r"] > self.WIDTH:
                    c["dx"] *= -1
                if c["y"] - c["r"] < 0 or c["y"] + c["r"] > self.HEIGHT:
                    c["dy"] *= -1

                # Draw circle
                pygame.draw.circle(self.screen, c["color"], (int(c["x"]), int(c["y"])), c["r"])

                if self.check_collision(c["x"], c["y"], c["r"]):
                    print("GAME OVER!")
                    running = False

                if self.player_won(c["x"], c["y"], c["r"], self.offset):
                    print("Player won!")
                    running = False

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    game = GameBoard(600, 400, 20)
    game.play()