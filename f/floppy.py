import pygame
import random

# Initialize Pygame
pygame.init()

# Set up the game window
WIDTH = 400
HEIGHT = 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy Bird")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Game variables
GRAVITY = 0.5
FLAP_STRENGTH = -10
PIPE_SPEED = 3
PIPE_GAP = 200
PIPE_FREQUENCY = 1500  # milliseconds

# Bird
class Bird:
    def __init__(self):
        self.x = 50
        self.y = HEIGHT // 2
        self.velocity = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self):
        pygame.draw.circle(window, YELLOW, (int(self.x), int(self.y)), 20)

# Pipe
class Pipe:
    def __init__(self):
        self.x = WIDTH
        self.height = random.randint(100, HEIGHT - PIPE_GAP - 100)

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self):
        pygame.draw.rect(window, GREEN, (self.x, 0, 50, self.height))
        pygame.draw.rect(window, GREEN, (self.x, self.height + PIPE_GAP, 50, HEIGHT - self.height - PIPE_GAP))

# Game loop
def game_loop():
    bird = Bird()
    pipes = []
    score = 0
    last_pipe = pygame.time.get_ticks()
    font = pygame.font.Font(None, 36)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.flap()

        bird.update()

        # Generate new pipes
        time_now = pygame.time.get_ticks()
        if time_now - last_pipe > PIPE_FREQUENCY:
            pipes.append(Pipe())
            last_pipe = time_now

        # Update and remove pipes
        for pipe in pipes:
            pipe.update()
            if pipe.x + 50 < 0:
                pipes.remove(pipe)
                score += 1

        # Check for collisions
        for pipe in pipes:
            if (bird.x + 20 > pipe.x and bird.x - 20 < pipe.x + 50) and \
               (bird.y - 20 < pipe.height or bird.y + 20 > pipe.height + PIPE_GAP):
                running = False

        if bird.y > HEIGHT or bird.y < 0:
            running = False

        # Draw everything
        window.fill(BLACK)
        bird.draw()
        for pipe in pipes:
            pipe.draw()

        # Draw score
        score_text = font.render(f"Score: {score}", True, WHITE)
        window.blit(score_text, (10, 10))

        pygame.display.update()
        clock.tick(60)

    pygame.quit()

game_loop()   