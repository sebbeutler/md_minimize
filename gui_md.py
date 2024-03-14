import pygame
import sys
import numpy as np
from md import *

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SPHERE_RADIUS = 30
SPHERE_COLOR = (255, 0, 0)
BACKGROUND_COLOR = (255, 255, 255)

# Sphere positionss
sphere_positions = np.array([(2.0, 2.0), (3.0, 2.0), (2.5, 3.0)])
sphere_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0) ]

# Define the button's properties
button_position = (50, 50)  # Top-left corner of the button
button_size = (100, 50)  # Width and height of the button
minimizing = False


# Flags for dragging
dragging = [False, False, False]
offsets = [0, 0, 0]

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Draggable Spheres")


pos_init = sphere_positions + (5**.5 * DT) + (0.5 * gradient(sphere_positions) * DT**2)

positions: list[np.ndarray] = [pos_init, sphere_positions]


# Main game loop
while True:
    if minimizing:
        sphere_positions = (2 * sphere_positions) - positions[-2] + (DT**2 * gradient(sphere_positions))
        positions.append(sphere_positions)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for i, (sphere_x, sphere_y) in enumerate(sphere_positions*100):
                distance = pygame.math.Vector2(mouse_x - sphere_x, mouse_y - sphere_y).length()
                if distance < SPHERE_RADIUS:
                    dragging[i] = True
                    offsets[i] = (sphere_x - mouse_x, sphere_y - mouse_y) # type: ignore
            if pygame.Rect(button_position, button_size).collidepoint(mouse_x, mouse_y):
                minimizing = not minimizing
                
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = [False, False, False]

    # Update sphere positions while dragging
    for i in range(3):
        if dragging[i]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            sphere_positions[i] = np.array((mouse_x + offsets[i][0], mouse_y + offsets[i][1])) / 100.0 # type: ignore

    # Draw background
    screen.fill(BACKGROUND_COLOR)

    # Draw lines connecting spheres
    for i in range(3):
        for j in range(i, 3):
            pygame.draw.line(screen, (66, 189, 219), sphere_positions[i]*100, sphere_positions[j]*100, 1)

    # Draw spheres
    for i, sphere_position in enumerate(sphere_positions*100):
        pygame.draw.circle(screen, sphere_colors[i], (int(sphere_position[0]), int(sphere_position[1])), SPHERE_RADIUS)

    # Draw the button
    pygame.draw.rect(screen, (255, 79, 0) if minimizing else (0, 255, 0), pygame.Rect(button_position, button_size))


    # Update display
    pygame.display.flip()

    # Control the frame rate
    pygame.time.Clock().tick(60)