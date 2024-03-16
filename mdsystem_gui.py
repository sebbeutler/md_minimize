import pygame
import sys

_screen: pygame.Surface = None # type: ignore

buttons = []
dragging = {}
text_font: pygame.font.Font = None # type: ignore

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SPHERE_RADIUS = 20
BACKGROUND_COLOR = (96, 124, 141)

class Button:
    def __init__(self, pos=(50,50), size=(70,30), color=(120,170,172), text="hello", callback=lambda:0):
        self.pos = pos
        self.size = size
        self.color = color
        self.text = text
        self.callback = callback

def init():
    global _screen
    global text_font
    # Initialize Pygame
    pygame.init()

    text_font = pygame.font.Font(None, 14)
    # Create the screen
    _screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Molecular Dynamics App")

def update(system):
    global buttons
    global dragging

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            for button in buttons:
                if pygame.Rect(button.pos, button.size).collidepoint(mouse_x, mouse_y):
                    button.callback()

            for i, atom in enumerate(system.atoms):
                ax, ay = atom.pos[:2]*100
                distance = pygame.math.Vector2(mouse_x - ax, mouse_y - ay).length()
                if distance < SPHERE_RADIUS:
                    dragging[atom] = (ax - mouse_x, ay - mouse_y)
        elif event.type == pygame.MOUSEBUTTONUP:
            dragging.clear()

    # Update Drag&Drop
    for atom, offset in dragging.items():
        mouse_x, mouse_y = pygame.mouse.get_pos()
        atom.pos[0] = (mouse_x + offset[0])/100.0
        atom.pos[1] = (mouse_y + offset[1])/100.0

    # Update metrics
    system.update_metrics()

def draw(system):
    global buttons

    # Draw background
    _screen.fill(BACKGROUND_COLOR)

    # Draw lines connecting spheres
    for bound in system.bounds:
        atom1, atom2 = bound
        pygame.draw.line(_screen, (121, 158, 196), (atom1.pos[:2]*100).astype(int), (atom2.pos[:2]*100).astype(int), 1)

    # Draw spheres
    for atom in system.atoms:
        pygame.draw.circle(_screen, (174, 194, 198), (int(atom.pos[0]*100), int(atom.pos[1]*100)), SPHERE_RADIUS)

    # Draw the button
    for button in buttons:
        pygame.draw.rect(_screen, button.color, pygame.Rect(button.pos, button.size))
        text_surface = text_font.render(button.text, True, (0, 0, 0))
        text_size = text_surface.get_size()
        text_x = button.pos[0] + (button.size[0] - text_size[0]) / 2
        text_y = button.pos[1] + (button.size[1] - text_size[1]) / 2
        _screen.blit(text_surface, (text_x, text_y))

    # Draw the metrics
    for i, (name, mesures) in enumerate(system.metrics.items()):
        if name == 'coordinates' or len(mesures) == 0:
            last_mesure = 0
        else:
            last_mesure = mesures[-1]
        mesure_pos = (0, 70+i*40)
        mesure_size = (120, 40)
        pygame.draw.rect(_screen, (103, 121, 229), pygame.Rect(mesure_pos, mesure_size))
        text_surface = text_font.render(f'{name:13s}: {last_mesure:6.4f}', True, (0, 0, 0))
        text_size = text_surface.get_size()
        text_x = mesure_pos[0] + (mesure_size[0] - text_size[0]) / 2
        text_y = mesure_pos[1] + (mesure_size[1] - text_size[1]) / 2
        _screen.blit(text_surface, (text_x, text_y))

    # Update display
    pygame.display.flip()

    # Control the frame rate
    pygame.time.Clock().tick(60)
