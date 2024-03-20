import sys
import pygame

from minimize import *

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
SPHERE_RADIUS = 20
BACKGROUND_COLOR = (96, 124, 141)

class Button:
    def __init__(
        self, pos=(50,50),
        size=(70,30),
        color=(120,170,172),
        text="hello",
        callback=lambda:0
        ):
        self.pos = pos
        self.size = size
        self.color = color
        self.text = text
        self.callback = callback

class SystemGui:
    def __init__(self, system: System):
        pygame.init()

        self.screen: pygame.Surface = None # type: ignore
        self.dragging = {}

        self.text_font = pygame.font.Font(None, 14)

        # Create the screen
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Molecular Dynamics App")

        self.system = system
        self.initial_coords = self.system.atoms_coordinates()
        self.system.reset_metrics()

        self.is_minimizing = False
        self.btn_minimize = Button(pos=(0, 0), text="minimize")
        self.btn_minimize.callback = self.toggle_minimize

        self.btn_plot = Button(pos=(100, 0), text="plot")
        self.btn_plot.callback = self.system.plot_all

        self.btn_reset = Button(pos=(200, 0), text="reset")
        self.btn_reset.callback = self.reset_playground

        self.btn_save = Button(pos=(300, 0), text="save")
        self.btn_save.callback = self.system.save

        self.buttons = [
            self.btn_minimize,
            self.btn_plot,
            self.btn_reset,
            self.btn_save
        ]

    def run(self):
        while True:
            if self.is_minimizing:
                self.system.step_minimize()
            else:
                self.system.update_all_distances()
                for atom in self.system.atoms:
                    self.system.gradient(atom)
                self.system.update_metrics()

            self.update()
            self.draw()

    def update(self):
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for button in self.buttons:
                    if pygame.Rect(button.pos, button.size).collidepoint(mouse_x, mouse_y):
                        button.callback()

                for i, atom in enumerate(self.system.atoms):
                    ax, ay = atom.pos[:2]*100
                    distance = pygame.math.Vector2(mouse_x - ax, mouse_y - ay).length()
                    if distance < SPHERE_RADIUS:
                        self.dragging[atom] = (ax - mouse_x, ay - mouse_y)
            elif event.type == pygame.MOUSEBUTTONUP:
                self.dragging.clear()

        # Update Drag&Drop
        for atom, offset in self.dragging.items():
            mouse_x, mouse_y = pygame.mouse.get_pos()
            atom.pos[0] = (mouse_x + offset[0])/100.0
            atom.pos[1] = (mouse_y + offset[1])/100.0

    def toggle_minimize(self):
        if self.is_minimizing:
            self.is_minimizing = False
            self.btn_minimize.color = (120, 170, 172)
        else:
            self.system.reset_metrics()
            self.is_minimizing = True
            self.btn_minimize.color = (164, 202, 202)

    def reset_playground(self):
        for i, atom in enumerate(self.system.atoms):
            atom.pos = self.initial_coords[i]

        self.system.reset_metrics()
        self.is_minimizing = False
        self.btn_minimize.color = (120, 170, 172)

    def draw(self):
        # Draw background
        self.screen.fill(BACKGROUND_COLOR)

        # Draw lines connecting spheres
        for bond in self.system.bonds:
            atom1, atom2 = bond
            pygame.draw.line(self.screen, (121, 158, 196), (atom1.pos[:2]*100).astype(int), (atom2.pos[:2]*100).astype(int), 1) # type: ignore

        # Draw spheres
        for atom in self.system.atoms:
            pygame.draw.circle(self.screen, (174, 194, 198), (int(atom.pos[0]*100), int(atom.pos[1]*100)), SPHERE_RADIUS)

        # Draw the button
        for button in self.buttons:
            pygame.draw.rect(self.screen, button.color, pygame.Rect(button.pos, button.size))
            text_surface = self.text_font.render(button.text, True, (0, 0, 0))
            text_size = text_surface.get_size()
            text_x = button.pos[0] + (button.size[0] - text_size[0]) / 2
            text_y = button.pos[1] + (button.size[1] - text_size[1]) / 2
            self.screen.blit(text_surface, (text_x, text_y))

        # Draw the metrics
        self.draw_stats(f'{"steps":>11s}: {self.system.step:<6d}', (0, 70))
        for i, (name, mesures) in enumerate(self.system.metrics.items()):
            if name in ['coordinates', 'gradients'] or len(mesures) == 0:
                last_mesure = 0
            else:
                last_mesure = mesures[-1]
            self.draw_stats(f'{name:>11s}: {last_mesure:<6.4f}', (0, 70+(i+1)*40))

        # Update display
        pygame.display.flip()

        # Control the frame rate
        # pygame.time.Clock().tick(120)

    def draw_stats(self, text: str, pos: tuple[int, int]):
        size = (120, 40)
        pygame.draw.rect(self.screen, (103, 121, 229), pygame.Rect(pos, size))
        text_surface = self.text_font.render(text, True, (0, 0, 0))
        text_size = text_surface.get_size()
        text_x = 5
        text_y = pos[1] + (size[1] - text_size[1]) / 2
        self.screen.blit(text_surface, (text_x, text_y))

if __name__ == '__main__':
    from water_system import system

    # 2D coordinates only
    for atom in system.atoms:
        atom.pos = atom.pos[:2]

    SystemGui(system).run()