import pymol
from pymol import cmd
import os
from PIL import Image

# Initialize PyMOL
pymol.finish_launching()

# Load your molecular structure
cmd.load("animation.pdb")

# Set up the viewport
cmd.viewport(800, 600)

# Delete all representations
cmd.hide("all")
# Show sticks representation
cmd.show("sticks")
cmd.zoom("all", 0.7)

# Define a function to render each frame
def render_frames():
    num_frames = cmd.count_states()
    images = []
    for frame in range(1, num_frames + 1, 10):
        # Set the frame
        cmd.frame(frame)
        # Use the ray function to render the scene
        cmd.ray()
        # Save the image
        image_file = f"frame_{frame}.png"
        cmd.png(image_file)
        images.append(image_file)
    return images

# Call the function to render each frame
images = render_frames()

# Combine images into a GIF file
gif_file = "animation.gif"
with Image.open(images[0]) as first_image:
    first_image.save(gif_file, save_all=True, append_images=[Image.open(img) for img in images], loop=0, duration=10, disposal = 2)

for img in images:
    os.remove(img)