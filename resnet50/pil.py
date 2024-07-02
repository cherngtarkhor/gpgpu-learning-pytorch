from PIL import Image
import os

# print(__file__)
# print(os.path.abspath(__file__))
# print(os.path.dirname(os.path.abspath(__file__)))
# print(os.getcwd())

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, "black.png")
saved_path = os.path.join(script_dir, "saved.png")

print(image_path)
print(saved_path)

file = "/app/test1/resnet50/black.png"
img = Image.open(image_path)
#img.show()
img.save(saved_path)
