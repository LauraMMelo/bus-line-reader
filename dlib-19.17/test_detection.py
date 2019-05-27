import os
import sys
import glob
from detector import detectar_letreiro
import dlib


bus_folder = "./bus_examples/"



# Now let's run the detector over the images in the faces folder and display the
# results.
print("Showing detections on the images in the bus folder...")
win = dlib.image_window()
for f in glob.glob(os.path.join(bus_folder, "*.jpg")):
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    print(type(img))
    print(img)
    dets = detectar_letreiro(img)
    print("Number of letreiros detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()

