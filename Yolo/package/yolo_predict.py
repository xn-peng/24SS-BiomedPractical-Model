from ultralytics import YOLO
from PIL import Image

model = YOLO("yolo_v5_20.pt")

# Define path to the image file
source = "0398_slice_1_84.png"

# Run inference on the source
results = model(source)  # list of Results objects

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Save results to disk
    #r.save(filename=f"results{i}.jpg")




