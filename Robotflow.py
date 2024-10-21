from inference_sdk import InferenceHTTPClient
import cv2
import supervision as sv
from tkinter import filedialog
from tkinter import Tk
from PIL import Image

# Initiate default value
image = cv2.imread("car.jpg")
model_id = "vehicle-detection-bz0yu/4"


def select_image():
    # Hide root window
    root = Tk()
    root.withdraw()

    # Open file dialog to select image
    image_file = filedialog.askopenfilenames(
        title="Select File",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
    )
    return image_file


# Select an image from the file system
image_file = select_image()
if not image_file:
    print("No image selected.")
else:
    # Get the first selected image file from the tuple
    image_path = image_file[0]  # Use [0] to get the first file
    print(image_path)
    image = cv2.imread(image_path)

# Configure client
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",  # route for local inference server
    api_key="srDkA4gHUsoSDzV9siZ2",  # api key for your workspace
)

# Run inference
result = client.infer(image, model_id=model_id)

# Load results into Supervision Detection API
detections = sv.Detections.from_inference(result)

# Create Supervision annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Extract labels array from inference results
labels = [p['class'] for p in result['predictions']]

# Apply results to image using Supervision annotators
annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

# Write annotated image to file or display image
with sv.ImageSink(target_dir_path="./results/", overwrite=True) as sink:
    sink.save_image(annotated_image)
# or sv.plot_image(annotated_image)

# Display the annotated image in a pop-up window using OpenCV
image_path = ("C:/Users/tanwa/OneDrive - MSFT/TWK developer/Documents/PycharmProjects/"
              "pythonProject2/results/image_00000.png")
# Change this to your specific path

# Open the image in the default viewer
img = Image.open(image_path)
img.show()
