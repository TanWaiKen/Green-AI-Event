# pip install gradio_client --user
from gradio_client import Client, handle_file

client = Client("https://faceonlive-face-liveness-detection-sdk.hf.space/")
result = client.predict(
		frame=handle_file('car.jpg'),
		api_name="/face_liveness"
)
print(result)
