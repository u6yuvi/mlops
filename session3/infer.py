import requests
import torch
import random
from torchvision import datasets
from torchvision import datasets, transforms
from PIL import Image
from io import BytesIO
import base64
import os
import concurrent.futures
import time
from model import Net
# Create requests directory if it doesn't exist
os.makedirs('requests', exist_ok=True)

# Load MNIST dataset
dataset = datasets.MNIST('./mnist/data', train=False, download=True)

# Load the model
model = Net()
model.load_state_dict(torch.load("./mnist/model/mnist_cnn.pt"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def run_inference():
    indexes = random.sample(range(0,len(dataset)),5)
    # Select a random image from the dataset for testing
    for index in indexes:
        image, label = dataset[index]

        # Save the image in a buffer
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        with torch.no_grad():
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension
            prediction = model(image)
            predicted_class = torch.argmax(prediction, dim=1)


        # Encode the image in base64 and send a POST request
        img_str = base64.b64encode(buffer.read()).decode('utf-8')
        #response = requests.post('http://localhost:8000/predict/', files={'file': ("image.png", BytesIO(base64.b64decode(img_str)), 'image/png')})

        # print(f'Index: {index}, Label: {label}, Prediction: {response.json()["class"]}')

        # Save the image to the requests directory
        filename = f'predicted_{predicted_class}_label_{label}_index_{index}.png'
        with open(f'./mnist/results/{filename}', 'wb') as f:
            f.write(base64.b64decode(img_str))

    # return index

# # Number of requests to make in the load test
# num_requests = 1_000

# start_time = time.time()

# # Use a ThreadPoolExecutor to perform the requests in parallel
# with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
#     indices = random.sample(range(len(dataset)), num_requests)
#     futures = [executor.submit(make_request, index) for index in indices]

#     for future in concurrent.futures.as_completed(futures):
#         print(f'Completed request {future.result()}')

# end_time = time.time()
# duration = end_time - start_time
# requests_per_second = num_requests / duration

# print(f'Completed {num_requests} in {duration} seconds')
# print(f'{requests_per_second} requests per second')

if __name__ == "__main__":
    run_inference()
