from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import torch
from torchvision import transforms
from io import BytesIO
from PIL import Image
from model.model import ASLClassifier
from typing import List
import cv2

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and map it to the appropriate device
modelASL = ASLClassifier(3, 10, 29)
modelASL.load_state_dict(torch.load("model/model.pth", map_location=device, weights_only=True))
modelASL.to(device)
modelASL.eval()

transform = transforms.Compose([
    transforms.Resize([255, 255]),
    transforms.ToTensor()
])


@app.get('/ping')
async def ping():
    return 'I am born'


def read_file_as_image(databyte):
    image = Image.open(BytesIO(databyte))
    imageAsTensor = transform(image).unsqueeze(dim=0)
    return imageAsTensor, image


@app.post("/predict")
async def predict_end(file: UploadFile = File(...)):
    input_batch, image = read_file_as_image(await file.read())
    print(input_batch.size())
    with torch.inference_mode():
        output = modelASL(input_batch)
        # Get the predicted class
        predicted_class = torch.argmax(output, dim=1).item()

        # Return all predictions as a list
    return {"predictions": predicted_class}


if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8000)
