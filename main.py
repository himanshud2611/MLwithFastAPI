import io
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Load the trained model
with open('mnist_model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the MNIST prediction API"}


@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')  # Open byte stream and convert to grayscale

    pil_image = PIL.ImageOps.invert(pil_image)  # Invert the image
    pil_image = pil_image.resize((28, 28), PIL.Image.LANCZOS)  # Resize to 28x28
    img_array = np.array(pil_image).reshape(1, -1)  # Flatten and reshape

    # Normalize if needed
    #img_array = img_array / 255.0


    prediction = model.predict(img_array)
    return {"prediction": int(prediction[0])}


# uvicorn main:app --reload
#if __name__ == "__main__":
 #   uvicorn.run(app, host="0.0.0.0", port=8000)

