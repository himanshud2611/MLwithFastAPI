MNIST Classifier API using FastAPI, featuring a Random Forest model trained on the MNIST dataset using scikit-learn. Implemented image preprocessing pipelines and CORS middleware for cross-origin requests.
The API is fully integrated with the frontend web interface. The user can upload an image, and the frontend code will communicate with the backend API to get the prediction result, which is then displayed on the web page.
The backend FastAPI application provides the */predict-image/* endpoint that accepts the uploaded image and returns the prediction result.

How to use this?
1. Fork the repo in your local machine
2. run train_model.py. This will serialize MNIST model and return the accuracy of model.
3. run pip install -r requirements.txt to install the dependencies
4. run main.py to check the execution
5. run python -m uvicorn main:app --reload to communicate with FASTApi using ASGI
6. run index.html to input images and clissfy the digits
