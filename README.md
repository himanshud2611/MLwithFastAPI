MNIST Classifier API using FastAPI, featuring a Random Forest model trained on the MNIST dataset using scikit-learn. Implemented image preprocessing pipelines and CORS middleware for cross-origin requests.

How to use this?
1. Fork the repo in your local machine
2. run train_model.py. This will serialize MNIST model and return the accuracy of model.
3. run main.py to check the execution
4. run uvicorn main:app --reload to communicate with FASTApi using ASGI
5. run index.html to input images and clissfy the digits
