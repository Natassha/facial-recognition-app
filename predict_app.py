import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from keras_vggface.vggface import VGGFace
import mtcnn
from mtcnn.mtcnn import MTCNN
import PIL
import os
import numpy as np
import cv2
import urllib.request
from PIL import Image
import base64
from PIL import Image
from io import BytesIO
from keras import backend as K

app = Flask(__name__)

# load the image

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True) # this gets all the request data
    imgsrcstring = message['image']
    imgdata = imgsrcstring.split(',')[1]
    decoded = base64.b64decode(imgdata)
    img = np.array(Image.open(BytesIO(decoded)))
    detector = MTCNN()
    target_size = (224,224)
    border_rel = 0
    detections = detector.detect_faces(img)
    # creating the model to predict the celebrity
    x1, y1, width, height = detections[0]['box']
    dw = round(width * border_rel)
    dh = round(height * border_rel)
    x2, y2 = x1 + width + dw, y1 + height + dh
    face = img[y1:y2, x1:x2]
    face = PIL.Image.fromarray(face)
    face = face.resize((224, 224))
    face = np.asarray(face)
    # convert to float32
    face_pp = face.astype('float32')
    face_pp = np.expand_dims(face_pp, axis = 0)
    from keras_vggface.utils import preprocess_input
    face_pp = preprocess_input(face_pp, version = 2)
    # Create the resnet50 Model
    model = VGGFace(model= 'resnet50')
    # predict the face with the input
    prediction = model.predict(face_pp)
    # convert predictions into names & probabilities
    from keras_vggface.utils import decode_predictions
    results = decode_predictions(prediction)
    results = decode_predictions(prediction,top=1)
    s = results[0][0][0]
    # regex to get everything between the '' and remove the underscore
    b = (s.split("' "))[1].split("':")[0]
    # replace the underscore with a space:
    c = b.replace("_", " ")
    response = c.replace("'", '')
    response = 'You look like '+response+'!'
    return(response)

if __name__ == "__main__":
    app.run(port='8086',threaded=False)
