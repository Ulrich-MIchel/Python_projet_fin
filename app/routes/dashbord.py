from app import app
import os
import numpy as np
from app.util import base64_to_pil
from PIL import Image, ImageOps  
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow
import tensorflow as tf
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image



# Variables 
# Change them if you are using custom model or pretrained model with saved weigths
#Model_json = ".json"#
#Model_weigths = ".h5"#

#app.config['UPLOAD_FOLDER'] = 'upload'

# Declare a flask app
#app = Flask(__name__)


models = load_model("app/model/keras_model.h5", compile=False)

# Load the labels
class_names = open("app/model/labels.txt", "r").readlines()


def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds

def prediction_star(img_path, models= models):
  # Create the array of the right shape to feed into the keras model
  # The 'length' or number of images you can put into the array is
  # determined by the first position in the shape tuple, in this case 1
  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  # Replace this with the path to your image
  #image = Image.open(img_path).convert("RGB")###
  image = img_path.convert("RGB")

  # resizing the image to be at least 224x224 and then cropping from the center
  size = (224, 224)
  image = ImageOps.fit(image, size,Image.LANCZOS)

  # turn the image into a numpy array
  image_array = np.asarray(image)

  # Normalize the image
  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  # Load the image into the array
  data[0] = normalized_image_array
  # Predicts the model
  prediction = models.predict(data)
  index = np.argmax(prediction)
  class_name = class_names[index]
  confidence_score = prediction[0][index]
  # Print prediction and confidence score
  #print("Class:", class_name[2:], end="")
  #print("Confidence Score:", confidence_score)
  print(confidence_score)

  return class_name[2:], confidence_score

@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('dashbord.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        
        # Get the image from post request
        img = base64_to_pil(request.json)
        
        # initialize model
        #model = get_ImageClassifierModel()#

        # Make prediction
        #preds = model_predict(img, model)#
        preds,img = prediction_star(img)

        #pred_proba = "{:.3f}".format(np.amax(preds))#    # Max probability
        #pred_class = decode_predictions(preds, top=1)#   # ImageNet Decode

        #result = str(pred_class[0][0][1])#               # Convert to string
        #result = result.replace('_', ' ').capitalize()
        #print(preds)
        # Serialize the result, you can add additional fields
        if img >= 0.98:
            return jsonify(result=preds)
        else:
            msg = "L'image que vous avez renseignée n'est pas assez claire et ou la personne n'est pas une star"
            return jsonify(result = msg)
    return None


@app.route('/demo', methods=['GET'])
def demo():
    '''
    Render the main page
    '''
    return render_template('demo.html')

@app.route('/description', methods=['GET'])
def description():
    '''
    Render the main page
    '''
    return render_template('description.html')
