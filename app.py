from flask import Flask, render_template
import numpy as np 
import tensorflow as tf
from PIL import Image
import base64
from io import BytesIO
import cv2
import os
import json
import random

import matplotlib
matplotlib.use('Agg')
import plotly
import plotly.graph_objs as go
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# import image processing
from image_utils import crop_image, normalize_image, convert_to_rgb, convert_to_np


# Dictionary with label codes
label_dict = {0:'ant', 1:'bat', 2:'bear', 3:'bee', 4:'butterfly', 
              5:'camel', 6:'cat', 7:'cow', 8:'crab', 9:'crocodile',
              10:'dog', 11:'dolphin', 12:'dragon', 13:'duck', 14:'elephant', 
              15:'flamingo', 16:'frog', 17:'giraffe', 18:'hedgehog', 19:'horse', 
              20:'kangaroo', 21:'lion', 22:'lobster', 23:'monkey', 24:'mosquito', 
              25:'mouse', 26:'octopus', 27:'owl', 28:'panda', 29:'parrot', 
              30:'penguin', 31:'pig', 32:'rabbit', 33:'raccoon', 34:'rhinoceros', 
              35:'scorpion', 36:'sea turtle', 37:'shark', 38:'sheep', 39:'snail', 
              40:'snake', 41:'spider', 42:'squirrel', 43:'swan', 44:'tiger', 
              45:'whale', 46:'zebra'}

def animal_picker():
    random_key = random.randint(0, 47)
    random_animal = label_dict[random_key]
    return random_animal, random_key

def loading_model(filepath='model/model_h5.h5'):
    print("Loading model from {} \n".format(filepath))
    model = load_model(filepath)
    graph = tf.compat.v1.get_default_graph()
    return model, graph

def make_prediction(model, input):
    input = cv2.resize(input, (96, 96))
    img_array = img_to_array(input)
    img_array = np.expand_dims(img_array, 0)
    preds = model.predict(img_array)[0]
    ind = (-preds).argsort()[:10]
    top_10_animals = [label_dict[x] for x in ind]
    label = ind[0]
    label_name = top_10_animals[0]
    preds.sort()
    top_10_values = preds[::-1][:10]
    return label, label_name, ind, preds, top_10_animals, top_10_values

app = Flask(__name__)
# load model
model, graph = loading_model()

@app.route('/')
@app.route('/index')
def index():
    random_animal, random_key = animal_picker()
    return render_template('index.html', random_animal=random_animal)


@app.route('/go/<dataURL>')
def pred(dataURL):

    # decode base64  '._-' -> '+/='
    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')

    # get the base64 string and convert string to bytes
    image_b64_str = dataURL
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    
    # open Image with PIL and convert image
    img = Image.open(image_data)
    img = img.convert("RGBA")
    image_cropped = crop_image(img) 
    image_normalized = normalize_image(image_cropped) 
    img_rgb = convert_to_rgb(image_normalized)
    image_np = convert_to_np(img_rgb) 

    # apply model and print prediction
    label, label_num, ind, preds, top_10_animals, top_10_values = make_prediction(model, image_np)
    print("This is a {}".format(label_num))

    # plt.style.use('tableau-colorblind10')
    # x = top_10_animals
    # y = top_10_values
    # sns.barplot(y, x)
    # plt.savefig('top10.png')

    # create plotly visualization
    graphs = [
        #plot with probabilities for each class of images
        {'data': [go.Bar(x = preds.ravel().tolist(),
                         y = [label_dict[pred] for pred in ind[::-1]],
                         orientation = 'h')],
         'layout': {'title': 'Class Probabilities',
                    'yaxis': {'title': "Classes"},
                    'xaxis': {'title': "Probability", }
            }
        }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render the hook.html passing prediction resuls
    return render_template(
        'hook.html',
        result = label_num, # predicted class label
        ids=ids, # plotly graph ids
        graphJSON=graphJSON, # json plotly graphs
        dataURL = dataURL # image to display with result)

if __name__ == '__main__':
    port=int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)