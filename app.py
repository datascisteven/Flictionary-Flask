from flask import Flask, render_template, request
import numpy as np 
import tensorflow as tf
from PIL import Image
import base64
import re
from io import BytesIO
import cv2
import time
import os
import json

from matplotlib.pyplot import imshow
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly
import chart_studio.plotly as py
import plotly.graph_objs as go

# import image processing
import sys
from image_utils import crop_image, normalize_image, convert_to_rgb, convert_to_np
from tensorflow.keras.models import load_model

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


def loading_model(filepath='model/h5/model_h5.h5'):
    print("Loading model from {} \n".format(filepath))
    model = load_model(filepath)
    graph = tf.compat.v1.get_default_graph()
    return model, graph

def make_prediction(model, input):
    input = cv2.resize(input, (96, 96))
    pred = model.predict(np.expand_dims(input, axis=0))[0]
    preds = (-pred).argsort()[:10]
    top_10 = [label_dict[x] for x in preds]
    label = preds[0]
    label_name = top_10[0]
    return label, label_name, preds

# def view_classify(img, preds):
#     preds = preds.squeeze()
#     fig, (ax1, ax2) = plt.subplots(figsize=(12,18), ncols=2)
#     ax1.imshow(img.squeeze())
#     ax1.axis('off')
#     ax2.barh(np.arange(47), preds)
#     ax2.set_aspect(0.1)
#     ax2.set_yticks(np.arange(10))
#     ax2.set_yticklabels(['bee', 'cat', 'cow', 'dog', 'duck', 'horse', 'pig', 'rabbit', 'snake', 'whale'], size='small');
#     ax2.set_title('Class Probability')
#     ax2.set_xlim(0, 1.1)

#     plt.tight_layout()

#     ts = time.time()
#     plt.savefig('prediction' + str(ts) + '.png')


app = Flask(__name__)
# load model
model, graph = loading_model()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/go/<dataURL>')
def pred(dataURL):

    # decode base64  '._-' -> '+/='
    dataURL = dataURL.replace('.', '+')
    dataURL = dataURL.replace('_', '/')
    dataURL = dataURL.replace('-', '=')

    # get the base64 string and  convert string to bytes
    image_b64_str = dataURL
    byte_data = base64.b64decode(image_b64_str)
    image_data = BytesIO(byte_data)
    
    # open Image with PIL and save original image as png (for debugging)
    img = Image.open(image_data)
    ts = time.time()
    img.save('image' + str(ts) + '.png', 'PNG')

    # convert image to RGBA and preprocess image for model
    img = img.convert("RGBA")
    image_cropped = crop_image(img) # crop image and resize to 28x28
    image_normalized = normalize_image(image_cropped) # normalize color after crop
    img_rgb = convert_to_rgb(image_normalized) # convert image from RGBA to RGB
    image_np = convert_to_np(img_rgb) # convert image to numpy

    # apply model and print prediction
    label, label_num, preds = make_prediction(model, image_np) # need to change
    print("This is a {}".format(label_num))

    # save classification results as a diagram
    # view_classify(image_np, preds)

    # create plotly visualization
    graphs = [
        #plot with probabilities for each class of images
        {
            'data': [
                go.Bar(
                        x = preds.ravel().tolist(),
                        y = [label_dict[pred] for pred in preds],
                        orientation = 'h')
            ],

            'layout': {
                'title': 'Class Probabilities',
                'yaxis': {
                    'title': "Classes"
                },
                'xaxis': {
                    'title': "Probability",
                }
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
        dataURL = dataURL # image to display with result
    )

if __name__ == '__main__':
    port=int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)