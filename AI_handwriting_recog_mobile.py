from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import re
import base64
from io import BytesIO
# from matplotlib.pyplot import imread, imsave


app = Flask(__name__)
global model, graph

num_models = 4
models = [None] * num_models
models[0] = load_model(r'.\model\mnist_Basic_model_batchsize64_epoch35.h5')
models[1] = load_model(r'.\model\mnist_Intermediate1_model_batchsize64_epoch35.h5')
models[2] = load_model(r'.\model\mnist_Intermediate2_model_batchsize64_epoch35.h5')
models[3] = load_model(r'.\model\mnist_Advanced_model_batchsize64_epoch35.h5')

flag_save_img_file = False
models_choice_dict = {'basic': 0, 'intermediate-I': 1, 'intermediate-II': 2, 'advanced': 3}
selected_model = 3

def parseImageFromInternet(imgDataStr):
    imgstr = re.search(r'base64,(.*)', str(imgDataStr)).group(1)
    if flag_save_img_file:
        with open('save_img_from_html.png', 'wb') as output:
            output.write(base64.b64decode(imgstr))
        img = Image.open('save_img_from_html.png')
        x = np.array(img.resize(size=(28, 28)))
        x = x[:, :, 0] / 255  # normalize to 0 - 1, the same way as the x_train that were previously used during training
    else:
        img = np.array(Image.open(BytesIO(base64.b64decode(imgstr))))
        x = img[:, :, 0] / 255
    down_ratio = 10
    target_size = x.shape[0] // down_ratio
    x = x.reshape(-1, down_ratio, target_size, down_ratio).mean((-1, -3))
    x[x < 0.25] = 0
    x[x >= 0.25] = 1
    # imsave('final_image_to_classify.jpg', x)
    return x


@app.route('/')
def index():
    return render_template(r'index.html')


@app.route('/predict/', methods=['POST'])
def predict():
    global selected_model
    # Predict function will be called when the users click  "Recognize" button
    # on the mobile webpage (transmitted over internet using http with POST method).
    imgDataStr = request.get_data()
    x = parseImageFromInternet(imgDataStr)
    x = x.reshape(1, 28, 28, 1)
    out = models[selected_model].predict(x)
    response = np.argmax(out, axis=1)
    return str(response[0])


@app.route('/changemodel/', methods=['POST'])
def changemodel():
    global selected_model
    # changemodel function will be called when the users change the drop-down menu of "select-model"
    # on the webpage (transmitted over internet using http with POST method).
    modelChoice = request.get_data().decode('utf-8')
    selected_model = models_choice_dict[modelChoice]
    if selected_model not in range(num_models):
        selected_model = num_models - 1
    return str(modelChoice)  # return string instead of index


if __name__ == "__main__":
    # Note: update the host ip address accordingly
    app.run(host='192.168.1.82', port=8080)
