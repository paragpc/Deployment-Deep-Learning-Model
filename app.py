from __future__ import division, print_function
import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('inception.h5')
print('Yoga pose classifier model loaded.')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict_yoga_pose', methods=['POST'])
def predict_yoga_pose():
    if request.method == 'POST':
        f = request.files['file']
        current_dir_path = os.path.dirname(__file__)
        image_file_path = os.path.join(
            current_dir_path, 'uploads', secure_filename(f.filename))
        f.save(image_file_path)

        img = image.load_img(image_file_path, target_size=(299, 299))
        preprocessed_img = image.img_to_array(img)
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        preprocessed_img = preprocess_input(preprocessed_img)
        predictions = model.predict(preprocessed_img)
        print("Prediction probabilities:" + str(predictions[0]))

        positions = {0: 'DownwardFacingDog',
                     1: 'LowLunge',
                     2: 'Planks',
                     3: 'ReversePlanks',
                     4: 'SeatedForwardBend',
                     5: 'SidePlanks',
                     6: 'TreePose',
                     7: 'TrianglePose',
                     8: 'WarriorPose'}
        cur_max_ind = 0
        cur_max = 0
        for j in range(len(predictions[0])):
            if predictions[0][j] > cur_max:
                cur_max = predictions[0][j]
                cur_max_ind = j
        return positions[cur_max_ind]
    return None


if __name__ == '__main__':
    app.run(debug=True)

