import os
import keras
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import numpy as np
import tensorflow 
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


model = load_model("models/tobacco_leaf_cnn_improved.h5")


class_labels = ["Hypermature", "Immature", "Mature"]


def predict_leaf(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    #confidence = round(np.max(prediction) * 100, 2)

    return class_labels[class_index]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Save Image
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            # Predict
            predicted_class = predict_leaf(file_path)

            return render_template("index.html", uploaded_image=file_path, result=predicted_class)

    return render_template("index.html", uploaded_image=None, result=None)


if __name__ == "__main__":
    app.run(debug=False)