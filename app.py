from flask import Flask, request, render_template
import pickle
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained SVM model and PCA transformers
with open('svm_algo_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('pca_transformer_hist.pkl', 'rb') as pca_file:
    pca_hist = pickle.load(pca_file)

with open('pca_transformer_canny.pkl', 'rb') as pca_file:
    pca_canny = pickle.load(pca_file)

with open('pca_transformer_sift.pkl', 'rb') as pca_file:
    pca_sift = pickle.load(pca_file)


# Function to calculate histogram and flatten it
def calc_histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten()


# Function to detect edges using Canny and flatten the result
def canny_edge_detection(img):
    edges = cv2.Canny(img, 100, 200)
    return edges.flatten()


# Function to extract SIFT features and ensure the length is consistent
def sift_features(img, num_features=128):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    if descriptors is None:
        descriptors = np.zeros((num_features, 128))
    if len(descriptors) < num_features:
        descriptors = np.vstack([descriptors, np.zeros((num_features - len(descriptors), 128))])
    elif len(descriptors) > num_features:
        descriptors = descriptors[:num_features]
    return descriptors.flatten()


# Function to preprocess the uploaded image
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Resize image to 128x128 pixels

    # Compute features
    hist_features = calc_histogram(img)
    canny_features = canny_edge_detection(img)
    sift_features_vec = sift_features(img)

    # Apply PCA
    hist_pca = pca_hist.transform(hist_features.reshape(1, -1))
    canny_pca = pca_canny.transform(canny_features.reshape(1, -1))
    sift_pca = pca_sift.transform(sift_features_vec.reshape(1, -1))

    # Combine PCA features
    combined_features = np.hstack((hist_pca, canny_pca, sift_pca))

    return combined_features


@app.route('/', methods=['GET', 'POST'])
def upload_image(): 
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_pca = preprocess_image(file)
            prediction = model.predict(img_pca)
            return render_template("result.html", prediction=prediction[0])
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)

