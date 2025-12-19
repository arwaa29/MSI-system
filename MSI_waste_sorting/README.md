to load files of pkl and run:

run this dependency:pip install numpy scikit-learn joblib

import joblib

# Load model
svm_model = joblib.load(MODEL_PATH)

print("SVM model loaded successfully")
print(svm_model)

knn_model = joblib.load("../models/knn_best.pkl")
scaler = joblib.load("../models/scaler_knn.pkl")
pca = joblib.load("../models/pca_knn.pkl")

print(knn_model)
print(scaler)
print(pca)
