#root is ..
# __file__ → path to config.py
# .resolve() → absolute path (safe)
# .parents[1] → go up two levels:
# [0] → src
# [1] → MSI_waste_sorting

from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
data_dir = root_dir / "dataset"
raw_dir = data_dir/ "raw"
aug_dir = data_dir / "augmented"
processed_dir = data_dir / "processed"
models_dir = root_dir / "models"

numberOfClasses = 7

dictionaryMapping = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash",
    6: "unknown"
}
reverseMapping = {v: k for k, v in dictionaryMapping.items()}
unknownID = 6
trainable = {0, 1, 2, 3, 4, 5}

imageSize = (224, 224)
color = "RGB"

# whether to extract color-based descriptors
# whether to extract texture descriptors
# whether to extract edge/shape descriptors

colorFeature = True
textureFeature = True
edgeFeature = True

colorFeatureMethod = "HSV"
textureFeatureMethod = "LBP"
edgeFeatureMethod = "HOG"

combinedFeature = "concatenate"

fixed_length = True

# ratios must sum to 1
train_ratio = 0.7
valid_ratio = 0.3
test_ratio = 0.0
stratified_split = True

# every run gives different results
# can't reproduce accuracy
#with random seed same spit, same training behavior, same reported numbers

# best practice is simple fixed integer
random_seed = 42

# training images are modified
augmentation_enabled = True
augmentation_target_per_class = 500

# rotating image is allowed with small range
rotation_range = (-10, 10)
# mirror the image
horizontal_flip = True
vertical_flip = False

#zooming
# also need to be small range
# original size 1.0
# 10% zoom in or out
scaling_range = (0.9, 1.1)
brightness_range = (0.8, 1.2)
contrast_range = (0.8, 1.2)
saturation_range = (0.8, 1.2)
noise_level = 0.01
blur_probability = 0.1

feature_scaling_enabled = True
# subtract mean divide by standard deviation
scaling_method = "standard"
scaler_persistence = True

# HSV has H (Hue), S (Saturation), V (Value)
color_hist_bins = (16, 16, 16)
# for HSV H: 0 → 180 (OpenCV), S: 0 → 255, V: 0 → 255
color_hist_range = ((0,180), (0,255), (0,255))
color_channels = ("H", "S", "V")

# far from the center pixel
# 1 micro , 2 slightly larger
lbp_radius = 1
# neighbors
# points = 8 * radius
lbp_points = 8
# encoded patterns
lbp_method = "uniform"

# edge directions
hog_orientation = 9
# pixels per cell common 8x8
hog_pixels = (8,8)
#cells per block common 2x2
hog_cells = (2,2)
#blocks overlap
hog_block = "L2-Hys"

svm_kernel = "rbf"
# margin hardness low -> softer margin, high -> overfitting
svm_C = 1.0
svm_gamma = "scale"
svm_degree = 3
svm_probability_enabled = True
#one-vs-rest, required for rejection logic
svm_multiclass_strategy = "ovr"

knn_k = 5
knn_distance_metric = "euclidean"
knn_weighting = "distance"
knn_search_algorithm = "auto"

rejection_enabled = True
rejection_threshold = 0.6
rejection_strategy = "max_probability"

primary_metric = "accuracy"
secondary_metrics = ["precision", "recall", "f1"]
confusion_matrix_enabled = True

camera_fps_target = 30
max_inference_time_ms = 50
frame_skip = 1

experiment_name = "baseline_hsv_lbp_hog_svm_knn"
save_predictions = True
save_confusion_matrix = True