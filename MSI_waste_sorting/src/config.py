#root is ..
# __file__ → path to config.py
# .resolve() → absolute path (safe)
# .parents[1] → go up two levels:
# [0] → src
# [1] → MSI_waste_sorting

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
