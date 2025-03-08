import tensorflow as tf
import segmentation_models as sm

IMAGE_SHAPE = (304, 304)
N_CHANNELS = 1
N_CLASSES = 1

EPOCHS = 100
BATCH_SIZE = 4

VERBOSE = 0

precision_score = tf.keras.metrics.Precision(name="precision", thresholds=0.5)
recall_score = tf.keras.metrics.Recall(name="recall", thresholds=0.5)
f_score = sm.metrics.FScore(threshold=0.5, per_image = True)
iou_score = sm.metrics.IOUScore(threshold=0.5, per_image = True)

DATASET_PATH = "datasets/304x304"