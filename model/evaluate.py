import numpy as np
from keras.models import load_model
from data_processing import load_data

# Load model
model = load_model("best_model.hdf5")

# Load validation/test data
test_images, test_labels = load_data('test_images.npy', 'test_labels.npy')

# Evaluate
loss, accuracy = model.evaluate(test_images, test_labels, verbose=1)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
