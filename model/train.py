import numpy as np
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from data_processing import load_data, augment_data
from alexnet import alexnet  # Hoặc từ model_lenet import le_net

# Load dataset
train_images, train_labels = load_data('train_images_alexnet.npy', 'train_labels_alexnet.npy')

# Split data
perm = np.random.permutation(len(train_images))
train_images, train_labels = train_images[perm], train_labels[perm]
val_images, val_labels = train_images[:1000], train_labels[:1000]
new_train, new_labels = train_images[1000:], train_labels[1000:]

# Model initialization
model = alexnet()
model.compile(loss='categorical_crossentropy', optimizer="Adam", metrics=['accuracy'])

# Callbacks
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), patience=2, min_lr=0.5e-6)
csv_logger = CSVLogger('train_log.csv')
early_stopper = EarlyStopping(min_delta=0.001, patience=30)
model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', save_best_only=True)

# Data Augmentation
datagen = augment_data()
datagen.fit(new_train)

# Train
model.fit(datagen.flow(new_train, new_labels, batch_size=12),
          steps_per_epoch=len(new_train) // 12,
          validation_data=(val_images, val_labels),
          epochs=30, callbacks=[lr_reducer, early_stopper, csv_logger, model_checkpoint])
