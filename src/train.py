from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory='./data/training',
    target_size=(128,128),
    classes=['food','non_food'],
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    directory='./data/validation',
    target_size=(128,128),
    classes=['food','non_food'],
    class_mode="categorical",
    color_mode="rgb",
    shuffle=True
)

early_stopping=EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min'
)

model = Sequential((
    Conv2D(64,
           kernel_initializer='he_normal',
           kernel_size=(3,3),
           input_shape=(128,128,3),
           activation='relu'),
    MaxPool2D(2,2),

    Conv2D(128,
           kernel_initializer='he_normal',
           kernel_size=(3,3),
           activation='relu'),
    MaxPool2D(2,2),

    Conv2D(256,
           kernel_initializer='he_normal',
           kernel_size=(3,3),
           activation='relu'),
    MaxPool2D(2,2),

    Flatten(),
    Dense(512,activation='relu'),
    Dense(2,activation='softmax')
))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

model.save("model/model1.keras")
