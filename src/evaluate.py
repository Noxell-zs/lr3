from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

model = load_model("model/model1.keras")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory='./data/evaluation',
    target_size=(128,128),
    classes=['food','non_food'],
    class_mode="categorical",
    color_mode="rgb",
    shuffle=False
)

result = model.evaluate(test_generator)

with open('metrics/accuracy1.txt', 'w+', encoding='utf-8') as file:
    file.write(str(result[1]))
