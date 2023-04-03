import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('lab5_model')

names = ['beagle', 'borzoi', 'chihuahua', 'pug', 'rottweiler']

while True:
    path = input('image path: ')

    if path == 'exit':
        break

    path = os.path.join(os.getcwd(), path)

    if (not os.path.exists(path)) or (not os.path.isfile(path)):
        print('no such file found')
        continue

    img = tf.keras.utils.load_img(path, target_size=(150, 150))
    img = tf.keras.utils.img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    
    num = np.argmax(pred)
    print(f'image ({path}) is a {names[num]}')