from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img  


datagen = ImageDataGenerator(rotation_range=50,
                             horizontal_flip=True,
                             vertical_flip=True,
                             zoom_range=0.5,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             brightness_range=[0.5, 1.5], fill_mode='nearest')

img = load_img('images.jpeg')
x = img_to_array(img)
x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='aug', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break