import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
#cv2: specialised library to help with image preprocessing e.g. resizing

'''
Performs onehot-encodings for each output class 
'''
def encode_onehot(pos, n_rows):
    # 4 classes
    y_onehot = [0] * 4
    # create onehot-encodings for 4 classes
    y_onehot[pos] = 1
    y_onehots = [y_onehot] * n_rows
    # convert python list to numpy array
    # as keras requires numpy array
    return np.array(y_onehots)

#Generating extra pictures for mixed fruits
'''def mixed_fruits_augmentation(path):
    for file in os.listdir(path):
        if file[0] == '.':  # skip hidden files
            continue

        # reading image file into memory
        image = Image.open("{}/{}".format(path, file))
        image_rot_90 = image.rotate(90)
        filename =  path + '/' + file[:-4] + '_rot_90.png'
        image_rot_90.save(filename)
        image_rot_90 = image.rotate(180)
        filename_1 =  path + '/' + file[:-4] + '_rot_180.png'
        image_rot_90.save(filename_1)
        image_flip = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        filename_2 = path + '/' + file[:-4] + '_image_flip.png'
        image_flip.save(filename_2)

'''
'''
Read image data one by one from each folder (class)
'''
def read_img_data(path):
    for file in os.listdir(path):
        if file[0] == '.':  # skip hidden files
            continue

        # reading image file into memory
        img = cv2.imread("{}/{}".format(path, file))
        #resize image into dimensions 64 by 64 (width,height) and default interpolation: inter-linear 
        resized = cv2.resize(img, dsize=(64,64))

        try:
            x_train = np.concatenate((x_train, resized))
            #concatenate the image
        except:
            x_train = resized   
            # initialise x_train first for first value before concatenation can occur

    # # -1 to let numpy computes the number of rows 
    return np.reshape(x_train, (-1, 64,64, 3))  
   
'''
Prepare data by folders (class). Save image data returned from read image function and encode the the folder class with one hot encoding method. 
1-apple 2-banana 3-orange 4-mixed
apple: y label: [1,0,0,0]
'''
def prep_data(paths):
    for i in range(len(paths)):
        data = read_img_data(paths[i])
        y_onehots = encode_onehot(i, data.shape[0])
        if i == 0:
            x = data
            y = y_onehots
        else:
            x = np.concatenate((x, data))
            y = np.concatenate((y, y_onehots))             
    
    #randomly shuffle and permutate the ordered pairs of data (x,y) with the indexes (len(x)): so that the validation split done later in the split model which takes the last 20% of train data for validation, will not seen the same classified 20 pictures in the validation set.
    shuffler = np.random.permutation(len(x))
    x = x[shuffler]
    y = y[shuffler]
    return x, y

'''
Prepare train data but adding the folder (class) path into the paths list.
'''
def prep_train_data():
    paths = []

    for i in range(4):
        paths.append('fruit_train/{}/'.format(i+1))

    return prep_data(paths)

'''
Prepare test data but adding the folder (class) path into the paths list.
'''
def prep_test_data():
    paths = []

    for i in range(4):
        paths.append('fruit_test/{}/'.format(i+1))

    return prep_data(paths)
'''
Create our model
'''
def create_model():
    
    model = tf.keras.Sequential()
    #model.add(tf.keras.layers.RandomContrast((0.8,2.5), seed=5))
    model.add(tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=5))
    model.add(tf.keras.layers.RandomRotation(0.2,fill_mode="reflect",interpolation="bilinear",seed=5,fill_value=0.0))
    #model.add(tf.keras.layers.RandomZoom( height_factor=(-0.3,-0.2),width_factor=None,fill_mode="reflect",interpolation="bilinear",seed=5,fill_value=0.0))
    #model.add(tf.keras.layers.RandomBrightness((-0.8,0.8), value_range=(0, 255), seed=5))
    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(7, 7), activation='relu', input_shape=(64,64,3)))
    model.add(tf.keras.layers.BatchNormalization(axis=1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Conv2D(filters=32,
        kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=28, activation='relu'))   
    model.add(tf.keras.layers.Dense(units=4, activation='softmax'))    
    #final layer need to have 4 output neurons for predicting 4 classes
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                    metrics=['accuracy'])
    return model

'''
Train our model.
'''
def train_model(model, x_train, y_train, callback):
    #increase batch size from default of 32 to 64, to speed up machine learning
    #indicating the optimum epoch number for training
    #set aside 20% of training data for validation: to see the graph plots and find the optimum epoch for the most optimised model
    #to find out the model that optimally fit out training, validation data
    return model.fit(x=x_train, y=y_train, validation_split=0.20,batch_size=64, epochs=100, callbacks=[callback])    

'''
Automatic evaluation of our model against test set
'''
def auto_eval(model, x_test, y_test):
    loss, accuracy = model.evaluate(x=x_test, y=y_test)

    print('loss = ', loss)
    print('accuracy = ', accuracy)    

    return loss, accuracy
'''
Do our own evaluation; printing out predictions given by our model.
'''
def manual_eval(model, x_test, y_test):
    # get predicted values from model
    predictions = model.predict(x=x_test)       
    # compute accuracy
    n_preds = len(predictions)       
    correct = 0
    wrong = 0
    for i in np.arange(n_preds):
        pred_max = np.argmax(predictions[i])
        # as predicted value come in this form = [1, 0, 0,0]. so find the index of the max value in this array to get the predicted value)
        actual_max = np.argmax(y_test[i])

        if pred_max == actual_max:
            correct += 1
        else:
            wrong += 1
    print('correct: {0}, wrong: {1}'.format(correct, wrong))
    print('accuracy =', correct/n_preds)

'''
Create loss and accuracy plots.
'''
def plot(hist):
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    ax[0].plot(hist.history['loss'])
    ax[0].plot(hist.history['val_loss'])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Curve')
    ax[0].legend(['train', 'validation'], loc='upper left')


    ax[1].plot(hist.history['accuracy'])
    ax[1].plot(hist.history['val_accuracy'])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Curve')
    ax[1].legend(['train', 'validation'], loc='upper left')

    plt.show()

'''
(OPTIONAL) Create callback.
'''
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.9999999):
            self.model.stop_training = True 

'''
Save model with good test accuracy/loss.
'''
def save_model(model, path, accuracy, loss):
    if(accuracy > 0.933333):
        model.save(path)   

'''
Main program.
'''
def main():

  # for re-producibility in the debugging stage with this, 
  # our generated "random" numbers will always be the same 
  # (easy to debug during development)
    np.random.seed(5)

    # create our CNN model
    model = create_model()

    #mixed_fruits_augmentation('fruit_train/4')

    # fetch training data and onehot-encoded labels
    x_train, y_train = prep_train_data()

    callback = myCallback()

    # normalize x_train to be between [0, 1]
    hist = train_model(model, x_train/255, y_train, callback)

    # loss and accuracy plots
    plot(hist)

    x_test, y_test = prep_test_data()

    # test how well our model performs against data
    # that it has not seen before
    loss, accuracy = auto_eval(model, x_test/255, y_test)

    # perform manual evaluation of our model using test set
    manual_eval(model, x_test/255, y_test)

    save_model(model, './models', accuracy, loss)

# running via "python mnist_sample.py"
if __name__ == '__main__':
  main()
