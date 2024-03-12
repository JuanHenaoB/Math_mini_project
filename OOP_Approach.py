#########################################################
###### Import all necessary libraries for project #######
#########################################################

import numpy as np
import keras
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from datasets import load_dataset # import Hugging Face datasets
from keras import regularizers
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from keras.applications.vgg16 import VGG16
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

###################################
####### Define all functions ######
###################################

#Define a function to resize the images
def resize_img(dataset):
    """
    Function used in balanced_artist_dataset to resize images as part 
    of pre processing for CMM implementation
    """
    #use list generators and hf_datasets methods
    dataset["image"] = [image.convert("RGB").resize((224, 224)) for image in dataset["image"]]
    return dataset

#Define a function to read and pre process dataset 
def balanced_artist_dataset(path, verbose=1):
    """
    Function that creates dataset using HF datasets and the public repository
    From Kaggle profile "Juan Henao". Uses resize_img function stated above
    """
    #load all images from 4 different artists using HF datasets
    print("Reading Dataset")
    raw_dataset = load_dataset("imagefolder", data_dir=path, drop_labels=False)
    
    #Check Raw data set
    print("Check Raw Dataset")
    print("\n")
    print(raw_dataset)

    #Check images size
    print(raw_dataset['train'][100])
    print(raw_dataset['train'][200])
    print("We can see that all the images have different sizes so we have to resize them for the CNN")
    print('\n')
    
    #Further details controlled by verbose
    if verbose == 1:
        #Check one image of the dataset
        print("Example Image from dataset: Vincent Van Gogh")
        print(raw_dataset['train'][750]['image'])
        print('\n')
        
    # use map function to resize our images
    print("Resizing Images")
    raw_dataset = raw_dataset.map(resize_img, batched=True, batch_size=100)
    
    #Further details
    if verbose == 1:
        #check observations again to confirm resizing
        print('Resized images')
        print(raw_dataset['train'][100])
        print(raw_dataset['train'][200])
        print('\n')
    
    #Shuffle the dataset and create a train, test split
    shuffled_dataset = raw_dataset.shuffle(seed=18)

    #Split into train/test 20% for test, and specify seed
    shuffled_dataset = shuffled_dataset['train'].train_test_split(test_size=0.2, seed=18)
    
    if verbose == 1:
        #check dataset
        print("Shuffled Dataset")
        print(shuffled_dataset)
        print('\n')
        
    #cast our hf dataset to np arrays so that we can use it in keras
    
    #Cast images to list of tensors
    X_train = [np.expand_dims(np.array(shuffled_dataset['train'][i]['image']),axis=0) 
               for i in range(len(list(shuffled_dataset['train']['image']) ) ) ]

    X_test = [np.expand_dims(np.array(shuffled_dataset['test'][i]['image']),axis=0) 
               for i in range(len(list(shuffled_dataset['test']['image']) ) ) ]

    #Stack list of tensors
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    #Normalize values
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    #Cast labels to np arrays
    y_train = np.array(list(shuffled_dataset['train']['label']))
    y_test = np.array(list(shuffled_dataset['test']['label']))


    #One hot encode labels
    y_train = keras.utils.to_categorical(y_train, 4)
    y_test = keras.utils.to_categorical(y_test, 4)
    
    return X_train, y_train, X_test, y_test

#function to define and train from scratch a VGG-like model
def train_scratch_model(X_train, y_train, X_test, y_test, verbose=1):
    """
    Function to instantiate and train VGG-like model from scratch
    """
    # Instantiate Architechture
    VGG_ish_model = Sequential()
    VGG_ish_model.name = "VGG_ish_model" #Name model

    #First Convolution layer
    VGG_ish_model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), 
                             activation='relu', padding='same', input_shape=(224,224,3)))
    VGG_ish_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # second Convolution layer
    VGG_ish_model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), 
                             activation='relu', padding='same'))
    VGG_ish_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # third Convolution layer
    VGG_ish_model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), 
                             activation='relu', padding='same'))
    VGG_ish_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # fourth Convolution Layer
    VGG_ish_model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), 
                             activation='relu', padding='same'))
    VGG_ish_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # fifth convolution layer
    VGG_ish_model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), 
                             activation='relu', padding='same'))
    VGG_ish_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    # Classification head
    VGG_ish_model.add(Flatten())
    VGG_ish_model.add(Dense(16, activation='relu'))
    VGG_ish_model.add(Dropout(0.5))
    VGG_ish_model.add(Dense(16, activation='relu'))
    VGG_ish_model.add(Dropout(0.5))
    VGG_ish_model.add(Dense(4, activation='softmax'))
    
    if verbose == 1:
        # Check Model Summary
        print("Check ",VGG_ish_model.name," summary")
        VGG_ish_model.summary()
        print('\n')
        
    # Set Optimizer and compile model
    #Schedule a reduce on lr when validation loss hits a plateau same as in VGG training
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1))

    #use the same optimezer used in VGG19
    optimizer = keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9)

    #compile model
    VGG_ish_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    #Train the model
    print("Train model from scratch")
    start = time.time()#To check amount of time spent in training
    hist = VGG_ish_model.fit(x=X_train, y=y_train, batch_size=5, epochs=25, 
                             validation_split=0.2, callbacks=[reduce_lr])
    end = time.time()
    print("Training time :",round(end-start)," s")
    print('\n')
    return VGG_ish_model, hist

#Function to define and train transfer learning model using VGG16 as a feature extractor
def transfer_learning_model(X_train, y_train, X_test, y_test, verbose=1):
    """
    Function to instantiate and train transfer learning model
    """
    #load VGG16 model trained on imagenet dataset
    vggmodel = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    
    # iterate through model layers and lock them to make them not trainable
    # except for the last 5th convolutional block
    for layer in vggmodel.layers[:-5]:
        layer.trainable = False
        
    # use “get_layer” method to save the last layer of the network
    last_layer = vggmodel.get_layer('global_average_pooling2d')

    # save the output of the last layer to be the input of the next layer
    last_output = last_layer.output

    # add our new softmax layer with 4 hidden units
    x = Dense(4, activation='softmax', name='softmax')(last_output)

    # instantiate a new_model using keras’s Model class
    tl_model = Model(inputs=vggmodel.input, outputs=x)
    tl_model.name = "Transfer_Learning_model"

    # print the new_model summary
    if verbose == 1:
        print("Transfer learning model summary")
        tl_model.summary()
        print('\n')
    
    #monitor accuracy to execute training early stopping
    #acc_monitor = keras.callbacks.EarlyStopping(monitor="val_accuracy", min_delta=0.00001, patience=2)

    #Compile new model
    tl_model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    #Train new model
    hist_2 = tl_model.fit(x=X_train, y=y_train, batch_size=5, epochs=20, validation_split=0.2)
    
    return tl_model, hist_2

#Define a function to evaluate models training and predictions
def evaluate_model(model, history, X_test, y_test):
    """
    Function to evaluate models
    """
    
    print("Evaluating "+model.name)
    
    # Plot training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and validation loss '+model.name)
    plt.legend()
    plt.show()
    
    # Plot training and validation Accuracy
    loss = history.history['accuracy']
    val_loss = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training accuracy')
    plt.plot(epochs, val_loss, 'b', label='Validation accuracy')
    plt.xlabel('Epochs')
    plt.title('Training and validation accuracy '+model.name)
    plt.legend()
    plt.show()
    
    # make predictions
    y_pred = model.predict(X_test)
    print("Accuracy :",np.round(accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1)), 2))
    y_pred_2 = keras.utils.to_categorical(y_pred.argmax(axis=1), 4) 
    print("F1 Score :",np.round(f1_score(y_test, y_pred_2, average='weighted'), 2))
    
    #Plot confusion matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title('Confusion Matrix '+model.name)
    plt.show()
  
#############################
####### Main function #######
#############################
if __name__ == "__main__":
    #Check the version of TensorFlow you are using and check for GPU
    print("Check Tensorflow version")
    print(tf.__version__)
    print("\n")
    print("Check available GPUs")
    print(tf.config.list_physical_devices('GPU'))
    print("\n")
    
    #Specify path for dataset, this is the path for my kaggle notebook
    path = "/kaggle/input/artist-balanced-sample/Artist_balanced_sample"
    X_train, y_train, X_test, y_test = balanced_artist_dataset(path=path, verbose=1)
    scratch_model, hist = train_scratch_model(X_train, y_train, X_test, y_test, verbose=1)
    tl_model, hist_2    = transfer_learning_model(X_train, y_train, X_test, y_test, verbose=1)
    evaluate_model(scratch_model, hist, X_test, y_test)
    evaluate_model(tl_model, hist_2, X_test, y_test)
