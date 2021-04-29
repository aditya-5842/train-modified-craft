import os
import time
import json
import imgaug
import numpy as np
import pandas as pd
from operator import imod
import tensorflow as tf
from tensorflow.python.keras.models import clone_model
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from sklearn.model_selection import train_test_split
from helper.dataset import get_image_data_generator, DataGenerator
from helper.build_model import build_keras_model, load_keras_model_vgg16, build_effiecient_basenetmodel

####################################### training parameters ###############################
lr = 1e-3
epochs = 1#250
patience = 10
batch_size = 32
train_size = 0.9 #(90% data for training)
############################################################################################

################################ load data (csv file) and data slplit ######################
df = pd.read_csv("./labeled.csv")
df['boxes'] = df['boxes'].apply(lambda x: json.loads(x))
labeles = []

for _, row in df.iterrows():
    boxes = []
    for box in row['boxes']:
        # boxes.append([(box, 'a')])
        boxes.append([(np.array(box, np.float), 'a')])
    labeles.append((row['filename'], boxes, 1))

# data split
train, validation = train_test_split(labeles, train_size=train_size, random_state=42)
############################################################################################


##################################  model training ########################################
def train_(model, train_gen, valid_gen, filename='vgg16_random_initilaization', lr=1e-3, is_efficient=False):
    current_train_loss = np.inf
    best_val_loss = np.inf
    min_loss_delta = 1e-5 #(atleast this much value is required to consider the improvement)
    losses = {'loss': [], 'val_loss': []}

    total_time = 0
    cnt_patience = 0
    # complile the model
    model.compile(loss='mse', optimizer='adam')
    # training loop
    for epoch in range(1, epochs+1):
        train_loss = 0
        val_loss = 0
        start = time.time()
        steps = len(train)//batch_size
        x_train, y_train = train_gen.get_item()
        x_val, y_val = valid_gen.get_item()
        if is_efficient:
            # x_train is normalized (between 0-1) but effiecient needs the values from 0-255
            x_train = (x_train*255).astype(np.uint8)
            x_val = (x_val*255).astype(np.uint8)
        for _ in range(steps):
            history = model.fit(x_train,y_train, batch_size=batch_size, validation_data=(x_val, y_val), verbose=0)
            train_loss += history.history['loss'][0]
            val_loss += history.history['val_loss'][0]
        
        # time
        time_take = time.time()-start
        total_time += time_take

        # loss
        current_train_loss = train_loss/steps 
        current_val_loss   = val_loss/steps
        losses['loss'].append(current_train_loss)
        losses['val_loss'].append(current_val_loss)

        # to print everything
        if epoch == 1:
            print("\n\n")
            print(f"{'Epoch':<10}{'Train-loss':<40}{'Validation-loss':<40}{'Time':<15}{'Learning-Rate':<10}")
            print("-"*125)
        print(f"{str(epoch)+'/'+str(epochs):<10}{str(round(current_train_loss, 6)):<40}{str(round(current_val_loss,6)):<40}{str(round(time_take, 5))+' s':<15}{round(lr,6):<10}")


        # logic of reducing the learning rate
        if epoch%5 == 0:
            lr /= 2 # update the learing rate
            # recompile the model updated learning rate
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=lr))


        # updating best-val-loss and save 
        if (best_val_loss>current_val_loss)>min_loss_delta:
            best_val_loss = current_val_loss
            cnt_patience = 0
            model.save("./weights/custom_craft_model_{}.h5".format(filename), include_optimizer=False)
        else:
            cnt_patience +=1
        
        if cnt_patience >= patience:
            print(f"stopping trainig because there is no improvement for validation loss for last {patience} epochs")
            df = pd.DataFrame(data=losses)
            df.reset_index(inplace=True)
            df.to_csv("./losses_{}.csv".format(filename), index=False)
            break
        
        if epoch == epochs:
            df = pd.DataFrame(data=losses)
            df.reset_index(inplace=True)
            df.to_csv("./losses_{}.csv".format(filename), index=False)
############################################################################################


################################ image augmentor and data genrator #########################
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GaussianBlur(sigma=(0, 3)), #Gaussian blur with mean=0 and std = 3
    imgaug.augmenters.Affine(scale=(1, 1), rotate = (-5, 5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2),
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0))]
)

generator_kwargs = {'width': 224, 'height': 224}

train_img_gen = get_image_data_generator(labels=train, augmenter=augmenter, **generator_kwargs)
valid_img_gen = get_image_data_generator(labels=validation, **generator_kwargs)

train_gen = DataGenerator(image_generator=train_img_gen, batch_size=batch_size)
valid_gen = DataGenerator(image_generator=valid_img_gen, batch_size=batch_size)
############################################################################################

################################ vgg16 random intialization ######################################################################
model = build_keras_model()
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='vgg16_random_initialization_(224,224)', lr=1e-3)
##################################################################################################################################


################################ vgg16  intialization + train all ################################################################
model = load_keras_model_vgg16()
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='vgg16_initialization_train_all_(224,224)', lr=(1e-3)/3)
##################################################################################################################################


################################ vgg16 intialization + don't train basenet #######################################################
model = load_keras_model_vgg16(basenet_trainable=False)
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='vgg16_initialization_basenet_train_false_(224,224)', lr=(1e-3)/5)
##################################################################################################################################

######################### EfficientNet intialization + don't train basenet #######################################################
model = build_effiecient_basenetmodel()
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='EfficientNet_initialization_train_all_(224,224)', lr=(1e-3)/3, is_efficient=True)
##################################################################################################################################

######################### EfficientNet intialization + don't train basenet #######################################################
model = build_effiecient_basenetmodel(basenet_trainable=False)
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='EfficientNet_initialization_basenet_train_false_(224,224)', lr=(1e-3)/5, is_efficient=True)
##################################################################################################################################











################################ image augmentor and data genrator ###############################################################
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.GaussianBlur(sigma=(0, 3)), #Gaussian blur with mean=0 and std = 3
    imgaug.augmenters.Affine(scale=(1, 1), rotate = (-5, 5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2),
    imgaug.augmenters.GammaContrast(gamma=(0.25, 3.0))]
)

generator_kwargs = {'width': 224, 'height': 96}

train_img_gen = get_image_data_generator(labels=train, augmenter=augmenter, **generator_kwargs)
valid_img_gen = get_image_data_generator(labels=validation, **generator_kwargs)

train_gen = DataGenerator(image_generator=train_img_gen, batch_size=batch_size)
valid_gen = DataGenerator(image_generator=valid_img_gen, batch_size=batch_size)
######################################################################################################################################

################################ vgg16 random intialization ##########################################################################
model = build_keras_model()
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='vgg16_random_initialization_(224,96)', lr=1e-3)
####################################################################################################################################


################################ vgg16  intialization + train all ################################################################
model = load_keras_model_vgg16()
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='vgg16_initialization_train_all_(224,96)', lr=(1e-3)/3)
##################################################################################################################################


################################ vgg16 intialization + don't train basenet #######################################################
model = load_keras_model_vgg16(basenet_trainable=False)
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='vgg16_initialization_basenet_train_false_(224,96)', lr=(1e-3)/5)
##################################################################################################################################

######################### EfficientNet intialization + don't train basenet #######################################################
model = build_effiecient_basenetmodel()
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='EfficientNet_initialization_train_all_(224,96)', lr=(1e-3)/3, is_efficient=True)
##################################################################################################################################

######################### EfficientNet intialization + don't train basenet #######################################################
model = build_effiecient_basenetmodel(basenet_trainable=False)
print("\n\n\n")
print(model.summary())

# train the created model
train_(model=model, train_gen=train_gen, valid_gen=valid_gen, filename='EfficientNet_initialization_basenet_train_false_(224,96)', lr=(1e-3)/5, is_efficient=True)
##################################################################################################################################
