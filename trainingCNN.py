



from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)



train_generator = train_datagen.flow_from_directory('./Training',target_size=(100, 100),batch_size=32,class_mode='categorical',color_mode="grayscale")


test_generator = test_datagen.flow_from_directory('./Test',target_size=(100, 100),batch_size=32,class_mode='categorical',color_mode="grayscale")
val_generator = test_datagen.flow_from_directory('./Validation',target_size=(100, 100),batch_size=32,class_mode='categorical',color_mode="grayscale")



from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint



def nn_model():

   model = Sequential()

   model.add(Conv2D(20, (5, 5), input_shape=(100, 100, 1), padding='same'))

   model.add(Activation('relu'))

   model.add(MaxPooling2D(pool_size=(2, 2)))

   model.add(Activation('relu'))

   model.add(Conv2D(40, (5, 5), padding='same'))

   model.add(Activation('relu'))

   model.add(MaxPooling2D(pool_size=(2, 2)))

   model.add(Activation('relu'))

   model.add(Conv2D(45, (5, 5), padding='same'))

   model.add(Activation('relu'))

   model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

   model.add(Activation('relu'))

   model.add(Flatten())
   model.add(Dropout(0.5))

   model.add(Dense(units=500))

   model.add(Activation('relu'))

   model.add(Dense(units=64))

   model.add(Activation('relu'))
   model.add(Dropout(0.5))


   model.add(Dense(units=train_generator.num_classes, activation='softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



   return model

model=nn_model()
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,mode='auto')

history=model.fit(train_generator, validation_data=val_generator,epochs=10,callbacks=[checkpoint])
test_loss,test_acc= model.evaluate(test_generator)
print("-------------------------------------------")
print(f'The accuracy for test data is: {test_acc*100}')
print(f'The loss for test data is: {test_loss}')
print("--------------------------------------------")

from matplotlib import pyplot as plt

plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],label='validation loss')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],label='validation accuracy')
plt.xlabel('# epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print('cnn model summary')
print("----------------------------------------------------------")

model.summary()
print("-----------------------------------------------------------")



