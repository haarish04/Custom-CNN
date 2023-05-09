model = keras.models.Sequential([
    #keras.layers.Conv2D(filters=16, (3,3), activation='relu', input_shape=(50,50,3)),
    keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (50, 50, 3)), 
    keras.layers.MaxPooling2D(2,2),
    #keras.layers.BatchNormalization(axis=-1),
    keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
    keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])
model.summary()

#Parameters for training Model
alpha=0.01
epochs=15
optim = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer = optim, loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Fit the model

early_stopping_monitor = EarlyStopping(patience=3)
history = model.fit(train_dataset,
                    steps_per_epoch=len(train_dataset),
                    epochs=epochs,
                    validation_data=test_dataset,
                    validation_steps=len(test_dataset),
                    callbacks=[early_stopping_monitor])

