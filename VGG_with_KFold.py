from sklearn.model_selection import KFold

# ===================================
# Merge inputs and targets
inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

acc_per_fold = []
loss_per_fold = []

for train, test in kfold.split(inputs, targets):
    model = Sequential()
    model.add(Conv2D(32, 3, activation='relu', padding='same',
                     input_shape=(179, 87, 1)))  # add my input_shape # Neural number = 32
    model.add(Conv2D(64, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, activation='relu', padding='same'))
    model.add(Conv2D(128, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))
    model.add(Conv2D(256, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 1))  # default stride is 2
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))
    model.add(Conv2D(512, 3, activation='relu', padding='same'))

    model.add(MaxPooling2D(2, 1))  # default stride is 2
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))

    # ===================================
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    print(f'Training for fold {fold_no} ...')

    model.fit(inputs[train], targets[train], batch_size=128, epochs=15)

    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(
        f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
