import numpy as np;
import requests;
import tensorflow as tf;
import tensorflow_hub as hub;
import tensorflow_text as tf_text;
from tensorflow.keras import layers, models;

Dataset = ["sad af","happy","sadder","happierlol","im hornyyyyyyy marine shut the fuck up we are a dataset"];
Dataset2 = ["JOYOUS","MARVELOUS","WAAAAAAAAAAAAAAAAAHH","SEDUCE ME","asdfdgfhjsdfgf"];
AnnotatedEmotion = [0,1,0,1,2];
AnnotatedEmotion2 = [1,1,0,2,4];

MAX_WORDS = 10000       # vocabulary size
MAX_SEQUENCE_LENGTH = 500  # maximum words per text sample
EMBEDDING_DIM = 100      # number of features per word/token

ClassCount = 5;
EmotionName = ["sad","joy","aroused","annoyed","neutral","insane"]

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_WORDS,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

vectorizer.adapt(Dataset)

X_train = vectorizer(np.array(Dataset))
y_train = tf.keras.utils.to_categorical(AnnotatedEmotion, ClassCount)
print(X_train)
print(y_train)
X_val = vectorizer(np.array(Dataset2))
y_val = tf.keras.utils.to_categorical(AnnotatedEmotion2, ClassCount)

model = models.Sequential([
    # Adding an Embedding layer of dimension 100 (without the deprecated input_length argument)
    layers.Embedding(MAX_WORDS,EMBEDDING_DIM),

    # First Convolutional Block
    layers.Conv1D(filters=100, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    # Second Convolutional Block
    layers.Conv1D(filters=100, kernel_size=4, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    # Third Convolutional Block
    layers.Conv1D(filters=100, kernel_size=5, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),

    # Feature Aggregation & Classification Head
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(ClassCount, activation='softmax')
])

# 3. Compile the model with 'categorical_crossentropy' to match one-hot targets
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 4. Train with Early Stopping
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)
