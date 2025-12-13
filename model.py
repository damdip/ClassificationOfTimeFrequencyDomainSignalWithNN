import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input

def build_2d_cnn_model(input_height=65, input_width=113):
    """
    Costruisce una CNN 2D basata sull'architettura 1D del paper originale.
    Input: Spettrogramma Mel (Height=Mel Bands, Width=Time Frames)
    """
    
    # Definizione della forma dell'input: (Frequenza, Tempo, Canali)
    # È necessario aggiungere 1 canale alla fine (come se fosse un'immagine in bianco e nero)
    input_shape = (input_height, input_width, 1)

    model = Sequential(name="Spectograms_2D_CNN")

    # --- BLOCCO 1
    # 64 filtri, kernel 3, same padding, ReLU, Batch Norm, MaxPool 2
    model.add(Input(shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # --- BLOCCO 2 
    # 128 filtri, kernel 3, same padding, Batch Norm, MaxPool 2, Dropout 0.3
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # --- BLOCCO 3
    # 256 filtri, kernel 3, same padding, Batch Norm, MaxPool 2, Dropout 0.5
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # --- CLASSIFICATORE (Testa della rete) ---
    # Il paper usava GlobalAveragePooling1D, qui usiamo Flatten per linearizzare la matrice 2D
    # Nota: Si può usare anche GlobalAveragePooling2D se si vuole ridurre drasticamente i parametri
    model.add(Flatten())

    # Fully Connected Layers come da paper
    model.add(Dense(128, activation='relu'))
    
    # Dropout prima dell'ultimo layer denso
    model.add(Dropout(0.5)) 
    
    model.add(Dense(32, activation='relu'))

    # Output Layer: 2 neuroni (Difetto vs Non-Difetto) con Softmax
    model.add(Dense(2, activation='softmax'))

    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01) # Learning rate dal paper: 10^-2
    
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Esempio di creazione del modello
#model_2d = build_2d_cnn_model(input_height=65, input_width=113)
#model_2d.summary()