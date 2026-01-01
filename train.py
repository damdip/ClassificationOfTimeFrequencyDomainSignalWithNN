"""
Script per il training del modello
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
from datetime import datetime
import os
from pathlib import Path

from model import build_2d_cnn_model
from data_preparation import load_preprocessed_data


def create_callbacks(model_name, patience=10, min_lr=1e-7):
    """
    Crea callbacks per il training
    
    Args:
        model_name: nome del modello per salvare i checkpoint
        patience: pazienza per early stopping
        min_lr: learning rate minimo per ReduceLROnPlateau
        
    Returns:
        lista di callbacks
    """
    # Crea cartelle per salvare risultati
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        # Salva il modello migliore
        ModelCheckpoint(
            filepath=checkpoint_dir / f"{model_name}_best.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping se non migliora
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Riduzione learning rate quando si ferma il miglioramento
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=min_lr,
            verbose=1
        ),
        
        # TensorBoard per visualizzazione
        TensorBoard(
            log_dir=logs_dir / f"{model_name}_{timestamp}",
            histogram_freq=1
        )
    ]
    
    return callbacks


def plot_training_history(history, output_path="training_history.png"):
    """
    Crea plot della storia del training
    
    Args:
        history: oggetto History di Keras
        output_path: percorso per salvare il plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot salvato in '{output_path}'")
    plt.close()

def freeze_layers_for_finetuning(model, num_trainable_layers=2):
    """
    Congela tutti i layer del modello tranne gli ultimi n layer.
    
    Args:
        model: modello Keras
        num_trainable_layers: numero di layer da mantenere trainable dalla fine
        
    Returns:
        model con layer congelati
    """
    total_layers = len(model.layers)
    
    # Congela tutti i layer tranne gli ultimi num_trainable_layers
    for i, layer in enumerate(model.layers):
        if i < total_layers - num_trainable_layers:
            layer.trainable = False
        else:
            layer.trainable = True
    
    # Ricompila il modello per applicare le modifiche
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # LR più basso per fine-tuning
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"\n=== Fine-tuning Configuration ===")
    print(f"Total layers: {total_layers}")
    print(f"Frozen layers: {total_layers - num_trainable_layers}")
    print(f"Trainable layers: {num_trainable_layers}")
    print("\nTrainable status per layer:")
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i} ({layer.name}): {'TRAINABLE' if layer.trainable else 'FROZEN'}")
    print()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=10, batch_size=32, model_name="model", fine_tuning=False, num_trainable_layers=2):
    """
    Esegue il training del modello
    
    Args:
        model: modello Keras da trainare
        X_train: dati di training
        y_train: label di training
        X_val: dati di validation
        y_val: label di validation
        epochs: numero di epochs
        batch_size: dimensione del batch
        model_name: nome per salvare il modello
        
    Returns:
        history: storia del training
        model: modello trainato
    """
    print("\n=== Inizio Training ===")
    print(f"Mode: {'FINE-TUNING' if fine_tuning else 'FULL TRAINING'}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"Output classes: {y_train.shape[1]}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    
    if fine_tuning:
        model = freeze_layers_for_finetuning(model, num_trainable_layers)
    
    print()
    # Crea callbacks
    callbacks = create_callbacks(model_name)
    
    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n=== Training completato ===")
    
    # Plot storia
    plot_training_history(history, f"{model_name}_history.png")
    
    return history, model


def save_model(model, model_name="model"):
    """
    Salva il modello in diversi formati
    
    Args:
        model: modello Keras
        model_name: nome base per i file
    """
    models_dir = Path("saved_models")
    models_dir.mkdir(exist_ok=True)
    
    # Salva in formato Keras nativo
    model.save(models_dir / f"{model_name}.keras")
    print(f"Modello salvato in 'saved_models/{model_name}.keras'")



def load_model(model_path):
    """
    Carica un modello salvato
    
    Args:
        model_path: percorso del modello
        
    Returns:
        modello caricato
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Modello caricato da '{model_path}'")
    return model


def get_model_summary(model):
    """
    Stampa e restituisce il summary del modello
    
    Args:
        model: modello Keras
        
    Returns:
        stringa con il summary
    """
    from io import StringIO
    import sys
    
    # Cattura il summary
    old_stdout = sys.stdout
    sys.stdout = summary_string = StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    summary = summary_string.getvalue()
    print(summary)
    return summary


if __name__ == "__main__":
    import argparse
    
    print("=== Training Script ===\n")
    
    # Parsing argomenti da command line
    parser = argparse.ArgumentParser(description='Train signal classifier')
    parser.add_argument('--fine-tuning', action='store_true',
                        help='Enable fine-tuning mode (freeze all layers except last 2)')
    parser.add_argument('--num-trainable-layers', type=int, default=2,
                        help='Number of layers to keep trainable in fine-tuning mode (default: 2)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to pretrained model for fine-tuning')
    
    args = parser.parse_args()
    
    # Configurazione
    DATA_FILE = "data_mel_3mm.pkl"
    MODEL_NAME = "defect_classifier_mel_object_1mm"
    if args.fine_tuning:
        MODEL_NAME += "_finetuned"
    
    # 1. Carica i dati preprocessati
    print("Caricamento dati preprocessati...")
    try:
        data = load_preprocessed_data(DATA_FILE)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        class_names = data.get('class_names', None)
        
        print(f"Dati caricati correttamente")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        if class_names:
            print(f"Classi: {class_names}")
        
    except FileNotFoundError:
        print(f"ERRORE: File '{DATA_FILE}' non trovato!")
        print("Esegui prima 'python data_preparation.py' per preparare i dati")
        exit(1)
    
    # 2. Costruisci o carica il modello
    print("\nCostruzione del modello...")
    input_height, input_width = X_train.shape[1], X_train.shape[2]
    num_classes = y_train.shape[1]
    
    if args.model_path:
        # Carica modello esistente per fine-tuning
        print(f"Caricamento modello da {args.model_path}...")
        model = load_model(args.model_path)
    else:
        # Costruisci nuovo modello
        model = build_2d_cnn_model(input_height=input_height, input_width=input_width)
    
    # Stampa summary
    print("\nArchitettura del modello:")
    get_model_summary(model)
    
    # 3. Training
    history, trained_model = train_model(
        model, 
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=MODEL_NAME,
        fine_tuning=args.fine_tuning,
        num_trainable_layers=args.num_trainable_layers
    )
    
    # 4. Valutazione finale sul test set
    print("\nValutazione sul test set...")
    test_loss, test_accuracy = trained_model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 5. Salva il modello finale
    save_model(trained_model, MODEL_NAME)
    
    # Stampa best results
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"\n=== Risultati Migliori ===")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    print("\n=== Training completato con successo! ===")
    print(f"Per visualizzare i log TensorBoard: tensorboard --logdir=logs")