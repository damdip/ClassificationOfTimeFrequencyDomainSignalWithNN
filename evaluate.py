"""
Script per la valutazione delle prestazioni del modello
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve, 
    auc,
    roc_auc_score
)
from pathlib import Path
import json

from train import load_model
from data_preparation import load_preprocessed_data


def evaluate_model(model, X, y_true, class_names=None):
    """
    Valuta il modello e calcola metriche di performance
    
    Args:
        model: modello Keras trainato
        X: dati di input
        y_true: label vere (one-hot encoded)
        class_names: nomi delle classi
        
    Returns:
        dizionario con le metriche
    """
    print("=== Valutazione Modello ===\n")
    
    # Predizioni
    y_pred_proba = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true_labels = np.argmax(y_true, axis=1)
    
    # Accuracy
    accuracy = accuracy_score(y_true_labels, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")
    
    # Precision, Recall, F1-Score
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_labels, y_pred, average='weighted'
    )
    
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}\n")
    
    # Classification Report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(y_true.shape[1])]
    
    print("Classification Report:")
    print(classification_report(y_true_labels, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true_labels, y_pred)
    
    # Calcola metriche per classe
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': precision_recall_fscore_support(
                y_true_labels, y_pred, labels=[i], average='weighted'
            )[0],
            'recall': precision_recall_fscore_support(
                y_true_labels, y_pred, labels=[i], average='weighted'
            )[1],
            'f1': precision_recall_fscore_support(
                y_true_labels, y_pred, labels=[i], average='weighted'
            )[2],
            'support': np.sum(y_true_labels == i)
        }
    
    # Calcola ROC-AUC se classificazione binaria o multi-class
    try:
        if y_true.shape[1] == 2:
            # Binaria
            roc_auc = roc_auc_score(y_true_labels, y_pred_proba[:, 1])
        else:
            # Multi-class
            roc_auc = roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')
        print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    except Exception as e:
        print(f"Impossibile calcolare ROC-AUC: {e}\n")
        roc_auc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_true': y_true_labels
    }
    
    return metrics


def plot_confusion_matrix(cm, class_names, output_path="confusion_matrix.png"):
    """
    Crea plot della confusion matrix
    
    Args:
        cm: confusion matrix
        class_names: nomi delle classi
        output_path: percorso per salvare il plot
    """
    plt.figure(figsize=(10, 8))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2%', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix salvata in '{output_path}'")
    plt.close()


def plot_roc_curves(y_true, y_pred_proba, class_names, output_path="roc_curves.png"):
    """
    Crea plot delle curve ROC per ogni classe
    
    Args:
        y_true: label vere (one-hot)
        y_pred_proba: probabilità predette
        class_names: nomi delle classi
        output_path: percorso per salvare il plot
    """
    n_classes = len(class_names)
    

    if n_classes == 2:
        y_true_labels = np.argmax(y_true, axis=1)
        fpr, tpr, _ = roc_curve(y_true_labels, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
    else:
        # Multi-class: una curva per classe
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multi-class')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"ROC curves salvate in '{output_path}'")
    plt.close()


def plot_class_distribution(y_true, class_names, output_path="class_distribution.png"):
    """
    Crea plot della distribuzione delle classi
    
    Args:
        y_true: label vere (labels, non one-hot)
        class_names: nomi delle classi
        output_path: percorso per salvare il plot
    """
    unique, counts = np.unique(y_true, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='black')
    
    # Colora barre diverse
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    
    # Aggiungi valori sopra le barre
    for i, (idx, count) in enumerate(zip(unique, counts)):
        plt.text(i, count + max(counts)*0.01, str(count), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distribuzione classi salvata in '{output_path}'")
    plt.close()


def plot_prediction_confidence(y_pred_proba, y_true, output_path="prediction_confidence.png"):
    """
    Crea plot della confidenza delle predizioni
    
    Args:
        y_pred_proba: probabilità predette
        y_true: label vere
        output_path: percorso per salvare il plot
    """
    # Confidenza è massima probabilità predetta
    confidence = np.max(y_pred_proba, axis=1)
    
    # Separa predizioni corrette e sbagliate
    y_pred = np.argmax(y_pred_proba, axis=1)
    correct = (y_pred == y_true)
    
    plt.figure(figsize=(12, 5))
    
    # Istogramma confidenza
    plt.subplot(1, 2, 1)
    plt.hist(confidence[correct], bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    plt.hist(confidence[~correct], bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot
    plt.subplot(1, 2, 2)
    data_to_plot = [confidence[correct], confidence[~correct]]
    plt.boxplot(data_to_plot, labels=['Correct', 'Incorrect'])
    plt.ylabel('Prediction Confidence')
    plt.title('Confidence Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Confidenza predizioni salvata in '{output_path}'")
    plt.close()


def save_metrics_to_file(metrics, class_names, output_path="evaluation_metrics.json"):
    """
    Salva le metriche in un file JSON
    
    Args:
        metrics: dizionario con le metriche
        class_names: nomi delle classi
        output_path: percorso per salvare il file
    """
    # Prepara dati serializzabili
    metrics_to_save = {
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'roc_auc': float(metrics['roc_auc']) if metrics['roc_auc'] else None,
        'confusion_matrix': metrics['confusion_matrix'].tolist(),
        'class_names': class_names,
        'per_class_metrics': {
            k: {
                'precision': float(v['precision']),
                'recall': float(v['recall']),
                'f1': float(v['f1']),
                'support': int(v['support'])
            }
            for k, v in metrics['per_class_metrics'].items()
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"Metriche salvate in '{output_path}'")


if __name__ == "__main__":
    print("=== Evaluation Script ===\n")
    
    # Configurazione
    DATA_FILE = "data_spec_1mm.pkl"
    MODEL_PATH = "saved_models/defect_classifier_spec_object_1mm_finetuned.keras"
    OUTPUT_DIR = Path("evaluation_results/spec_1mm_finetuned")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 1. Carica i dati preprocessati
    print("Caricamento dati...")
    try:
        data = load_preprocessed_data(DATA_FILE)
        X_test = data['X_test']
        y_test = data['y_test']
        class_names = data.get('class_names', None)
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(y_test.shape[1])]
        
        print(f"Test set: {X_test.shape}")
        print(f"Classi: {class_names}\n")
        
    except FileNotFoundError:
        print(f"ERRORE: File '{DATA_FILE}' non trovato!")
        exit(1)
    
    # 2. Carica il modello
    print("Caricamento modello...")
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il modello da '{MODEL_PATH}'")
        print(f"Dettagli: {e}")
        exit(1)
    
    # 3. Valuta il modello
    metrics = evaluate_model(model, X_test, y_test, class_names)
    
    # 4. Crea visualizzazioni
    print("\nCreazione visualizzazioni...")
    
    # Confusion Matrix
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names,
        output_path=OUTPUT_DIR / "confusion_matrix.png"
    )
    
    # ROC Curves
    try:
        plot_roc_curves(
            y_test,
            metrics['y_pred_proba'],
            class_names,
            output_path=OUTPUT_DIR / "roc_curves.png"
        )
    except Exception as e:
        print(f"Impossibile creare ROC curves: {e}")
    
    # Class Distribution
    plot_class_distribution(
        metrics['y_true'],
        class_names,
        output_path=OUTPUT_DIR / "class_distribution.png"
    )
    
    # Prediction Confidence
    plot_prediction_confidence(
        metrics['y_pred_proba'],
        metrics['y_true'],
        output_path=OUTPUT_DIR / "prediction_confidence.png"
    )
    
    # 5. Salva metriche
    save_metrics_to_file(
        metrics,
        class_names,
        output_path=OUTPUT_DIR / "evaluation_metrics.json"
    )
    
    print("\n=== Valutazione completata! ===")
    print(f"Tutti i risultati sono salvati in '{OUTPUT_DIR}/'")
    print(f"\nRiepilogo:")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1-Score: {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
