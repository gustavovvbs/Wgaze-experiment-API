import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_history(history):
    """
    Plot training curves for loss, top-1 accuracy, and top-4 accuracy.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(18, 6))

    # -- Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    # -- Top-1 Acc
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Top-1 Accuracy')
    plt.legend()

    # -- Top-4 Acc
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_top4'], label='Train Top-4')
    plt.plot(epochs, history['val_top4'], label='Val Top-4')
    plt.xlabel('Epoch')
    plt.ylabel('Top-4 Accuracy (%)')
    plt.title('Top-4 Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm, target_names):
    """
    Plot confusion matrix heatmap.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def visualize_gaze_path(gaze_points, key_positions, word=None):
    """
    Visualize a gaze path overlaid on the keyboard layout.
    """
    plt.figure(figsize=(12, 8))
    
    # plotting key positions
    for key, pos in key_positions.items():
        if key == 'BOTAO_ACABAR':
            plt.scatter(pos[0], pos[1], color='green', s=150, alpha=0.7)
            plt.text(pos[0], pos[1], 'FINISH', ha='center', va='center', color='white')
        else:
            plt.scatter(pos[0], pos[1], color='blue', s=100, alpha=0.5)
            plt.text(pos[0], pos[1], key, ha='center', va='center')
    
    gp = np.array(gaze_points)
    plt.plot(gp[:, 0], gp[:, 1], 'r-', alpha=0.5)
    colors = np.linspace(0.2, 1, len(gp))
    plt.scatter(gp[:, 0], gp[:, 1], c=colors, cmap='Reds', alpha=0.7, s=30)
    
    plt.annotate("Start", (gp[0, 0], gp[0, 1]), xytext=(-30, 20), 
                textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    plt.annotate("End", (gp[-1, 0], gp[-1, 1]), xytext=(30, 20), 
                textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    
    if word:
        plt.title(f'Gaze Path for Word: "{word}"')
    else:
        plt.title('Gaze Path Visualization')
    
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.show()