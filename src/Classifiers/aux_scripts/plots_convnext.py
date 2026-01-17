import matplotlib.pyplot as plt
import os

def plot_training_metrics(history, exp_dir, experiment_name):
    """Create and save plots showing training metrics evolution"""
    plt.style.use('default')
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Training Metrics Evolution - {experiment_name}', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Plot 3: F1 Score
    axes[0, 2].plot(epochs, history['val_f1'], 'g-', label='Validation F1', linewidth=2)
    axes[0, 2].set_title('Validation F1 Score', fontweight='bold')
    axes[0, 2].set_xlabel('Epochs')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # Plot 4: Precision and Recall (if available)
    if 'val_precision' in history and 'val_recall' in history:
        axes[1, 0].plot(epochs, history['val_precision'], 'm-', label='Validation Precision', linewidth=2)
        axes[1, 0].plot(epochs, history['val_recall'], 'c-', label='Validation Recall', linewidth=2)
        axes[1, 0].set_title('Validation Precision and Recall', fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
    else:
        axes[1, 0].axis('off')
    
    # Plot 5: Learning Rate (if available)
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
    else:
        axes[1, 1].axis('off')
    
    # Plot 6: Combined Overview
    ax_combined = axes[1, 2]
    ax2 = ax_combined.twinx()
    
    # Plot accuracy on left y-axis
    line1 = ax_combined.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax_combined.set_xlabel('Epochs')
    ax_combined.set_ylabel('Accuracy', color='r')
    ax_combined.tick_params(axis='y', labelcolor='r')
    ax_combined.set_ylim(0, 1)
    
    # Plot loss on right y-axis
    line2 = ax2.plot(epochs, history['val_loss'], 'b-', label='Val Loss', linewidth=2)
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax_combined.set_title('Validation Accuracy vs Loss', fontweight='bold')
    ax_combined.grid(True, alpha=0.3)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_combined.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(exp_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training metrics plot saved to: {plot_path}")
    
    # Also create individual plots for each metric
    create_individual_plots(history, exp_dir, experiment_name)
    
    plt.close(fig)  # Close to free memory


def create_individual_plots(history, exp_dir, experiment_name):
    """Create individual plots for each metric"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Individual Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title(f'Loss Evolution - {experiment_name}', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'loss_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'Accuracy Evolution - {experiment_name}', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'accuracy_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Individual F1 plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['val_f1'], 'g-', label='Validation F1 Score', linewidth=2)
    plt.title(f'F1 Score Evolution - {experiment_name}', fontweight='bold', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'f1_evolution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual metric plots saved to: {exp_dir}")