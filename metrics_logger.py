import json
import time
import math
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class MetricsLogger:
    """
    Wrapper para capturar y guardar métricas de entrenamiento.
    """
    
    def __init__(self, save_path: str = "training_metrics.json"):
        self.save_path = Path(save_path)
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "train_perplexity": [],
            "val_perplexity": [],
            "epoch_times": [],
            "total_time": 0,
            "epochs": 0
        }
        self.start_time = None
        self.epoch_start_time = None
    
    def start_training(self):
        """Inicia el cronómetro de entrenamiento."""
        self.start_time = time.time()
    
    def start_epoch(self):
        """Inicia el cronómetro de época."""
        self.epoch_start_time = time.time()
    
    def log_epoch(self, train_loss: float, val_loss: float):
        """
        Registra las métricas de una época.
        
        Args:
            train_loss: Pérdida de entrenamiento
            val_loss: Pérdida de validación
        """
        # Calcular perplexity
        train_perplexity = math.exp(min(train_loss, 10))  # Clamp para evitar overflow
        val_perplexity = math.exp(min(val_loss, 10))
        
        # Calcular tiempo de época
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Guardar métricas
        self.metrics["train_loss"].append(train_loss)
        self.metrics["val_loss"].append(val_loss)
        self.metrics["train_perplexity"].append(train_perplexity)
        self.metrics["val_perplexity"].append(val_perplexity)
        self.metrics["epoch_times"].append(epoch_time)
        self.metrics["epochs"] += 1
        
        print(f"Época {self.metrics['epochs']} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        print(f"Train Perplexity: {train_perplexity:.4f} - Val Perplexity: {val_perplexity:.4f}")
        print(f"Tiempo de época: {epoch_time:.2f}s")
    
    def end_training(self):
        """Finaliza el entrenamiento y calcula el tiempo total."""
        if self.start_time:
            self.metrics["total_time"] = time.time() - self.start_time
            print(f"Entrenamiento completo. Tiempo total: {self.metrics['total_time']:.2f}s")
    
    def save_metrics(self):
        """Guarda las métricas en un archivo JSON."""
        with open(self.save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Métricas guardadas en {self.save_path}")
    
    def get_metrics(self) -> Dict:
        """Retorna las métricas actuales."""
        return self.metrics.copy()

def load_and_plot_metrics(json_path: str, save_plots: bool = True, output_dir: str = "./plots"):
    """
    Carga métricas desde un archivo JSON y genera gráficos.
    
    Args:
        json_path: Ruta al archivo JSON con métricas
        save_plots: Si guardar los gráficos como archivos
        output_dir: Directorio donde guardar los gráficos
    """
    # Cargar métricas
    with open(json_path, 'r') as f:
        metrics = json.load(f)
    
    epochs = list(range(1, metrics["epochs"] + 1))
    
    # Crear directorio de salida si es necesario
    if save_plots:
        Path(output_dir).mkdir(exist_ok=True)
    
    # Configurar estilo
    plt.style.use('default')
    
    # Gráfico 1: Loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["train_loss"], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, metrics["val_loss"], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Pérdida por Época')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico 2: Perplexity
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics["train_perplexity"], 'b-', label='Train Perplexity', linewidth=2)
    plt.plot(epochs, metrics["val_perplexity"], 'r-', label='Validation Perplexity', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Perplexity')
    plt.title('Perplejidad por Época')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/loss_and_perplexity.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Gráfico 3: Tiempos de entrenamiento
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, metrics["epoch_times"], 'g-', linewidth=2, marker='o')
    plt.xlabel('Época')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Tiempo por Época')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    cumulative_time = np.cumsum(metrics["epoch_times"])
    plt.plot(epochs, cumulative_time, 'purple', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Tiempo Acumulado (segundos)')
    plt.title('Tiempo Acumulado de Entrenamiento')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{output_dir}/training_times.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Mostrar estadísticas finales
    print("\n=== Resumen de Entrenamiento ===")
    print(f"Épocas totales: {metrics['epochs']}")
    print(f"Tiempo total: {metrics['total_time']:.2f}s ({metrics['total_time']/60:.1f} min)")
    print(f"Tiempo promedio por época: {np.mean(metrics['epoch_times']):.2f}s")
    print(f"Loss final - Train: {metrics['train_loss'][-1]:.4f}, Val: {metrics['val_loss'][-1]:.4f}")
    print(f"Perplexity final - Train: {metrics['train_perplexity'][-1]:.4f}, Val: {metrics['val_perplexity'][-1]:.4f}")
    
    return metrics

def compare_models_metrics(metrics_paths: List[str], model_names: List[str], save_plot: bool = True):
    """
    Compara métricas de múltiples modelos.
    
    Args:
        metrics_paths: Lista de rutas a archivos JSON de métricas
        model_names: Lista de nombres para cada modelo
        save_plot: Si guardar el gráfico de comparación
    """
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (path, name) in enumerate(zip(metrics_paths, model_names)):
        with open(path, 'r') as f:
            metrics = json.load(f)
        
        epochs = list(range(1, metrics["epochs"] + 1))
        color = colors[i % len(colors)]
        
        # Loss de validación
        plt.subplot(2, 2, 1)
        plt.plot(epochs, metrics["val_loss"], color=color, label=name, linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Validation Loss')
        plt.title('Comparación - Pérdida de Validación')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Perplexity de validación
        plt.subplot(2, 2, 2)
        plt.plot(epochs, metrics["val_perplexity"], color=color, label=name, linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Validation Perplexity')
        plt.title('Comparación - Perplejidad de Validación')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Tiempo por época
        plt.subplot(2, 2, 3)
        plt.plot(epochs, metrics["epoch_times"], color=color, label=name, linewidth=2)
        plt.xlabel('Época')
        plt.ylabel('Tiempo por Época (s)')
        plt.title('Comparación - Tiempo por Época')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Tiempo total
        plt.subplot(2, 2, 4)
        plt.bar(name, metrics["total_time"], color=color, alpha=0.7)
        plt.ylabel('Tiempo Total (s)')
        plt.title('Comparación - Tiempo Total de Entrenamiento')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plot:
        plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()