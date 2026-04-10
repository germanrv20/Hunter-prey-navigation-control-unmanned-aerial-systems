import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

file_path = os.path.expanduser("~/drone_ws/src/drone_lap/data/distance_data_v2.npy")

# 1. DEFINICIÓN DE LA RED NEURONAL (MLP)
class MlpDistance(nn.Module):
    def __init__(self):
        super(MlpDistance, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 32),   
            nn.ELU(),
            nn.Linear(32, 16), 
            nn.ELU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


# 2. FUNCIÓN DE ENTRENAMIENTO
def train_epoch(model, dataloader, optimizer, criterion, noise_factor=0.0):
    model.train()  # Ponemos el modelo en modo entrenamiento
    total_loss = 0.0
    
    for inputs, targets in dataloader:
        # Añadir ruido Gaussiano a los inputs para hacer la IA más robusta
        noisy_inputs = inputs + torch.randn_like(inputs) * noise_factor
        
        # Resetear gradientes
        optimizer.zero_grad()
        
        # Forward pass (Predicción)
        outputs = model(noisy_inputs)
        
        # Calcular el error (Loss)
        loss = criterion(outputs, targets)
        
        # Backward pass (Ajustar pesos)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        
    return total_loss / len(dataloader.dataset)


# 3. FUNCIÓN DE EVALUACIÓN
def eval_epoch(model, dataloader, criterion):
    model.eval()  # Ponemos el modelo en modo evaluación
    total_loss = 0.0
    
    with torch.no_grad(): # No calculamos gradientes (ahorra memoria y CPU)
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
    return total_loss / len(dataloader.dataset)


# 4. BUCLE PRINCIPAL
def main():
    if not os.path.exists(file_path):
        print(f"Error: No se encuentra el archivo {file_path}")
        return

    # -- Cargar y preparar datos --
    data = np.load(file_path)
    distances = data[:, 0]
    areas = data[:, 1]

    valid_idx = (areas > 50) & (distances < 15.0)
    distances = distances[valid_idx].reshape(-1, 1)
    sizes = (1.0 / np.sqrt(areas[valid_idx])).reshape(-1, 1)

    # Escalado de datos (VITAL para Redes Neuronales)
    scaler_x = StandardScaler()
    sizes_scaled = scaler_x.fit_transform(sizes)

    # Dividir en Train y Test
    X_train, X_test, y_train, y_test = train_test_split(sizes_scaled, distances, test_size=0.2, random_state=42)

    # Convertir a Tensores de PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Crear DataLoaders (para procesar en lotes/batches)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    # -- Inicializar Modelo, Optimizador y Función de Pérdida --
    model = MlpDistance()
    criterion = nn.MSELoss() # Error Cuadrático Medio
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=50, factor=0.5, verbose=True)
    
    epochs = 1500
    train_losses = []
    val_losses = []
    
    print("Comenzando entrenamiento en PyTorch...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # -- GUARDAR EL MODELO DE FORMA NATIVA PARA PYTHON --
    os.makedirs(os.path.expanduser("~/drone_ws/src/drone_lap/models/"), exist_ok=True)
    
    # Cambiamos la extensión a .pth (estándar de PyTorch normal)
    model_save_path = os.path.expanduser("~/drone_ws/src/drone_lap/models/mlp_distance_model.pth")
    
    # Creamos un diccionario con TODO lo que necesitará tu nuevo nodo
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler_x.mean_[0],
        'scaler_scale': scaler_x.scale_[0]
    }
    
    torch.save(checkpoint, model_save_path)
    
    print("-" * 60)
    print(f"Modelo y Scaler exportados exitosamente en:\n{model_save_path}")
    print("Listo para ser cargado por tu nuevo nodo en Python.")
    print("-" * 60)

    # -- Visualización del Entrenamiento --
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.title('Curva de Aprendizaje del MLP (Área Cruda)')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()