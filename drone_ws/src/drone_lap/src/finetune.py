import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 1. RUTA DE TUS NUEVOS DATOS ---
new_file_path = os.path.expanduser("~/drone_ws/src/drone_lap/data/distance_data_v2.npy")
model_save_path = os.path.expanduser("~/drone_ws/src/drone_lap/models/mlp_distance_model.pth")

# DEFINICIÓN DE LA RED NEURONAL (Debe ser EXACTAMENTE igual a la que vas a cargar)
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

def train_epoch(model, dataloader, optimizer, criterion, noise_factor=0.0):
    model.train() 
    total_loss = 0.0
    for inputs, targets in dataloader:
        noisy_inputs = inputs + torch.randn_like(inputs) * noise_factor
        optimizer.zero_grad()
        outputs = model(noisy_inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def main():
    if not os.path.exists(new_file_path):
        print(f"Error: No se encuentra el archivo {new_file_path}")
        return
    if not os.path.exists(model_save_path):
        print(f"Error: No se encuentra el modelo base en {model_save_path}")
        return

    print("Cargando nuevos datos...")
    data = np.load(new_file_path)
    distances = data[:, 0]
    areas = data[:, 1]

    # Aplicamos la física (Inversa de la Raíz)
    valid_idx = (areas > 50) & (distances < 15.0)
    distances = distances[valid_idx].reshape(-1, 1)
    sizes = (1.0 / np.sqrt(areas[valid_idx])).reshape(-1, 1)

    # --- 2. CARGAR EL MODELO BASE Y EL SCALER ---
    print("Cargando el cerebro anterior (.pth)...")
    checkpoint = torch.load(model_save_path)
    
    # Extraemos el scaler viejo
    old_mean = checkpoint['scaler_mean']
    old_scale = checkpoint['scaler_scale']

    # NORMALIZAMOS LOS DATOS NUEVOS CON EL SCALER VIEJO (Muy importante)
    sizes_scaled = (sizes - old_mean) / old_scale

    X_train, X_test, y_train, y_test = train_test_split(sizes_scaled, distances, test_size=0.2, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=64, shuffle=False)

    # Inicializamos el modelo y le inyectamos los pesos viejos
    model = MlpDistance()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = nn.MSELoss() 
    
    # --- 3. REDUCIR EL LEARNING RATE ---
    # Usamos 0.0001 (Ajuste fino) en lugar de 0.001 (Entrenamiento desde cero)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    
    # El scheduler también lo hacemos más paciente
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=20, factor=0.5, verbose=True)
    
    epochs = 1000 # No necesitas 1500 épocas para finetunear, con unas pocas suele bastar
    train_losses = []
    val_losses = []
    
    print("Comenzando Fine-Tuning en PyTorch...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = eval_epoch(model, test_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Sobrescribimos el modelo con la versión mejorada
    checkpoint_actualizado = {
        'model_state_dict': model.state_dict(),
        'scaler_mean': old_mean,      # Mantenemos el mismo scaler
        'scaler_scale': old_scale
    }
    torch.save(checkpoint_actualizado, model_save_path)
    
    print("-" * 60)
    print(f"Fine-Tuning completado. Modelo actualizado en:\n{model_save_path}")
    print("-" * 60)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss (MSE)')
    plt.plot(val_losses, label='Validation Loss (MSE)')
    plt.title('Curva de Aprendizaje - Fine Tuning')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()