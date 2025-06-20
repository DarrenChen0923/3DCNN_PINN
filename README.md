# CNN3D-PINN: Physics-Informed Neural Networks for Springback Prediction in SPIF
This repository implements a novel hybrid deep learning approach for springback prediction in Single Point Incremental Forming (SPIF) processes. By combining 3D Convolutional Neural Networks (CNN3D) with Physics-Informed Neural Networks (PINN), our model accurately predicts springback errors without requiring detailed material parameters.

# Overview
Single Point Incremental Forming (SPIF) is an advanced sheet metal forming technique that offers exceptional flexibility for small-batch and customized manufacturing. However, the springback phenomenon—elastic deformation that occurs after tool force removal—significantly affects dimensional accuracy. This project presents a deep learning approach to predict springback errors in SPIF processes.
Our method uses a hybrid CNN3D-PINN architecture that:

1) Processes geometry data directly from point sequences without requiring detailed material parameters
2) Incorporates physical constraints through both explicit loss terms and a dedicated PINN branch
3) Provides accurate springback predictions that can be visualised as error heatmaps

# Model Architecture
The model combines geometric feature extraction with physics-based constraints in a parallel structure:
![image](https://github.com/user-attachments/assets/5846209f-a8ff-4ebe-9921-eaeec64d162a)

# CNN3D Module
Extracts geometric features from 3D voxelized point data:
```bash
class CNN3D_Module(nn.Module):
    def __init__(self):
        super(CNN3D_Module, self).__init__()
        
        self.cnn_layers = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 1, 1), padding=0),
            nn.ReLU(),
        )
        
        self.flatten_size = 64 * 3 * 3 * 1
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
```

# PINN Module
Captures physical relationships between adjacent points:
```bash
class PINN_Module(nn.Module):
    def __init__(self):
        super(PINN_Module, self).__init__()
        
        self.physics_net = nn.Sequential(
            nn.Linear(8, 32),  # Input is 8 adjacent point spreads
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
```

# Fusion Layer
Combines predictions from both branches:
```bash
class CNN3D_PINN_Model(nn.Module):
    def __init__(self):
        super(CNN3D_PINN_Model, self).__init__()
        
        self.cnn_module = CNN3D_Module()
        self.pinn_module = PINN_Module()
        self.fusion_layer = nn.Linear(2, 1)
```
The dual-branch architecture allows the model to simultaneously learn from data patterns and physical principles, with the fusion layer dynamically weighting their contributions.

# Physics-Informed Loss Function
A key innovation in our approach is the physics-informed loss function that incorporates:
```bash
class PhysicsLoss(nn.Module):
    def __init__(self, boundary_weight=0.1, smoothness_weight=0.1):
        super(PhysicsLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight
    
    def forward(self, pred, target, point_series):
        # Data fitting loss
        data_loss = self.mse_loss(pred, target)
        
        # Smoothness constraint
        if pred.size(0) > 1:
            pred_diff = pred[1:] - pred[:-1]
            smoothness_loss = torch.mean(pred_diff**2)
            
            # Boundary constraints
            batch_point_series = point_series.view(pred.size(0), -1)
            border_indices = [0, 2, 6, 8]  # Corner points are boundary points
            border_points = batch_point_series[:, border_indices]
            boundary_loss = torch.mean(torch.abs(border_points))
            
            # Total loss with physics constraints
            total_loss = data_loss + \
                         self.smoothness_weight * smoothness_loss + \
                         self.boundary_weight * boundary_loss
        else:
            total_loss = data_loss
            
        return total_loss
```

The total loss combines:

1) Data Loss: Standard MSE between predictions and ground truth
2) Smoothness Constraint: Penalises high-frequency oscillations in the predicted deformation field
3) Boundary Constraint: Enforces near-zero displacement at clamped boundaries

These physics-based constraints ensure the predictions conform to the expected physical behaviour of sheet metal during springback.

# Requirements
Install the required packages:
```bash
pip install -r requirements.txt
```
Requirements include:
```bash
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
torch>=1.10.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

# Usage
# Command Line Interface
Run the model with customised parameters:
```bash
python main.py --grid_size 20 --batch_size 16 --epochs 1000 --patience 100 --boundary_weight 0.05 --smoothness_weight 0.05
```
# Key parameters:

| Argument | Description |
|----------|-------------|
| `--grid_size` | Data grid size in mm (5, 10, 15, 20) |
| `--batch_size` | Batch size for training |
| `--epochs` | Maximum training epochs |
| `--patience` | Early stopping patience |
| `--boundary_weight` | Weight for boundary constraint |
| `--smoothness_weight` | Weight for smoothness constraint |
| `--visualize` | Flag to generate visualizations |
| `--lr` | Learning rate (default: 0.001) |
| `--test_size` | Test set ratio (default: 0.2) |
| `--val_size` | Validation set ratio (default: 0.25) |
| `--seed` | Random seed for reproducibility (default: 42) |
| `--device` | Training device: 'cuda', 'cpu', or 'auto' |
| `--output_dir` | Directory to save results (default: 'results') |

# Results
<img width="466" alt="image" src="https://github.com/user-attachments/assets/402a7eb5-4d03-4073-a763-7b4d12077c18" />


