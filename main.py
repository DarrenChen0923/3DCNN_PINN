"""
Single point incremental forming springback error prediction system - main program
=================================
How to use:
python main.py --grid_size 20 --batch_size 16 --epochs 1000 --patience 1000
"""

import os
import argparse
import torch
import time
import datetime

from data_utils import load_data_by_grid_size, preprocess_data, create_data_loaders
from models import CNN3D_PINN_Model, PhysicsLoss
from trainer import train_model, evaluate_model, init_training
from visualization import (visualize_predictions, visualize_error_heatmap, 
                           visualize_training_history, visualize_grid_representation)
from utils import save_results, save_model_info, inverse_transform_predictions, set_seed


def parse_args():
    """Parsing command line arguments"""
    parser = argparse.ArgumentParser(description='Single point incremental forming springback error prediction system')
    
    # Data related parameters
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.25, help='Val set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random Seed')
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size(mm)')

    
    # Model related parameters
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learing rate')
    parser.add_argument('--patience', type=int, default=20, help='Early Stop Patience Value')
    parser.add_argument('--boundary_weight', type=float, default=0.05, help='Boundary Constraint Weight')
    parser.add_argument('--smoothness_weight', type=float, default=0.05, help='Smoothness constraint weight')
    
    # System related parameters
    parser.add_argument('--device', type=str, default='auto', help='Training device, optional: cuda, cpu, auto')
    parser.add_argument('--output_dir', type=str, default='results', help='Output Directory')
    parser.add_argument('--visualize', action='store_true',default=False, help='Whether to visualize the results')
    
    return parser.parse_args()


def main():
    """Main funciton"""
    # Parsing command line arguments
    args = parse_args()
    
    # Build output directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f"grid_{args.grid_size}mm_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # set random seed
    set_seed(args.seed)
    
    # set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Use device: {device}")
    
    # Modified to use grid size to load data
    print(f"Load data with a grid size of {args.grid_size}mm")
    point_series, errors = load_data_by_grid_size(args.grid_size)
    print(f"Loading completed: {len(point_series)} samples")
    
    # Display data example
    if args.visualize:
        print("Sample point series visualization...")
        sample_idx = 0
        sample_point_series = point_series[sample_idx]
        visualize_grid_representation(
            sample_point_series, 
            save_path=os.path.join(output_dir, 'sample_grid.png')
        )
    
    # Preprocessing Data
    print("Preprocessing Data...")
    data_dict = preprocess_data(
        point_series, errors, 
        test_size=args.test_size, 
        val_size=args.val_size, 
        random_state=args.seed
    )
    
    # Creating a Data Loader
    print(f"Creating a Data Loader, batch size: {args.batch_size}")
    loaders = create_data_loaders(data_dict, batch_size=args.batch_size)
    
    # Init model
    print("Init model...")
    model = CNN3D_PINN_Model()
    
    # save model information
    save_model_info(
        model, 
        os.path.join(output_dir, 'model_info.json'),
        additional_info={
            'data_size': len(point_series),
            'train_size': len(data_dict['X_train']),
            'val_size': len(data_dict['X_val']),
            'test_size': len(data_dict['X_test']),
            'batch_size': args.batch_size,
            'device': str(device)
        }
    )
    
    # Initialize training components
    print(f"Initialize optimizer and learning rate scheduler, learning rate: {args.lr}")
    optimizer, scheduler = init_training(model, lr=args.lr, patience=10)
    
    # Define the loss function
    print(f"Define loss function, boundary weight: {args.boundary_weight}, smoothness weight: {args.smoothness_weight}")
    criterion = PhysicsLoss(
        boundary_weight=args.boundary_weight, 
        smoothness_weight=args.smoothness_weight
    )
    
    # Train model
    print(f"Start training, maximum number of epochs: {args.epochs}, early stopping patience value: {args.patience}")
    start_time = time.time()
    
    model, history = train_model(
        model=model,
        train_loader=loaders['train_loader'],
        val_loader=loaders['val_loader'],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        scheduler=scheduler,
        patience=args.patience,
        save_dir=output_dir
    )
    
    training_time = time.time() - start_time
    print(f"Training completed, took: {training_time:.2f} seconds")
    
    # Visualizing training history
    if args.visualize:
        print("Visualizing training history...")
        visualize_training_history(
            history, 
            save_path=os.path.join(output_dir, 'training_history.png')
        )
    
    # Evaluating the Model
    print("Evaluating the Model...")
    eval_results = evaluate_model(
        model=model,
        test_loader=loaders['test_loader'],
        device=device
    )
    
    # Denormalized prediction results
    orig_predictions, orig_targets = inverse_transform_predictions(
        eval_results['predictions'], 
        eval_results['targets'], 
        data_dict['y_scaler']
    )
    
    # Save the evaluation results
    eval_results.update({
        'orig_predictions': orig_predictions,
        'orig_targets': orig_targets,
        'training_time': training_time
    })
    
    save_results(
        eval_results, 
        os.path.join(output_dir, 'evaluation_results.json')
    )
    
    # Print Print evaluation metrics
    print("\n=== Ealuation metrics ===")
    print(f"MAE: {eval_results['mae']:.4f}")
    print(f"MSE: {eval_results['mse']:.4f}")
    print(f"RMSE: {eval_results['rmse']:.4f}")
    print(f"R²: {eval_results['r2']:.4f}")
    
    # Visualizing prediction results
    if args.visualize:
        print("Visualizing prediction results...")
        visualize_predictions(
            orig_predictions, 
            orig_targets, 
            save_path=os.path.join(output_dir, 'predictions.png')
        )
        
        print("Generate error heatmap...")
        visualize_error_heatmap(
            data_dict['X_test'], 
            eval_results['predictions'], 
            eval_results['targets'], 
            save_path=os.path.join(output_dir, 'error_heatmap.png')
        )
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()