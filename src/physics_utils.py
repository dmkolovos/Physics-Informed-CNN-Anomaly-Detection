import numpy as np

def calculate_kinematic_complexity(signal):
    """
    Calculates a Physics-Informed complexity penalty based on the 
    first (velocity) and second (acceleration) derivatives of the signal.
    """
    # 1st derivative (velocity approximation)
    velocity = np.diff(signal, axis=0)
    
    # 2nd derivative (acceleration approximation)
    acceleration = np.diff(velocity, axis=0)
    
    # Standard deviations representing kinematic energy
    std_pos = np.std(signal)
    std_vel = np.std(velocity) if len(velocity) > 0 else 0
    std_acc = np.std(acceleration) if len(acceleration) > 0 else 0
    
    # Complexity Score (C = std_pos * std_vel * std_acc)
    complexity_score = std_pos * std_vel * std_acc
    
    return complexity_score

def compute_anomaly_score(mse_loss, signal, alpha=0.7, beta=0.3):
    """
    Fuses the Deep Learning reconstruction loss (MSE) with the Physics-Informed Penalty.
    """
    kinematic_penalty = calculate_kinematic_complexity(signal)
    
    # Final fused score
    total_score = (alpha * mse_loss) + (beta * kinematic_penalty)
    return total_score
