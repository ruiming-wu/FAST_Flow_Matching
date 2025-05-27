import numpy as np

def expand_pid_neighborhood(base_pid, std_multipliers=(0.1, 0.1, 0.1), num_variants=20):
    """
    Generate PID variants by adding Gaussian noise to each component.

    Parameters:
        base_pid (tuple): Base (Kp, Ki, Kd) PID parameters.
        std_multipliers (tuple): Multipliers for standard deviation of each component.
                                 Set to 0.0 for components you want to keep fixed.
        num_variants (int): Number of variants to generate.

    Returns:
        list of tuples: List of perturbed (Kp, Ki, Kd) values.
    """
    Kp, Ki, Kd = base_pid
    variants = []

    for _ in range(num_variants):
        dKp = np.random.normal(0, Kp * std_multipliers[0]) if std_multipliers[0] != 0 else 0.0
        dKi = np.random.normal(0, Ki * std_multipliers[1]) if std_multipliers[1] != 0 else 0.0
        dKd = np.random.normal(0, Kd * std_multipliers[2]) if std_multipliers[2] != 0 else 0.0
        new_pid = (Kp + dKp, Ki + dKi, Kd + dKd)
        variants.append(new_pid)

    return variants

def save_variants_to_txt(variants, filepath="pid_variants.txt"):
    """
    Save list of PID tuples to a plain text file.
    """
    with open(filepath, "w") as f:
        for kp, ki, kd in variants:
            f.write(f"{kp:.6f}, {ki:.6f}, {kd:.6f}\n")

def save_variants_to_npy(variants, filepath="pid_variants.npy"):
    """
    Save list of PID tuples to a .npy file (NumPy array).
    """
    np.save(filepath, np.array(variants))

if __name__ == "__main__":
    # artificially modify the parameters
    base_pid = (3.0, 0.0, 0.0)  
    std_multipliers = (0.1, 0.0, 0.0)  
    num_variants = 10
    txt_path = "pid_variants.txt"
    npy_path = "pid_variants.npy"

    variants = expand_pid_neighborhood(base_pid, std_multipliers, num_variants)

    print(f"\nGenerated {num_variants} PID variants from base PID {base_pid}:")
    for i, (kp, ki, kd) in enumerate(variants, start=1):
        print(f"Variant {i:>2}: Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}")
    
    save_variants_to_txt(variants, txt_path)
    save_variants_to_npy(variants, npy_path)
    print(f"\nSaved variants to:\n- Text file: {txt_path}\n- NumPy file: {npy_path}")
