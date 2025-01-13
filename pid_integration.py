import numpy as np
import logging
from pid_class import PIDController
from ot2_class import OT2Env

# Set up logging
log_file = "adaptive_pid_optimized_ot2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logging.info("Starting Optimized PID Controller Test with OT2Env")

# Initialize the OT2 environment
env = OT2Env()

# Initialize PID controllers with starting parameters
pid_x = PIDController(Kp=10, Ki=0.1, Kd=2, dt=0.5, integral_limit=5.0)
pid_y = PIDController(Kp=10, Ki=0.1, Kd=2, dt=0.5, integral_limit=5.0)
pid_z = PIDController(Kp=10, Ki=0.1, Kd=2, dt=0.5, integral_limit=1.0)

# Test one random target position
target_position = np.array([np.random.uniform(-0.1872, 0.253),
                            np.random.uniform(-0.1705, 0.2195),
                            np.random.uniform(0.1693, 0.2895)], dtype=np.float32)
logging.info(f"Generated Target Position: {target_position}")

# Set PID setpoints
pid_x.setpoint = target_position[0]
pid_y.setpoint = target_position[1]
pid_z.setpoint = target_position[2]

# Reset the environment
observation, info = env.reset()

# Initialize loop variables
terminated = False
epoch = 0
all_error = 0
recent_errors = []

# Simulation loop with improvements
while not terminated and epoch < 200:
    epoch += 1

    # Current pipette position
    current_position = observation[:3]

    # Compute individual axis errors
    error_x = abs(target_position[0] - current_position[0])
    error_y = abs(target_position[1] - current_position[1])
    error_z = abs(target_position[2] - current_position[2])
    error = np.linalg.norm(target_position - current_position)
    all_error += error

    # Log errors and coordinates
    logging.info(f"Epoch: {epoch}")
    logging.info(f"Current Position: X={current_position[0]:.4f}, Y={current_position[1]:.4f}, Z={current_position[2]:.4f}")
    logging.info(f"Errors: X={error_x:.4f}, Y={error_y:.4f}, Z={error_z:.4f}")
    logging.info(f"Euclidean Error: {error:.4f}")

    # Success criterion
    if error <= 0.005:
        logging.info(f"Target reached successfully in {epoch} epochs. Final Error: {error:.4f}")
        break

    # Adjust PID parameters dynamically
    if error > 0.1:
        pid_x.Ki, pid_x.Kd = 0.20, 3
        pid_y.Ki, pid_y.Kd = 0.20, 3
        pid_z.Ki, pid_z.Kd = 0.18, 2.8
    elif 0.05 < error <= 0.1:
        pid_x.Ki, pid_x.Kd = 0.12, 1.8
        pid_y.Ki, pid_y.Kd = 0.12, 1.8
        pid_z.Ki, pid_z.Kd = 0.1, 1.5
    else:
        pid_x.Ki, pid_x.Kd = 0.1, 1.2
        pid_y.Ki, pid_y.Kd = 0.1, 1.2
        pid_z.Ki, pid_z.Kd = 0.08, 1.0

    # Compute control actions with action saturation
    control_x = np.clip(pid_x.compute(current_position[0]), -1, 1)
    control_y = np.clip(pid_y.compute(current_position[1]), -1, 1)
    control_z = np.clip(pid_z.compute(current_position[2]), -1, 1)
    action = np.array([control_x, control_y, control_z])

    # Take a step in the environment
    observation, _, terminated, truncated, _ = env.step(action)


    # Track recent errors for early termination
    recent_errors.append(error)
    if len(recent_errors) > 20:
        recent_errors.pop(0)
        avg_error_change = abs(recent_errors[-1] - recent_errors[0]) / 5
        if avg_error_change < 0.0005:
            logging.info(f"Terminating early due to negligible error change after {epoch} epochs.")
            break

    # Handle truncation
    if truncated:
        logging.warning("Environment truncated. Resetting...")
        observation, info = env.reset()
        pid_x.setpoint = target_position[0]
        pid_y.setpoint = target_position[1]
        pid_z.setpoint = target_position[2]

# Log final results
logging.info(f"Final Position: {current_position}")
logging.info(f"Final Error: {error:.4f}")
logging.info(f"Total Error Across Epochs: {all_error:.4f}")
logging.info(f"Total Epochs: {epoch}")

print("Optimized PID Controller Test Complete. Check the log file for details.")
