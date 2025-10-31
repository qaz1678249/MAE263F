import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from singlestepsim import simulate_beam, get_middle_node_xy

def mpc_objective_stateful(control_input_horizon_flat, desired_window, current_q, current_u, control_input_prev1, control_input_prev2, lambda_vel, lambda_accel, dt, nv, RodLength):
    H = len(desired_window)
    control_input_horizon = control_input_horizon_flat.reshape((H, 3))
    q0 = current_q.copy()
    u0 = current_u.copy()
    tracking_error = 0
    for i in range(H):
        q_new, u_new = simulate_beam(u0, q0,control_input_horizon[i, :],dt,nv,RodLength, draw_figure=False)
        tracking_error += np.sum((get_middle_node_xy(nv,q_new) - desired_window[i, :])**2)
        q0 = q_new.copy() # New position becomes old position
        u0 = u_new.copy() # New velocity becomes old velocity
    vel_penalty = 0
    if H > 0:
        vel_penalty += np.sum((control_input_horizon[0] - control_input_prev1)**2)
        for i in range(1, H):
            vel_penalty += np.sum((control_input_horizon[i] - control_input_horizon[i-1])**2)
    accel_penalty = 0
    if H > 0:
        vel_curr = control_input_horizon[0] - control_input_prev1; vel_prev = control_input_prev1 - control_input_prev2
        accel_penalty += np.sum((vel_curr - vel_prev)**2)
    if H > 1:
        vel_curr = control_input_horizon[1] - control_input_horizon[0]; vel_prev = control_input_horizon[0] - control_input_prev1
        accel_penalty += np.sum((vel_curr - vel_prev)**2)
    for i in range(2, H):
        vel_curr = control_input_horizon[i] - control_input_horizon[i-1]; vel_prev = control_input_horizon[i-1] - control_input_horizon[i-2]
        accel_penalty += np.sum((vel_curr - vel_prev)**2)
    return tracking_error + lambda_vel * vel_penalty + lambda_accel * accel_penalty

if __name__ == '__main__':
    # Total time
    totalTime = 100 # second
    
    # Time step
    dt = 0.1 # second
    nv = 19 # number of nodes/vertices

    # Rod length
    RodLength = 1.0 # meter

    # Discrete length / reference length
    deltaL = RodLength / (nv - 1)

    # Geometry
    nodes = np.zeros((nv, 2))
    for c in range(nv):
      nodes[c, 0] = c * deltaL # x-coordinate
      nodes[c, 1] = 0.0 # y-coordinate
    
    # Initial conditions
    q0 = np.zeros(2 * nv)
    for c in range(nv):
      q0[2*c] = nodes[c, 0] # x coordinate
      q0[2*c+1] = nodes[c, 1] # y coordinate

    u0 = np.zeros(2 * nv) # old velocity   

    # Number of steps
    Nsteps = round( totalTime / dt )    
    
    ctime = 0 # Current time
    # --- Generate Desired Trajectory ---
    t = np.linspace(0, totalTime, Nsteps)
    x_d = RodLength/2 * np.cos(np.pi/2*t/totalTime)
    y_d = -RodLength/2 * np.sin(np.pi/2*t/totalTime)
    desired_trajectory = np.stack([x_d, y_d], axis=1)
    #print(desired_trajectory)
    

    # --- Main MPC Control Loop ---

    # MPC Parameters
    prediction_horizon = 10
    lambda_vel = 0.01
    lambda_accel = 0.01

    # --- DEFINE BOUNDS FOR YOUR INPUTS ---
    # Define the (min, max) range for each variable. 
    xc_bounds = (0, 1.5)         
    yc_bounds = (-1, 1) 
    theta_bounds = (-np.pi/2, np.pi/2)         

    # Arrays to store results
    control_input_mpc_solution = np.zeros((Nsteps, 3))
    achieved_trajectory = np.zeros((Nsteps, 2))
    q_state_history = np.zeros((Nsteps + 1, 2*nv))
    u_state_history = np.zeros((Nsteps + 1, 2*nv))

    # Initialization
    control_input_mpc_solution[0] = np.array([RodLength, 0.0, 0.0])
    achieved_trajectory[0] = get_middle_node_xy(nv, q0)
    q_state_history[0] = q0.copy()
    q_state_history[1] = q0.copy()
    control_input_prev1 = control_input_mpc_solution[0].copy()
    control_input_prev2 = control_input_mpc_solution[0].copy()

    print("--- Starting Stateful MPC with Bound Constraints ---")
    start_time = time.time()

    for k in range(1, Nsteps):
        print(f"Optimizing step {k+1}/{Nsteps}...")
        
        current_q = q_state_history[k]
        current_u = u_state_history[k]
        h_k = min(prediction_horizon, Nsteps - k)
        desired_window = desired_trajectory[k : k + h_k]
        initial_guess_horizon = np.tile(control_input_prev1, h_k)
        
        # --- 2. CONSTRUCT BOUNDS FOR THE HORIZON ---
        # The bounds list must match the length of the flattened optimization variable.
        horizon_bounds = []
        for _ in range(h_k):
            horizon_bounds.append(xc_bounds)
            horizon_bounds.append(yc_bounds)
            horizon_bounds.append(theta_bounds)

        # --- 3. ADD BOUNDS TO THE OPTIMIZER CALL ---
        result = minimize(
            fun=mpc_objective_stateful,
            x0=initial_guess_horizon,
            args=(desired_window, current_q, current_u, control_input_prev1, control_input_prev2, lambda_vel, lambda_accel, dt, nv, RodLength),
            method='L-BFGS-B',
            bounds=horizon_bounds, # <-- Pass the bounds here
            options={'maxiter': 2000, 'ftol': 1e-7}
        )
        
        first_input = result.x.reshape((h_k, 3))[0, :]
        
        # Apply, store, and update history
        q_new, u_new = simulate_beam(current_u, current_q, first_input,dt,nv,RodLength, draw_figure=False)
        control_input_mpc_solution[k, :] = first_input
        achieved_trajectory[k, :] = get_middle_node_xy(nv,q_new)
        q_state_history[k+1, :] = q_new.copy()
        u_state_history[k+1, :] = u_new.copy()
        control_input_prev2 = control_input_prev1
        control_input_prev1 = first_input
        
    end_time = time.time()
    print(f"\nMPC finished in {end_time - start_time:.2f} seconds.")

    # --- Plotting (no changes) ---
    plt.figure(figsize=(12, 6))
    # ... (plotting code is identical to the previous version)
    plt.subplot(1, 2, 1)
    plt.plot(desired_trajectory[:, 0], desired_trajectory[:, 1], 'bo-', label='Desired Trajectory')
    plt.plot(achieved_trajectory[:, 0], achieved_trajectory[:, 1], 'rx--', label='Achieved (MPC)')
    plt.plot(achieved_trajectory[0, 0], achieved_trajectory[0, 1], 'g*', markersize=15, label='Exact Start')
    plt.title("Output Trajectory [x, y]")
    plt.xlabel("X"); plt.ylabel("Y")
    plt.legend(); plt.axis('equal'); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(t, U_true[:, 0], 'b-', alpha=0.3, label='True $a$')
    plt.plot(t, U_mpc_solution[:, 0], 'b-', label='Found $a$ (MPC)')
    plt.plot(t, U_true[:, 1], 'g-', alpha=0.3, label='True $b$')
    plt.plot(t, U_mpc_solution[:, 1], 'g-', label='Found $b$ (MPC)')
    plt.plot(t, U_true[:, 2], 'r-', alpha=0.3, label='True $c$')
    plt.plot(t, U_mpc_solution[:, 2], 'r-', label='Found $c$ (MPC)')
    plt.title("Input Trajectories [a, b, c]")
    plt.xlabel("Time Step (k)"); plt.ylabel("Value")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show()