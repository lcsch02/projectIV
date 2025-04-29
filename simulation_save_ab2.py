# Import necessary packages 
import math
import numpy as np
from timeit import default_timer as timer
from numba import njit, prange
np.set_printoptions(linewidth=400)

# Begin tracking total compute time
start = timer()

# Initialise system constants and choose particle number
pi = math.pi
npoints = 100
viscosity = 1
gravity = 1.0
timestep = 0.01
t_max = 500

# Select the number of timesteps to elapse before saving positions; A space-saving measure
save_every = 50

# Set force vector for gravity
F = np.tile([0, 0, -gravity], npoints)

# Initialize position array
r = np.zeros(3 * npoints)

# Generate uniformly random particle positions
for i in range(npoints):
    while True:
        r_x, r_y, r_z = np.random.uniform(-1, 1, 3)
        if np.sqrt(r_x**2 + r_y**2 + r_z**2) <= 1:
            r[3*i:3*i+3] = [r_x, r_y, r_z]
            break


# Define the function which computes the Oseen tensor at a point in time
@njit(parallel=True)
def compute_oseen_tensor(T, r, npoints, inv_8pi_viscosity):
    for i in prange(npoints):
        for j in prange(i+1, npoints):
            R = r[3*j:3*(j+1)] - r[3*i:3*(i+1)]
            AbsR = np.linalg.norm(R)
            Tij = (np.eye(3) + np.outer(R, R) / AbsR**2) * inv_8pi_viscosity / AbsR
            T[3*i:3*i+3, 3*j:3*j+3] = Tij
            T[3*j:3*j+3, 3*i:3*i+3] = Tij  # Exploit symmetry
    return T


# Define function to execute the simulation
@njit
def main(r):
    # Define the number of timesteps to be simulated
    num_steps = int(t_max / timestep) + 1

    # Initialise the array to which our simulation data will be written and later saved
    r_steps = np.zeros((num_steps//save_every + 1, 3 * npoints))
    r_steps[0] = r.copy()

    # Initialise scaling constant and the Oseen tensor
    inv_8pi_viscosity = 1 / (8 * pi * viscosity)
    T = np.zeros((3 * npoints, 3 * npoints))
    
    # Calculate initial Oseen tensor
    T = compute_oseen_tensor(T, r, npoints, inv_8pi_viscosity)
    
    # Store the initial velocity (needed for AB2)
    u_previous = T @ F
    
    # Use Euler method for the first step
    r += timestep * u_previous
    if save_every == 1:
        r_steps[1] = r
    
    # Perform the simulation
    for step in range(2, num_steps):
        # Calculate Oseen tensor at the current timestep
        T = compute_oseen_tensor(T, r, npoints, inv_8pi_viscosity)

        # Compute current velocities and update positions using AB2
        u_current = T @ F
        r += timestep * (1.5 * u_current - 0.5 * u_previous)
        
        # Update previous velocity for next iteration
        u_previous = u_current.copy()
        
        # Save the current positions if the appropriate number of timesteps elapsed
        if step % save_every == 0:
            r_steps[step//save_every] = r
        
        # Print current timestep number for progress tracking
        print(step, "/", num_steps)

    return r_steps

# Execute the simulation
r_steps = main(r)

# Save the simulation data to a binary file
np.save("100_particle_500_second_data_ab2_1.npy", r_steps)

# Finish tracking and display total compute time
end = timer()
print(end - start)