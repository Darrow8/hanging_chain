import numpy as np
from scipy.special import j0, jn_zeros
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

# Physical parameters
g = 9.81  # Acceleration due to gravity, m.s-2
L = 1   # Chain length, m
A = 0.1  # Maximum amplitude scale, m

# Discretization
nz = 2000  # Number of points along the chain
z = np.linspace(0, L, nz)  # Vertical axis (m)
u = 2 * np.sqrt(z/g)       # Scaled vertical axis

# Number of modes to consider in the simulation
num_modes = 30

# Calculate frequencies for each mode
frequencies = [jn_zeros(0, n+1)[n] * np.sqrt(g/L) / 2 for n in range(num_modes)]

# Mode shapes (Bessel functions)
mode_shapes = np.array([j0(frequencies[n] * u) for n in range(num_modes)])


# Calculate mode coefficients for given initial conditions
def calculate_mode_coefficients(x0, v0):
    """
    Calculate the coefficients for each mode based on initial conditions.
    
    Parameters:
    -----------
    x0 : numpy.ndarray
        Initial position along the chain
    v0 : numpy.ndarray
        Initial velocity along the chain
    
    Returns:
    --------
    A : numpy.ndarray
        Amplitude coefficients for each mode
    B : numpy.ndarray
        Phase coefficients for each mode
    """
    # Weight function for the inner product
    weight = z  # For hanging chain: ρ(z) = constant * z
    
    # Calculate normalization factors for each mode
    norm_factors = np.array([np.sum(weight * mode_shapes[n]**2) for n in range(num_modes)])
    
    # Calculate position coefficients (A)
    A = np.array([np.sum(weight * x0 * mode_shapes[n]) / norm_factors[n] for n in range(num_modes)])
    
    # Calculate velocity coefficients (B)
    B = np.array([np.sum(weight * v0 * mode_shapes[n]) / (frequencies[n] * norm_factors[n]) for n in range(num_modes)])
    
    return A, B

# Calculate chain position at time t
def chain_position(t, A_coefs, B_coefs):
    """
    Calculate the position of the chain at time t.
    
    Parameters:
    -----------
    t : float
        Time in seconds
    A_coefs : numpy.ndarray
        Amplitude coefficients for each mode
    B_coefs : numpy.ndarray
        Phase coefficients for each mode
    
    Returns:
    --------
    x : numpy.ndarray
        Position along the chain at time t
    """
    x = np.zeros_like(z)
    
    for n in range(num_modes):
        x += mode_shapes[n] * (A_coefs[n] * np.cos(frequencies[n] * t) + 
                             B_coefs[n] * np.sin(frequencies[n] * t))
    
    return x

def set_initial_position(mode='vertical', func=None, amplitude=A):
    """
    Set the initial position of the chain.
    
    Parameters:
    -----------
    mode : str
        Mode of initial position ('vertical', 'custom', 'pluck', or 'sine')
    func : callable, optional
        Custom function for initial position if mode='custom'
    amplitude : float
        Maximum amplitude of displacement (default is global A)
    """
    if mode == 'vertical':
        return np.zeros_like(z)
    elif mode == 'angled':
        # Simple linear function x(z) = L - z
        return z * 0.4 #(L - z) * 0.2
    elif mode == 'custom' and callable(func):
        return func(z)
    elif mode == 'pluck':
        mid_point = L/2
        x0 = np.zeros_like(z)
        x0[z <= mid_point] = amplitude * z[z <= mid_point] / mid_point
        x0[z > mid_point] = amplitude * (L - z[z > mid_point]) / (L - mid_point)
        return x0
    elif mode == 'sine':
        # Use the third mode shape (Bessel function) instead of sine
        mode_number = 2  # index 2 gives the third mode (0-based indexing)
        return amplitude * mode_shapes[mode_number] * 2
    else:
        return np.zeros_like(z)

def set_initial_velocity(mode='zero', amplitude=0.1):
    """
    Set the initial velocity of the chain.
    
    Parameters:
    -----------
    mode : str
        Mode of initial velocity ('zero', 'uniform', or 'sine')
    amplitude : float
        Maximum amplitude of velocity
    
    Returns:
    --------
    v0 : numpy.ndarray
        Initial velocity along the chain
    """
    if mode == 'uniform':
        return amplitude * np.ones_like(z) * 10
    elif mode == 'sine':
        return amplitude * np.sin(np.pi * z / L)
    else:  # 'zero' or any other input
        return np.zeros_like(z)

x0 = set_initial_position('sine')  # or simply set_initial_position()
v0 = set_initial_velocity('zero')      # chain at rest

# Calculate the fundamental period (period of the first mode)
T = 2 * np.pi / frequencies[0]  # Period of first mode

# Create snapshots at t=0, T/2, T
fig_snapshots, axs = plt.subplots(1, 3, figsize=(12, 4))
fig_snapshots.suptitle(f'Chain Position Snapshots (Period = {T:.2f} s)')

# Calculate mode coefficients once
A_coefs, B_coefs = calculate_mode_coefficients(x0, v0)

# Create snapshots at different times
times = [0, 0.25, 0.5]  # Start, half period, full period
for i, t in enumerate(times):
    axs[i].axis('off')
    axs[i].set_xlim((-L, L))
    axs[i].set_ylim((0, L))
    axs[i].set_title(f't = {t:.2f} s')
    
    # Calculate and plot chain position
    x = chain_position(t, A_coefs, B_coefs)
    axs[i].plot(x, z, 'k', dashes=[5, 2], lw=3)

plt.tight_layout()

# Modify animation function with adjusted view
def create_animation(x0, v0):
    """
    Create an animation of the chain oscillation.
    """
    # Calculate mode coefficients ONCE at the start
    A_coefs, B_coefs = calculate_mode_coefficients(x0, v0)
    
    # Set fixed duration instead of using period
    duration = 3.0  # Animation duration in seconds
    
    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    ax.set_xlim((-L, L))  # Expanded limits to show full 45-degree angle
    ax.set_ylim((0, L))
    ax.set_title('Hanging Chain')
    
    # Add time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                       verticalalignment='top')
    
    line, = ax.plot([], [], 'k', dashes=[5, 2], lw=3)
    
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line, time_text
    
    # Delay between frames (ms)
    interval = 20  # milliseconds
    fps = 1000/interval  # frames per second
    nframes = int(duration * fps)
    
    def animate(frame):
        # Calculate time more precisely
        t = frame / fps  # This gives time in seconds
        x = chain_position(t, A_coefs, B_coefs)
        line.set_data(x, z)
        
        # Update time text
        time_text.set_text(f'Time: {t:.2f} s')
        return line, time_text
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=nframes, interval=interval, blit=True)
    
    return anim, fig

# Create and display animation
anim, fig_anim = create_animation(x0, v0)

# Show both the snapshots and animation
plt.show()
