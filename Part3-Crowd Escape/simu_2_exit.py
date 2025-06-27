import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import tqdm
import imageio.v3 as iio
import matplotlib
import random


save_dir = "output_2_exit"
os.makedirs(f"{save_dir}/heatmaps", exist_ok=True)

seed = 0
random.seed(seed)
np.random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)
matplotlib.rcParams['image.lut'] = 256


# ---------------- Simulation Parameters ----------------
Lx, Ly = 20.0, 10.0
Nx, Ny = 100, 50
dx, dy = Lx / Nx, Ly / Ny
dt = 0.01
total_time = 100
steps = int(total_time / dt)
heatmap_interval = int(1.0 / dt)  # Save heatmap every second


W_exit_top = 1.0  # Width of the top exit
W_exit_bottom = 1.0 # Width of the bottom exit
Y_exit_top_center = Ly - W_exit_top / 2 # Center Y-coordinate for top exit
Y_exit_bottom_center = W_exit_bottom / 2 # Center Y-coordinate for bottom exit

# ---------------- Physical Parameters ----------------
gamma = 0.1
k_a = 0.2
beta = 10
mu = 0.01
gamma_p = 0.5
alpha = 0.1
lambda_p = 0.2
rho_c = 8.0 * dx * dy  # Critical density
v_out = 7
v_max = 3
max_rho = 30.0

# ---------------- Initialization ----------------
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
rho = np.full_like(X, 15.0 * dx * dy) + 0.05 * dx * dy * np.random.randn(*X.shape)
vx = np.zeros_like(rho)
vy = np.zeros_like(rho)
px = np.zeros_like(rho)
py = np.zeros_like(rho)


exit_mask_top = (X >= Lx - dx) & (np.abs(Y - Y_exit_top_center) <= W_exit_top / 2)
exit_mask_bottom = (X >= Lx - dx) & (np.abs(Y - Y_exit_bottom_center) <= W_exit_bottom / 2)


combined_exit_mask = exit_mask_top | exit_mask_bottom

if not np.any(combined_exit_mask):
    raise ValueError("Combined exit mask is empty. Check exit widths or grid size.")

flow_rate = []

# ---------------- Utility Functions ----------------
def gradient(f, dx):
    fx = np.zeros_like(f); fy = np.zeros_like(f)
    fx[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    fx[0] = (f[1] - f[0]) / dx
    fx[-1] = (f[-1] - f[-2]) / dx
    fy[:,1:-1] = (f[:,2:] - f[:,:-2]) / (2 * dy)
    fy[:,0] = (f[:,1] - f[:,0]) / dy
    fy[:,-1] = (f[:,-1] - f[:,-2]) / dy
    return fx, fy

def laplacian(f, dx):
    f_xx = (np.roll(f,1,0) - 2*f + np.roll(f,-1,0)) / dx**2
    f_yy = (np.roll(f,1,1) - 2*f + np.roll(f,-1,1)) / dy**2
    return np.nan_to_num(f_xx + f_yy)

def apply_boundary(vx, vy, rho):
    vy[:, 0] = vy[:, -1] = 0  # No vertical velocity at top/bottom boundaries
    # vx[-1] = np.where(combined_exit_mask[-1], v_out * np.tanh(rho[-1] / rho_c), 0) # This was for a single exit
    
    # New: Apply outflow boundary condition at both exits

    vx[-1, combined_exit_mask[-1]] = v_out * np.tanh(rho[-1, combined_exit_mask[-1]] / rho_c)
    vx[-1, ~combined_exit_mask[-1]] = 0 # No outflow where there's no exit

    vy[-1, :] = 0 #
    vx[0, :] = vy[0, :] = 0  
    vx[:, 0] = vx[:, -1] = 0 
    return vx, vy, rho



# New: Calculate distances to the center of each exit
# X coordinate for both exits is Lx - dx (rightmost column)
exit_x_coord = Lx - dx


dx_to_top_exit = exit_x_coord - X
dy_to_top_exit = Y_exit_top_center - Y
dist_to_top_exit = np.sqrt(dx_to_top_exit**2 + dy_to_top_exit**2)


dx_to_bottom_exit = exit_x_coord - X
dy_to_bottom_exit = Y_exit_bottom_center - Y
dist_to_bottom_exit = np.sqrt(dx_to_bottom_exit**2 + dy_to_bottom_exit**2)

dx_exit = np.zeros_like(X)
dy_exit = np.zeros_like(Y)



top_exit_is_closer = dist_to_top_exit <= dist_to_bottom_exit

dx_exit[top_exit_is_closer] = dx_to_top_exit[top_exit_is_closer]
dy_exit[top_exit_is_closer] = dy_to_top_exit[top_exit_is_closer]

dx_exit[~top_exit_is_closer] = dx_to_bottom_exit[~top_exit_is_closer]
dy_exit[~top_exit_is_closer] = dy_to_bottom_exit[~top_exit_is_closer]


e_exit = np.sqrt(dx_exit**2 + dy_exit**2)
e_exit[e_exit == 0] = 1 
dx_exit /= e_exit
dy_exit /= e_exit


class FieldMonitor:
    def __init__(self, positions):
        """
        Initialize field monitor
        :param positions: List of positions to monitor, each a (x, y) tuple (in meters)
                          Example: [(1.5, 5.0), (2.0, 5.0), (1.5, 3.0)]
        """
        self.positions = positions
        self.monitor_data = {pos: [] for pos in positions}
        self.grid_indices = {}
        

        for pos in positions:
            x_pos, y_pos = pos
            i = np.argmin(np.abs(x - (Lx - x_pos)))
            j = np.argmin(np.abs(y - y_pos))
            self.grid_indices[pos] = (i, j)


            #print(f"Monitoring position: ({x_pos:.2f}m, {y_pos:.2f}m) -> grid cell ({i}, {j})")
    
    def record(self, field):
       
        for pos in self.positions:
            i, j = self.grid_indices[pos]
            self.monitor_data[pos].append(field[i, j])
    
    def analyze_and_plot(self, save_dir, field_name="vx"):
        
        for pos in self.positions:
            x_pos, y_pos = pos
            data_arr = np.array(self.monitor_data[pos])
            
          
            pos_id = f"x{x_pos:.1f}_y{y_pos:.1f}"
            
           
            plt.figure(figsize=(8,4))
            plt.plot(np.arange(0, total_time, dt)[:len(data_arr)], data_arr)
            plt.xlabel('Time (s)')
            plt.ylabel(f'{field_name} at ({x_pos:.1f}m, {y_pos:.1f}m)')
            plt.title(f'{field_name} vs Time at ({x_pos:.1f}m, {y_pos:.1f}m)')
            plt.grid(True)
            plt.savefig(f"{save_dir}/{field_name}_time_series_{pos_id}.png")
            plt.close()
            

            if len(data_arr) > 1: 
                freqs = np.fft.rfftfreq(len(data_arr), d=dt)
                spectrum = np.abs(np.fft.rfft(data_arr))
                spectrum[0] = 0  
                
               
                mask = (freqs<=5)
                plt.figure()
                plt.plot(freqs[mask], spectrum[mask])
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Amplitude')
                plt.title(f'{field_name} Spectrum at ({x_pos:.1f}m, {y_pos:.1f}m)')
                plt.savefig(f"{save_dir}/{field_name}_spectrum_{pos_id}.png")
                plt.close()
            
          
            print(f"\n--- Analysis at ({x_pos:.1f}m, {y_pos:.1f}m) ---")
            print(f"Max {field_name}: {np.max(data_arr):.4f}")
            print(f"Min {field_name}: {np.min(data_arr):.4f}")
            print(f"Mean {field_name}: {np.mean(data_arr):.4f}")
            print(f"Std dev {field_name}: {np.std(data_arr):.4f}")

monitor_top_vx = FieldMonitor(positions=[(0.5, Y_exit_top_center), (1.0, Y_exit_top_center), (1.5, Y_exit_top_center), (2.0, Y_exit_top_center)])
monitor_bottom_vx = FieldMonitor(positions=[(0.5, Y_exit_bottom_center), (1.0, Y_exit_bottom_center), (1.5, Y_exit_bottom_center), (2.0, Y_exit_bottom_center)])
for step in tqdm.tqdm(range(steps), desc="Simulation", unit="step"):

    rho_safe = np.clip(rho, 1e-4, 30)
    P = rho_safe**beta
    Px, Py = gradient(P, dx)
    lap_vx, lap_vy = laplacian(vx, dx), laplacian(vy, dx)
    


    
    dvx = (-gamma * vx + px + k_a * np.abs(vx - v_max)*dx_exit - Px / rho_safe + mu * lap_vx) * dt
    dvy = (-gamma * vy + py + k_a * np.abs(vy - v_max)*dy_exit - Py / rho_safe + mu * lap_vy) * dt
    panic = lambda_p * (rho > rho_c)
    cross = np.nan_to_num(alpha**2 * (px * vy - py * vx))
    
    dpx = (-gamma_p * px + gamma_p * vx - cross * py + panic * dx_exit) * dt
    dpy = (-gamma_p * py + gamma_p * vy + cross * px + panic * dy_exit) * dt

    vx += dvx; vy += dvy; px += dpx; py += dpy

    monitor_top_vx.record(vx)

    monitor_bottom_vx.record(vx)

    px = np.clip(px, -1e3, 1e3); py = np.clip(py, -1e3, 1e3)
    vx = np.clip(vx, -10, 10); vy = np.clip(vy, -10, 10)

    


    flux_x = rho * vx * dt / dx
    flux_y = rho * vy * dt / dy

    flux_x_left = np.roll(flux_x, 1, axis=0)
    flux_x_left[0,:] = 0
    flux_y_down = np.roll(flux_y, 1, axis=1)
    flux_y_down[:,0] = 0

    flux_x_right = np.roll(flux_x, -1, axis=0)
    flux_x_right[-1,:] = 0

    flux_y_up = np.roll(flux_y, -1, axis=1)
    flux_y_up[:,-1] = 0


    inflow_left   = np.maximum(flux_x_left, 0)
    inflow_bottom = np.maximum(flux_y_down, 0)
    inflow_right  = -np.minimum(flux_x_right, 0)
    inflow_top    = -np.minimum(flux_y_up, 0)


    outflow_right = abs(flux_x)
    outflow_right[0,:] = 0
    outflow_right[-1,:] = 0 
    outflow_top    = abs(flux_y)
  



    rho += inflow_left + inflow_bottom + inflow_right + inflow_top
    rho -= outflow_right + outflow_top 
    


    rho[-1, combined_exit_mask[-1]] -= np.maximum(flux_x[-1, combined_exit_mask[-1]], 0) 

    rho = np.clip(rho, 0, max_rho*dx*dy) 


    vx, vy, rho = apply_boundary(vx, vy, rho)


    outflow = np.sum(np.abs(flux_x[-1, combined_exit_mask[-1]])) 
    flow_rate.append(outflow)


    if step % heatmap_interval == 0:
        plt.pcolormesh(X, Y, rho, shading='auto', vmin=0, vmax=max_rho*dx*dy)
        plt.colorbar(label='Density'); plt.title(f't = {step*dt:.1f} s')
        plt.savefig(f"{save_dir}/heatmaps/density_{step:05d}.png"); plt.close()

# ---------------- Spectrum and Curve Analysis ----------------

flow_arr = np.array(flow_rate)
freqs = np.fft.rfftfreq(flow_arr.size, d=dt)
spectrum = np.abs(np.fft.rfft(flow_arr))
spectrum[0]=0
np.savetxt(f"{save_dir}/flow_spectrum.txt", np.vstack((freqs, spectrum)).T, header='freq spectrum')
mask = freqs <= 5
plt.figure(); plt.plot(freqs[mask], spectrum[mask])
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude'); plt.title('Flow Spectrum'); plt.savefig(f"{save_dir}/flow_spectrum.png"); plt.close()

# Flow rate vs Time curve
plt.figure(figsize=(8,4)); plt.plot(np.arange(0, total_time, dt), flow_rate)
plt.xlabel('Time (s)'); plt.ylabel('Flow Rate (people/s)'); plt.title('Flow Rate vs Time'); plt.grid(True)
plt.savefig(f"{save_dir}/flow_time_series.png"); plt.close()


print(f"Max Flow Rate: {np.max(flow_rate):.2f}")
print(f"Avg Flow Rate: {np.mean(flow_rate):.2f}")
print(f"Total Evacuated: {int(np.sum(flow_rate) * dt)}")

# Animation generation
frames = sorted(os.listdir(f"{save_dir}/heatmaps"))
fig, ax = plt.subplots(figsize=(6,3))
def animate(i):
    img = iio.imread(f"{save_dir}/heatmaps/{frames[i]}")
    ax.clear(); ax.imshow(img); ax.axis('off')
ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=500)
ani.save(f"{save_dir}/evacuation_animation.gif", writer='pillow')
plt.close()




monitor_top_vx.analyze_and_plot(save_dir, field_name="vx")

monitor_bottom_vx.analyze_and_plot(save_dir, field_name="vx")
