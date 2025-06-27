import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import tqdm
import imageio.v3 as iio
import matplotlib
import random
# 输出目录
save_dir = "output"
os.makedirs(f"{save_dir}/heatmaps", exist_ok=True)

seed = 0
random.seed(seed)
np.random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)
matplotlib.rcParams['image.lut'] = 256


# ---------------- 模拟参数 ----------------
Lx, Ly = 20.0, 10.0
W_exit = 1.0
Nx, Ny = 100, 50
dx, dy = Lx / Nx, Ly / Ny
dt = 0.01
total_time = 100
steps = int(total_time / dt)
heatmap_interval = int(1.0 / dt)  # 每秒保存热图

# ---------------- 物理参数 ----------------
gamma = 0.1
k_a = 0.2
beta = 10
mu = 0.01
gamma_p = 0.5
alpha = 0.1
lambda_p = 0.2
rho_c = 8.0 * dx * dy  # 临界密度
v_out = 7
v_max = 3
max_rho = 30.0

# ---------------- 初始化 ----------------
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
rho = np.full_like(X, 3.0 * dx * dy) + 0.05 * dx * dy * np.random.randn(*X.shape)
vx = np.zeros_like(rho)
vy = np.zeros_like(rho)
px = np.zeros_like(rho)
py = np.zeros_like(rho)
exit_mask = (X >= Lx - dx) & (np.abs(Y - Ly / 2) <= W_exit / 2)
if not np.any(exit_mask):
    raise ValueError("出口掩码为空，请检查出口宽度或网格大小。")
flow_rate = []

# ---------------- 工具函数 ----------------
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

def apply_boundary(vx, vy, rho, outflow=0):
    vy[:, 0] = vy[:, -1] = 0  # 上下边界无垂直速度
    vx[-1] = np.where(exit_mask[-1], v_out * np.tanh(rho[-1] / rho_c), 0)
    vy[-1] = 0
    vx[0] = vy[0] = 0  # 上下边界无速度
    vx[:, 0] = vx[:, -1] = 0  # 左右边界无水平速度
    return vx, vy, rho



target_vx = []
 
dx_exit = Lx - X; dy_exit = Ly / 2 - Y
e_exit = np.sqrt(dx_exit**2 + dy_exit**2)
dx_exit /= e_exit; dy_exit /= e_exit


target_x = Lx - 1.5
i_target = np.argmin(np.abs(x - target_x))
j_center = np.argmin(np.abs(y - Ly/2))


class FieldMonitor:
    def __init__(self, positions):
        """
        初始化场量监测器
        :param positions: 要监测的位置列表，每个位置是 (x, y) 元组 (单位米)
                          例如: [(1.5, 5.0), (2.0, 5.0), (1.5, 3.0)]
        """
        self.positions = positions
        self.monitor_data = {pos: [] for pos in positions}
        self.grid_indices = {}
        
        # 计算每个位置的网格索引
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
                
                """# 保存频谱数据
                np.savetxt(f"{save_dir}/{field_name}_spectrum_{pos_id}.txt", 
                           np.vstack((freqs, spectrum)).T, 
                           header='freq spectrum')"""
                
                
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
            
            """
            np.savetxt(f"{save_dir}/{field_name}_timeseries_{pos_id}.txt", 
                       data_arr, 
                       header=f'{field_name} time series at ({x_pos:.1f}m, {y_pos:.1f}m)')
"""

monitor = FieldMonitor(positions=[(0.5, Ly / 2),( 1.0, Ly / 2),(1.5,  Ly / 2),(2.0, Ly /2 )])
monitor_vy = FieldMonitor(positions=[(0.5, Ly / 2+0.5),(0.5, Ly /2 -1)])
# ---------------- main ----------------
for step in tqdm.tqdm(range(steps), desc="Simulation", unit="step"):

    rho_safe = np.clip(rho, 1e-4, 10)
    P = rho_safe**beta
    Px, Py = gradient(P, dx)
    lap_vx, lap_vy = laplacian(vx, dx), laplacian(vy, dx)
    

    # 动量与势能更新
    
    dvx = (-gamma * vx + px + k_a * np.abs(vx - v_max)*dx_exit - Px / rho_safe + mu * lap_vx) * dt
    dvy = (-gamma * vy + py + k_a * np.abs(vy - v_max)*dy_exit - Py / rho_safe + mu * lap_vy) * dt
    panic = lambda_p * (rho > rho_c)
    cross = np.nan_to_num(alpha**2 * (px * vy - py * vx))
    
    dpx = (-gamma_p * px + gamma_p * vx - cross * py + panic * dx_exit) * dt
    dpy = (-gamma_p * py + gamma_p * vy + cross * px + panic * dy_exit) * dt

    vx += dvx; vy += dvy; px += dpx; py += dpy

    # 记录目标位置的速度


    monitor.record(vx)
    monitor_vy.record(vy)
    px = np.clip(px, -1e3, 1e3); py = np.clip(py, -1e3, 1e3)
    vx = np.clip(vx, -10, 10); vy = np.clip(vy, -10, 10)

    

    # 质量守恒：方向敏感的入/出流处理
    flux_x = rho * vx * dt / dx  # 右侧通量：>0 为向右出，<0 为向右侧流入
    flux_y = rho * vy * dt / dy  # 上侧通量：>0 为向上出，<0 为从上侧流入

    #print(flux_x)
    flux_x_left = np.roll(flux_x, 1, axis=0)   # 左侧邻居
    #print("-----------")
    #print(flux_x_left)
    flux_x_left[0,:] = 0  # 左侧边界
    #print(flux_x_left)
    flux_y_down = np.roll(flux_y, 1, axis=1)   # 下方邻居
    flux_y_down[:,0] = 0  # 下侧边界

    flux_x_right = np.roll(flux_x, -1, axis=0) # 右侧邻居
    flux_x_right[-1,:] = 0  # 右侧边界

    flux_y_up = np.roll(flux_y, -1, axis=1)    # 上方邻居
    flux_y_up[:,-1] = 0  # 上侧边界

    # 计算流入 (正值都是流入本格)
    inflow_left   = np.maximum(flux_x_left, 0)
    inflow_bottom = np.maximum(flux_y_down, 0)
    inflow_right  = -np.minimum(flux_x_right, 0)
    inflow_top    = -np.minimum(flux_y_up, 0)
    #print(f"{inflow_left.shape} {inflow_bottom.shape} {inflow_right.shape} {inflow_top.shape}")

    # 计算流出 (都是流出)
    outflow_right = abs(flux_x)
    #outflow_right[0,:] = np.maximum(flux_x[0,:], 0)
    #outflow_right[-1,:] =  np.abs(-np.minimum(flux_x[-1,:], 0))
    outflow_right[0,:] = 0
    outflow_right[-1,:] = 0
    outflow_top  =  abs(flux_y)

    #outflow_top[:,0] = np.maximum(flux_y[:,0], 0)
    #outflow_top[:,-1] = np.abs(-np.minimum(flux_y[:,-1], 0))
    outflow_top[:,0] = 0
    outflow_top[:,-1] = 0


    exit_rows = np.where(exit_mask[-1])[0]
    #outflow_right[-1, exit_rows] = np.maximum(rho[-1, exit_rows] * vx[-1, exit_rows] * dt / dx, 0)

    # 更新密度
    rho += inflow_left + inflow_bottom + inflow_right + inflow_top
    rho -= outflow_right + outflow_top 

    #print((inflow_left.sum() + inflow_bottom.sum() + inflow_right.sum()+ inflow_top.sum())-(outflow_right.sum() + outflow_top.sum()))
     
    rho = np.clip(rho, 0, max_rho*dx*dy)  


    vx, vy, rho = apply_boundary(vx, vy, rho)

    # 记录
    outflow = np.sum(abs(flux_x)[-1, exit_rows])
    flow_rate.append(outflow)


    # 保存热图
    if step % heatmap_interval == 0:
        plt.pcolormesh(X, Y, rho, shading='auto', vmin=0, vmax=max_rho*dx*dy)
        plt.colorbar(label='Density'); plt.title(f't = {step*dt:.1f} s')
        plt.savefig(f"{save_dir}/heatmaps/density_{step:05d}.png"); plt.close()

    #每次获得输入以后再进行下一个循环，方便调试
    #input("Press Enter to continue...")  # Uncomment for step-by-step debugging
# ---------------- 频谱与曲线分析 ----------------




flow_arr = np.array(flow_rate)
freqs = np.fft.rfftfreq(flow_arr.size, d=dt)
spectrum = np.abs(np.fft.rfft(flow_arr))
spectrum[0]=0
np.savetxt(f"{save_dir}/flow_spectrum.txt", np.vstack((freqs, spectrum)).T, header='freq spectrum')
mask = freqs <= 5
plt.figure(); plt.plot(freqs[mask], spectrum[mask])
plt.xlabel('Frequency (Hz)'); plt.ylabel('Amplitude'); plt.title('Flow Spectrum'); plt.savefig(f"{save_dir}/flow_spectrum.png"); plt.close()

# 流率-时间曲线
plt.figure(figsize=(8,4)); plt.plot(np.arange(0, total_time, dt), flow_rate)
plt.xlabel('Time (s)'); plt.ylabel('Flow Rate (people/s)'); plt.title('Flow Rate vs Time'); plt.grid(True)
plt.savefig(f"{save_dir}/flow_time_series.png"); plt.close()


print(f"Max Flow Rate: {np.max(flow_rate):.2f}")
print(f"Avg Flow Rate: {np.mean(flow_rate):.2f}")
print(f"Total Evacuated: {int(np.sum(flow_rate) * dt)}")

# 动画生成
frames = sorted(os.listdir(f"{save_dir}/heatmaps"))
fig, ax = plt.subplots(figsize=(6,3))
def animate(i):
    img = iio.imread(f"{save_dir}/heatmaps/{frames[i]}")
    ax.clear(); ax.imshow(img); ax.axis('off')
ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=100)
ani.save(f"{save_dir}/evacuation_animation.gif", writer='pillow')
plt.close()



monitor.analyze_and_plot(save_dir)
monitor_vy.analyze_and_plot(save_dir, field_name="vy")