import numpy as np
import matplotlib.pyplot as plt


class RocketConfig:
    def __init__(self):
       
        self.wet_mass = 29900.0   
        self.dry_mass = 6500.0    
        
       
        self.thrust = 420000.0 
        
        self.burn_time = 143.0    
        self.diameter = 1.78      
        self.area = np.pi * (self.diameter / 2)**2 
        self.drag_coeff = 0.75    
        
        
        self.launch_angle = 89.0 
        
    
        fuel_mass = self.wet_mass - self.dry_mass
        self.mass_flow_rate = fuel_mass / self.burn_time 

    def info(self):
        print(f"--- ROCKET CONFIGURATION ---")
        print(f"Launch Mass: {self.wet_mass} kg")
        print(f"Thrust:      {self.thrust} N")
        print(f"Burn Time:   {self.burn_time} s")
        print(f"----------------------------")


def get_air_density(altitude):
    rho_0 = 1.225
    scale_height = 8500.0
    if altitude < 0: return rho_0
    elif altitude > 150000: return 0.0
    return rho_0 * np.exp(-altitude / scale_height)

def get_gravity(altitude):
    g0 = 9.81
    R_earth = 6371000.0
    return g0 * (R_earth / (R_earth + altitude))**2


def rocket_derivatives(t, state, config):
    x, y, vx, vy, m = state
    
 
    rho = get_air_density(y)
    g = get_gravity(y)
    v = np.sqrt(vx**2 + vy**2)
    
  
    theta = np.arctan2(vy, vx)

  
    drag = 0.5 * rho * v**2 * config.drag_coeff * config.area
    
    if t < config.burn_time and m > config.dry_mass:
        thrust = config.thrust
        dm_dt = -config.mass_flow_rate
    else:
        thrust = 0.0
        dm_dt = 0.0
        
   
    Fx = (thrust - drag) * np.cos(theta)
    Fy = (thrust - drag) * np.sin(theta) - (m * g)
    
    ax = Fx / m
    ay = Fy / m
    
    return np.array([vx, vy, ax, ay, dm_dt])


def runge_kutta_step(t, state, dt, config):
    k1 = rocket_derivatives(t, state, config)
    k2 = rocket_derivatives(t + 0.5*dt, state + 0.5*dt*k1, config)
    k3 = rocket_derivatives(t + 0.5*dt, state + 0.5*dt*k2, config)
    k4 = rocket_derivatives(t + dt, state + dt*k3, config)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def solve_trajectory_manual(config):
   
    v_start = 60.0 
    
    angle_rad = np.deg2rad(config.launch_angle)
    vx = v_start * np.cos(angle_rad)
    vy = v_start * np.sin(angle_rad)
    
    state = np.array([0.0, 0.0, vx, vy, config.wet_mass])
    
    t = 0.0
    dt = 0.1
    max_time = 1000.0
    
    t_hist = [t]
    state_hist = [state]
    
    print("Simulating Mission...")
    while t < max_time:
        if state[1] < 0 and t > 5:
            print(f"Impact detected at t={t:.2f}s")
            break
            
        state = runge_kutta_step(t, state, dt, config)
        t += dt
        
        t_hist.append(t)
        state_hist.append(state)
        
    return np.array(t_hist), np.array(state_hist)


if __name__ == "__main__":
    rocket = RocketConfig()
    rocket.info()
    
    times, states = solve_trajectory_manual(rocket)
    
    x = states[:, 0]
    y = states[:, 1]
    vx = states[:, 2]
    vy = states[:, 3]
    m = states[:, 4]
    velocity = np.sqrt(vx**2 + vy**2)
    
    print(f"\n--- MISSION REPORT ---")
    print(f"Apogee:      {np.max(y)/1000:.2f} km")
    print(f"Range:       {x[-1]/1000:.2f} km")
    print(f"Max Speed:   {np.max(velocity):.2f} m/s")
    print(f"Flight Time: {times[-1]:.2f} s")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x/1000, y/1000, 'b-')
    plt.title('Trajectory (Side View)')
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.axis('equal')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(times, y/1000, 'r-')
    plt.title('Altitude vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(times, velocity, 'g-')
    plt.title('Velocity vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(times, m, 'k-')
    plt.title('Mass vs Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()