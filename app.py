import os
import webbrowser
from threading import Timer
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import plotly.graph_objects as go
from groq import Groq
import random
import threading
import time

class MartianOrbitalEnv:
    """A multi-dimensional 1D physics-based orbital environment for satellite stabilization."""
    def __init__(self, target_alt, initial_alt=None):
        self.target_altitude = target_alt
        if initial_alt is not None:
            self.current_altitude = initial_alt
        else:
            self.current_altitude = target_alt + random.uniform(-100, 100) 
        self.velocity = random.uniform(-2, 2)  # km per frame vertical velocity
        self.fuel_remaining = 100.0            # Total capacity metric units
        self.max_steps = 40
        self.current_step = 0

    def get_state_index(self):
        """Converts raw physics dimensions into table coordinates for the Q-Table."""
        error = self.current_altitude - self.target_altitude
        
        # Discretize Altitude Error
        if error < -50: error_idx = 0
        elif error < -5: error_idx = 1
        elif abs(error) <= 5: error_idx = 2
        elif error <= 50: error_idx = 3
        else: error_idx = 4

        # Discretize Vertical Velocity (Distinguishes direction of travel)
        if self.velocity < -2: vel_idx = 0
        elif abs(self.velocity) <= 2: vel_idx = 1
        else: vel_idx = 2

        # Discretize Remaining Fuel
        if self.fuel_remaining < 20: fuel_idx = 0
        elif self.fuel_remaining <= 70: fuel_idx = 1
        else: fuel_idx = 2

        return (error_idx, vel_idx, fuel_idx)

    def step(self, action):
        self.current_step += 1
        
        action_mapping = {
            0: 0.0,   # Coast
            1: 1.0,   # Small Up
            2: 3.0,   # Medium Up
            3: 5.0,   # Large Up
            4: -1.0,  # Small Down
            5: -3.0,  # Medium Down
            6: -5.0   # Large Down
        }
        thrust = action_mapping.get(action, 0.0)
        
        fuel_burned = abs(thrust) * 0.15 
        if self.fuel_remaining <= 0:
            thrust = 0.0  
            fuel_burned = 0.0
            
        self.fuel_remaining = max(0.0, self.fuel_remaining - fuel_burned)
        orbital_decay_accel = np.random.normal(0, 0.5) 
        
        self.velocity = self.velocity + thrust + orbital_decay_accel
        self.current_altitude += self.velocity
        
        error = self.current_altitude - self.target_altitude
        
        w1 = 0.005
        w2 = 5.0
        reward = -w1 * (error ** 2) - w2 * fuel_burned
        
        r_sat = 3389.5 + self.current_altitude
        dist_phobos = abs(r_sat - 9376.0)
        dist_deimos = abs(r_sat - 23463.0)
        
        penalty_phobos = 30.0 * np.exp(-0.01 * dist_phobos)
        penalty_deimos = 30.0 * np.exp(-0.01 * dist_deimos)
        penalty_congestion = 15.0 * np.exp(-0.04 * abs(error))
        
        reward -= (penalty_phobos + penalty_deimos + penalty_congestion)
            
        done = self.current_step >= self.max_steps
        if abs(error) > 250 or (self.fuel_remaining <= 0 and abs(error) > 15):
            done = True
            
        return self.get_state_index(), reward, done, fuel_burned

class QTrackingAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.episode_count = 0  
        
        self.num_error_bins = 5
        self.num_vel_bins = 3
        self.num_fuel_bins = 3
        self.num_actions = 7 
        
        self.q_table = np.zeros((self.num_error_bins, self.num_vel_bins, self.num_fuel_bins, self.num_actions))

    def reset_epsilon(self):
        self.epsilon = self.initial_epsilon
        self.episode_count = 0

    def reset_q_table(self):
        self.q_table = np.zeros((self.num_error_bins, self.num_vel_bins, self.num_fuel_bins, self.num_actions))

    def choose_action(self, state_idx, force_inference=False):
        if not force_inference and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return int(np.argmax(self.q_table[state_idx]))

    def learn(self, state_idx, action, reward, next_state_idx, done):
        current_q = self.q_table[state_idx][action]
        if done:
            target = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            target = reward + self.gamma * max_next_q
            
        self.q_table[state_idx][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.episode_count += 1
        if self.episode_count % 300 == 0:
            self.epsilon = max(self.epsilon, 0.5)

app = Flask(__name__, static_folder='.', template_folder='.')

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8050))
DEBUG = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
OPEN_BROWSER = os.environ.get("OPEN_BROWSER", "false").lower() in ("1", "true", "yes")

agent = QTrackingAgent()

rl_training_logs = {
    "episode_reward": [],
    "epsilon": [],
    "max_q": []
}
last_trained_altitude = None

def run_background_training(target_altitude, episodes=800):
    agent.reset_q_table()
    agent.reset_epsilon()
    
    rl_training_logs["episode_reward"] = []
    rl_training_logs["epsilon"] = []
    rl_training_logs["max_q"] = []
    
    for episode in range(episodes):
        env = MartianOrbitalEnv(target_altitude)
        state_idx = env.get_state_index()
        total_reward = 0
        
        while True:
            action = agent.choose_action(state_idx, force_inference=False)
            next_state_idx, reward, done, _ = env.step(action)
            agent.learn(state_idx, action, reward, next_state_idx, done)
            state_idx = next_state_idx
            total_reward += reward
            if done:
                break
                
        agent.decay_epsilon()
        rl_training_logs["episode_reward"].append(total_reward)
        rl_training_logs["epsilon"].append(agent.epsilon)
        rl_training_logs["max_q"].append(float(np.max(agent.q_table)))

        if episode % 5 == 0:
            time.sleep(0.001)

def get_coords(r, anomaly, lan, inc):
    x_p = r * np.cos(anomaly)
    y_p = r * np.sin(anomaly)
    x_s = x_p * np.cos(lan) - y_p * np.sin(lan) * np.cos(inc)
    y_s = x_p * np.sin(lan) + y_p * np.cos(lan) * np.cos(inc)
    z_s = y_p * np.sin(inc)
    return x_s, y_s, z_s

def build_animated_orbital_plot(num_satellites, orbit_altitude, frame_duration, debris_density=0.3, training_mode=False):
    R_mars = 3389.5  
    GM_mars = 42828.3  
    
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_m = R_mars * np.outer(np.cos(u), np.sin(v))
    y_m = R_mars * np.outer(np.sin(u), np.sin(v))
    z_m = R_mars * np.outer(np.ones(np.size(u)), np.cos(v))

    r_orbit = R_mars + orbit_altitude  
    v_orbit_kms = np.sqrt(GM_mars / r_orbit)  
    omega = (v_orbit_kms / r_orbit) * 250  

    r_phobos = 9376.0
    r_deimos = 23463.0
    omega_phobos = (np.sqrt(GM_mars / r_phobos) / r_phobos) * 250
    omega_deimos = (np.sqrt(GM_mars / r_deimos) / r_deimos) * 250

    sat_inc = np.random.uniform(-np.pi/4, np.pi/4, size=num_satellites)
    sat_lan = np.random.uniform(0, 2*np.pi, size=num_satellites)

    # Dynamic generation based on UI debris slider
    num_asteroids = max(0, int(debris_density * 300))
    ast_r = np.random.uniform(9000, 13000, size=num_asteroids) if num_asteroids > 0 else np.array([])
    ast_inc = np.random.uniform(-np.pi/12, np.pi/12, size=num_asteroids) if num_asteroids > 0 else np.array([])
    ast_lan = np.random.uniform(0, 2*np.pi, size=num_asteroids) if num_asteroids > 0 else np.array([])
    ast_phase = np.random.uniform(0, 2*np.pi, size=num_asteroids) if num_asteroids > 0 else np.array([])
    ast_omega = (np.sqrt(GM_mars / ast_r) / ast_r) * 250 if num_asteroids > 0 else np.array([])

    fig = go.Figure()
    theta_track = np.linspace(0, 2 * np.pi, 100)
    
    sat_path_x, sat_path_y, sat_path_z = [], [], []
    for i in range(num_satellites):
        for t_val in theta_track:
            x, y, z = get_coords(r_orbit, t_val, sat_lan[i], sat_inc[i])
            sat_path_x.append(x)
            sat_path_y.append(y)
            sat_path_z.append(z)
        sat_path_x.append(None)
        sat_path_y.append(None)
        sat_path_z.append(None)
        
    p_path_x, p_path_y, p_path_z = [], [], []
    for t_val in theta_track:
        x, y, z = get_coords(r_phobos, t_val, 0.0, 0.019)
        p_path_x.append(x)
        p_path_y.append(y)
        p_path_z.append(z)
        
    d_path_x, d_path_y, d_path_z = [], [], []
    for t_val in theta_track:
        x, y, z = get_coords(r_deimos, t_val, 0.0, 0.010)
        d_path_x.append(x)
        d_path_y.append(y)
        d_path_z.append(z)

    frames = []
    num_frames = 40
    
    sat_envs = [MartianOrbitalEnv(orbit_altitude) for _ in range(num_satellites)]
    total_fuel_consumed = [0.0] * num_satellites

    for t in range(num_frames):
        s_x, s_y, s_z, s_hover = [], [], [], []
        s_color, s_size = [], []
        for i in range(num_satellites):
            env = sat_envs[i]
            state = env.get_state_index()
            
            action = agent.choose_action(state, force_inference=not training_mode)
            next_state, reward, done, fuel_spent = env.step(action)
            total_fuel_consumed[i] += fuel_spent
            
            if training_mode:
                agent.learn(state, action, reward, next_state, done)
            
            dynamic_r = R_mars + env.current_altitude
            ma = (2 * np.pi / num_satellites) * i + (omega * t)
            x, y, z = get_coords(dynamic_r, ma, sat_lan[i], sat_inc[i])
            
            s_x.append(x)
            s_y.append(y)
            s_z.append(z)
            
            err_val = env.current_altitude - env.target_altitude
            action_names = {
                0: "Coast", 1: "Small Up", 2: "Med Up", 3: "Large Up",
                4: "Small Down", 5: "Med Down", 6: "Large Down"
            }
            act_text = action_names.get(action, "Coast")
            
            s_hover.append(f"🛰️ Sat {i+1}<br>Action: {act_text}<br>Dev: {err_val:+.1f} km<br>Fuel: {env.fuel_remaining:.1f}%")
            if action == 0:
                s_color.append('gold')
                s_size.append(5)
            elif action in [1, 2, 3]:
                s_color.append('#33ff33')
                s_size.append(9)
            else:
                s_color.append('#ff3333')
                s_size.append(9)

        a_x, a_y, a_z = [], [], []
        for j in range(num_asteroids):
            aa = ast_phase[j] + (ast_omega[j] * t)
            x, y, z = get_coords(ast_r[j], aa, ast_lan[j], ast_inc[j])
            a_x.append(x)
            a_y.append(y)
            a_z.append(z)

        p_x, p_y, p_z = get_coords(r_phobos, omega_phobos * t, 0.0, 0.019)
        d_x, d_y, d_z = get_coords(r_deimos, omega_deimos * t, 0.0, 0.010)

        elapsed_mins = int((t * 250) // 60)
        elapsed_secs = int((t * 250) % 60)
        time_text = f"⏳ Elapsed: {elapsed_mins}m {elapsed_secs}s"

        if t == 0:
            fig.add_trace(
                go.Surface(
                    x=x_m, y=y_m, z=z_m,
                    surfacecolor=np.zeros_like(z_m), colorscale="Earth",
                    cmin=0, cmax=1, showscale=False, opacity=0.80, name="🪐 Mars",
                    showlegend=True,
                    lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1, roughness=0.9, fresnel=0.2),
                    lightposition=dict(x=100000, y=100000, z=100000)
                )
            )
            fig.add_trace(go.Scatter3d(x=a_x, y=a_y, z=a_z, mode='markers', marker=dict(size=2.5, color='darkgray'), name="Asteroids", hoverinfo='none'))
            fig.add_trace(go.Scatter3d(
                x=s_x, y=s_y, z=s_z, 
                mode='markers', 
                marker=dict(size=s_size, color=s_color, symbol='diamond'), 
                name="Satellites", 
                text=s_hover, 
                hoverinfo="text"
            ))
            fig.add_trace(go.Scatter3d(x=[p_x], y=[p_y], z=[p_z], mode='markers', marker=dict(size=9, color='#c1a48c', symbol='circle'), name="🌕 Phobos"))
            fig.add_trace(go.Scatter3d(x=[d_x], y=[d_y], z=[d_z], mode='markers', marker=dict(size=6, color='#e5e5e5', symbol='circle'), name="🌑 Deimos"))
            
            fig.add_trace(go.Scatter3d(x=sat_path_x, y=sat_path_y, z=sat_path_z, mode='lines', line=dict(color='rgb(250, 204, 21)', width=1), name="Satellite Tracks", hoverinfo='none'))
            fig.add_trace(go.Scatter3d(x=p_path_x, y=p_path_y, z=p_path_z, mode='lines', line=dict(color='rgb(193, 164, 140)', width=1.2), name="Phobos Track", hoverinfo='none'))
            fig.add_trace(go.Scatter3d(x=d_path_x, y=d_path_y, z=d_path_z, mode='lines', line=dict(color='rgb(229, 229, 229)', width=1.2), name="Deimos Track", hoverinfo='none'))
        else:
            frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=a_x, y=a_y, z=a_z),
                    go.Scatter3d(x=s_x, y=s_y, z=s_z, text=s_hover, marker=dict(color=s_color, size=s_size)),
                    go.Scatter3d(x=[p_x], y=[p_y], z=[p_z]), 
                    go.Scatter3d(x=[d_x], y=[d_y], z=[d_z])  
                ],
                traces=[1, 2, 3, 4], 
                name=f"frame_{t}",
                layout=dict(title=dict(text=time_text))
            ))

    fig.frames = frames
    fig.update_layout(
        margin=dict(l=10, r=10, b=10, t=10), height=550, 
        paper_bgcolor='black',  
        plot_bgcolor='black',   
        uirevision='constant_camera_state', 
        scene=dict(
            xaxis=dict(title='X (km)', backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False),
            yaxis=dict(title='Y (km)', backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False),
            zaxis=dict(title='Z (km)', backgroundcolor="rgba(0,0,0,0)", gridcolor="gray", showbackground=False),
            aspectmode='data', camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        updatemenus=[
            {
                "type": "buttons", "showactive": False, "x": 0.05, "y": 0.12, "xanchor": "left", "yanchor": "bottom",
                "pad": {"t": 10, "b": 10}, "font": {"color": "gold", "size": 13}, 
                "bgcolor": "#1e1e1e", "bordercolor": "gold", "borderwidth": 1,
                "buttons": [{
                    "label": "▶ Animate Real-World Dynamics", "method": "animate",
                    "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]
                }]
            },
            {
                "type": "buttons", 
                "direction": "left", 
                "showactive": True, 
                "x": 0.95, 
                "y": 0.98, 
                "xanchor": "right", 
                "yanchor": "top",
                "font": {"color": "#e5e7eb", "size": 11},  
                "bgcolor": "#374151",                      
                "bordercolor": "#4b5563",                  
                "borderwidth": 1,
                "buttons": [
                    {
                        "label": "🌐 Grid On",
                        "method": "relayout",
                        "args": [{
                            "scene.xaxis.showgrid": True,
                            "scene.yaxis.showgrid": True,
                            "scene.zaxis.showgrid": True
                        }]
                    },
                    {
                        "label": "🕳️ Grid Off",
                        "method": "relayout",
                        "args": [{
                            "scene.xaxis.showgrid": False,
                            "scene.yaxis.showgrid": False,
                            "scene.zaxis.showgrid": False
                        }]
                    }
                ]
            }
        ],
        title=dict(text="⏳ Elapsed: 0m 0s", x=0.05, y=0.04, xanchor='left', yanchor='top', font=dict(color="rgba(255, 255, 255, 0.55)", size=12)),
        legend=dict(x=0.05, y=0.95, xanchor='left', yanchor='top', font=dict(color="white"))
    )
    return fig.to_json(), v_orbit_kms, r_orbit, sat_lan, sat_inc, omega, total_fuel_consumed, num_frames

@app.route('/api/update', methods=['POST'])
def update_dashboard():
    data = request.json or {}
    num_satellites = int(data.get('num_satellites', 20))
    sat_mass = float(data.get('sat_mass', 200))
    orbit_altitude = float(data.get('orbit_altitude', 1000))
    frame_duration = int(data.get('frame_duration', 40))
    training_mode = bool(data.get('training_mode', False))
    debris_density = float(data.get('debris_density', 0.3))

    global last_trained_altitude

    if training_mode and (
        len(rl_training_logs["episode_reward"]) == 0
        or last_trained_altitude is None
        or abs(last_trained_altitude - orbit_altitude) > 1e-6
    ):
        last_trained_altitude = orbit_altitude
        training_thread = threading.Thread(
            target=run_background_training, 
            args=(orbit_altitude, 800)
        )
        training_thread.start()

    plot_json, current_velocity, r_orbit, sat_lan, sat_inc, omega, fuel_consumed, num_frames = build_animated_orbital_plot(
        num_satellites, orbit_altitude, frame_duration, debris_density=debris_density, training_mode=training_mode
    )

    collision_array = np.zeros(num_satellites, dtype=int)
    positions = []
    
    for i in range(num_satellites):
        ma = (2 * np.pi / num_satellites) * i + (omega * 10)
        x, y, z = get_coords(r_orbit, ma, sat_lan[i], sat_inc[i])
        positions.append(np.array([x, y, z]))
        
    proximity_tolerance = 10.0 
    for i in range(num_satellites):
        for j in range(i + 1, num_satellites):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < proximity_tolerance:
                collision_array[i] += 1
                collision_array[j] += 1

    uptime_percentage = np.round(np.random.uniform(94.0, 99.9, size=num_satellites), 1).tolist()
    
    telemetry_table = []
    for k in range(num_satellites):
        telemetry_table.append({
            "satellite": f"Satellite {k+1}",
            "collisions": int(collision_array[k]),
            "fuel": round(fuel_consumed[k] * (sat_mass / 500.0), 2), 
            "uptime": uptime_percentage[k]
        })

    step_stride = max(1, len(rl_training_logs["episode_reward"]) // num_frames)
    chart_rewards = rl_training_logs["episode_reward"][::step_stride][:num_frames]
    chart_max_q = rl_training_logs["max_q"][::step_stride][:num_frames]
    line_chart_data = list(zip(chart_rewards, chart_max_q))

    return jsonify({
        "plot_data": plot_json,
        "current_velocity": round(current_velocity, 3),
        "telemetry": telemetry_table,
        "line_chart": line_chart_data
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json or {}
    messages = data.get('messages', [])
    
    if GROQ_API_KEY == "replace_with_your_groq_api_key" or not GROQ_API_KEY.strip():
        return jsonify({
            "success": False,
            "error": "GROQ_API_KEY is not configured."
        })

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful aerospace assistant specialized in deep-space satellite routing, reinforcement learning, and Martian orbital mechanics."},
                *messages
            ],
            model="llama-3.3-70b-versatile",
        )
        assistant_message = response.choices[0].message.content
        return jsonify({"success": True, "content": assistant_message})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/framework')
def framework(): return send_from_directory('.', 'framework.html')
@app.route('/components')
def components(): return send_from_directory('.', 'components.html')
@app.route('/documentation')
def documentation(): return send_from_directory('.', 'documentation.html') 
@app.route('/')
def landing(): return send_from_directory('.', 'landing.html')
@app.route('/dashboard')
def dashboard(): return send_from_directory('.', 'index.html')

def open_browser():
    webbrowser.open_new(f"http://{HOST}:{PORT}/dashboard")

if __name__ == '__main__':
    if OPEN_BROWSER: Timer(1.5, open_browser).start()
    app.run(host=HOST, port=PORT, debug=DEBUG)