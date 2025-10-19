# bloch_sphere_logic.py

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# ------------------- Constants & Styling -------------------
BACKGROUND_COLOR = "#111111"
TEXT_COLOR = "#ffffff"
GRID_COLOR = "#444444"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
INITIAL_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1))

# ------------------- Core Quantum & Coordinate Functions (Unchanged) -------------------
def get_bloch_vector_coordinates(theta_rad, phi_rad):
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return x, y, z

def state_to_bloch(state_vector):
    rho = np.outer(state_vector, np.conj(state_vector))
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    x = np.trace(rho @ pauli_x).real
    y = np.trace(rho @ pauli_y).real
    z = np.trace(rho @ pauli_z).real
    return x, y, z

def apply_gate_to_state(theta_deg, phi_deg, gate_name):
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    current_state_vector = np.array([np.cos(theta_rad / 2), np.exp(1j * phi_rad) * np.sin(theta_rad / 2)])
    gate_circuit = QuantumCircuit(1)
    gate_map = {'X': gate_circuit.x, 'Y': gate_circuit.y, 'Z': gate_circuit.z, 'H': gate_circuit.h, 'S': gate_circuit.s, 'T': gate_circuit.t}
    if gate_name in gate_map:
        gate_map[gate_name](0)
        gate_operator = Operator(gate_circuit)
        new_state_vector = gate_operator.data @ current_state_vector
    else:
        new_state_vector = current_state_vector
    x, y, z = state_to_bloch(new_state_vector)
    new_theta_rad = np.arccos(np.clip(z, -1, 1))
    new_phi_rad = np.arctan2(y, x)
    return np.rad2deg(new_theta_rad), np.rad2deg(new_phi_rad % (2 * np.pi))

# ------------------- The New, All-in-One Figure Creation Function -------------------
def create_figure_for_state(theta_deg, phi_deg):
    """Creates a complete Plotly figure for a given qubit state."""
    fig = go.Figure()

    # --- Base Sphere and Grid ---
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x_sphere, y_sphere, z_sphere = np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v)
    fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, colorscale=[[0, '#aaaaaa'], [1, '#dddddd']], opacity=0.15, showscale=False))
    for i in np.arange(-np.pi/2, np.pi/2, np.pi/6):
        t = np.linspace(0, 2*np.pi, 100)
        x_line, y_line = np.cos(t) * np.cos(i), np.sin(t) * np.cos(i)
        fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=np.sin(i) * np.ones(100), mode='lines', line=dict(color=GRID_COLOR, width=1)))
    for i in np.arange(0, 2*np.pi, np.pi/6):
        s = np.linspace(0, np.pi, 100)
        x_line, y_line = np.cos(i) * np.sin(s), np.sin(i) * np.sin(s)
        fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=np.cos(s), mode='lines', line=dict(color=GRID_COLOR, width=1)))

    # --- State-Dependent Arrow ---
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    x, y, z = get_bloch_vector_coordinates(theta_rad, phi_rad)
    fig.add_trace(go.Scatter3d(x=[0,x], y=[0,y], z=[0,z], mode='lines', line=dict(color="#ff3b30", width=8), name='arrow'))
    fig.add_trace(go.Cone(x=[x], y=[y], z=[z], u=[x], v=[y], w=[z], sizemode="absolute", sizeref=0.15, anchor="tip", colorscale=[[0, "#ff3b30"], [1, "#ff3b30"]], showscale=False, name='arrowhead'))

    # --- Axes and Layout ---
    fig.add_trace(go.Scatter3d(x=[0, 1.2], y=[0,0], z=[0,0], mode='lines', line=dict(color='red', width=5), name='axis_x'))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0, 1.2], z=[0,0], mode='lines', line=dict(color='green', width=5), name='axis_y'))
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0, 1.2], mode='lines', line=dict(color='blue', width=5), name='axis_z'))

    fig.update_layout(
        width=600, height=600, showlegend=False,
        scene=dict(
            xaxis=dict(title='X', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            yaxis=dict(title='Y', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            zaxis=dict(title='', tickvals=[-1, 1], ticktext=['|1⟩', '|0⟩'], backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=INITIAL_CAMERA
        ),
        margin=dict(l=0, r=0, b=0, t=0), paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY)
    )
    return fig