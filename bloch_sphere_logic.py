# bloch_sphere_logic.py

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

# ------------------- Constants & Styling -------------------
BACKGROUND_COLOR = "#111111" # Darker for better contrast
TEXT_COLOR = "#ffffff"
ACCENT_COLOR = "#007aff"
GRID_COLOR = "#444444" # Softer grid
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
INITIAL_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1))

# ------------------- Core Quantum & Coordinate Functions -------------------
def get_bloch_vector_coordinates(theta_rad, phi_rad):
    """Converts spherical coordinates to Cartesian for the Bloch vector."""
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return x, y, z

def state_to_bloch(state_vector):
    """Converts a state vector to Bloch sphere Cartesian coordinates."""
    rho = np.outer(state_vector, np.conj(state_vector))
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    x = np.trace(rho @ pauli_x).real
    y = np.trace(rho @ pauli_y).real
    z = np.trace(rho @ pauli_z).real
    return x, y, z

def apply_gate_to_state(theta_deg, phi_deg, gate_name):
    """Applies a quantum gate to a state defined by theta and phi."""
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)

    # Create the initial state vector
    current_state_vector = np.array([
        np.cos(theta_rad / 2),
        np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    ])

    # Get the unitary matrix for the gate
    gate_circuit = QuantumCircuit(1)
    gate_map = {
        'X': gate_circuit.x, 'Y': gate_circuit.y, 'Z': gate_circuit.z,
        'H': gate_circuit.h, 'S': gate_circuit.s, 'T': gate_circuit.t
    }
    if gate_name in gate_map:
        gate_map[gate_name](0)
        gate_operator = Operator(gate_circuit)
        # Apply the gate
        new_state_vector = gate_operator.data @ current_state_vector
    else:
        new_state_vector = current_state_vector

    # Convert the new state back to Bloch coordinates and then to angles
    x, y, z = state_to_bloch(new_state_vector)
    new_theta_rad = np.arccos(np.clip(z, -1, 1))
    new_phi_rad = np.arctan2(y, x)

    return np.rad2deg(new_theta_rad), np.rad2deg(new_phi_rad % (2 * np.pi))


# ------------------- Figure Creation & Update Functions -------------------
def create_initial_figure():
    """Creates the base Plotly figure for the Bloch Sphere."""
    fig = go.Figure()

    # Sphere surface
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, '#aaaaaa'], [1, '#dddddd']],
        opacity=0.15, showscale=False, name="sphere"
    ))

    # Grid lines
    for i in np.arange(-np.pi/2, np.pi/2, np.pi/6):
        t = np.linspace(0, 2*np.pi, 100)
        x_line, y_line = np.cos(t) * np.cos(i), np.sin(t) * np.cos(i)
        fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=np.sin(i) * np.ones(100), mode='lines', line=dict(color=GRID_COLOR, width=1)))
    for i in np.arange(0, 2*np.pi, np.pi/6):
        s = np.linspace(0, np.pi, 100)
        x_line, y_line = np.cos(i) * np.sin(s), np.sin(i) * np.sin(s)
        fig.add_trace(go.Scatter3d(x=x_line, y=y_line, z=np.cos(s), mode='lines', line=dict(color=GRID_COLOR, width=1)))


    # State vector arrow and cone (tip)
    fig.add_trace(go.Scatter3d(x=[0,0], y=[0,0], z=[0,1], mode='lines', line=dict(color="#ff3b30", width=8), name='arrow'))
    fig.add_trace(go.Cone(x=[0], y=[0], z=[1], u=[0], v=[0], w=[0.1], sizemode="absolute", sizeref=0.15, anchor="tip", colorscale=[[0, "#ff3b30"], [1, "#ff3b30"]], showscale=False, name='arrowhead'))

    # Axes
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

def update_figure_state(fig, theta_deg, phi_deg):
    """Updates the figure's arrow based on the new qubit state."""
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    x, y, z = get_bloch_vector_coordinates(theta_rad, phi_rad)

    # Create a new figure data object to avoid mutation issues with Dash
    new_fig = go.Figure(fig)

    # CORRECTED INDICES: Use 19 for the arrow line and 20 for the arrowhead cone
    new_fig.data[19].x, new_fig.data[19].y, new_fig.data[19].z = [0, x], [0, y], [0, z]
    new_fig.data[20].x, new_fig.data[20].y, new_fig.data[20].z = [x], [y], [z]
    new_fig.data[20].u, new_fig.data[20].v, new_fig.data[20].w = [x], [y], [z]

    return new_fig

