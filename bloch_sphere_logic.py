import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import requests
import json
import os
import hvac

# Constants & Styling
BACKGROUND_COLOR = "#111111"  
TEXT_COLOR = "#ffffff"        
GRID_COLOR = "#444444"        
FONT_FAMILY = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)                                         
INITIAL_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1))  

# Initialize Hashicorp's Vault
vault_url = 'http://localhost:8200'
vault_token = 'your_vault_token'

client = hvac.Client(url=vault_url, token=vault_token)

# Read API key from Vault
api_key = client.secrets.kv.v2.read_secret_version(
    path='your_secret_path'
)['data']['data']['GEMINI_API_KEY']

# Core Quantum & Coordinate Functions
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
    current_state_vector = np.array([
        np.cos(theta_rad / 2),                          
        np.exp(1j * phi_rad) * np.sin(theta_rad / 2)    
    ])

    gate_circuit = QuantumCircuit(1)
    gate_map = {
        'X': gate_circuit.x,
        'Y': gate_circuit.y,
        'Z': gate_circuit.z,
        'H': gate_circuit.h,
        'S': gate_circuit.s,
        'T': gate_circuit.t,
    }

    if gate_name in gate_map:
        gate_map[gate_name](0)                    
        gate_operator = Operator(gate_circuit)    
        new_state_vector = gate_operator.data @ current_state_vector
        new_phi_rad = np.angle(new_state_vector[1] / new_state_vector[0])
        new_theta_rad = 2 * np.arccos(np.abs(new_state_vector[0]))
        new_theta_deg = np.rad2deg(new_theta_rad)
        new_phi_deg = np.rad2deg(new_phi_rad)
        return new_theta_deg, new_phi_deg
    else:
        return theta_deg, phi_deg


def rotate_api_key():
    new_api_key = client.secrets.kv.v2.generate_random_bytes(
        num_bytes=32
    )['data']['decoded']
    client.secrets.kv.v2.create_or_update_secret(
        path='your_secret_path',
        secret=dict(GEMINI_API_KEY=new_api_key)
    )
    return new_api_key


def get_api_key():
    global api_key
    try:
        api_key = client.secrets.kv.v2.read_secret_version(
            path='your_secret_path'
        )['data']['data']['GEMINI_API_KEY']
    except:
        api_key = rotate_api_key()
    return api_key


def main():
    # Your code here
    pass


if __name__ == "__main__":
    main()