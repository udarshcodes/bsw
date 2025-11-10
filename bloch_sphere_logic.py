# --- Imports ---
import numpy as np                              # Numerical operations and linear algebra
import plotly.graph_objects as go               # 3D plotting for the Bloch sphere
from qiskit import QuantumCircuit              # To build single-qubit circuits for gates
from qiskit.quantum_info import Operator       # To extract unitary matrices from circuits
import requests                                 # HTTP client for calling external APIs
import json                                     # JSON encoding/decoding for API payloads
import os                                       # OS utilities (e.g., read API key from file)

# --- Constants & Styling ---
BACKGROUND_COLOR = "#111111"  # Dark background for figure canvas
TEXT_COLOR = "#ffffff"        # Default text color in the figure
GRID_COLOR = "#444444"        # Gridline color on the sphere
FONT_FAMILY = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)                                         # Font stack used by Plotly layout
INITIAL_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1))  # Initial 3D camera position

# --- Core Quantum & Coordinate Functions ---
def get_bloch_vector_coordinates(theta_rad, phi_rad):
    """Convert spherical angles (θ, φ) to Cartesian Bloch vector (x, y, z).

    Args:
        theta_rad (float): Polar angle θ in radians (0 at |0⟩, π at |1⟩).
        phi_rad (float): Azimuthal angle φ in radians (phase around Z-axis).

    Returns:
        tuple[float, float, float]: (x, y, z) coordinates on the unit sphere.
    """
    x = np.sin(theta_rad) * np.cos(phi_rad)  # x = sin(θ) cos(φ)
    y = np.sin(theta_rad) * np.sin(phi_rad)  # y = sin(θ) sin(φ)
    z = np.cos(theta_rad)                    # z = cos(θ)
    return x, y, z


def state_to_bloch(state_vector):
    """Map a pure state vector |ψ⟩ in C^2 to its Bloch vector (x, y, z).

    The Bloch vector components are expectation values of Pauli operators
    with respect to the density matrix ρ = |ψ⟩⟨ψ|.

    Args:
        state_vector (np.ndarray): Column vector [α, β] for a 1-qubit pure state.

    Returns:
        tuple[float, float, float]: Bloch coordinates (x, y, z).
    """
    rho = np.outer(state_vector, np.conj(state_vector))  # ρ = |ψ⟩⟨ψ|
    # Pauli matrices
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    # Expectation values: Tr(ρ σ_i)
    x = np.trace(rho @ pauli_x).real
    y = np.trace(rho @ pauli_y).real
    z = np.trace(rho @ pauli_z).real
    return x, y, z


def apply_gate_to_state(theta_deg, phi_deg, gate_name):
    """Apply a single-qubit gate to a state specified by (θ, φ) on the Bloch sphere.

    This constructs the current state vector from spherical angles, applies the
    requested gate via Qiskit's unitary, then converts the resulting state back
    into (θ, φ) for UI consumption.

    Args:
        theta_deg (float): Polar angle θ in degrees.
        phi_deg (float): Azimuthal angle φ in degrees.
        gate_name (str): One of {'X','Y','Z','H','S','T'}.

    Returns:
        tuple[float, float]: (new_theta_deg, new_phi_deg) after gate application.
    """
    # Build the current state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩
    theta_rad = np.deg2rad(theta_deg)
    phi_rad = np.deg2rad(phi_deg)
    current_state_vector = np.array([
        np.cos(theta_rad / 2),                     # α
        np.exp(1j * phi_rad) * np.sin(theta_rad / 2)  # β = e^{iφ} sin(θ/2)
    ])

    # Create a 1-qubit circuit and map gate names to builder methods
    gate_circuit = QuantumCircuit(1)
    gate_map = {
        'X': gate_circuit.x,
        'Y': gate_circuit.y,
        'Z': gate_circuit.z,
        'H': gate_circuit.h,
        'S': gate_circuit.s,
        'T': gate_circuit.t,
    }

    # If recognized gate, append it to the circuit and get its unitary; otherwise, no-op
    if gate_name in gate_map:
        gate_map[gate_name](0)                   # Apply gate to qubit 0
        gate_operator = Operator(gate_circuit)   # Extract unitary from the circuit
        new_state_vector = gate_operator.data @ current_state_vector  # |ψ'⟩ = U|ψ⟩
    else:
        new_state_vector = current_state_vector  # Unknown gate → keep state unchanged

    # Convert the new state back to spherical angles
    x, y, z = state_to_bloch(new_state_vector)   # Get Bloch components
    new_theta_rad = np.arccos(np.clip(z, -1, 1)) # θ = arccos(z), clip for numerical safety
    new_phi_rad = np.arctan2(y, x)               # φ = atan2(y, x)

    # Return degrees, with φ wrapped to [0, 360)
    return np.rad2deg(new_theta_rad), np.rad2deg(new_phi_rad % (2 * np.pi))


# --- Figure Creation Function ---
def create_figure_for_state(theta_deg, phi_deg):
    """Create a complete Plotly 3D figure of the Bloch sphere and current state.

    The figure includes the translucent sphere surface, latitude/longitude grid,
    XYZ axes, and an arrow + cone indicating the qubit state direction.

    Args:
        theta_deg (float): Polar angle θ in degrees.
        phi_deg (float): Azimuthal angle φ in degrees.

    Returns:
        go.Figure: Configured Plotly figure ready to render.
    """
    fig = go.Figure()  # Start with an empty figure

    # Parametric sphere surface (u ∈ [0, 2π), v ∈ [0, π])
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)

    # Add translucent sphere surface (no colorbar)
    fig.add_trace(
        go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, '#aaaaaa'], [1, '#dddddd']],  # Light gray scale
            opacity=0.15,
            showscale=False,
        )
    )

    # Latitude lines (constant z), sweep t around the circle for each elevation i
    for i in np.arange(-np.pi/2, np.pi/2, np.pi/6):
        t = np.linspace(0, 2*np.pi, 100)
        x_line = np.cos(t) * np.cos(i)
        y_line = np.sin(t) * np.cos(i)
        z_line = np.sin(i) * np.ones(100)
        fig.add_trace(
            go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color=GRID_COLOR, width=1)
            )
        )

    # Longitude lines (constant azimuth), sweep s from pole to pole for each azimuth i
    for i in np.arange(0, 2*np.pi, np.pi/6):
        s = np.linspace(0, np.pi, 100)
        x_line = np.cos(i) * np.sin(s)
        y_line = np.sin(i) * np.sin(s)
        z_line = np.cos(s)
        fig.add_trace(
            go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color=GRID_COLOR, width=1)
            )
        )

    # Compute the current state's Bloch vector from input angles
    theta_rad, phi_rad = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    x, y, z = get_bloch_vector_coordinates(theta_rad, phi_rad)

    # Arrow shaft from origin to (x, y, z)
    fig.add_trace(
        go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color="#ff3b30", width=8),  # Red arrow
            name='arrow'
        )
    )

    # Cone at the tip to form an arrowhead (aligned with vector direction)
    fig.add_trace(
        go.Cone(
            x=[x], y=[y], z=[z],
            u=[x], v=[y], w=[z],  # Direction
            sizemode="absolute", sizeref=0.15,  # Fixed-size cone
            anchor="tip",                        # Place cone tip at (x,y,z)
            colorscale=[[0, "#ff3b30"], [1, "#ff3b30"]],  # Solid red
            showscale=False,
            name='arrowhead'
        )
    )

    # X, Y, Z axes for reference (color-coded)
    fig.add_trace(go.Scatter3d(x=[0, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red',   width=5), name='axis_x'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 1.2], z=[0, 0], mode='lines', line=dict(color='green', width=5), name='axis_y'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.2], mode='lines', line=dict(color='blue',  width=5), name='axis_z'))

    # Layout: equal aspect ratio, dark theme, hidden tick labels on x/y
    fig.update_layout(
        width=600, height=600, showlegend=False,
        scene=dict(
            xaxis=dict(title='X', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            yaxis=dict(title='Y', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            zaxis=dict(title='', tickvals=[-1, 1], ticktext=['|1⟩', '|0⟩'], backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=INITIAL_CAMERA
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY)
    )
    return fig


def get_ai_explanation(state_data, last_action):
    """Generate a natural-language explanation for the current qubit state via Gemini.

    The function reads an API key from a secret file, validates it, builds a
    system/user prompt pair, calls the Gemini `generateContent` endpoint, and
    returns the model's text response. Several guardrails and friendly error
    messages are included to make failures clearer to users.

    Args:
        state_data (dict): Contains angles, formatted state string, and probabilities.
        last_action (str): A description of the user's last interaction.

    Returns:
        str: Markdown-formatted explanation text, or an error message.
    """
    api_key = ""  # Will be populated from secret file if present
    secret_path = '/etc/secrets/GEMINI_API_KEY'  # Render-style secret mount path

    # Attempt to read API key from the secret file if it exists
    if os.path.exists(secret_path):
        with open(secret_path, 'r') as f:
            api_key = f.read().strip()

    # If no key is configured, instruct the operator how to enable the feature
    if not api_key:
        return (
            "**AI Service Not Configured**\n\n"
            "The Gemini API key has not been configured on the server. "
            "Please set the `GEMINI_API_KEY` secret file in the deployment environment to enable this feature."
        )

    # Lightweight sanity check on key format (Gemini keys often begin with 'AIza')
    if not api_key.startswith("AIza"):
        return (
            "**Invalid API Key Format**\n\n"
            "The API key configured on the server does not appear to be in the correct format. A valid Gemini API key typically starts with `AIza`. "
            "Please generate a new key from Google AI Studio and ensure it is correctly placed in the `GEMINI_API_KEY` secret file on Render."
        )

    # REST endpoint for the chosen Gemini model variant
    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    )

    # System prompt: tone, scope, and strict formatting rules for plain-text rendering
    system_prompt = (
        "You are a quantum computing expert and an excellent educator. Your role is to explain the state of a qubit on the Bloch Sphere to a student. "
        "Be clear, concise, and use analogies where helpful. Start with a direct explanation of the current state and then connect it to the user's last action. "
        "Explain the concepts of superposition and probability in the context of the given state. "
        "**Crucially, explain the measurement probabilities in all three bases (Z, X, and Y) and how they relate to the state vector's position.** "
        "Do not greet the user. Get straight to the explanation. "
        "Structure your response in Markdown, using headings, bold text, and lists to improve readability.\n\n"
        "**CRITICAL FORMATTING RULES (No Exceptions):**\n"
        "1.  **NO LaTeX:** You MUST NOT use LaTeX, dollar signs ($), or any LaTeX-style syntax (like \\sqrt, \\frac, \\psi).\n"
        "2.  **USE UNICODE:** You MUST use plain Unicode characters for all symbols (e.g., θ, φ, ψ, |0⟩, |+⟩, |−⟩, |+i⟩, |−i⟩).\n"
        "3.  **FOR EXPONENTS:** You MUST use the caret symbol (^). Example: Write 'cos(θ/2)^2', NOT 'cos²(θ/2)' or 'cos$^2$(θ/2)'.\n"
        "4.  **FOR FRACTIONS:** You MUST use the slash symbol (/). Example: Write '1/sqrt(2)', NOT '1/\\sqrt{2}' or '$\\frac{1}{\\sqrt{2}}$'.\n"
        "5.  **FOR SQUARE ROOTS:** You MUST write 'sqrt(...)'. Example: '1/sqrt(2)'.\n\n"
        "This is not a suggestion. You must follow these formatting rules exactly, as the output is being rendered in a plain text environment that does not support LaTeX."
    )

    # Extract angles/state/probabilities from the state_data dictionary
    theta_deg = state_data.get('theta', 0)
    phi_deg = state_data.get('phi', 0)
    state_str = state_data.get('state_str', 'N/A')

    prob_z = state_data.get('prob_z', [0, 0])
    prob_x = state_data.get('prob_x', [0, 0])
    prob_y = state_data.get('prob_y', [0, 0])

    # Human-readable probability lines for the prompt
    prob_z_text = f"P(|0⟩): {prob_z[0]:.1%}, P(|1⟩): {prob_z[1]:.1%}"
    prob_x_text = f"P(|+⟩): {prob_x[0]:.1%}, P(|−⟩): {prob_x[1]:.1%}"
    prob_y_text = f"P(|+i⟩): {prob_y[0]:.1%}, P(|−i⟩): {prob_y[1]:.1%}"

    # User prompt ties the UI action to the physics and asks for a didactic explanation
    user_prompt = (
        f"The user performed the action: **'{last_action}'**.\n\n"
        f"This resulted in the following qubit state:\n"
        f"- **Spherical Coordinates:** Theta (θ) = {theta_deg:.2f} degrees, Phi (φ) = {phi_deg:.2f} degrees.\n"
        f"- **State Vector |ψ⟩:** {state_str}\n"
        f"- **Measurement Probabilities:**\n"
        f"  - **Z-Basis:** {prob_z_text}\n"
        f"  - **X-Basis:** {prob_x_text}\n"
        f"  - **Y-Basis:** {prob_y_text}\n\n"
        "Please provide a detailed, theoretical explanation of this result, focusing on how the state's position on the sphere determines all three sets of probabilities."
    )

    # Gemini JSON payload and headers
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    headers = {'Content-Type': 'application/json'}

    try:
        # Send request to Gemini with a modest timeout
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=25)
        response.raise_for_status()  # Raise if non-2xx HTTP
        result = response.json()     # Parse JSON body

        # Drill into candidates → content → parts → text per API schema
        candidate = result.get("candidates", [{}])[0]
        text_part = candidate.get("content", {}).get("parts", [{}])[0]
        explanation = text_part.get("text", "Error: Could not retrieve explanation from the AI model.")
        return explanation

    except requests.exceptions.RequestException as e:
        # Network/server/HTTP issues are surfaced here with a friendly message
        detailed_error = f"API Request Error: {e}"
        print(detailed_error)
        return (
            "**Error: Could not connect to the AI service.**\n\n"
            "This is often due to network restrictions on the free hosting plan that. "
            "Please also double-check that your Gemini API key is correctly configured as a secret file in your Render dashboard.\n\n"
            f"*Details: {e}*"
        )
    except Exception as e:
        # Any other unexpected exception
        print(f"An unexpected error occurred: {e}")
        return "**An unexpected error occurred while generating the explanation.**"
