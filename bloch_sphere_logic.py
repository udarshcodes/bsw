"""
Quantum computation and visual representation module.

This module implements the core linear algebra operations for a single-qubit system.
It provides functions to convert spherical coordinate parametrizations into discrete 
complex state vectors, process operator matrix products via Qiskit's backend, and 
project representations into real Cartesian coordinates for 3D visualization using Plotly.
It also includes an LLM integration layer via the Groq API to auto-generate contextual 
explanations for state transformations.
"""

import numpy as np
import plotly.graph_objects as go
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import requests
import json
import os

BACKGROUND_COLOR = "rgba(0,0,0,0)"  # Transparent to let the app gradient shine through
TEXT_COLOR = "#F5F5F7"            # Apple style off-white
GRID_COLOR = "rgba(255,255,255,0.15)"  # Subtle glass grid
FONT_FAMILY = (
    "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)
INITIAL_CAMERA = dict(eye=dict(x=1.5, y=1.5, z=1))


def get_bloch_vector_coordinates(theta_rad: float, phi_rad: float) -> tuple[float, float, float]:
    """
    Project spherical probability coordinates onto the 3D unit sphere.

    Maps pure state angles (theta, phi) mapping to an isomorphic representation 
    in R3 space.

    Args:
        theta_rad (float): Polar angle θ in radians (Z-axis rotation metric).
        phi_rad (float): Azimuthal angle φ in radians (phase tracking around Z-axis).

    Returns:
        tuple[float, float, float]: Corresponding Cartesian coordinates (x, y, z).
    """
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)
    return x, y, z


def state_to_bloch(state_vector: np.ndarray) -> tuple[float, float, float]:
    """
    Evaluate the Bloch vector mapping via expectation values of Pauli matrices.

    Extracts Cartesian representations from the system's density matrix ρ by 
    tracing it with the Pauli group operators.

    Args:
        state_vector (np.ndarray): 1D complex probability amplitude column vector.

    Returns:
        tuple[float, float, float]: Equivalent point in the Bloch sphere coordinate space.
    """
    # Density matrix formulation is necessary to construct observables correctly.
    rho = np.outer(state_vector, np.conj(state_vector))
    
    pauli_x = np.array([[0, 1], [1, 0]])
    pauli_y = np.array([[0, -1j], [1j, 0]])
    pauli_z = np.array([[1, 0], [0, -1]])
    
    # Calculate expectation values to project pure state onto axes: Tr(ρ * σ).
    x = np.trace(rho @ pauli_x).real
    y = np.trace(rho @ pauli_y).real
    z = np.trace(rho @ pauli_z).real
    
    return x, y, z


pass


def create_figure_for_state(theta_deg: float, phi_deg: float) -> go.Figure:
    """
    Construct a visual rendering topology for the Bloch sphere interface.

    Synthesizes the topological mesh required to view the qubit state representation, 
    mapping the requested rotational vector into an interactive Plotly scene.

    Args:
        theta_deg (float): Evaluated polar boundary angle θ.
        phi_deg (float): Evaluated azimuthal boundary angle φ.

    Returns:
        go.Figure: Instantiated Plotly engine entity configured for UI display.
    """
    fig = go.Figure()

    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:100j]
    x_sphere = np.cos(u) * np.sin(v)
    y_sphere = np.sin(u) * np.sin(v)
    z_sphere = np.cos(v)

    fig.add_trace(
        go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, '#aaaaaa'], [1, '#dddddd']],
            opacity=0.15,
            showscale=False,
        )
    )

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

    theta_rad, phi_rad = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
    x, y, z = get_bloch_vector_coordinates(theta_rad, phi_rad)

    fig.add_trace(
        go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color="#ff3b30", width=8),
            name='arrow'
        )
    )

    fig.add_trace(
        go.Cone(
            x=[x], y=[y], z=[z],
            u=[x], v=[y], w=[z],
            sizemode="absolute", sizeref=0.15,
            anchor="tip",
            colorscale=[[0, "#ff3b30"], [1, "#ff3b30"]],
            showscale=False,
            name='arrowhead'
        )
    )

    fig.add_trace(go.Scatter3d(x=[0, 1.2], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red',   width=5), name='axis_x'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 1.2], z=[0, 0], mode='lines', line=dict(color='green', width=5), name='axis_y'))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1.2], mode='lines', line=dict(color='blue',  width=5), name='axis_z'))

    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.05],
        mode='text',
        text=['|0⟩'],
        textfont=dict(color=TEXT_COLOR, size=18, family=FONT_FAMILY),
        hoverinfo='none',
        showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-1.05],
        mode='text',
        text=['|1⟩'],
        textfont=dict(color=TEXT_COLOR, size=18, family=FONT_FAMILY),
        hoverinfo='none',
        showlegend=False
    ))
    
    fig.update_layout(
        width=600, height=600, showlegend=False,
        scene=dict(
            xaxis=dict(title='X', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            yaxis=dict(title='Y', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            zaxis=dict(title='', showticklabels=False, backgroundcolor=BACKGROUND_COLOR, gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, range=[-1.2, 1.2]),
            aspectratio=dict(x=1, y=1, z=1),
            camera=INITIAL_CAMERA
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR, family=FONT_FAMILY)
    )
    return fig


def get_ai_explanation(state_data: dict, last_action: str) -> str:
    """
    Interface with Groq API for programmatic pedagogical response generation.

    Coordinates REST connectivity with Groq's completions endpoint. Structures and enforces
    domain specific prompt generation against formatting guardrails before returning valid Markdown.

    Args:
        state_data (dict): Mapping containing the extracted vector representations and observable probabilities.
        last_action (str): Event ledger descriptor tracking the interaction trigger point.

    Returns:
        str: Response string parsed from the LLM or graceful fallback error message string.
    """
    api_key = os.environ.get('GROQ_API_KEY', "")
    secret_path = '/etc/secrets/GROQ_API_KEY'
    local_key_path = 'groq_key.txt'
    
    if not api_key and os.path.exists(secret_path):
        with open(secret_path, 'r') as f:
            api_key = f.read().strip()
            
    if not api_key and os.path.exists(local_key_path):
        with open(local_key_path, 'r') as f:
            api_key = f.read().strip()

    if not api_key:
        return (
            "**AI Service Not Configured**\n\n"
            "The Groq API key has not been configured. "
            "To enable this feature locally, create a file named `groq_key.txt` in your project directory "
            "and paste your API key inside it, or set the `GROQ_API_KEY` environment variable."
        )

    if not api_key.startswith("gsk_"):
        return (
            "**Invalid API Key Format**\n\n"
            "The API key configured on the server does not appear to be in the correct format. A valid Groq API key typically starts with `gsk_`. "
            "Please generate a new key from the Groq console and ensure it is correctly placed in the `GROQ_API_KEY` secret file on Render."
        )

    api_url = "https://api.groq.com/openai/v1/chat/completions"

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
        "5.  **FOR SQUARE ROOTS:** You MUST write 'sqrt(...)'. Example: '1/sqrt(2)'.\n"
        "6.  **FOR TABLES:** You MUST use clean, simple Markdown pipe tables. Do not add complex formatting inside them. Example:\n"
        "    | Outcome | Probability | Interpretation |\n"
        "    | :--- | :--- | :--- |\n"
        "    | P(|0⟩) | 50.0% | Explanation here. |\n"
        "    | P(|1⟩) | 50.0% | Explanation here. |\n\n"
        "This is not a suggestion. You must follow these formatting rules exactly, as the output is being rendered in a plain text environment that does not support LaTeX."
    )
    
    theta_deg = state_data.get('theta', 0)
    phi_deg = state_data.get('phi', 0)
    state_str = state_data.get('state_str', 'N/A')

    prob_z = state_data.get('prob_z', [0, 0])
    prob_x = state_data.get('prob_x', [0, 0])
    prob_y = state_data.get('prob_y', [0, 0])

    prob_z_text = f"P(|0⟩): {prob_z[0]:.1%}, P(|1⟩): {prob_z[1]:.1%}"
    prob_x_text = f"P(|+⟩): {prob_x[0]:.1%}, P(|−⟩): {prob_x[1]:.1%}"
    prob_y_text = f"P(|+i⟩): {prob_y[0]:.1%}, P(|−i⟩): {prob_y[1]:.1%}"

    user_prompt_summary = (
        f"The user performed the action: **'{last_action}'**.\n"
        f"The state is Theta={theta_deg:.2f}, Phi={phi_deg:.2f}. "
        f"State Vector: {state_str}\n"
        "Please provide a 2-sentence 'Quick Intuition' summary of what this state means physically."
    )

    payload_summary = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a quantum physics tutor. Be concise and use simple analogies."},
            {"role": "user", "content": user_prompt_summary}
        ],
        "max_tokens": 150
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    try:
        # Call 1: Fast 3.1 model for Quick Intuition
        res_summary = requests.post(api_url, headers=headers, data=json.dumps(payload_summary), timeout=15)
        res_summary.raise_for_status()
        summary_text = res_summary.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Call 2: Powerful 3.3 model for Deep Dive
        user_prompt_deep = (
            f"The user performed the action: **'{last_action}'**.\n\n"
            f"This resulted in the following qubit state:\n"
            f"- **Spherical Coordinates:** Theta (θ) = {theta_deg:.2f} degrees, Phi (φ) = {phi_deg:.2f} degrees.\n"
            f"- **State Vector |ψ⟩:** {state_str}\n"
            f"- **Measurement Probabilities:**\n"
            f"  - **Z-Basis:** {prob_z_text}\n"
            f"  - **X-Basis:** {prob_x_text}\n"
            f"  - **Y-Basis:** {prob_y_text}\n\n"
            f"Here is a quick intuition summary generated earlier: '{summary_text}'\n\n"
            "Now, please provide the detailed mathematical and theoretical breakdown, focusing on how the state's position determines all three sets of probabilities."
        )

        payload_deep = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_deep}
            ]
        }

        res_deep = requests.post(api_url, headers=headers, data=json.dumps(payload_deep), timeout=35)
        res_deep.raise_for_status()
        deep_text = res_deep.json().get("choices", [{}])[0].get("message", {}).get("content", "Error: Could not retrieve explanation from the AI model.")
        
        return f"### Quick Intuition\n{summary_text}\n\n### Mathematical Deep Dive\n{deep_text}"

    except requests.exceptions.RequestException as e:
        detailed_error = f"API Request Error: {e}"
        print(detailed_error)
        return (
            "**Error: Could not connect to the AI service.**\n\n"
            "This is often due to network restrictions on the free hosting plan that. "
            "Please also double-check that your Groq API key is correctly configured as a secret file in your Render dashboard.\n\n"
            f"*Details: {e}*"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return "**An unexpected error occurred while generating the explanation.**"
