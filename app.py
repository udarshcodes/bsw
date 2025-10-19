# app.py

import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import random

# Import functions from our logic file
from bloch_sphere_logic import (
    create_initial_figure,
    update_figure_state,
    apply_gate_to_state
)

# Initialize the Dash app (which uses Flask as its server)
app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server  # Expose the Flask server for deployment

# --- App Layout ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'fontFamily': 'Inter'}, children=[
    html.H1("Interactive Bloch Sphere", style={'textAlign': 'center', 'padding': '20px'}),

    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'gap': '30px'}, children=[
        # Left Side: The 3D Plot
        dcc.Graph(id='bloch-sphere-graph', figure=create_initial_figure()),

        # Right Side: Controls
        html.Div(style={'width': '400px'}, children=[
            html.H3("State Controls"),
            html.Div(className='control-group', children=[
                html.Label(html.B("Theta (θ) degrees")),
                dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=0, marks={i: str(i) for i in range(0, 181, 45)}),
            ]),
            html.Div(className='control-group', style={'marginTop': '20px'}, children=[
                html.Label(html.B("Phi (φ) degrees")),
                dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=0, marks={i: str(i) for i in range(0, 361, 90)}),
            ]),

            html.H3("Quantum Gates", style={'marginTop': '30px'}),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'}, children=[
                html.Button('X Gate', id='gate-x', n_clicks=0),
                html.Button('Y Gate', id='gate-y', n_clicks=0),
                html.Button('Z Gate', id='gate-z', n_clicks=0),
                html.Button('H Gate', id='gate-h', n_clicks=0),
                html.Button('S Gate', id='gate-s', n_clicks=0),
                html.Button('T Gate', id='gate-t', n_clicks=0),
            ]),

            html.H3("Presets", style={'marginTop': '20px'}),
             html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                html.Button('Reset to |0⟩', id='reset-button', n_clicks=0),
                html.Button('Set to |+⟩', id='plus-button', n_clicks=0),
                html.Button('Set to |-⟩', id='minus-button', n_clicks=0),
                html.Button('Random State', id='random-button', n_clicks=0),
            ]),


            html.H3("Live Readouts", style={'marginTop': '30px'}),
            html.Div(id='state-vector-readout', style={'fontSize': '1.1em', 'fontFamily': 'monospace'}),
            html.Div(id='prob-readout', style={'marginTop': '10px'}),
        ])
    ])
])

# --- Callbacks for Interactivity ---
@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('state-vector-readout', 'children'),
    Output('prob-readout', 'children'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    # Inputs from all buttons and sliders
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value'),
    Input('gate-x', 'n_clicks'),
    Input('gate-y', 'n_clicks'),
    Input('gate-z', 'n_clicks'),
    Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'),
    Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'),
    Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'),
    Input('random-button', 'n_clicks'),
    # State of the figure to update it
    State('bloch-sphere-graph', 'figure')
)
def update_sphere_and_readouts(
    theta_deg, phi_deg,
    n_x, n_y, n_z, n_h, n_s, n_t,
    n_reset, n_plus, n_minus, n_random,
    current_figure):

    # Determine which input triggered the callback
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    new_theta, new_phi = theta_deg, phi_deg

    # --- Handle Gate and Preset Buttons ---
    gate_map = {
        'gate-x': 'X', 'gate-y': 'Y', 'gate-z': 'Z',
        'gate-h': 'H', 'gate-s': 'S', 'gate-t': 'T'
    }

    if triggered_id in gate_map:
        new_theta, new_phi = apply_gate_to_state(theta_deg, phi_deg, gate_map[triggered_id])
    elif triggered_id == 'reset-button':
        new_theta, new_phi = 0, 0
    elif triggered_id == 'plus-button':
        new_theta, new_phi = 90, 0
    elif triggered_id == 'minus-button':
        new_theta, new_phi = 90, 180
    elif triggered_id == 'random-button':
        new_theta = np.rad2deg(np.arccos(2 * random.random() - 1))
        new_phi = 360 * random.random()

    # --- Update Figure ---
    updated_figure = update_figure_state(current_figure, new_theta, new_phi)

    # --- Update Readouts ---
    theta_rad = np.deg2rad(new_theta)
    phi_rad = np.deg2rad(new_phi)
    alpha = np.cos(theta_rad / 2)
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    
    state_str = f"|ψ⟩ = {alpha:.2f} |0⟩ + ({beta.real:.2f}{beta.imag:+.2f}j) |1⟩"
    
    prob0 = alpha**2 * 100
    prob1 = abs(beta)**2 * 100
    prob_html = [
        html.B("Measurement Probabilities:"),
        html.Div(f"P(|0⟩): {prob0:.1f}%"),
        html.Div(f"P(|1⟩): {prob1:.1f}%"),
    ]

    return updated_figure, state_str, prob_html, new_theta, new_phi

# This is for running the app locally
if __name__ == '__main__':
    app.run_server(debug=True)