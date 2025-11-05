# app.py

import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import random

# Import functions from our logic file
from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

# Initialize the Dash app (which uses Flask as its server)
app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server  # Expose the Flask server for deployment

# --- Add MathJax configuration script ---
app.scripts.config.serve_locally = True
app.scripts.append_script({
    "external_url": [
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"
    ]
})

# --- AESTHETICS: Common style for all buttons ---
common_button_style = {
    'backgroundColor': '#007aff',
    'color': 'white',
    'border': 'none',
    'borderRadius': '8px',
    'padding': '10px 15px',
    'fontSize': '14px',
    'fontWeight': '500',
    'cursor': 'pointer',
    'transition': 'background-color 0.2s ease',
    'width': '100%'
}

# --- AESTHETICS: Common style for section headers ---
section_header_style = {
    'marginTop': '25px',
    'marginBottom': '10px',
    'borderBottom': '1px solid #444',
    'paddingBottom': '5px'
}

# --- App Layout ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'fontFamily': 'Inter', 'minHeight': '100vh'}, children=[
    
    # Hidden store for state management
    dcc.Store(id='current-state-store'),

    html.H1("Interactive Bloch Sphere", style={'textAlign': 'center', 'padding': '20px', 'fontWeight': '600'}),
    
    # --- AESTHETICS: Main content wrapper with responsive layout ---
    html.Div(style={
        'display': 'flex',
        'flexDirection': 'row',
        'flexWrap': 'wrap', # Allow wrapping on small screens
        'justifyContent': 'center',
        'gap': '30px',
        'padding': '0 20px'
    }, children=[
        
        # --- Left Side: The 3D Plot ---
        # AESTHETICS: Added wrapper for better alignment
        html.Div(
            dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0)),
            style={'flex': '1 1 600px', 'minWidth': '400px', 'maxWidth': '600px'}
        ),
        
        # --- Right Side: Controls ---
        # AESTHETICS: Updated panel style
        html.Div(style={
            'flex': '1 1 450px',
            'minWidth': '400px',
            'maxWidth': '500px',
            'padding': '20px',
            'border': '1px solid #333',
            'borderRadius': '12px',
            'backgroundColor': '#1c1c1e' # Slightly lighter dark shade
        }, children=[
            
            html.H3("State Controls", style=section_header_style),
            html.Label(html.B("Theta (θ) degrees")),
            dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=0, marks={i: str(i) for i in range(0, 181, 45)}),
            
            html.Div([
                html.Label(html.B("Phi (φ) degrees"), style={'marginTop': '20px', 'display': 'block'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                    html.Div(
                        dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=0, marks={i: str(i) for i in range(0, 361, 90)}),
                        style={'flex': '1'}
                    ),
                    dcc.Input(id='phi-input', type='number', placeholder='φ', min=0, max=360, step=1, value=0, style={'width': '70px', 'textAlign': 'center'})
                ])
            ]),
            
            html.H3("Quantum Gates", style=section_header_style),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'}, children=[
                html.Button('X Gate', id='gate-x', n_clicks=0, style=common_button_style),
                html.Button('Y Gate', id='gate-y', n_clicks=0, style=common_button_style),
                html.Button('Z Gate', id='gate-z', n_clicks=0, style=common_button_style),
                html.Button('H Gate', id='gate-h', n_clicks=0, style=common_button_style),
                html.Button('S Gate', id='gate-s', n_clicks=0, style=common_button_style),
                html.Button('T Gate', id='gate-t', n_clicks=0, style=common_button_style),
            ]),
            
            html.H3("Presets", style=section_header_style),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                html.Button('Reset to |0⟩', id='reset-button', n_clicks=0, style=common_button_style),
                html.Button('Set to |+⟩', id='plus-button', n_clicks=0, style=common_button_style),
                html.Button('Set to |-⟩', id='minus-button', n_clicks=0, style=common_button_style),
                html.Button('Random State', id='random-button', n_clicks=0, style=common_button_style),
            ]),
            
            html.H3("Live Readouts", style=section_header_style),
            html.Div(id='state-vector-readout', style={'fontSize': '1.1em', 'fontFamily': 'monospace', 'padding': '10px', 'backgroundColor': '#2c2c2e', 'borderRadius': '8px'}),
            
            # --- NEW: Probability Display Area ---
            html.Div(id='probability-display-area', style={'marginTop': '15px'}),
            
            # --- AI Explanation Area ---
            html.H3("AI Explanation", style=section_header_style),
            html.Div(style={'marginTop': '20px', 'textAlign': 'center'}, children=[
                 html.Button("Explain with AI", id="ai-explain-button", n_clicks=0, style=common_button_style),
            ]),
            html.Div(
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    children=html.Div(
                        id="ai-explanation-output",
                        # --- FIX 2: Added styles for overflow ---
                        style={
                            'maxHeight': '400px', 
                            'overflowY': 'auto', 
                            'textAlign': 'left',
                            'paddingRight': '10px' # Add padding for the scrollbar
                        }
                    )
                ),
                style={
                    'marginTop': '15px', 
                    'padding': '15px', 
                    'border': '1px solid #333', 
                    'borderRadius': '8px', 
                    'minHeight': '50px', 
                    'backgroundColor': '#2c2c2e',
                    'overflowWrap': 'break-word' # Ensure long words wrap
                }
            )
        ])
    ])
])

# --- Main Callback for Core Logic ---
# This callback now calculates EVERYTHING and stores it in dcc.Store
@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    Output('phi-input', 'value'),
    Output('current-state-store', 'data'), # <-- NEW: Output to state store
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value'),
    Input('phi-input', 'value'),
    Input('gate-x', 'n_clicks'), Input('gate-y', 'n_clicks'),
    Input('gate-z', 'n_clicks'), Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'), Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'), Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'), Input('random-button', 'n_clicks'),
)
def update_sphere_and_readouts(
    theta_deg, phi_from_slider, phi_from_input,
    n_x, n_y, n_z, n_h, n_s, n_t,
    n_reset, n_plus, n_minus, n_random
):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'
    
    if triggered_id == 'phi-input':
        if phi_from_input is None:
            new_phi = phi_from_slider
        else:
            new_phi = max(0, min(360, phi_from_input))
    else:
        new_phi = phi_from_slider

    new_theta = theta_deg
    gate_map = {'gate-x':'X', 'gate-y':'Y', 'gate-z':'Z', 'gate-h':'H', 'gate-s':'S', 'gate-t':'T'}

    if triggered_id in gate_map:
        new_theta, new_phi = apply_gate_to_state(theta_deg, new_phi, gate_map[triggered_id])
    elif triggered_id == 'reset-button': new_theta, new_phi = 0, 0
    elif triggered_id == 'plus-button': new_theta, new_phi = 90, 0
    elif triggered_id == 'minus-button': new_theta, new_phi = 90, 180
    elif triggered_id == 'random-button':
        new_theta = np.rad2deg(np.arccos(2 * random.random() - 1))
        new_phi = 360 * random.random()
    
    # --- NEW: All Calculations Happen Here ---
    updated_figure = create_figure_for_state(new_theta, new_phi)
    
    theta_rad, phi_rad = np.deg2rad(new_theta), np.deg2rad(new_phi)
    alpha = np.cos(theta_rad / 2)
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    
    # Use .real and .imag to handle potential floating point inaccuracies
    state_str = f"|ψ⟩ = {alpha.real:.2f}{alpha.imag:+.2f}j |0⟩ + ({beta.real:.2f}{beta.imag:+.2f}j) |1⟩"
    
    # Z-Basis Probs
    p_z_0 = (np.abs(alpha)**2)
    p_z_1 = (np.abs(beta)**2)
    
    # X-Basis Probs: |+⟩ = 1/sqrt(2)(|0⟩ + |1⟩), |−⟩ = 1/sqrt(2)(|0⟩ - |1⟩)
    p_x_plus = 0.5 * (np.abs(alpha + beta)**2)
    p_x_minus = 0.5 * (np.abs(alpha - beta)**2)
    
    # Y-Basis Probs: |+i⟩ = 1/sqrt(2)(|0⟩ + i|1⟩), |-i⟩ = 1/sqrt(2)(|0⟩ - i|1⟩)
    p_y_plus = 0.5 * (np.abs(alpha - 1j * beta)**2) 
    p_y_minus = 0.5 * (np.abs(alpha + 1j * beta)**2) 

    store_data = {
        'theta': new_theta,
        'phi': new_phi,
        'state_str': state_str,
        'prob_z': [p_z_0, p_z_1],
        'prob_x': [p_x_plus, p_x_minus],
        'prob_y': [p_y_plus, p_y_minus],
        'last_action': triggered_id
    }
    
    return updated_figure, new_theta, new_phi, new_phi, store_data


# --- NEW: Callback for Displaying Readouts ---
# This callback just reads from the store and formats the display.
@app.callback(
    Output('state-vector-readout', 'children'),
    Output('probability-display-area', 'children'),
    Input('current-state-store', 'data')
)
def update_readouts(data):
    if not data:
        # Default state on first load
        state_html = "|ψ⟩ = 1.00+0.00j |0⟩ + (0.00+0.00j) |1⟩"
        prob_cards = []
        for basis, states in [
            ('Z-Basis', [('|0⟩', 1.0), ('|1⟩', 0.0)]),
            ('X-Basis', [('|+⟩', 0.5), ('|−⟩', 0.5)]),
            ('Y-Basis', [('|+i⟩', 0.5), ('|−i⟩', 0.5)]),
        ]:
            prob_cards.append(
                html.Div([
                    html.H4(basis, style={'textAlign': 'center', 'margin': '0 0 10px 0', 'color': '#aaa'}),
                    html.Div([
                        html.Div(f"P({states[0][0]})", style={'fontWeight': '500'}),
                        html.Div(f"{states[0][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.1em'})
                    ], style={'textAlign': 'center'}),
                    html.Div([
                        html.Div(f"P({states[1][0]})", style={'fontWeight': '500'}),
                        html.Div(f"{states[1][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.1em'})
                    ], style={'textAlign': 'center', 'marginTop': '10px'}),
                ], style={
                    'flex': '1',
                    'minWidth': '100px',
                    'padding': '15px',
                    'backgroundColor': '#2c2c2e',
                    'borderRadius': '8px'
                })
            )
        prob_html = [
            html.B("Measurement Probabilities:"),
            html.Div(prob_cards, style={'display': 'flex', 'gap': '10px', 'marginTop': '10px', 'flexWrap': 'wrap'})
        ]
        return state_html, prob_html

    # This runs on every update after the first load
    state_html = data['state_str']
    
    # --- AESTHETICS: Build Probability Cards ---
    prob_cards = []
    for basis, states in [
        ('Z-Basis', [('|0⟩', data['prob_z'][0]), ('|1⟩', data['prob_z'][1])]),
        ('X-Basis', [('|+⟩', data['prob_x'][0]), ('|−⟩', data['prob_x'][1])]),
        ('Y-Basis', [('|+i⟩', data['prob_y'][0]), ('|−i⟩', data['prob_y'][1])]),
    ]:
        prob_cards.append(
            html.Div([
                html.H4(basis, style={'textAlign': 'center', 'margin': '0 0 10px 0', 'color': '#aaa'}),
                html.Div([
                    html.Div(f"P({states[0][0]})", style={'fontWeight': '500'}),
                    html.Div(f"{states[0][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.1em'})
                ], style={'textAlign': 'center'}),
                html.Div([
                    html.Div(f"P({states[1][0]})", style={'fontWeight': '500'}),
                    html.Div(f"{states[1][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.1em'})
                ], style={'textAlign': 'center', 'marginTop': '10px'}),
            ], style={
                'flex': '1',
                'minWidth': '100px',
                'padding': '15px',
                'backgroundColor': '#2c2c2e',
                'borderRadius': '8px'
            })
        )
        
    prob_html = [
        html.B("Measurement Probabilities:"),
        html.Div(prob_cards, style={'display': 'flex', 'gap': '10px', 'marginTop': '10px', 'flexWrap': 'wrap'})
    ]

    return state_html, prob_html


# --- Updated Callback for AI Explanation ---
# This callback now reads from the state store, which is much cleaner.
@app.callback(
    Output('ai-explanation-output', 'children'),
    Input('ai-explain-button', 'n_clicks'),
    State('current-state-store', 'data'), # <-- NEW: Reads from the state store
    prevent_initial_call=True
)
def update_ai_explanation(n_clicks, state_data):
    if not state_data:
        return "Please interact with the sphere first to generate a state."
    
    last_action = state_data.get('last_action', 'User requested explanation')
    if last_action == 'ai-explain-button':
        last_action = "User requested an explanation of the current state."

    explanation = get_ai_explanation(state_data, last_action)
    
    # --- FIX 1: Added mathjax=True ---
    return dcc.Markdown(explanation, dangerously_allow_html=True, link_target="_blank", mathjax=True)

if __name__ == '__main__':
    app.run(debug=True)