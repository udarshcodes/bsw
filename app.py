# app.py

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import numpy as np
import random
import json # Make sure json is imported, it's used in the callback

# Import functions from our logic file
from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

# Initialize the Dash app (which uses Flask as its server)
app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server  # Expose the Flask server for deployment

# --- AESTHETICS: Common style for all buttons ---
common_button_style = {
    'background': '#333', 'border': '1px solid #555', 'color': 'white',
    'padding': '10px', 'borderRadius': '5px', 'cursor': 'pointer', 'width': '100%',
    'fontSize': '14px', 'textAlign': 'center'
}
# Style for when a button is hovered over
common_button_hover_style = {**common_button_style, 'background': '#444', 'borderColor': '#777'}


# --- App Layout ---
app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'fontFamily': 'Inter', 'minHeight': '100vh'}, children=[
    
    # --- Hidden Store for State ---
    dcc.Store(id='current-state-store'),

    # --- Header ---
    html.H1("Interactive Bloch Sphere", style={'textAlign': 'center', 'padding': '20px 0', 'margin': '0'}),

    # --- Main Content Area ---
    html.Div(style={
        'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap',
        'justifyContent': 'center', 'gap': '30px', 'padding': '0 20px'
    }, children=[
        
        # --- Left Column: Bloch Sphere ---
        html.Div(
            dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0)),
            style={'flex': '1', 'minWidth': '400px', 'maxWidth': '600px'}
        ),

        # --- Right Column: Controls & Data ---
        html.Div(style={'flex': '1', 'minWidth': '400px', 'maxWidth': '600px'}, children=[
            
            # --- Controls Section ---
            html.Div(style={'background': '#222', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #333'}, children=[
                html.H3("State Controls", style={'marginTop': '0', 'borderBottom': '1px solid #444', 'paddingBottom': '10px'}),
                
                html.Label(html.B("Theta (θ) degrees")),
                dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=0, marks={i: str(i) for i in range(0, 181, 45)}),
                
                html.Label(html.B("Phi (φ) degrees"), style={'marginTop': '20px', 'display': 'block'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                    html.Div(
                        dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=0, marks={i: str(i) for i in range(0, 361, 90)}),
                        style={'flex': '1'}
                    ),
                    dcc.Input(id='phi-input', type='number', placeholder='φ', min=0, max=360, step=1, value=0, style={'width': '70px', 'textAlign': 'center', 'background': '#333', 'color': 'white', 'border': '1px solid #555', 'borderRadius': '4px'})
                ]),

                html.H3("Quantum Gates", style={'marginTop': '30px', 'borderBottom': '1px solid #444', 'paddingBottom': '10px'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'}, children=[
                    html.Button('X Gate', id='gate-x', n_clicks=0, style=common_button_style),
                    html.Button('Y Gate', id='gate-y', n_clicks=0, style=common_button_style),
                    html.Button('Z Gate', id='gate-z', n_clicks=0, style=common_button_style),
                    html.Button('H Gate', id='gate-h', n_clicks=0, style=common_button_style),
                    html.Button('S Gate', id='gate-s', n_clicks=0, style=common_button_style),
                    html.Button('T Gate', id='gate-t', n_clicks=0, style=common_button_style),
                ]),
                
                html.H3("Presets", style={'marginTop': '30px', 'borderBottom': '1px solid #444', 'paddingBottom': '10px'}),
                html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                    html.Button('Reset to |0⟩', id='reset-button', n_clicks=0, style=common_button_style),
                    html.Button('Set to |+⟩', id='plus-button', n_clicks=0, style=common_button_style),
                    html.Button('Set to |-⟩', id='minus-button', n_clicks=0, style=common_button_style),
                    html.Button('Random State', id='random-button', n_clicks=0, style=common_button_style),
                ]),
            ]),

            # --- Data Readout Section ---
            html.Div(style={'background': '#222', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #333', 'marginTop': '20px'}, children=[
                html.H3("Live Data Readouts", style={'marginTop': '0', 'borderBottom': '1px solid #444', 'paddingBottom': '10px'}),
                html.Div(id='state-vector-readout', style={'fontSize': '1.2em', 'fontFamily': 'monospace', 'color': '#00BFFF', 'marginBottom': '15px'}),
                html.Div(id='prob-readout-container', children=[
                    # This will be populated by the callback
                ]),
            ]),

            # --- AI Explanation Area ---
            html.Div(style={'background': '#222', 'padding': '20px', 'borderRadius': '8px', 'border': '1px solid #333', 'marginTop': '20px'}, children=[
                html.H3("AI Explanation", style={'marginTop': '0', 'borderBottom': '1px solid #444', 'paddingBottom': '10px'}),
                html.Button("Explain with AI", id="ai-explain-button", n_clicks=0, style=common_button_style),
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    children=html.Div(id="ai-explanation-output"),
                    color="#00BFFF",
                    style={'marginTop': '15px'}
                ),
                html.Div(
                    id='ai-explanation-scroll-box',
                    style={
                        'marginTop': '15px', 'padding': '15px', 'border': '1px solid #444', 'borderRadius': '5px',
                        'minHeight': '100px', 'backgroundColor': '#1a1a1a',
                        'maxHeight': '300px', 'overflowY': 'auto', # Makes it scrollable
                        'whiteSpace': 'pre-wrap', 'wordBreak': 'break-word' # Controls text wrapping
                    },
                    children=["Click the button above to get an AI-powered explanation of the current state."]
                )
            ])
        ])
    ]),

    # --- NEW: Add the footer line here ---
    html.P(
        "Made with love by Vitians",
        style={
            'textAlign': 'center',
            'marginTop': '40px',
            'marginBottom': '20px',
            'color': '#888',
            'fontSize': '0.9em'
        }
    )
    # --- END NEW FOOTER ---

]) # <-- This is the closing bracket for the main html.Div

# --- Main Callback for Core Logic & UI Updates ---
@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    Output('phi-input', 'value'),
    Output('current-state-store', 'data'), # Store all computed data here
    Output('state-vector-readout', 'children'),
    Output('prob-readout-container', 'children'),
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value'),
    Input('phi-input', 'value'),
    Input('gate-x', 'n_clicks'), Input('gate-y', 'n_clicks'),
    Input('gate-z', 'n_clicks'), Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'), Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'), Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'), Input('random-button', 'n_clicks'),
)
def update_main_state(
    theta_deg, phi_from_slider, phi_from_input,
    n_x, n_y, n_z, n_h, n_s, n_t,
    n_reset, n_plus, n_minus, n_random
):
    ctx = callback_context
    triggered_id = 'initial_load'
    if ctx.triggered:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # --- 1. Determine new Theta and Phi ---
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

    # --- 2. Create Figure ---
    updated_figure = create_figure_for_state(new_theta, new_phi)

    # --- 3. Calculate State Vector & Probabilities ---
    theta_rad, phi_rad = np.deg2rad(new_theta), np.deg2rad(new_phi)
    
    # State Vector
    alpha = np.cos(theta_rad / 2)
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    state_str = f"|ψ⟩ = {alpha:.2f} |0⟩ + ({beta.real:+.2f}{beta.imag:+.2f}j) |1⟩"
    
    # Z-Basis Probs
    prob_z = [np.abs(alpha)**2, np.abs(beta)**2]
    
    # X-Basis Probs: |+⟩ = 1/√2(|0⟩ + |1⟩), |−⟩ = 1/√2(|0⟩ - |1⟩)
    prob_x = [
        np.abs(1/np.sqrt(2) * (alpha + beta))**2,
        np.abs(1/np.sqrt(2) * (alpha - beta))**2
    ]
    
    # Y-Basis Probs: |+i⟩ = 1/√2(|0⟩ + i|1⟩), |−i⟩ = 1/√2(|0⟩ - i|1⟩)
    prob_y = [
        np.abs(1/np.sqrt(2) * (alpha + 1j * beta))**2,
        np.abs(1/np.sqrt(2) * (alpha - 1j * beta))**2
    ]
    
    # --- 4. Store All Data ---
    state_data = {
        'theta': new_theta,
        'phi': new_phi,
        'state_str': state_str,
        'prob_z': prob_z,
        'prob_x': prob_x,
        'prob_y': prob_y,
        'last_action': triggered_id
    }

    # --- 5. Create Probability UI ---
    prob_layout = html.Div([
        html.Div("Z-Basis:", style={'fontWeight': 'bold'}),
        html.Div(f"P(|0⟩): {prob_z[0]:.1%}", style={'paddingLeft': '10px'}),
        html.Div(f"P(|1⟩): {prob_z[1]:.1%}", style={'paddingLeft': '10px'}),
        
        html.Div("X-Basis:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
        html.Div(f"P(|+⟩): {prob_x[0]:.1%}", style={'paddingLeft': '10px'}),
        html.Div(f"P(|−⟩): {prob_x[1]:.1%}", style={'paddingLeft': '10px'}),
        
        html.Div("Y-Basis:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
        html.Div(f"P(|+i⟩): {prob_y[0]:.1%}", style={'paddingLeft': '10px'}),
        html.Div(f"P(|−i⟩): {prob_y[1]:.1%}", style={'paddingLeft': '10px'}),
    ], style={'fontFamily': 'monospace', 'fontSize': '14px'})

    # --- 6. Return All Outputs ---
    return (
        updated_figure,
        new_theta,
        new_phi,
        new_phi,
        state_data, # Send to dcc.Store
        state_str,
        prob_layout
    )

# --- Callback for AI Explanation ---
@app.callback(
    Output('ai-explanation-scroll-box', 'children'), # Update the content of the scroll box
    Input('ai-explain-button', 'n_clicks'),
    State('current-state-store', 'data'), # Read from the state store
    prevent_initial_call=True
)
def update_ai_explanation(n_clicks, state_data):
    if not state_data:
        return "No state data available yet. Please interact with the controls first."

    last_action = state_data.get('last_action', 'unknown action')
    if last_action == 'ai-explain-button':
        last_action = "User requested an explanation of the current state."

    explanation = get_ai_explanation(state_data, last_action)
    
    # Return the explanation inside a Markdown component
    # This will render math formulas correctly
    return dcc.Markdown(explanation, link_target="_blank", mathjax=True, dangerously_allow_html=True)

if __name__ == '__main__':
    app.run(debug=True)