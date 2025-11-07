# app.py

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import numpy as np
import random
import json

# Import functions from our logic file
from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

# Initialize the Dash app (which uses Flask as its server)
app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server  # Expose the Flask server for deployment

# --- NEW AESTHETICS: Common style for all buttons ---
common_button_style = {
    'backgroundColor': '#7B68EE', # MediumSlateBlue
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

# --- NEW AESTHETICS: Common style for section headers ---
section_header_style = {
    'marginTop': '25px',
    'marginBottom': '10px',
    'borderBottom': '1px solid #444',
    'paddingBottom': '5px'
}

# --- App Layout ---
app.layout = html.Div(style={
    'backgroundColor': '#121212', # Darker background
    'color': '#E0E0E0', # Lighter text
    'fontFamily': 'Inter', 
    'minHeight': '100vh'
}, children=[
    
    # --- Hidden Store for State ---
    dcc.Store(id='current-state-store'),

    # --- Header/Navbar ---
    html.Header(style={
        'backgroundColor': '#1E1E1E', # Dark panel color
        'borderBottom': '1px solid #333',
        'padding': '15px 30px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center'
    }, children=[
        html.Div(
            "INTERACTIVE BLOCH SPHERE",
            style={'fontSize': '20px', 'fontWeight': '600', 'color': '#7B68EE'} # Accent color
        ),
        html.A(
            "GITHUB",
            # --- IMPORTANT: REPLACE THIS WITH YOUR REPO LINK ---
            href="https://github.com/your-username/your-repo-name",
            target="_blank", # Opens in a new tab
            style={
                'fontSize': '14px',
                'fontWeight': '500',
                'color': 'white',
                'textDecoration': 'none',
                'padding': '8px 12px',
                'borderRadius': '6px',
                'border': '1px solid #444',
                'transition': 'background-color 0.2s ease'
            }
        )
    ]),
    # --- END HEADER ---

    # --- Main Content Area ---
    html.Div(style={
        'display': 'flex',
        'flexDirection': 'row',
        'flexWrap': 'wrap', # Allow wrapping on small screens
        'justifyContent': 'center',
        'gap': '30px',
        'padding': '30px 40px' # Added more padding
    }, children=[
        
        # --- Left Side: The 3D Plot ---
        html.Div(
            dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0)),
            # --- "ZOOM IN" -> Increased max width ---
            style={'flex': '1 1 700px', 'minWidth': '400px', 'maxWidth': '700px'}
        ),
        
        # --- Right Side: Controls ---
        html.Div(style={
            # --- "ZOOM IN" -> Increased max width ---
            'flex': '1 1 500px',
            'minWidth': '400px',
            'maxWidth': '550px',
            'padding': '20px',
            'border': '1px solid #333',
            'borderRadius': '12px',
            'backgroundColor': '#1E1E1E' # Dark panel color
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
                    dcc.Input(id='phi-input', type='number', placeholder='φ', min=0, max=360, step=1, value=0, style={
                        'width': '70px', 'textAlign': 'center', 'background': '#2a2a2a', 
                        'color': 'white', 'border': '1px solid #444', 'borderRadius': '4px'
                    })
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
            html.Div(id='state-vector-readout', style={
                'fontSize': '1.1em', 'fontFamily': 'monospace', 'padding': '10px', 
                'backgroundColor': '#2a2a2a', 'borderRadius': '8px', 'color': '#87CEEB' # Sky blue
            }),
            
            html.Div(id='probability-display-area', style={'marginTop': '15px'}),
            
            html.H3("AI Explanation", style=section_header_style),
            html.Button("Explain with AI", id="ai-explain-button", n_clicks=0, style={
                **common_button_style, 'backgroundColor': '#28A745' # Green AI button
            }),
            html.Div(
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    children=html.Div(
                        id="ai-explanation-output",
                        style={
                            'maxHeight': '400px', 
                            'overflowY': 'auto', 
                            'textAlign': 'left',
                            'paddingRight': '10px'
                        }
                    ),
                    color="#7B68EE", # Match accent
                    style={'marginTop': '15px'}
                ),
                style={
                    'marginTop': '15px', 
                    'padding': '15px', 
                    'border': '1px solid #333', 
                    'borderRadius': '8px', 
                    'minHeight': '50px', 
                    'backgroundColor': '#2a2a2a',
                    'overflowWrap': 'break-word'
                }
            )
        ])
    ]),

    # --- Footer ---
    html.Footer(
        children=[
            "Made with ",
            html.Span("❤️", style={'color': '#E31B23'}),
            " by VITIANs"
        ],
        style={
            'textAlign': 'center',
            'marginTop': '40px',
            'paddingBottom': '20px',
            'paddingTop': '20px',
            'borderTop': '1px solid #333',
            'color': '#888',
            'fontSize': '0.9em'
        }
    )
    # --- END FOOTER ---
])

# --- Main Callback for Core Logic ---
@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    Output('phi-input', 'value'),
    Output('current-state-store', 'data'),
    Output('state-vector-readout', 'children'),
    Output('probability-display-area', 'children'),
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
    
    updated_figure = create_figure_for_state(new_theta, new_phi)
    
    theta_rad, phi_rad = np.deg2rad(new_theta), np.deg2rad(new_phi)
    alpha = np.cos(theta_rad / 2)
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    
    state_str = f"|ψ⟩ = {alpha.real:.2f}{alpha.imag:+.2f}j |0⟩ + ({beta.real:.2f}{beta.imag:+.2f}j) |1⟩"
    
    p_z_0 = (np.abs(alpha)**2)
    p_z_1 = (np.abs(beta)**2)
    p_x_plus = 0.5 * (np.abs(alpha + beta)**2)
    p_x_minus = 0.5 * (np.abs(alpha - beta)**2)
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
    
    prob_cards = []
    for basis, states in [
        ('Z-Basis', [('|0⟩', store_data['prob_z'][0]), ('|1⟩', store_data['prob_z'][1])]),
        ('X-Basis', [('|+⟩', store_data['prob_x'][0]), ('|−⟩', store_data['prob_x'][1])]),
        ('Y-Basis', [('|+i⟩', store_data['prob_y'][0]), ('|−i⟩', store_data['prob_y'][1])]),
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
                'flex': '1', 'minWidth': '100px', 'padding': '15px',
                'backgroundColor': '#2a2a2a', 'borderRadius': '8px' # Updated card color
            })
        )
        
    prob_html = [
        html.B("Measurement Probabilities:"),
        html.Div(prob_cards, style={'display': 'flex', 'gap': '10px', 'marginTop': '10px', 'flexWrap': 'wrap'})
    ]

    return updated_figure, new_theta, new_phi, new_phi, store_data, state_str, prob_html


# --- Updated Callback for AI Explanation ---
@app.callback(
    Output('ai-explanation-output', 'children'),
    Input('ai-explain-button', 'n_clicks'),
    State('current-state-store', 'data'),
    prevent_initial_call=True
)
def update_ai_explanation(n_clicks, state_data):
    if not state_data:
        return dcc.Markdown("Please interact with the sphere first to generate a state.")
    
    last_action = state_data.get('last_action', 'User requested explanation')
    if last_action == 'ai-explain-button':
        last_action = "User requested an explanation of the current state."

    explanation = get_ai_explanation(state_data, last_action)
    
    # Remove mathjax=True. This allows the unicode symbols to render correctly.
    return dcc.Markdown(explanation, link_target="_blank")

if __name__ == '__main__':
    app.run(debug=True)