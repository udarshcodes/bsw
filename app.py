# app.py

import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import random
import json # Import json, which is used later

# Import functions from our logic file
from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

# Initialize the Dash app (which uses Flask as its server)
# The assets/custom.css file will be loaded automatically by Dash
app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server  # Expose the Flask server for deployment

# --- NEW AESTHETICS: Apple-themed button style ---
common_button_style = {
    'backgroundColor': '#007AFF', # Apple's accent blue
    'color': 'white',
    'border': 'none',
    'borderRadius': '12px', # Softer corners
    'padding': '12px 18px', # More padding
    'fontSize': '15px', # Bigger text
    'fontWeight': '600', # Bolder
    'cursor': 'pointer',
    'transition': 'all 0.3s ease', # Changed for click animation
    'width': '100%',
    'outline': 'none' # Remove default outline
}

# --- NEW AESTHETICS: Apple-themed section headers ---
section_header_style = {
    'marginTop': '30px',
    'marginBottom': '15px',
    'borderBottom': '1px solid #333', # Softer border
    'paddingBottom': '10px',
    'fontSize': '1.3rem', # Bigger text
    'fontWeight': '600'
}

# --- GitHub Logo SVG Fix ---
github_logo_data_uri = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' width='18' height='18' fill='white'%3E%3Cpath d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z'%3E%3C/path%3E%3C/svg%3E"


# --- App Layout ---
app.layout = html.Div(style={
    'backgroundColor': '#1D1D1F', # Apple's dark grey
    'color': '#F5F5F7', # Apple's off-white
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif', 
    'minHeight': '100vh',
    'fontSize': '16px' # Larger base font
}, children=[
    
    # --- Hidden Store for State ---
    dcc.Store(id='current-state-store'),
    
    # --- The html.STYLE block has been REMOVED from here ---
    # The assets/custom.css file replaces it.

    # --- Header/Navbar ---
    html.Header(style={
        'backgroundColor': '#333333', # Apple's header bar
        'borderBottom': '1px solid #333',
        'padding': '15px 40px', # More padding
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center'
    }, children=[
        html.Div(
            "INTERACTIVE BLOCH SPHERE",
            style={'fontSize': '22px', 'fontWeight': '600', 'color': '#007AFF'} # Apple Blue
        ),
        html.A(
            children=[
                html.Img(src=github_logo_data_uri, style={'marginRight': '8px', 'verticalAlign': 'text-bottom'}),
                " GITHUB"
            ],
            href="https://github.com/udarshcodes/bsw", # --- UPDATED REPO LINK ---
            target="_blank", # Opens in a new tab
            style={
                'display': 'flex', 
                'alignItems': 'center',
                'fontSize': '14px',
                'fontWeight': '500',
                'color': 'white',
                'textDecoration': 'none',
                'padding': '8px 16px', 
                'borderRadius': '999px', # Pill shape
                'backgroundColor': '#333',
                'border': '1px solid #555',
                'transition': 'all 0.2s ease'
            }
        )
    ]),
    # --- END HEADER ---

    # --- Main Content Area ---
    html.Div(style={
        'display': 'flex',
        'flexDirection': 'row',
        'flexWrap': 'wrap',
        'justifyContent': 'center',
        'gap': '40px', 
        'padding': '40px 40px' 
    }, children=[
        
        # --- Left Side: The 3D Plot ---
        html.Div(
            dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0)),
            style={'flex': '1 1 700px', 'minWidth': '400px', 'maxWidth': '700px'}
        ),
        
        # --- Right Side: Controls ---
        html.Div(style={
            'flex': '1 1 500px',
            'minWidth': '400px',
            'maxWidth': '550px',
            'padding': '25px', 
            'border': '1px solid #333',
            'borderRadius': '18px', 
            'backgroundColor': '#2C2C2E', 
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.2)'
        }, children=[
            
            html.H2("State Controls", style={**section_header_style, 'marginTop': '0'}), 
            
            # --- NEW: Theta Input ---
            html.Label(html.B("Theta (θ) degrees")),
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                html.Div(
                    dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=0, marks={i: str(i) for i in range(0, 181, 45)}),
                    style={'flex': '1'}
                ),
                dcc.Input(id='theta-input', type='number', placeholder='θ', min=0, max=180, step=1, value=0, style={
                    'width': '70px', 'textAlign': 'center', 'background': '#333333', 
                    'color': 'white', 'border': '1px solid #555', 'borderRadius': '8px' 
                })
            ]),
            # --- END NEW ---
            
            html.Div([
                html.Label(html.B("Phi (φ) degrees"), style={'marginTop': '20px', 'display': 'block'}),
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                    html.Div(
                        dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=0, marks={i: str(i) for i in range(0, 361, 90)}),
                        style={'flex': '1'}
                    ),
                    dcc.Input(id='phi-input', type='number', placeholder='φ', min=0, max=360, step=1, value=0, style={
                        'width': '70px', 'textAlign': 'center', 'background': '#333333', 
                        'color': 'white', 'border': '1px solid #555', 'borderRadius': '8px'
                    })
                ])
            ]),
            
            html.H2("Quantum Gates", style=section_header_style), 
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'}, children=[
                html.Button('X Gate', id='gate-x', n_clicks=0, style=common_button_style),
                html.Button('Y Gate', id='gate-y', n_clicks=0, style=common_button_style),
                html.Button('Z Gate', id='gate-z', n_clicks=0, style=common_button_style),
                html.Button('H Gate', id='gate-h', n_clicks=0, style=common_button_style),
                html.Button('S Gate', id='gate-s', n_clicks=0, style=common_button_style),
                html.Button('T Gate', id='gate-t', n_clicks=0, style=common_button_style),
            ]),
            
            html.H2("Presets", style=section_header_style), 
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                html.Button('Reset to |0⟩', id='reset-button', n_clicks=0, style=common_button_style),
                html.Button('Set to |+⟩', id='plus-button', n_clicks=0, style=common_button_style),
                html.Button('Set to |-⟩', id='minus-button', n_clicks=0, style=common_button_style),
                html.Button('Random State', id='random-button', n_clicks=0, style=common_button_style),
            ]),
            
            html.H2("Live Readouts", style=section_header_style), 
            html.Div(id='state-vector-readout', style={
                'fontSize': '1.2em', 
                'fontFamily': 'monospace', 'padding': '15px', 
                'backgroundColor': '#333333', 'borderRadius': '12px', 'color': '#87CEEB'
            }),
            
            html.Div(id='probability-display-area', style={'marginTop': '20px'}),
        ])
    ]),
    
    # --- AI Explanation Section (Moved to Bottom) ---
    html.Div(style={
        'padding': '0 40px', # Match horizontal padding
        'maxWidth': '1200px', # Control max width
        'margin': '20px auto 0 auto' # Center the section
    }, children=[
        html.Div(style={
            'padding': '25px', 
            'border': '1px solid #333',
            'borderRadius': '18px', 
            'backgroundColor': '#2C2C2E',
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.2)'
        }, children=[
            html.H2("AI Explanation", style={**section_header_style, 'marginTop': '0'}),
            html.Button("Explain with AI", id="ai-explain-button", n_clicks=0, style={
                **common_button_style, 
                'backgroundColor': '#34C759', # Apple Green
                'fontWeight': '600',
                'maxWidth': '400px', # Give button a max width
                'margin': '0 auto', # Center button
                'display': 'block'
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
                            'paddingRight': '10px',
                            'marginTop': '20px',
                            'lineHeight': '1.6'
                        }
                    ),
                    color="#007AFF",
                    style={'marginTop': '15px'} # Kept for loading spinner spacing
                ),
                style={
                    'marginTop': '15px', 
                    'padding': '20px',
                    'border': '1px solid #333', 
                    'borderRadius': '12px', 
                    'minHeight': '50px', 
                    'backgroundColor': '#333333',
                    'overflowWrap': 'break-word',
                }
            )
        ])
    ]),
    # --- END AI SECTION ---

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
            'paddingBottom': '30px', 
            'paddingTop': '30px', 
            'borderTop': '1px solid #333',
            'color': '#007AFF',
            'fontSize': '18px' 
        }
    )
    # --- END FOOTER ---

]) # <-- This is the closing bracket for the main html.Div


# --- Main Callback for Core Logic ---
@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    Output('theta-input', 'value'), # --- NEW OUTPUT ---
    Output('phi-input', 'value'),
    Output('current-state-store', 'data'),
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value'),
    Input('theta-input', 'value'), # --- NEW INPUT ---
    Input('phi-input', 'value'),
    Input('gate-x', 'n_clicks'), Input('gate-y', 'n_clicks'),
    Input('gate-z', 'n_clicks'), Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'), Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'), Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'), Input('random-button', 'n_clicks'),
)
def update_sphere_and_readouts(
    theta_from_slider, phi_from_slider, # --- RENAMED ---
    theta_from_input, phi_from_input,  # --- NEW ARGS ---
    n_x, n_y, n_z, n_h, n_s, n_t,
    n_reset, n_plus, n_minus, n_random
):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'
    
    # --- NEW: Logic for syncing Theta input/slider ---
    if triggered_id == 'theta-input':
        if theta_from_input is None:
            new_theta = theta_from_slider
        else:
            new_theta = max(0, min(180, theta_from_input)) # Clamp 0-180
    else:
        new_theta = theta_from_slider
    # --- END NEW ---

    if triggered_id == 'phi-input':
        if phi_from_input is None:
            new_phi = phi_from_slider
        else:
            new_phi = max(0, min(360, phi_from_input))
    else:
        new_phi = phi_from_slider

    gate_map = {'gate-x':'X', 'gate-y':'Y', 'gate-z':'Z', 'gate-h':'H', 'gate-s':'S', 'gate-t':'T'}

    if triggered_id in gate_map:
        new_theta, new_phi = apply_gate_to_state(new_theta, new_phi, gate_map[triggered_id]) # Use new_theta
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
    
    return updated_figure, new_theta, new_phi, new_theta, new_phi, store_data


# --- Callback for Displaying Readouts ---
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
                    html.H4(basis, style={'textAlign': 'center', 'margin': '0 0 10px 0', 'color': '#aaa', 'fontWeight': '500'}),
                    html.Div([
                        html.Div(f"P({states[0][0]})", style={'fontWeight': '500', 'fontSize': '15px'}),
                        html.Div(f"{states[0][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.2em'})
                    ], style={'textAlign': 'center'}),
                    html.Div([
                        html.Div(f"P({states[1][0]})", style={'fontWeight': '500', 'fontSize': '15px'}),
                        html.Div(f"{states[1][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.2em'})
                    ], style={'textAlign': 'center', 'marginTop': '10px'}),
                ], style={
                    'flex': '1', 'minWidth': '110px', 'padding': '15px',
                    'backgroundColor': '#333333', 'borderRadius': '12px'
                })
            )
        prob_html = [
            html.B("Measurement Probabilities:", style={'fontSize': '1.1em'}),
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
                html.H4(basis, style={'textAlign': 'center', 'margin': '0 0 10px 0', 'color': '#aaa', 'fontWeight': '500'}),
                html.Div([
                    html.Div(f"P({states[0][0]})", style={'fontWeight': '500', 'fontSize': '15px'}),
                    html.Div(f"{states[0][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.2em'})
                ], style={'textAlign': 'center'}),
                html.Div([
                    html.Div(f"P({states[1][0]})", style={'fontWeight': '500', 'fontSize': '15px'}),
                    html.Div(f"{states[1][1]:.1%}", style={'fontWeight': 'bold', 'fontSize': '1.2em'})
                ], style={'textAlign': 'center', 'marginTop': '10px'}),
            ], style={
                'flex': '1', 'minWidth': '110px', 'padding': '15px',
                'backgroundColor': '#333333', 'borderRadius': '12px'
            })
        )
        
    prob_html = [
        html.B("Measurement Probabilities:", style={'fontSize': '1.1em'}),
        html.Div(prob_cards, style={'display': 'flex', 'gap': '10px', 'marginTop': '10px', 'flexWrap': 'wrap'})
    ]

    return state_html, prob_html


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
    
    return dcc.Markdown(explanation, link_target="_blank")


# --- Clientside Callback for Button Click Animation ---
app.clientside_callback(
    """
    function(n_x, n_y, n_z, n_h, n_s, n_t, n_reset, n_plus, n_minus, n_random) {
        // Get the button that was just clicked
        const triggered = dash_clientside.callback_context.triggered[0];
        if (!triggered) {
            return; // No button was triggered
        }
        
        const buttonId = triggered.prop_id.split('.')[0];
        const element = document.getElementById(buttonId);
        
        if (element) {
            // Add the 'clicked' class
            element.classList.add('button-clicked');
            
            // Remove the class after a short delay
            setTimeout(() => {
                element.classList.remove('button-clicked');
            }, 150); // 150ms matches the animation
        }
        return dash_clientside.no_update; // Don't update the output
    }
    """,
    Output('current-state-store', 'data', allow_duplicate=True), # Dummy output
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
    prevent_initial_call=True
)
# --- END NEW ---


if __name__ == '__main__':
    app.run(debug=True)