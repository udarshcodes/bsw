"""
Main application entry point for the QuantumLens visualizer.

This module sets up a Dash web application that provides a real-time, 3D interactive 
visualization of a single-qubit quantum state. It coordinates user inputs (sliders, 
buttons, inputs) with the underlying quantum state computations and renders the resulting
Bloch sphere and probability distributions. 

The architecture delegates state logic and rendering to `bloch_sphere_logic.py` to maintain 
a clean separation of concerns between UI and business logic.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context
import numpy as np
import secrets

from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'], title="QuantumLens", update_title=None, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1"}])
server = app.server

section_header_style = {
    'marginTop': '30px',
    'marginBottom': '15px',
    'borderBottom': '1px solid var(--bg-border)',
    'paddingBottom': '10px',
    'fontSize': '1.3rem',
    'fontWeight': '600',
    'color': 'var(--white)'
}

app.layout = html.Div(style={
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    'minHeight': '100vh',
    'paddingBottom': '60px',
    'fontSize': '16px',
    'background': 'var(--bg)'
}, children=[
    dcc.Store(id='current-state-store'),
    
    # Landing Page Container (100vh)
    html.Div(style={
        'minHeight': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'justifyContent': 'center',
        'alignItems': 'center',
        'position': 'relative',
        'padding': '0 20px',
        'backgroundImage': 'url("/assets/hero_bg.png")',
        'backgroundSize': 'cover',
        'backgroundPosition': 'center',
        'borderBottom': '1px solid rgba(139, 47, 240, 0.2)'
    }, children=[
        # Dark overlay
        html.Div(style={
            'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
            'backgroundColor': 'rgba(5, 5, 7, 0.80)', 'zIndex': 0
        }),
        # Navbar overlay
        html.Nav(style={
            'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'zIndex': 2,
            'padding': '25px 50px', 'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'
        }, children=[
            html.Div("QuantumLens", style={'fontWeight': '800', 'fontSize': '1.3rem', 'letterSpacing': '-0.5px', 'color': 'var(--white)'})
        ]),
        
        # Hero Content
        html.Div([
            html.Img(src="/assets/logo.png", className="hero-logo", style={
                'display': 'block',
                'margin': '0 auto 20px auto', 
                'mixBlendMode': 'screen',
                'animation': 'float 6s ease-in-out infinite'
            }),
            html.H1("QuantumLens", className="gradient-text hero-title", style={'display': 'block', 'margin': '0 auto 15px auto', 'fontWeight': '800', 'letterSpacing': '-1.5px'}),
            html.P("A stunning, interactive 3D Bloch Sphere visualizer to explore single-qubit quantum states in real-time.", 
                   className="hero-subtitle",
                   style={'color': 'var(--white)', 'margin': '0 auto 40px auto', 'lineHeight': '1.6', 'fontWeight': '400', 'textShadow': '0 2px 10px rgba(0,0,0,0.8)'}),
            html.A("Start Exploring", href="#app-container", className="glass-button primary-gradient-bg hero-btn", 
                   style={'textDecoration': 'none', 'display': 'inline-block', 'fontWeight': '600', 'boxShadow': '0 10px 20px rgba(139, 47, 240, 0.3)'})
        ], style={'position': 'relative', 'zIndex': 1, 'textAlign': 'center', 'animation': 'fadeInUp 1s ease-out', 'padding': '0 15px'}),
        
        # Scroll Indicator
        html.A(
            href="#app-container",
            className="scroll-indicator",
            children=[
                "SCROLL BELOW",
                html.Div("↓", style={'fontSize': '18px', 'marginTop': '4px'})
            ]
        )
    ]),
    
    # Main App Container
    html.Div(id="app-container", style={'paddingTop': '80px', 'background': 'radial-gradient(circle at 50% -20%, rgba(139, 47, 240, 0.12), transparent 60%)'}, children=[
        
        # Main Application Area
        html.Div(className="app-wrapper", children=[
            
            # Left: Bloch Sphere Plot
            html.Div(
                className="bloch-container",
                children=[dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0), config={'displayModeBar': False})]
            ),
            
            # Right: Controls Panel
            html.Div(className="glass-panel controls-container", children=[
                
                html.H2("State Controls", style={**section_header_style, 'marginTop': '0'}),
                
                html.Label(html.B("Theta (θ) degrees"), style={'color': 'var(--text-muted)'}),
                html.Div(className="slider-input-group", children=[
                    html.Div(
                        dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=0, marks={i: str(i) for i in range(0, 181, 45)}),
                        style={'flex': '1'}
                    ),
                    dcc.Input(id='theta-input', type='number', placeholder='θ', min=0, max=180, step=1, value=0, className="num-input")
                ]),
                
                html.Div([
                    html.Label(html.B("Phi (φ) degrees"), style={'marginTop': '25px', 'display': 'block', 'color': 'var(--text-muted)'}),
                    html.Div(className="slider-input-group", children=[
                        html.Div(
                            dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=0, marks={i: str(i) for i in range(0, 361, 90)}),
                            style={'flex': '1'}
                        ),
                        dcc.Input(id='phi-input', type='number', placeholder='φ', min=0, max=360, step=1, value=0, className="num-input")
                    ])
                ]),
                
                html.H2("Quantum Gates", style=section_header_style),
                html.Div(className='quantum-gates-grid', children=[
                    html.Button('X Gate', id='gate-x', n_clicks=0, className="glass-button"),
                    html.Button('Y Gate', id='gate-y', n_clicks=0, className="glass-button"),
                    html.Button('Z Gate', id='gate-z', n_clicks=0, className="glass-button"),
                    html.Button('H Gate', id='gate-h', n_clicks=0, className="glass-button"),
                    html.Button('S Gate', id='gate-s', n_clicks=0, className="glass-button"),
                    html.Button('T Gate', id='gate-t', n_clicks=0, className="glass-button"),
                ]),
                
                html.H2("Presets", style=section_header_style),
                html.Div(className='presets-grid', children=[
                    html.Button('Reset to |0⟩', id='reset-button', n_clicks=0, className="glass-button"),
                    html.Button('Set to |+⟩', id='plus-button', n_clicks=0, className="glass-button"),
                    html.Button('Set to |-⟩', id='minus-button', n_clicks=0, className="glass-button"),
                    html.Button('Random State', id='random-button', n_clicks=0, className="glass-button"),
                ]),
                
                html.H2("Live Readouts", style=section_header_style),
                html.Div(id='state-vector-readout', className="readout-box", style={
                    'fontFamily': 'monospace',
                    'backgroundColor': 'rgba(0,0,0,0.3)', 'color': 'var(--cyan)',
                    'border': '1px solid var(--bg-border)'
                }),
                
                html.Div(id='probability-display-area', style={'marginTop': '25px'}),
            ])
        ]),
        
        # AI Explanation Section
        html.Div(className="ai-panel-wrapper", children=[
            html.Div(className="glass-panel ai-panel-inner", children=[
                html.H2("AI Insight Lens", style={**section_header_style, 'marginTop': '0'}),
                html.Button("Analyze State with AI", id="ai-explain-button", n_clicks=0, className="glass-button primary-gradient-bg", style={
                    'maxWidth': '400px',
                    'margin': '0 auto',
                    'display': 'block',
                    'border': 'none',
                    'padding': '16px 24px',
                    'fontSize': '16px'
                }),
                html.Div(
                    dcc.Loading(
                        id="loading-spinner",
                        type="default",
                        children=html.Div(
                            id="ai-explanation-output",
                            style={
                                'maxHeight': '500px',
                                'overflowY': 'auto', 
                                'textAlign': 'left',
                                'paddingRight': '10px',
                                'marginTop': '25px',
                                'lineHeight': '1.7',
                                'fontSize': '1.05rem',
                                'color': 'rgba(255,255,255,0.85)'
                            }
                        ),
                        color="var(--blue)",
                        style={'marginTop': '20px'}
                    ),
                    style={
                        'marginTop': '20px', 
                        'padding': '25px',
                        'border': '1px solid var(--bg-border)', 
                        'borderRadius': '16px', 
                        'minHeight': '50px', 
                        'backgroundColor': 'rgba(0,0,0,0.2)',
                        'overflowWrap': 'break-word',
                    }
                )
            ])
        ]),
    
        html.Footer(
            children=[
                html.Div("© 2026 Udarsh Goyal. All rights reserved."),
                html.Div(
                    html.A(
                        "About the Developer",
                        href="https://www.linkedin.com/in/udarsh-goyal-256095383/",
                        target="_blank",
                        style={
                            'color': 'var(--text-muted)', 
                            'textDecoration': 'underline',
                            'fontSize': '13px',
                            'marginTop': '8px',
                            'display': 'inline-block',
                            'opacity': '0.8'
                        }
                    )
                )
            ],
            style={
                'textAlign': 'center',
                'marginTop': '60px',
                'paddingTop': '30px',
                'borderTop': '1px solid var(--bg-border)',
                'color': 'var(--text-muted)',
                'fontSize': '14px',
                'fontWeight': '500'
            }
        )
    ])
])


@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    Output('theta-input', 'value'),
    Output('phi-input', 'value'),
    Output('current-state-store', 'data'),
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value'),
    Input('theta-input', 'value'),
    Input('phi-input', 'value'),
    Input('gate-x', 'n_clicks'), Input('gate-y', 'n_clicks'),
    Input('gate-z', 'n_clicks'), Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'), Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'), Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'), Input('random-button', 'n_clicks'),
)
def update_sphere_and_readouts(
    theta_from_slider: float, phi_from_slider: float, 
    theta_from_input: float, phi_from_input: float,  
    n_x: int, n_y: int, n_z: int, n_h: int, n_s: int, n_t: int,
    n_reset: int, n_plus: int, n_minus: int, n_random: int
) -> tuple:
    """
    Core state reconciliation and projection loop.
    
    Synchronizes UI input states (sliders vs numerical inputs) and applies discrete 
    quantum transformations before mapping the continuous amplitudes to observable probabilities.
    
    Args:
        theta_from_slider (float): Polar angle from slider input.
        phi_from_slider (float): Azimuthal angle from slider input.
        theta_from_input (float): Polar angle from exact numeric input.
        phi_from_input (float): Azimuthal angle from exact numeric input.
        n_x, n_y, n_z, n_h, n_s, n_t (int): Click counters for Pauli and phase gates.
        n_reset, n_plus, n_minus, n_random (int): Click counters for basis presets.
        
    Returns:
        tuple: Formatted as (figure, theta, phi, theta_input, phi_input, store_data) reflecting 
               the newly evaluated quantum state representation.
    """
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'
    
    if triggered_id == 'theta-input':
        new_theta = theta_from_slider if theta_from_input is None else max(0, min(180, theta_from_input))
    else:
        new_theta = theta_from_slider
    
    if triggered_id == 'phi-input':
        new_phi = phi_from_slider if phi_from_input is None else max(0, min(360, phi_from_input))
    else:
        new_phi = phi_from_slider

    gate_map = {'gate-x':'X', 'gate-y':'Y', 'gate-z':'Z', 'gate-h':'H', 'gate-s':'S', 'gate-t':'T'}

    # Process discrete operations. Applying a gate transitions the state vector deterministically.
    # The uniform random assignment samples cos(theta) uniformly on [-1, 1] to ensure an unbiased 
    # distribution over the spherical surface, avoiding coordinate singularity clustering at the poles.
    if triggered_id in gate_map:
        new_theta, new_phi = apply_gate_to_state(new_theta, new_phi, gate_map[triggered_id])
    elif triggered_id == 'reset-button': 
        new_theta, new_phi = 0, 0
    elif triggered_id == 'plus-button': 
        new_theta, new_phi = 90, 0
    elif triggered_id == 'minus-button': 
        new_theta, new_phi = 90, 180
    elif triggered_id == 'random-button':
        new_theta = np.rad2deg(np.arccos(2 * secrets.randbelow(1000000) / 1000000 - 1))
        new_phi = 360 * random.random()
    
    updated_figure = create_figure_for_state(new_theta, new_phi)
    
    theta_rad, phi_rad = np.deg2rad(new_theta), np.deg2rad(new_phi)
    
    # State parameterization utilizing standard convention:
    # |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩.
    # Phase factors only apply to |1⟩ component to factor out global phase.
    alpha = np.cos(theta_rad / 2)
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)
    
    state_str = f"|ψ⟩ = {alpha.real:.2f}{alpha.imag:+.2f}j |0⟩ + ({beta.real:.2f}{beta.imag:+.2f}j) |1⟩"
    
    # Measurement probabilities evaluated as Born rule projections (Tr(ρ Π)).
    # We resolve components against Pauli Z, X, and Y bases directly from pure state amplitudes.
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


@app.callback(
    Output('state-vector-readout', 'children'),
    Output('probability-display-area', 'children'),
    Input('current-state-store', 'data')
)
def update_readouts(data: dict) -> tuple:
    """
    Renders state probability matrices to the frontend layer.
    
    Args:
        data (dict): The serialized quantum state metrics, evaluated in the main callback loop.
                     
    Returns:
        tuple: Formatted HTML elements bridging numeric probabilities into UI cards.
    """
    def create_prob_card(basis_name, states):
        return html.Div([
            html.H4(basis_name, style={'textAlign': 'center', 'margin': '0 0 12px 0', 'color': 'var(--text-muted)', 'fontWeight': '600'}),
            html.Div([
                html.Div(f"P({states[0][0]})", style={'fontWeight': '500', 'fontSize': '14px', 'color': 'var(--white)'}),
                html.Div(f"{states[0][1]:.1%}", style={'fontWeight': '700', 'fontSize': '1.3em', 'color': 'var(--cyan)'})
            ], style={'textAlign': 'center'}),
            html.Div([
                html.Div(f"P({states[1][0]})", style={'fontWeight': '500', 'fontSize': '14px', 'color': 'var(--white)'}),
                html.Div(f"{states[1][1]:.1%}", style={'fontWeight': '700', 'fontSize': '1.3em', 'color': 'var(--cyan)'})
            ], style={'textAlign': 'center', 'marginTop': '12px'}),
        ], style={
            'flex': '1', 'minWidth': '110px', 'padding': '20px',
            'backgroundColor': 'rgba(0,0,0,0.2)', 'borderRadius': '16px',
            'border': '1px solid var(--bg-border)'
        })

    if not data:
        state_html = "|ψ⟩ = 1.00+0.00j |0⟩ + (0.00+0.00j) |1⟩"
        prob_cards = []
        for basis, states in [
            ('Z-Basis', [('|0⟩', 1.0), ('|1⟩', 0.0)]),
            ('X-Basis', [('|+⟩', 0.5), ('|−⟩', 0.5)]),
            ('Y-Basis', [('|+i⟩', 0.5), ('|−i⟩', 0.5)]),
        ]:
            prob_cards.append(create_prob_card(basis, states))
            
        prob_html = [
            html.B("Measurement Probabilities", style={'fontSize': '1.1em', 'color': 'var(--white)'}),
            html.Div(prob_cards, style={'display': 'flex', 'gap': '12px', 'marginTop': '15px', 'flexWrap': 'wrap'})
        ]
        return state_html, prob_html

    state_html = data['state_str']
    
    prob_cards = []
    for basis, states in [
        ('Z-Basis', [('|0⟩', data['prob_z'][0]), ('|1⟩', data['prob_z'][1])]),
        ('X-Basis', [('|+⟩', data['prob_x'][0]), ('|−⟩', data['prob_x'][1])]),
        ('Y-Basis', [('|+i⟩', data['prob_y'][0]), ('|−i⟩', data['prob_y'][1])]),
    ]:
        prob_cards.append(create_prob_card(basis, states))
        
    prob_html = [
        html.B("Measurement Probabilities", style={'fontSize': '1.1em', 'color': 'var(--white)'}),
        html.Div(prob_cards, style={'display': 'flex', 'gap': '12px', 'marginTop': '15px', 'flexWrap': 'wrap'})
    ]

    return state_html, prob_html


@app.callback(
    Output('ai-explanation-output', 'children'),
    Input('ai-explain-button', 'n_clicks'),
    State('current-state-store', 'data'),
    prevent_initial_call=True
)
def update_ai_explanation(n_clicks: int, state_data: dict):
    """
    Asynchronous hook to interface with LLM agent for pedagogical analysis.
    
    Args:
        n_clicks (int): Interaction counter, utilized to bypass initial render constraints.
        state_data (dict): The serialized quantum state mapping needed for LLM context generation.
        
    Returns:
        dcc.Markdown: Rendered output containing the dynamically generated explanation.
    """
    if not state_data:
        return dcc.Markdown("Please interact with the sphere first to generate a state.")
    
    last_action = state_data.get('last_action', 'User requested explanation')
    if last_action == 'ai-explain-button':
        last_action = "User requested an explanation of the current state."

    explanation = get_ai_explanation(state_data, last_action)
    
    return dcc.Markdown(explanation, link_target="_blank")


app.clientside_callback(
    """
    function(n_x, n_y, n_z, n_h, n_s, n_t, n_reset, n_plus, n_minus, n_random, ai_btn) {
        const triggered = dash_clientside.callback_context.triggered[0];
        if (!triggered) {
            return;
        }
        
        const buttonId = triggered.prop_id.split('.')[0];
        const element = document.getElementById(buttonId);
        
        if (element) {
            element.classList.add('button-clicked');
            
            setTimeout(() => {
                element.classList.remove('button-clicked');
            }, 150);
        }
        return dash_clientside.no_update;
    }
    """,
    Output('current-state-store', 'data', allow_duplicate=True),
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
    Input('ai-explain-button', 'n_clicks'),
    prevent_initial_call=True
)


if __name__ == '__main__':
    app.run(debug=True)
