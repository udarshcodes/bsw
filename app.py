# Import core Dash library for building web apps
import dash  # Dash is a framework for building analytic web apps in Python
from dash import dcc, html, Input, Output, State, callback_context  # Common Dash components and callback primitives
import numpy as np  # Numerical computing (angles, complex numbers, etc.)
import random  # Random numbers (for random Bloch states)
import json  # JSON utilities (not directly used here but handy for debugging)

# Import functions that handle Bloch sphere plotting and quantum state logic
from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

# Initialize the Dash app; by default Dash uses a Flask server under the hood
# External stylesheet pulls Inter font for a modern UI look
app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server  # Expose the underlying Flask server object (useful for deployment platforms)

# Reusable button styling (kept in a dict to apply across many buttons)
common_button_style = {
    'backgroundColor': '#007AFF',  # iOS blue
    'color': 'white',  # White text
    'border': 'none',  # No border
    'borderRadius': '12px',  # Rounded corners
    'padding': '12px 18px',  # Comfortable padding
    'fontSize': '15px',  # Readable size
    'fontWeight': '600',  # Semi-bold
    'cursor': 'pointer',  # Pointer cursor on hover
    'transition': 'all 0.3s ease',  # Smooth hover/click transitions
    'width': '100%',  # Full width in grid cells
    'outline': 'none'  # Remove default focus outline (visual focus handled via CSS classes)
}

# Section header style used for panel headings
section_header_style = {
    'marginTop': '30px',  # Space before header
    'marginBottom': '15px',  # Space after header
    'borderBottom': '1px solid #333',  # Subtle divider line
    'paddingBottom': '10px',  # Breathing room below text
    'fontSize': '1.3rem',  # Larger font size
    'fontWeight': '600'  # Semi-bold weight
}

# Inline SVG for GitHub logo encoded as a data URI (so no external asset file is needed)
github_logo_data_uri = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' width='18' height='18' fill='white'%3E%3Cpath d='M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z'%3E%3C/path%3E%3C/svg%3E"


# Define the overall page layout tree for the app
app.layout = html.Div(style={
    'backgroundColor': '#1D1D1F',  # Dark background
    'color': '#F5F5F7',  # Light foreground text
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',  # Font stack
    'minHeight': '100vh',  # Full viewport height
    'fontSize': '16px'  # Base font size
}, children=[
    
    # Hidden store to hold the current quantum state and probabilities between callbacks
    dcc.Store(id='current-state-store'),
    
    # --- Top navigation / header bar ---
    html.Header(style={
        'backgroundColor': '#333333',  # Header background
        'borderBottom': '1px solid #333',  # Bottom border
        'padding': '15px 40px',  # Spacing
        'display': 'flex',  # Flex layout
        'justifyContent': 'space-between',  # Space items apart
        'alignItems': 'center'  # Vertically center contents
    }, children=[
        html.Div(
            "INTERACTIVE BLOCH SPHERE",  # App title text
            style={'fontSize': '22px', 'fontWeight': '600', 'color': '#007AFF'}  # Styled title
        ),
        html.A(
            children=[
                html.Img(src=github_logo_data_uri, style={'marginRight': '8px', 'verticalAlign': 'text-bottom'}),  # GitHub mark
                " GITHUB"  # Link label
            ],
            href="https://github.com/udarshcodes/bsw",  # Repo link
            target="_blank",  # Open in new tab
            style={
                'display': 'flex',  # Inline icon + text
                'alignItems': 'center',
                'fontSize': '14px',
                'fontWeight': '500',
                'color': 'white',
                'textDecoration': 'none',
                'padding': '8px 16px',  # Chip-like look
                'borderRadius': '999px',  # Pill shape
                'backgroundColor': '#333',
                'border': '1px solid #555',
                'transition': 'all 0.2s ease'  # Smooth hover
            }
        )
    ]),
    # END HEADER

    # --- Main content area (left: plot, right: controls) ---
    html.Div(style={
        'display': 'flex',  # Two columns
        'flexDirection': 'row',
        'flexWrap': 'wrap',  # Wrap on narrow screens
        'justifyContent': 'center',
        'gap': '40px',  # Space between columns
        'padding': '40px 40px'  # Outer padding
    }, children=[
        
        # Left: Bloch sphere 3D plot
        html.Div(
            dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0)),  # Initial figure at |0> (theta=0, phi=0)
            style={'flex': '1 1 700px', 'minWidth': '400px', 'maxWidth': '700px'}  # Responsive sizing
        ),
        
        # Right: Control panel
        html.Div(style={
            'flex': '1 1 500px',
            'minWidth': '400px',
            'maxWidth': '550px',
            'padding': '25px',  # Inner padding
            'border': '1px solid #333',  # Card border
            'borderRadius': '18px',  # Rounded corners
            'backgroundColor': '#2C2C2E',  # Card background
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.2)'  # Soft shadow
        }, children=[
            
            html.H2("State Controls", style={**section_header_style, 'marginTop': '0'}),  # Panel header
            
            # Theta control row: slider + numeric input
            html.Label(html.B("Theta (θ) degrees")),  # Label for theta
            html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                html.Div(
                    dcc.Slider(id='theta-slider', min=0, max=180, step=1, value=0, marks={i: str(i) for i in range(0, 181, 45)}),  # θ slider
                    style={'flex': '1'}  # Take remaining width
                ),
                dcc.Input(id='theta-input', type='number', placeholder='θ', min=0, max=180, step=1, value=0, style={
                    'width': '70px', 'textAlign': 'center', 'background': '#333333', 
                    'color': 'white', 'border': '1px solid #555', 'borderRadius': '8px' 
                })  # Numeric input for θ (kept in sync with slider)
            ]),
            
            # Phi control row: slider + numeric input
            html.Div([
                html.Label(html.B("Phi (φ) degrees"), style={'marginTop': '20px', 'display': 'block'}),  # Label for φ
                html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '15px'}, children=[
                    html.Div(
                        dcc.Slider(id='phi-slider', min=0, max=360, step=1, value=0, marks={i: str(i) for i in range(0, 361, 90)}),  # φ slider
                        style={'flex': '1'}
                    ),
                    dcc.Input(id='phi-input', type='number', placeholder='φ', min=0, max=360, step=1, value=0, style={
                        'width': '70px', 'textAlign': 'center', 'background': '#333333', 
                        'color': 'white', 'border': '1px solid #555', 'borderRadius': '8px'
                    })  # Numeric input for φ
                ])
            ]),
            
            # Quantum gate buttons grid
            html.H2("Quantum Gates", style=section_header_style),  # Section header
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'}, children=[
                html.Button('X Gate', id='gate-x', n_clicks=0, style=common_button_style),  # Pauli-X
                html.Button('Y Gate', id='gate-y', n_clicks=0, style=common_button_style),  # Pauli-Y
                html.Button('Z Gate', id='gate-z', n_clicks=0, style=common_button_style),  # Pauli-Z
                html.Button('H Gate', id='gate-h', n_clicks=0, style=common_button_style),  # Hadamard
                html.Button('S Gate', id='gate-s', n_clicks=0, style=common_button_style),  # Phase (S)
                html.Button('T Gate', id='gate-t', n_clicks=0, style=common_button_style),  # T (π/8) gate
            ]),
            
            # Preset state buttons
            html.H2("Presets", style=section_header_style),  # Section header
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                html.Button('Reset to |0⟩', id='reset-button', n_clicks=0, style=common_button_style),  # |0> state
                html.Button('Set to |+⟩', id='plus-button', n_clicks=0, style=common_button_style),  # |+> state
                html.Button('Set to |-⟩', id='minus-button', n_clicks=0, style=common_button_style),  # |-> state
                html.Button('Random State', id='random-button', n_clicks=0, style=common_button_style),  # Random Bloch state
            ]),
            
            # Live readout of state vector and probabilities
            html.H2("Live Readouts", style=section_header_style),  # Section header
            html.Div(id='state-vector-readout', style={
                'fontSize': '1.2em',  # Slightly larger text
                'fontFamily': 'monospace', 'padding': '15px',  # Code-like font
                'backgroundColor': '#333333', 'borderRadius': '12px', 'color': '#87CEEB'  # Styled card
            }),
            
            html.Div(id='probability-display-area', style={'marginTop': '20px'}),  # Container for probability cards
        ])
    ]),
    
    # --- AI explanation area (uses LLM to describe current state) ---
    html.Div(style={
        'padding': '0 40px',  # Horizontal padding
        'maxWidth': '1200px',  # Max content width
        'margin': '20px auto 0 auto'  # Center the container
    }, children=[
        html.Div(style={
            'padding': '25px',  # Card padding
            'border': '1px solid #333',  # Border
            'borderRadius': '18px',  # Rounded corners
            'backgroundColor': '#2C2C2E',  # Card background
            'boxShadow': '0 8px 32px rgba(0, 0, 0, 0.2)'  # Shadow
        }, children=[
            html.H2("AI Explanation", style={**section_header_style, 'marginTop': '0'}),  # Section header
            html.Button("Explain with AI", id="ai-explain-button", n_clicks=0, style={
                **common_button_style,  # Base style
                'backgroundColor': '#34C759',  # Green call-to-action
                'fontWeight': '600',
                'maxWidth': '400px',  # Centered button width
                'margin': '0 auto',  # Center horizontally
                'display': 'block'  # Block-level for centering
            }),
            html.Div(
                dcc.Loading(  # Spinner while explanation is generated
                    id="loading-spinner",
                    type="default",
                    children=html.Div(
                        id="ai-explanation-output",  # Placeholder for AI markdown
                        style={
                            'maxHeight': '400px',  # Scroll if too long
                            'overflowY': 'auto', 
                            'textAlign': 'left',
                            'paddingRight': '10px',
                            'marginTop': '20px',
                            'lineHeight': '1.6'
                        }
                    ),
                    color="#007AFF",  # Spinner color
                    style={'marginTop': '15px'}  # Spacing above spinner
                ),
                style={
                    'marginTop': '15px', 
                    'padding': '20px',
                    'border': '1px solid #333', 
                    'borderRadius': '12px', 
                    'minHeight': '50px', 
                    'backgroundColor': '#333333',
                    'overflowWrap': 'break-word',  # Wrap long tokens
                }
            )
        ])
    ]),
    # END AI SECTION

    # --- Footer with credit ---
    html.Footer(
        children=[
            "Made with ",  # Text prefix
            html.Span("❤️", style={'color': '#E31B23'}),  # Heart icon
            " by VITIANs"  # Attribution
        ],
        style={
            'textAlign': 'center',  # Center text
            'marginTop': '40px',
            'paddingBottom': '30px',  # Bottom padding
            'paddingTop': '30px',  # Top padding
            'borderTop': '1px solid #333',  # Divider
            'color': '#007AFF',  # Accent color
            'fontSize': '18px'  # Slightly larger text
        }
    )
    

])


# Primary callback: computes new state when controls/gates/presets change and returns UI updates
@app.callback(
    Output('bloch-sphere-graph', 'figure'),  # Updated 3D figure
    Output('theta-slider', 'value'),         # Sync θ slider
    Output('phi-slider', 'value'),           # Sync φ slider
    Output('theta-input', 'value'),          # Sync θ number input
    Output('phi-input', 'value'),            # Sync φ number input
    Output('current-state-store', 'data'),   # Persist computed state + probabilities
    Input('theta-slider', 'value'),          # θ slider input
    Input('phi-slider', 'value'),            # φ slider input
    Input('theta-input', 'value'),           # θ input box
    Input('phi-input', 'value'),             # φ input box
    Input('gate-x', 'n_clicks'), Input('gate-y', 'n_clicks'),  # Gate clicks
    Input('gate-z', 'n_clicks'), Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'), Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'), Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'), Input('random-button', 'n_clicks'),
)
def update_sphere_and_readouts(
    theta_from_slider, phi_from_slider, 
    theta_from_input, phi_from_input,  
    n_x, n_y, n_z, n_h, n_s, n_t,
    n_reset, n_plus, n_minus, n_random
):
    ctx = callback_context  # Access info about what triggered this callback
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'initial_load'  # Element id
    
    # Reconcile θ: if the numeric input was the trigger, clamp and use it; otherwise keep slider value
    if triggered_id == 'theta-input':
        if theta_from_input is None:
            new_theta = theta_from_slider  # Fallback to slider if input is empty
        else:
            new_theta = max(0, min(180, theta_from_input))  # Clamp to [0, 180]
    else:
        new_theta = theta_from_slider  # No change; use slider
    
    # Reconcile φ similarly
    if triggered_id == 'phi-input':
        if phi_from_input is None:
            new_phi = phi_from_slider  # Fallback to slider
        else:
            new_phi = max(0, min(360, phi_from_input))  # Clamp to [0, 360]
    else:
        new_phi = phi_from_slider  # Keep slider value

    # Map button ids to gate labels understood by apply_gate_to_state
    gate_map = {'gate-x':'X', 'gate-y':'Y', 'gate-z':'Z', 'gate-h':'H', 'gate-s':'S', 'gate-t':'T'}

    # If a gate button or preset was clicked, update (θ, φ) accordingly
    if triggered_id in gate_map:
        new_theta, new_phi = apply_gate_to_state(new_theta, new_phi, gate_map[triggered_id])  # Apply gate
    elif triggered_id == 'reset-button': new_theta, new_phi = 0, 0  # |0⟩
    elif triggered_id == 'plus-button': new_theta, new_phi = 90, 0  # |+⟩
    elif triggered_id == 'minus-button': new_theta, new_phi = 90, 180  # |-⟩
    elif triggered_id == 'random-button':
        new_theta = np.rad2deg(np.arccos(2 * random.random() - 1))  # Uniform on sphere for cosθ
        new_phi = 360 * random.random()  # Uniform φ in [0, 360)
    
    updated_figure = create_figure_for_state(new_theta, new_phi)  # Redraw Bloch sphere with new state
    
    # Convert angles to radians for amplitude calculations
    theta_rad, phi_rad = np.deg2rad(new_theta), np.deg2rad(new_phi)
    alpha = np.cos(theta_rad / 2)  # Amplitude for |0⟩
    beta = np.exp(1j * phi_rad) * np.sin(theta_rad / 2)  # Amplitude for |1⟩ with phase φ
    
    # Pretty-printed state string (complex components shown as a+bi)
    state_str = f"|ψ⟩ = {alpha.real:.2f}{alpha.imag:+.2f}j |0⟩ + ({beta.real:.2f}{beta.imag:+.2f}j) |1⟩"
    
    # Basis measurement probabilities from amplitudes
    p_z_0 = (np.abs(alpha)**2)  # P(|0⟩) in Z-basis
    p_z_1 = (np.abs(beta)**2)   # P(|1⟩) in Z-basis
    p_x_plus = 0.5 * (np.abs(alpha + beta)**2)  # P(|+⟩) in X-basis
    p_x_minus = 0.5 * (np.abs(alpha - beta)**2)  # P(|−⟩) in X-basis
    p_y_plus = 0.5 * (np.abs(alpha - 1j * beta)**2)  # P(|+i⟩) in Y-basis
    p_y_minus = 0.5 * (np.abs(alpha + 1j * beta)**2)  # P(|−i⟩) in Y-basis

    # Pack all computed info into the dcc.Store for use by other callbacks
    store_data = {
        'theta': new_theta,
        'phi': new_phi,
        'state_str': state_str,
        'prob_z': [p_z_0, p_z_1],
        'prob_x': [p_x_plus, p_x_minus],
        'prob_y': [p_y_plus, p_y_minus],
        'last_action': triggered_id  # So AI can mention what changed
    }
    
    # Return updates to all targets (order must match Outputs)
    return updated_figure, new_theta, new_phi, new_theta, new_phi, store_data


# Secondary callback: renders the text readouts from the stored state
@app.callback(
    Output('state-vector-readout', 'children'),       # Human-readable state vector
    Output('probability-display-area', 'children'),   # Cards showing probabilities
    Input('current-state-store', 'data')              # Trigger when store changes
)
def update_readouts(data):
    if not data:
        # Default contents on first load (|0⟩ state)
        state_html = "|ψ⟩ = 1.00+0.00j |0⟩ + (0.00+0.00j) |1⟩"
        prob_cards = []  # Will hold three basis cards
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
        return state_html, prob_html  # Early return on initial load

    # If we have state in the store, use it to populate the UI
    state_html = data['state_str']  # Already formatted string
    
    # Build probability cards for each basis using stored probabilities
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

    return state_html, prob_html  # Updated readouts


# Callback to generate an AI explanation for the current state when button is clicked
@app.callback(
    Output('ai-explanation-output', 'children'),  # Markdown text output
    Input('ai-explain-button', 'n_clicks'),       # Trigger button
    State('current-state-store', 'data'),         # Current state and metadata
    prevent_initial_call=True                     # Do not run on initial page load
)
def update_ai_explanation(n_clicks, state_data):
    if not state_data:
        return dcc.Markdown("Please interact with the sphere first to generate a state.")  # Guard if no state
    
    last_action = state_data.get('last_action', 'User requested explanation')  # What changed last
    if last_action == 'ai-explain-button':
        last_action = "User requested an explanation of the current state."  # Make label more readable

    explanation = get_ai_explanation(state_data, last_action)  # Delegate to logic helper
    
    return dcc.Markdown(explanation, link_target="_blank")  # Render markdown (allow links to open in new tab)


# Client-side callback to add a quick click animation class to pressed buttons (no server roundtrip)
app.clientside_callback(
    """
    function(n_x, n_y, n_z, n_h, n_s, n_t, n_reset, n_plus, n_minus, n_random) {
        // Determine which input fired this clientside callback
        const triggered = dash_clientside.callback_context.triggered[0];
        if (!triggered) {
            return; // No button was triggered
        }
        
        const buttonId = triggered.prop_id.split('.')[0];  // Extract component id
        const element = document.getElementById(buttonId);  // Get the DOM element
        
        if (element) {
            // Add the CSS class that defines the animation
            element.classList.add('button-clicked');
            
            // Remove the class shortly after so it can be re-applied on the next click
            setTimeout(() => {
                element.classList.remove('button-clicked');
            }, 150); // Duration matches CSS transition for a crisp tap effect
        }
        return dash_clientside.no_update; // Do not modify any outputs; purely visual side-effect
    }
    """,
    Output('current-state-store', 'data', allow_duplicate=True),  # Dummy output to satisfy Dash API
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
    prevent_initial_call=True  # Only run after user interaction
)


# Run the development server when executing this script directly
if __name__ == '__main__':
    app.run(debug=True)  # Enable hot reloading and debug info for local development
