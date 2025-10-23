import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import numpy as np
import random

from bloch_sphere_logic import create_figure_for_state, apply_gate_to_state, get_ai_explanation

app = dash.Dash(__name__, external_stylesheets=['https://rsms.me/inter/inter.css'])
server = app.server

app.scripts.config.serve_locally = True
app.scripts.append_script({
    "external_url": [
        "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML"
    ]
})


app.layout = html.Div(style={'backgroundColor': '#111111', 'color': '#FFFFFF', 'fontFamily': 'Inter'}, children=[
    html.H1("Interactive Bloch Sphere", style={'textAlign': 'center', 'padding': '20px'}),
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center', 'gap': '30px'}, children=[
        dcc.Graph(id='bloch-sphere-graph', figure=create_figure_for_state(0, 0)),
        html.Div(style={'width': '400px'}, children=[
            html.H3("State Controls"),
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
            
            html.H3("Quantum Gates", style={'marginTop': '30px'}),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '10px'}, children=[
                html.Button('X Gate', id='gate-x', n_clicks=0), html.Button('Y Gate', id='gate-y', n_clicks=0),
                html.Button('Z Gate', id='gate-z', n_clicks=0), html.Button('H Gate', id='gate-h', n_clicks=0),
                html.Button('S Gate', id='gate-s', n_clicks=0), html.Button('T Gate', id='gate-t', n_clicks=0),
            ]),
            html.H3("Presets", style={'marginTop': '20px'}),
            html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '10px'}, children=[
                html.Button('Reset to |0⟩', id='reset-button', n_clicks=0), html.Button('Set to |+⟩', id='plus-button', n_clicks=0),
                html.Button('Set to |-⟩', id='minus-button', n_clicks=0), html.Button('Random State', id='random-button', n_clicks=0),
            ]),
            html.H3("Live Readouts", style={'marginTop': '30px'}),
            html.Div(id='state-vector-readout', style={'fontSize': '1.1em', 'fontFamily': 'monospace'}),
            html.Div(id='prob-readout', style={'marginTop': '10px'}),
            
            # --- AI Explanation Area ---
            html.Div(style={'marginTop': '20px', 'textAlign': 'center'}, children=[
                 html.Button("Explain with AI", id="ai-explain-button", n_clicks=0, style={'width': '100%'}),
            ]),
            html.Div(
                dcc.Loading(
                    id="loading-spinner",
                    type="default",
                    children=html.Div(id="ai-explanation-output")
                ),
                style={'marginTop': '15px', 'padding': '10px', 'border': '1px solid #444', 'borderRadius': '5px', 'minHeight': '50px', 'backgroundColor': '#222'}
            )
        ])
    ])
])

@app.callback(
    Output('bloch-sphere-graph', 'figure'),
    Output('state-vector-readout', 'children'),
    Output('prob-readout', 'children'),
    Output('theta-slider', 'value'),
    Output('phi-slider', 'value'),
    Output('phi-input', 'value'),
    Input('theta-slider', 'value'),
    Input('phi-slider', 'value'),
    Input('phi-input', 'value'),
    Input('gate-x', 'n_clicks'), Input('gate-y', 'n_clicks'),
    Input('gate-z', 'n_clicks'), Input('gate-h', 'n_clicks'),
    Input('gate-s', 'n_clicks'), Input('gate-t', 'n_clicks'),
    Input('reset-button', 'n_clicks'), Input('plus-button', 'n_clicks'),
    Input('minus-button', 'n_clicks'), Input('random-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_sphere_and_readouts(
    theta_deg, phi_from_slider, phi_from_input,
    n_x, n_y, n_z, n_h, n_s, n_t,
    n_reset, n_plus, n_minus, n_random
):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
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
    state_str = f"|ψ⟩ = {alpha:.2f} |0⟩ + ({beta.real:.2f}{beta.imag:+.2f}j) |1⟩"
    prob0, prob1 = alpha**2 * 100, abs(beta)**2 * 100
    prob_html = [html.B("Measurement Probabilities:"), html.Div(f"P(|0⟩): {prob0:.1f}%"), html.Div(f"P(|1⟩): {prob1:.1f}%")]
    
    return updated_figure, state_str, prob_html, new_theta, new_phi, new_phi

@app.callback(
    Output('ai-explanation-output', 'children'),
    Input('ai-explain-button', 'n_clicks'),
    State('theta-slider', 'value'),
    State('phi-slider', 'value'),
    State('state-vector-readout', 'children'),
    State('prob-readout', 'children'),
    prevent_initial_call=True
)
def update_ai_explanation(n_clicks, theta_deg, phi_deg, state_str_children, prob_html_children):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == 'ai-explain-button':
        last_action = "User requested an explanation of the current state."
    else:
        last_action = f"User interacted with {triggered_id}"
    
    state_str = state_str_children if isinstance(state_str_children, str) else "Not available"
    prob_text = ""
    if prob_html_children and isinstance(prob_html_children, list):
        prob_text = ' '.join([child['props']['children'] for child in prob_html_children if 'props' in child and 'children' in child['props']])

    explanation = get_ai_explanation(theta_deg, phi_deg, state_str, prob_text, last_action)
    
    return dcc.Markdown(explanation, dangerously_allow_html=True)

if __name__ == '__main__':
    app.run(debug=True)

