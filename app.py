# Import core Dash library for building web apps
import dash  # Dash is a framework for building analytic web apps in Python
from dash import dcc, html, Input, Output, State, callback_context  # Common Dash components and callback primitives
import numpy as np  # Numerical computing (angles, complex numbers, etc.)
import secrets  # Cryptographically secure pseudo-random generator
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
                html.Img(src=github_logo_data_uri, style={'height': '18px', 'marginRight': '10px'}),
                "View on GitHub"
            ],
            href="https://github.com/your-github-username/your-repo-name",
            target="_blank",
            style={'color': 'white', 'textDecoration': 'none'}
        )
    ]),
    
    # Rest of your app layout...
])

# Replace the 'random' module with the 'secrets' module
# Example:
# import secrets
# random_number = secrets.randbelow(100)

# If you are generating a random number within a range, use secrets.randbelow
# For floating point numbers, you can use secrets.choice and a list of possible values
# Or implement your own function to generate a random float

# Initialize the app
if __name__ == '__main__':
    app.run_server(debug=True)