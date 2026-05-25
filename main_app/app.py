import os
import dash
from dash import Dash, html

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title='Panel Educativo Saber 11 — Santander',
)
server = app.server

app.layout = html.Div([
    dash.page_container
], className='app-container')

if __name__ == '__main__':
    port  = int(os.environ.get('PORT', 8050))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)
