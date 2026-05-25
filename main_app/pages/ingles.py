import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go

dash.register_page(__name__, path='/ingles', name='Modelo 1 — Inglés', order=1)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get('MODELS_DIR', str(BASE_DIR / '..' / 'models')))
MODEL_PATH = MODELS_DIR / 'modelo_binario_supera_nivel_bajo_ingles.keras'
PREP_PATH  = MODELS_DIR / 'preprocessor_binario_ingles.pkl'

model        = None
preprocessor = None

try:
    import tensorflow as tf
    from tensorflow import keras
    model = keras.models.load_model(str(MODEL_PATH))
    print(f"[ingles] Modelo cargado: {MODEL_PATH}")
except Exception as exc:
    print(f"[ingles] Error cargando modelo: {exc}", file=sys.stderr)

try:
    preprocessor = joblib.load(str(PREP_PATH))
    print(f"[ingles] Preprocesador cargado: {PREP_PATH}")
except Exception as exc:
    print(f"[ingles] Error cargando preprocesador: {exc}", file=sys.stderr)

# ── Constantes ────────────────────────────────────────────────────────────────
EDUC_LEVELS = [
    'Ninguno', 'Primaria incompleta', 'Primaria completa',
    'Secundaria (Bachillerato) incompleta', 'Secundaria (Bachillerato) completa',
    'Técnica o tecnológica incompleta', 'Técnica o tecnológica completa',
    'Educación profesional incompleta', 'Educación profesional completa',
    'Postgrado', 'No sabe',
]

MUNICIPIOS = sorted([
    'AGUADA', 'ALBANIA', 'ARATOCA', 'BARBOSA', 'BARICHARA',
    'BARRANCABERMEJA', 'BETULIA', 'BOLIVAR', 'BUCARAMANGA', 'CABRERA',
    'CALIFORNIA', 'CAPITANEJO', 'CARCASI', 'CEPITA', 'CERRITO',
    'CHARALA', 'CHARTA', 'CHIMA', 'CHIPATA', 'CIMITARRA',
    'CONCEPCION', 'CONFINES', 'CONTRATACION', 'COROMORO', 'CURITI',
    'EL CARMEN DE CHUCURI', 'EL GUACAMAYO', 'EL PEÑON', 'EL PLAYON',
    'ENCINO', 'ENCISO', 'FLORIAN', 'FLORIDABLANCA', 'GALAN', 'GAMBITA',
    'GIRON', 'GUACA', 'GUADALUPE', 'GUAPOTA', 'GUAVATA', 'GUEPSA',
    'HATO', 'JESUS MARIA', 'JORDAN', 'LA BELLEZA', 'LA PAZ',
    'LANDAZURI', 'LEBRIJA', 'LOS SANTOS', 'MACARAVITA', 'MALAGA',
    'MATANZA', 'MOGOTES', 'MOLAGAVITA', 'OCAMONTE', 'OIBA', 'ONZAGA',
    'PALESTINA', 'PARAMO', 'PIEDECUESTA', 'PINCHOTE', 'PUENTE NACIONAL',
    'PUERTO PARRA', 'PUERTO WILCHES', 'RIONEGRO', 'SABANA DE TORRES',
    'SAN ANDRES', 'SAN BENITO', 'SAN GIL', 'SAN JOAQUIN',
    'SAN JOSE DE MIRANDA', 'SAN MIGUEL', 'SAN VICENTE DE CHUCURI',
    'SANTA BARBARA', 'SANTA HELENA DEL OPON', 'SIMACOTA', 'SOCORRO',
    'SUAITA', 'SUCRE', 'SURATA', 'TONA', 'VALLE DE SAN JOSE',
    'VELEZ', 'VETAS', 'VILLANUEVA', 'ZAPATOCA',
])


def opts(vals):
    return [{'label': v, 'value': v} for v in vals]


def form_group(label, component):
    return html.Div([
        html.Label(label, className='form-label'),
        component,
    ], className='form-group')


def metric_card(label, value):
    return html.Div([
        html.P(label, className='metric-label'),
        html.P(value, className='metric-value'),
    ], className='metric-card')


# ── Layout ────────────────────────────────────────────────────────────────────
layout = html.Div([

    html.Div([
        html.Div([
            html.Span('Secretaría de Educación de Santander', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.Span('Modelo 1 — Clasificación Binaria', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.A('← Panel principal', href='/', className='back-link'),
        ], className='header-badges'),
        html.H1('Predicción de desempeño mínimo en inglés', className='header-title'),
        html.H3('Modelo predictivo Saber 11 — Santander', className='header-subtitle'),
        html.P(
            'Este tablero estima la probabilidad de que un estudiante supere '
            'el nivel mínimo A- en inglés en las pruebas Saber 11, a partir '
            'de sus características escolares, familiares y socioeconómicas.',
            className='header-desc'
        ),
    ], className='header'),

    html.Div([

        html.Div([
            html.H2('Métricas del mejor modelo (Red Neuronal Binaria)',
                    className='section-title'),
            html.Div([
                metric_card('Accuracy',  '0.6843'),
                metric_card('Precision', '0.6994'),
                metric_card('Recall',    '0.7151'),
                metric_card('F1-score',  '0.7072'),
                metric_card('ROC-AUC',   '0.7554'),
            ], className='metrics-row'),
        ], className='section'),

        html.Div([
            html.Div([
                html.H2('Datos del estudiante', className='section-title'),

                html.H4('Información del colegio', className='subsection-title'),
                html.Div([
                    html.Div([
                        form_group('Área de ubicación del colegio',
                            dcc.Dropdown(id='ing-COLE_AREA_UBICACION',
                                options=opts(['URBANO', 'RURAL']),
                                value='URBANO', clearable=False)),
                        form_group('¿El colegio es bilingüe?',
                            dcc.Dropdown(id='ing-COLE_BILINGUE',
                                options=opts(['N', 'S']),
                                value='N', clearable=False)),
                        form_group('Calendario escolar',
                            dcc.Dropdown(id='ing-COLE_CALENDARIO',
                                options=opts(['A', 'B']),
                                value='A', clearable=False)),
                        form_group('Carácter del colegio',
                            dcc.Dropdown(id='ing-COLE_CARACTER',
                                options=opts(['ACADÉMICO', 'TÉCNICO/ACADÉMICO',
                                              'TÉCNICO', 'NORMALISTA']),
                                value='ACADÉMICO', clearable=False)),
                        form_group('Género del colegio',
                            dcc.Dropdown(id='ing-COLE_GENERO',
                                options=opts(['MIXTO', 'FEMENINO', 'MASCULINO']),
                                value='MIXTO', clearable=False)),
                    ], className='form-col'),
                    html.Div([
                        form_group('Jornada escolar',
                            dcc.Dropdown(id='ing-COLE_JORNADA',
                                options=opts(['MAÑANA', 'TARDE', 'NOCHE',
                                              'COMPLETA', 'ÚNICA',
                                              'SABATINA', 'MAÑANA Y TARDE']),
                                value='MAÑANA', clearable=False)),
                        form_group('Municipio del colegio',
                            dcc.Dropdown(id='ing-COLE_MCPIO_UBICACION',
                                options=opts(MUNICIPIOS),
                                value='BUCARAMANGA', clearable=False,
                                searchable=True)),
                        form_group('Naturaleza del colegio',
                            dcc.Dropdown(id='ing-COLE_NATURALEZA',
                                options=opts(['OFICIAL', 'NO OFICIAL']),
                                value='OFICIAL', clearable=False)),
                        form_group('¿Es sede principal?',
                            dcc.Dropdown(id='ing-COLE_SEDE_PRINCIPAL',
                                options=opts(['S', 'N']),
                                value='S', clearable=False)),
                    ], className='form-col'),
                ], className='form-row'),

                html.H4('Información del estudiante', className='subsection-title'),
                html.Div([
                    html.Div([
                        form_group('Género del estudiante',
                            dcc.Dropdown(id='ing-ESTU_GENERO',
                                options=opts(['M', 'F']),
                                value='M', clearable=False)),
                        form_group('¿Está privado de la libertad?',
                            dcc.Dropdown(id='ing-ESTU_PRIVADO_LIBERTAD',
                                options=opts(['No', 'Si']),
                                value='No', clearable=False)),
                    ], className='form-col'),
                    html.Div([
                        form_group('Edad del estudiante',
                            dcc.Slider(id='ing-EDAD', min=14, max=25, step=1,
                                       value=16,
                                       marks={i: str(i) for i in range(14, 26)},
                                       tooltip={'placement': 'bottom',
                                                'always_visible': True})),
                    ], className='form-col'),
                ], className='form-row'),

                html.H4('Información familiar y socioeconómica',
                        className='subsection-title'),
                html.Div([
                    html.Div([
                        form_group('Número de cuartos en el hogar',
                            dcc.Dropdown(id='ing-FAMI_CUARTOSHOGAR',
                                options=opts(['1', '2', '3', '4', '5 o más']),
                                value='3', clearable=False)),
                        form_group('Educación de la madre',
                            dcc.Dropdown(id='ing-FAMI_EDUCACIONMADRE',
                                options=opts(EDUC_LEVELS),
                                value='Secundaria (Bachillerato) completa',
                                clearable=False)),
                        form_group('Educación del padre',
                            dcc.Dropdown(id='ing-FAMI_EDUCACIONPADRE',
                                options=opts(EDUC_LEVELS),
                                value='Secundaria (Bachillerato) completa',
                                clearable=False)),
                        form_group('Estrato de vivienda',
                            dcc.Dropdown(id='ing-FAMI_ESTRATOVIVIENDA',
                                options=opts(['Estrato 1', 'Estrato 2', 'Estrato 3',
                                              'Estrato 4', 'Estrato 5', 'Estrato 6',
                                              'Sin Estrato']),
                                value='Estrato 2', clearable=False)),
                        form_group('Personas en el hogar',
                            dcc.Dropdown(id='ing-FAMI_PERSONASHOGAR',
                                options=opts(['1 a 2', '3 a 4', '5 a 6', '7 a más']),
                                value='3 a 4', clearable=False)),
                    ], className='form-col'),
                    html.Div([
                        form_group('Tiene automóvil',
                            dcc.Dropdown(id='ing-FAMI_TIENEAUTOMOVIL',
                                options=opts(['No', 'Si']),
                                value='No', clearable=False)),
                        form_group('Tiene computador en casa',
                            dcc.Dropdown(id='ing-FAMI_TIENECOMPUTADOR',
                                options=opts(['No', 'Si']),
                                value='Si', clearable=False)),
                        form_group('Tiene internet en casa',
                            dcc.Dropdown(id='ing-FAMI_TIENEINTERNET',
                                options=opts(['No', 'Si']),
                                value='Si', clearable=False)),
                        form_group('Tiene lavadora',
                            dcc.Dropdown(id='ing-FAMI_TIENELAVADORA',
                                options=opts(['No', 'Si']),
                                value='Si', clearable=False)),
                    ], className='form-col'),
                ], className='form-row'),

                html.Div([
                    html.Button('Generar predicción', id='ing-predict-btn',
                                className='btn-primary', n_clicks=0),
                ], className='btn-container'),

            ], className='form-panel'),

            html.Div([
                html.H2('Resultado', className='section-title'),
                html.Div(id='ing-result', children=[
                    html.P(
                        'Complete el formulario y haga clic en "Generar predicción".',
                        className='result-placeholder'
                    ),
                ]),
            ], className='result-panel'),

        ], className='main-content'),

        html.Div([
            html.Div([
                html.Span('ℹ', className='info-icon'),
                html.P(
                    'El resultado del modelo es una herramienta de apoyo para '
                    'la toma de decisiones educativas. No debe interpretarse '
                    'como una evaluación definitiva del estudiante.',
                    className='disclaimer-text'
                ),
            ], className='disclaimer-inner'),
        ], className='disclaimer'),

    ], className='content-wrapper'),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@callback(
    Output('ing-result', 'children'),
    Input('ing-predict-btn', 'n_clicks'),
    State('ing-COLE_AREA_UBICACION',   'value'),
    State('ing-COLE_BILINGUE',         'value'),
    State('ing-COLE_CALENDARIO',       'value'),
    State('ing-COLE_CARACTER',         'value'),
    State('ing-COLE_GENERO',           'value'),
    State('ing-COLE_JORNADA',          'value'),
    State('ing-COLE_MCPIO_UBICACION',  'value'),
    State('ing-COLE_NATURALEZA',       'value'),
    State('ing-COLE_SEDE_PRINCIPAL',   'value'),
    State('ing-ESTU_GENERO',           'value'),
    State('ing-ESTU_PRIVADO_LIBERTAD', 'value'),
    State('ing-EDAD',                  'value'),
    State('ing-FAMI_CUARTOSHOGAR',     'value'),
    State('ing-FAMI_EDUCACIONMADRE',   'value'),
    State('ing-FAMI_EDUCACIONPADRE',   'value'),
    State('ing-FAMI_ESTRATOVIVIENDA',  'value'),
    State('ing-FAMI_PERSONASHOGAR',    'value'),
    State('ing-FAMI_TIENEAUTOMOVIL',   'value'),
    State('ing-FAMI_TIENECOMPUTADOR',  'value'),
    State('ing-FAMI_TIENEINTERNET',    'value'),
    State('ing-FAMI_TIENELAVADORA',    'value'),
    prevent_initial_call=True,
)
def predict(n_clicks,
            cole_area, cole_bil, cole_cal, cole_car, cole_gen, cole_jor,
            cole_mcp, cole_nat, cole_sed,
            estu_gen, estu_lib, edad,
            fami_cuar, fami_edm, fami_edp, fami_est, fami_per,
            fami_aut, fami_com, fami_int, fami_lav):

    if model is None or preprocessor is None:
        return html.P(
            'Error: el modelo o el preprocesador no están disponibles. '
            'Verifique que los archivos en models/ existan.',
            className='error-text'
        )

    try:
        row = {
            'EDAD':                  int(edad) if edad is not None else 16,
            'COLE_AREA_UBICACION':   cole_area or 'Desconocido',
            'COLE_BILINGUE':         cole_bil  or 'Desconocido',
            'COLE_CALENDARIO':       cole_cal  or 'Desconocido',
            'COLE_CARACTER':         cole_car  or 'Desconocido',
            'COLE_GENERO':           cole_gen  or 'Desconocido',
            'COLE_JORNADA':          cole_jor  or 'Desconocido',
            'COLE_MCPIO_UBICACION':  cole_mcp  or 'Desconocido',
            'COLE_NATURALEZA':       cole_nat  or 'Desconocido',
            'COLE_SEDE_PRINCIPAL':   cole_sed  or 'Desconocido',
            'ESTU_GENERO':           estu_gen  or 'Desconocido',
            'ESTU_PRIVADO_LIBERTAD': estu_lib  or 'Desconocido',
            'FAMI_CUARTOSHOGAR':     fami_cuar or 'Desconocido',
            'FAMI_EDUCACIONMADRE':   fami_edm  or 'Desconocido',
            'FAMI_EDUCACIONPADRE':   fami_edp  or 'Desconocido',
            'FAMI_ESTRATOVIVIENDA':  fami_est  or 'Desconocido',
            'FAMI_PERSONASHOGAR':    fami_per  or 'Desconocido',
            'FAMI_TIENEAUTOMOVIL':   fami_aut  or 'Desconocido',
            'FAMI_TIENECOMPUTADOR':  fami_com  or 'Desconocido',
            'FAMI_TIENEINTERNET':    fami_int  or 'Desconocido',
            'FAMI_TIENELAVADORA':    fami_lav  or 'Desconocido',
        }
        df_input = pd.DataFrame([row])
        X_proc   = preprocessor.transform(df_input)
        prob     = float(model.predict(X_proc, verbose=0).ravel()[0])
    except Exception as exc:
        return html.P(f'Error en la predicción: {exc}', className='error-text')

    prob_pct  = prob * 100
    bar_color = '#27ae60' if prob >= 0.70 else ('#f39c12' if prob >= 0.50 else '#e74c3c')

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(prob_pct, 1),
        number={'suffix': '%', 'font': {'size': 40, 'color': bar_color}},
        title={'text': 'Probabilidad de superar nivel A-',
               'font': {'size': 15, 'color': '#2c3e50'}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%', 'tickcolor': '#7f8c8d'},
            'bar':  {'color': bar_color, 'thickness': 0.25},
            'steps': [
                {'range': [0,  50], 'color': '#fdecea'},
                {'range': [50, 70], 'color': '#fff3e0'},
                {'range': [70, 100],'color': '#e8f5e9'},
            ],
            'threshold': {'line': {'color': '#7f8c8d', 'width': 3},
                          'thickness': 0.75, 'value': 50},
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=10),
                      height=270, paper_bgcolor='white')

    res_text = ('Probablemente supera el nivel más bajo de inglés'
                if prob >= 0.50 else 'Probablemente permanece en el nivel más bajo A-')
    res_cls  = 'result-positive' if prob >= 0.50 else 'result-negative'

    if prob >= 0.70:
        reco     = 'Probabilidad alta. Se recomienda mantener estrategias de fortalecimiento y seguimiento.'
        reco_cls = 'reco-high'
    elif prob >= 0.50:
        reco     = 'Probabilidad media. Se recomienda reforzar el aprendizaje de inglés.'
        reco_cls = 'reco-medium'
    else:
        reco     = 'Probabilidad baja. Se recomienda priorizar acompañamiento académico.'
        reco_cls = 'reco-low'

    return [
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.P(f'{prob_pct:.1f}%', className='prob-pct'),
            html.P(res_text, className=f'result-text {res_cls}'),
        ], className='result-summary'),
        html.Div([html.P(reco, className='reco-text')], className=f'reco-box {reco_cls}'),
    ]
