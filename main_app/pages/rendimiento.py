import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import joblib
import dash
from dash import dcc, html, callback, Input, Output, State
import plotly.graph_objects as go

dash.register_page(__name__, path='/rendimiento', name='Modelo 2 — Rendimiento', order=2)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get('MODELS_DIR', str(BASE_DIR / '..' / 'models')))


def _find(filename):
    for p in [MODELS_DIR / filename, BASE_DIR / filename]:
        if p.exists():
            return str(p)
    return None


model     = None
le_dict   = None
le_target = None
scaler    = None

try:
    import tensorflow as tf
    from tensorflow import keras
    p = _find('modelo_clasificacion_binaria.h5')
    if p:
        model = keras.models.load_model(p)
        print(f"[rendimiento] Modelo cargado: {p}")
    else:
        print("[rendimiento] modelo_clasificacion_binaria.h5 no encontrado", file=sys.stderr)
except Exception as exc:
    print(f"[rendimiento] Error cargando modelo: {exc}", file=sys.stderr)

for name, var_name in [
    ('encoders_clasificacion_binaria.pkl',      'le_dict'),
    ('encoder_target_clasificacion_binaria.pkl', 'le_target'),
    ('scaler_clasificacion_binaria.pkl',         'scaler'),
]:
    try:
        p = _find(name)
        if p:
            globals()[var_name] = joblib.load(p)
            print(f"[rendimiento] {name} cargado")
    except Exception as exc:
        print(f"[rendimiento] Error cargando {name}: {exc}", file=sys.stderr)

# ── Constantes ────────────────────────────────────────────────────────────────
FEATURES = [
    'cole_area_ubicacion', 'cole_calendario', 'cole_caracter',
    'cole_jornada', 'cole_naturaleza', 'cole_mcpio_ubicacion',
    'estu_genero', 'estu_privado_libertad',
    'fami_cuartoshogar', 'fami_educacionmadre', 'fami_educacionpadre',
    'fami_estratovivienda', 'fami_personashogar',
    'fami_tieneautomovil', 'fami_tienecomputador',
    'fami_tieneinternet', 'fami_tienelavadora',
]

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


def metric_card(label, value, accent=None):
    style = {'borderTopColor': accent} if accent else {}
    return html.Div([
        html.P(label, className='metric-label'),
        html.P(value, className='metric-value'),
    ], className='metric-card', style=style)


def step_card(num, title, desc):
    return html.Div([
        html.Div(str(num), className='step-num'),
        html.Div([
            html.Strong(title, className='step-title'),
            html.P(desc, className='step-desc'),
        ]),
    ], className='step')


# ── Layout ────────────────────────────────────────────────────────────────────
layout = html.Div([

    html.Div([
        html.Div([
            html.Span('Secretaría de Educación de Santander', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.Span('Modelo 2 — Clasificación Binaria', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.A('← Panel principal', href='/', className='back-link'),
        ], className='header-badges'),
        html.H1('Predicción de Nivel de Rendimiento Global', className='header-title'),
        html.H3(
            '¿Alcanzará el estudiante un rendimiento Alto (≥ 300 pts) o Bajo (< 300 pts)?',
            className='header-subtitle'
        ),
        html.P(
            'Este tablero usa una red neuronal entrenada con datos Saber 11 de Santander '
            'para estimar si un estudiante alcanzará rendimiento Alto o Bajo en el '
            'puntaje global, a partir de características institucionales, personales y familiares.',
            className='header-desc'
        ),
    ], className='header'),

    html.Div([

        html.Div([
            html.H2('Métricas del modelo  ·  3 capas ocultas · 128 neuronas · Adam lr=0.001',
                    className='section-title'),
            html.Div([
                metric_card('Accuracy',           '71.6 %',    '#2980b9'),
                metric_card('AUC-ROC',            '0.779',     '#8e44ad'),
                metric_card('F1 — Alto',          '0.79',      '#27ae60'),
                metric_card('F1 — Bajo',          '0.56',      '#e74c3c'),
                metric_card('Umbral de decisión', '≥ 300 pts', '#7f8c8d'),
            ], className='metrics-row'),
        ], className='section'),

        html.Div([
            html.H2('¿Cómo usar este tablero?', className='section-title'),
            html.Div([
                step_card(1, 'Complete los datos del colegio',
                          'Seleccione área, calendario, carácter, jornada, naturaleza y municipio.'),
                step_card(2, 'Complete los datos del estudiante',
                          'Indique el género y si se encuentra privado de la libertad.'),
                step_card(3, 'Complete el contexto familiar',
                          'Nivel educativo de los padres, estrato, cuartos, personas y bienes.'),
                step_card(4, 'Haga clic en "Generar predicción"',
                          'El modelo estimará si el rendimiento será Alto (≥ 300) o Bajo.'),
            ], className='steps-container'),
            html.Div([
                html.Span('Clasificación: ', className='legend-label'),
                html.Span('ALTO', className='badge badge-alto'),
                html.Span('  puntaje global ≥ 300  ·  ', className='legend-sep'),
                html.Span('BAJO', className='badge badge-bajo'),
                html.Span('  puntaje global < 300', className='legend-sep'),
            ], className='legend-row'),
        ], className='section'),

        html.Div([

            html.Div([
                html.H2('Datos del estudiante', className='section-title'),

                html.H4('Información del colegio', className='subsection-title'),
                html.Div([
                    html.Div([
                        form_group('Área de ubicación del colegio',
                            dcc.Dropdown(id='rend-cole_area_ubicacion',
                                options=opts(['URBANO', 'RURAL']),
                                value='URBANO', clearable=False)),
                        form_group('Calendario escolar',
                            dcc.Dropdown(id='rend-cole_calendario',
                                options=opts(['A', 'B']),
                                value='A', clearable=False)),
                        form_group('Carácter del colegio',
                            dcc.Dropdown(id='rend-cole_caracter',
                                options=opts(['ACADÉMICO', 'TÉCNICO/ACADÉMICO',
                                              'TÉCNICO', 'NORMALISTA']),
                                value='ACADÉMICO', clearable=False)),
                    ], className='form-col'),
                    html.Div([
                        form_group('Jornada escolar',
                            dcc.Dropdown(id='rend-cole_jornada',
                                options=opts(['MAÑANA', 'TARDE', 'NOCHE',
                                              'COMPLETA', 'ÚNICA',
                                              'SABATINA', 'MAÑANA Y TARDE']),
                                value='MAÑANA', clearable=False)),
                        form_group('Naturaleza del colegio',
                            dcc.Dropdown(id='rend-cole_naturaleza',
                                options=opts(['OFICIAL', 'NO OFICIAL']),
                                value='OFICIAL', clearable=False)),
                        form_group('Municipio del colegio',
                            dcc.Dropdown(id='rend-cole_mcpio_ubicacion',
                                options=opts(MUNICIPIOS),
                                value='BUCARAMANGA', clearable=False,
                                searchable=True)),
                    ], className='form-col'),
                ], className='form-row'),

                html.H4('Información del estudiante', className='subsection-title'),
                html.Div([
                    html.Div([
                        form_group('Género del estudiante',
                            dcc.Dropdown(id='rend-estu_genero',
                                options=opts(['M', 'F']),
                                value='M', clearable=False)),
                    ], className='form-col'),
                    html.Div([
                        form_group('¿Está privado de la libertad?',
                            dcc.Dropdown(id='rend-estu_privado_libertad',
                                options=opts(['No', 'Si']),
                                value='No', clearable=False)),
                    ], className='form-col'),
                ], className='form-row'),

                html.H4('Información familiar y socioeconómica',
                        className='subsection-title'),
                html.Div([
                    html.Div([
                        form_group('Número de cuartos en el hogar',
                            dcc.Dropdown(id='rend-fami_cuartoshogar',
                                options=opts(['1', '2', '3', '4', '5 o más']),
                                value='3', clearable=False)),
                        form_group('Educación de la madre',
                            dcc.Dropdown(id='rend-fami_educacionmadre',
                                options=opts(EDUC_LEVELS),
                                value='Secundaria (Bachillerato) completa',
                                clearable=False)),
                        form_group('Educación del padre',
                            dcc.Dropdown(id='rend-fami_educacionpadre',
                                options=opts(EDUC_LEVELS),
                                value='Secundaria (Bachillerato) completa',
                                clearable=False)),
                        form_group('Estrato de vivienda',
                            dcc.Dropdown(id='rend-fami_estratovivienda',
                                options=opts(['Estrato 1', 'Estrato 2', 'Estrato 3',
                                              'Estrato 4', 'Estrato 5', 'Estrato 6',
                                              'Sin Estrato']),
                                value='Estrato 2', clearable=False)),
                        form_group('Personas en el hogar',
                            dcc.Dropdown(id='rend-fami_personashogar',
                                options=opts(['1 a 2', '3 a 4', '5 a 6', '7 a más']),
                                value='3 a 4', clearable=False)),
                    ], className='form-col'),
                    html.Div([
                        form_group('¿Tiene automóvil?',
                            dcc.Dropdown(id='rend-fami_tieneautomovil',
                                options=opts(['No', 'Si']),
                                value='No', clearable=False)),
                        form_group('¿Tiene computador en casa?',
                            dcc.Dropdown(id='rend-fami_tienecomputador',
                                options=opts(['No', 'Si']),
                                value='Si', clearable=False)),
                        form_group('¿Tiene internet en casa?',
                            dcc.Dropdown(id='rend-fami_tieneinternet',
                                options=opts(['No', 'Si']),
                                value='Si', clearable=False)),
                        form_group('¿Tiene lavadora?',
                            dcc.Dropdown(id='rend-fami_tienelavadora',
                                options=opts(['No', 'Si']),
                                value='Si', clearable=False)),
                    ], className='form-col'),
                ], className='form-row'),

                html.Div([
                    html.Button('Generar predicción', id='rend-predict-btn',
                                className='btn-primary', n_clicks=0),
                ], className='btn-container'),

            ], className='form-panel'),

            html.Div([
                html.H2('Resultado', className='section-title'),
                html.Div(id='rend-result', children=[
                    html.P(
                        'Complete el formulario y haga clic en "Generar predicción".',
                        className='result-placeholder'
                    ),
                ]),
            ], className='result-panel'),

        ], className='main-content'),

        html.Div([
            html.Div([
                html.Span('i', className='info-icon'),
                html.P(
                    'El resultado del modelo es una herramienta de apoyo para la toma '
                    'de decisiones educativas. No reemplaza el juicio pedagógico ni debe '
                    'interpretarse como una evaluación definitiva del estudiante.',
                    className='disclaimer-text'
                ),
            ], className='disclaimer-inner'),
        ], className='disclaimer'),

    ], className='content-wrapper'),
])


# ── Callback ──────────────────────────────────────────────────────────────────
@callback(
    Output('rend-result', 'children'),
    Input('rend-predict-btn', 'n_clicks'),
    State('rend-cole_area_ubicacion',   'value'),
    State('rend-cole_calendario',       'value'),
    State('rend-cole_caracter',         'value'),
    State('rend-cole_jornada',          'value'),
    State('rend-cole_naturaleza',       'value'),
    State('rend-cole_mcpio_ubicacion',  'value'),
    State('rend-estu_genero',           'value'),
    State('rend-estu_privado_libertad', 'value'),
    State('rend-fami_cuartoshogar',     'value'),
    State('rend-fami_educacionmadre',   'value'),
    State('rend-fami_educacionpadre',   'value'),
    State('rend-fami_estratovivienda',  'value'),
    State('rend-fami_personashogar',    'value'),
    State('rend-fami_tieneautomovil',   'value'),
    State('rend-fami_tienecomputador',  'value'),
    State('rend-fami_tieneinternet',    'value'),
    State('rend-fami_tienelavadora',    'value'),
    prevent_initial_call=True,
)
def predict(n_clicks,
            cole_area, cole_cal, cole_car, cole_jor, cole_nat, cole_mcp,
            estu_gen, estu_lib,
            fami_cuar, fami_edm, fami_edp, fami_est, fami_per,
            fami_aut, fami_com, fami_int, fami_lav):

    if model is None or le_dict is None or scaler is None:
        return html.P(
            'Error: los artefactos del modelo no están disponibles. '
            'Verifique que los archivos .h5 y .pkl existan en la carpeta models/.',
            className='error-text'
        )

    input_vals = {
        'cole_area_ubicacion':   cole_area, 'cole_calendario':       cole_cal,
        'cole_caracter':         cole_car,  'cole_jornada':          cole_jor,
        'cole_naturaleza':       cole_nat,  'cole_mcpio_ubicacion':  cole_mcp,
        'estu_genero':           estu_gen,  'estu_privado_libertad': estu_lib,
        'fami_cuartoshogar':     fami_cuar, 'fami_educacionmadre':   fami_edm,
        'fami_educacionpadre':   fami_edp,  'fami_estratovivienda':  fami_est,
        'fami_personashogar':    fami_per,  'fami_tieneautomovil':   fami_aut,
        'fami_tienecomputador':  fami_com,  'fami_tieneinternet':    fami_int,
        'fami_tienelavadora':    fami_lav,
    }

    try:
        encoded = []
        for feat in FEATURES:
            v  = str(input_vals[feat])
            le = le_dict[feat]
            try:
                enc_val = le.transform([v])[0]
            except ValueError:
                enc_val = 0
            encoded.append(float(enc_val))

        X        = np.array(encoded).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prob_bajo = float(model.predict(X_scaled, verbose=0).ravel()[0])
        prob_alto = 1.0 - prob_bajo
    except Exception as exc:
        return html.P(f'Error en la predicción: {exc}', className='error-text')

    if prob_bajo >= 0.5:
        categoria = 'BAJO';  prob_show = prob_bajo; color = '#e74c3c'
        res_cls   = 'result-negative'; badge_cls = 'badge badge-bajo badge-lg'
        pts_label = 'puntaje global estimado < 300'
    else:
        categoria = 'ALTO';  prob_show = prob_alto; color = '#27ae60'
        res_cls   = 'result-positive'; badge_cls = 'badge badge-alto badge-lg'
        pts_label = 'puntaje global estimado ≥ 300'

    prob_pct = prob_show * 100

    steps = ([{'range': [0,50],'color':'#fdecea'}, {'range':[50,70],'color':'#fff3e0'},
               {'range':[70,100],'color':'#e8f5e9'}] if categoria == 'ALTO'
             else [{'range':[0,50],'color':'#e8f5e9'}, {'range':[50,70],'color':'#fff3e0'},
                   {'range':[70,100],'color':'#fdecea'}])
    gauge_title = f'Probabilidad de Rendimiento {categoria}'

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=round(prob_pct, 1),
        number={'suffix': '%', 'font': {'size': 40, 'color': color}},
        title={'text': gauge_title, 'font': {'size': 14, 'color': '#2c3e50'}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%', 'tickcolor': '#7f8c8d'},
            'bar':  {'color': color, 'thickness': 0.25},
            'steps': steps,
            'threshold': {'line': {'color': '#7f8c8d', 'width': 3},
                          'thickness': 0.75, 'value': 50},
        },
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=10),
                      height=270, paper_bgcolor='white')

    if categoria == 'ALTO' and prob_alto >= 0.70:
        reco = 'Alta probabilidad de rendimiento alto. Se recomienda fortalecer con programas de excelencia académica.'
        reco_cls = 'reco-box reco-high'
    elif categoria == 'ALTO':
        reco = 'Probabilidad moderada de rendimiento alto. Se recomienda seguimiento académico continuo.'
        reco_cls = 'reco-box reco-medium'
    elif prob_bajo >= 0.70:
        reco = 'Alta probabilidad de rendimiento bajo. Se recomienda intervención prioritaria y acompañamiento pedagógico.'
        reco_cls = 'reco-box reco-low'
    else:
        reco = 'Probabilidad moderada de rendimiento bajo. Se recomienda refuerzo académico oportuno.'
        reco_cls = 'reco-box reco-medium'

    return [
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        html.Div([
            html.Span(categoria, className=badge_cls),
            html.P(f'{prob_pct:.1f}%', className='prob-pct'),
            html.P(pts_label, className=f'result-text {res_cls}'),
        ], className='result-summary'),
        html.Div([html.P(reco, className='reco-text')], className=reco_cls),
    ]
