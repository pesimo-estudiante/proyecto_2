import os
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import joblib
import dash
from dash import dcc, html, callback, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go

dash.register_page(__name__, path='/escenarios', name='Modelo 3 — Escenarios', order=3)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get('MODELS_DIR', str(BASE_DIR / '..' / 'models')))
DATA_DIR   = Path(os.environ.get('DATA_DIR',   str(BASE_DIR / '..' / 'data_exploration')))

DATA_PATH         = DATA_DIR   / 'data_samp.csv'
MODEL_PATH        = MODELS_DIR / 'modelo_punt_global.keras'
PREPROCESSOR_PATH = MODELS_DIR / 'preprocessor.joblib'

TARGET       = 'PUNT_GLOBAL'
N_PROFILES   = 6
RANDOM_STATE = 42
N_BOOTSTRAPS = 300


# ── Carga de artefactos ───────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_assets():
    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"No se encontró '{TARGET}' en los datos.")
    preprocessor = joblib.load(str(PREPROCESSOR_PATH))
    try:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.load_model(str(MODEL_PATH))
        print(f"[escenarios] Modelo cargado: {MODEL_PATH}")
    except Exception as exc:
        print(f"[escenarios] Error cargando modelo: {exc}", file=sys.stderr)
        model = None
    feature_cols = [col for col in df.columns if col not in (TARGET, 'PROFILE_ID')]
    return df, feature_cols, preprocessor, model


def transform_features(X, preprocessor):
    X_t = preprocessor.transform(X)
    if hasattr(X_t, 'toarray'):
        X_t = X_t.toarray()
    return X_t


def predict_model(df_features, preprocessor, model):
    return model.predict(transform_features(df_features, preprocessor), verbose=0).ravel()


# ── Perfiles ──────────────────────────────────────────────────────────────────
@lru_cache(maxsize=2)
def build_profile_data(prediction: bool = False):
    df, feature_cols, preprocessor, model = load_assets()
    df = df.copy()
    if prediction and model is not None:
        df['PRED_PUNT_GLOBAL'] = predict_model(df[feature_cols], preprocessor, model)
    else:
        df['PRED_PUNT_GLOBAL'] = df[TARGET]
    return df, summarize_profiles(df, feature_cols), feature_cols


def summarize_profiles(df, feature_cols):
    total = len(df)
    rows  = []
    for pid, grp in df.groupby('PROFILE_ID'):
        n = len(grp)
        rows.append({
            'PROFILE_ID':             int(pid),
            'PROFILE_NAME':           f'Perfil {int(pid) + 1}',
            'DESCRIPTION':            _profile_desc(grp, feature_cols),
            'N_STUDENTS':             n,
            'CURRENT_PROPORTION':     n / total,
            'CURRENT_PROPORTION_PCT': n / total * 100,
            'ACTUAL_MEAN_SCORE':      grp[TARGET].mean(),
            'PRED_MEAN_SCORE':        grp['PRED_PUNT_GLOBAL'].mean(),
            'ACTUAL_STD_SCORE':       grp[TARGET].std(),
        })
    ps = pd.DataFrame(rows).sort_values('PRED_MEAN_SCORE').reset_index(drop=True)
    ps['PROFILE_ORDER'] = np.arange(1, len(ps) + 1)
    return ps


def _profile_desc(group, feature_cols, max_vars=4):
    parts = []
    for col in feature_cols[:max_vars]:
        mode_vals = group[col].mode(dropna=True)
        if len(mode_vals):
            parts.append(f'{col}: {mode_vals.iloc[0]}')
    return ' | '.join(parts)


# ── Simulación ────────────────────────────────────────────────────────────────
def normalize_weights(values):
    v = np.array(values, dtype=float)
    if v.sum() <= 0:
        v = np.ones_like(v)
    return v / v.sum()


def simulate_scenario(profile_summary, scenario_weights):
    cw = profile_summary['CURRENT_PROPORTION'].values
    ps = profile_summary['PRED_MEAN_SCORE'].values
    return {
        'current_score':    float(np.sum(cw * ps)),
        'scenario_score':   float(np.sum(scenario_weights * ps)),
        'delta':            float(np.sum((scenario_weights - cw) * ps)),
        'current_weights':  cw,
        'scenario_weights': scenario_weights,
        'profile_scores':   ps,
        'contributions':    (scenario_weights - cw) * ps,
    }


def bootstrap_uncertainty(df, profile_summary, scenario_weights):
    n     = len(df)
    stats = (df.groupby('PROFILE_ID')['PRED_PUNT_GLOBAL']
               .agg(['mean', 'var', 'count']).reset_index())
    stats = profile_summary[['PROFILE_ID']].merge(stats, on='PROFILE_ID', how='left')
    means     = stats['mean'].values
    variances = stats['var'].fillna(0).values
    w  = np.array(scenario_weights, dtype=float);  w /= w.sum()
    mu = np.sum(w * means)
    se = np.sqrt(np.sum(w * (variances + means ** 2)) - mu ** 2) / np.sqrt(n)
    rng = np.random.default_rng(RANDOM_STATE)
    return {
        'mean': float(mu),
        'p05':  float(mu - 1.645 * se),
        'p50':  float(mu),
        'p95':  float(mu + 1.645 * se),
        'distribution': rng.normal(loc=mu, scale=se, size=N_BOOTSTRAPS).tolist(),
    }


# ── Helpers UI ────────────────────────────────────────────────────────────────
def tab_header(title, subtitle=None):
    return html.Div([
        html.H2(title, className='tab-title'),
        html.P(subtitle, className='tab-subtitle') if subtitle else None,
    ], className='tab-header')


def kpi_card(title, value, subtitle=None):
    return html.Div([
        html.P(title, className='kpi-label'),
        html.P(value, className='kpi-value'),
        html.P(subtitle, className='kpi-subtitle') if subtitle else None,
    ], className='kpi-card')


def chart_card(figure):
    return html.Div(
        dcc.Graph(figure=figure, config={'displayModeBar': False}),
        className='chart-card'
    )


def _table_style():
    return {
        'style_table': {'overflowX': 'auto'},
        'style_cell': {'textAlign': 'left', 'padding': '8px',
                       'fontFamily': 'Segoe UI, sans-serif', 'fontSize': '13px'},
        'style_header': {'fontWeight': 'bold', 'backgroundColor': '#f0f4f8'},
    }


# ── Vistas de pestañas ────────────────────────────────────────────────────────
def diagnosis_tab():
    df, ps, _ = build_profile_data(prediction=False)
    ps_ord = ps.sort_values('PROFILE_ID')
    po     = ps_ord['PROFILE_NAME'].tolist()

    def _bar_layout(fig):
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f9fafb',
                          margin=dict(t=50, b=30))
        return fig

    fig_hist  = _bar_layout(px.histogram(
        df, x=TARGET, nbins=40, title='Distribución actual del puntaje global'))
    fig_comp  = _bar_layout(px.bar(
        ps_ord, x='PROFILE_NAME', y='CURRENT_PROPORTION_PCT',
        text=ps_ord['CURRENT_PROPORTION_PCT'].round(1),
        title='Composición actual por perfil',
        labels={'PROFILE_NAME': 'Perfil', 'CURRENT_PROPORTION_PCT': 'Proporción (%)'},
        category_orders={'PROFILE_NAME': po}))
    fig_comp.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_score = _bar_layout(px.bar(
        ps_ord, x='PROFILE_NAME', y='PRED_MEAN_SCORE',
        text=ps_ord['PRED_MEAN_SCORE'].round(1),
        title='Puntaje predicho promedio por perfil',
        labels={'PROFILE_NAME': 'Perfil', 'PRED_MEAN_SCORE': 'Puntaje predicho'},
        category_orders={'PROFILE_NAME': po}))
    fig_score.update_traces(texttemplate='%{text:.1f}', textposition='outside')

    return html.Div([
        tab_header(
            '¿Cómo está compuesta actualmente la población estudiantil?',
            'Distribución de perfiles, puntaje global observado y desempeño esperado por perfil.'
        ),
        html.Div([
            kpi_card('Estudiantes analizados',   f"{len(df):,.0f}"),
            kpi_card('Puntaje promedio real',     f"{df[TARGET].mean():,.1f}"),
            kpi_card('Desviación estándar',       f"{df[TARGET].std():,.1f}"),
            kpi_card('Puntaje promedio predicho', f"{df['PRED_PUNT_GLOBAL'].mean():,.1f}"),
        ], className='kpi-grid'),
        html.Div([chart_card(fig_hist), chart_card(fig_comp), chart_card(fig_score)],
                 className='chart-grid'),
        html.Div([
            html.H3('Resumen de perfiles', className='section-title'),
            dash_table.DataTable(
                columns=[
                    {'name': 'Perfil',                    'id': 'PROFILE_NAME'},
                    {'name': 'Descripción dominante',     'id': 'DESCRIPTION'},
                    {'name': 'Estudiantes',               'id': 'N_STUDENTS',             'type': 'numeric'},
                    {'name': 'Proporción actual (%)',     'id': 'CURRENT_PROPORTION_PCT', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Puntaje real promedio',     'id': 'ACTUAL_MEAN_SCORE',      'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Puntaje predicho promedio', 'id': 'PRED_MEAN_SCORE',        'type': 'numeric', 'format': {'specifier': '.1f'}},
                ],
                data=ps_ord.to_dict('records'), page_size=10, **_table_style(),
            ),
        ], className='section'),
    ])


def simulation_tab():
    _, ps, _ = build_profile_data(prediction=False)
    sliders  = []
    for _, row in ps.iterrows():
        pid = int(row['PROFILE_ID'])
        sliders.append(html.Div([
            html.Div([
                html.Strong(row['PROFILE_NAME'], className='slider-name'),
                html.Span(
                    f" · Actual: {row['CURRENT_PROPORTION_PCT']:.1f}%"
                    f"  ·  Puntaje esperado: {row['PRED_MEAN_SCORE']:.1f}",
                    className='slider-stats'
                ),
            ], className='slider-header'),
            html.P(row['DESCRIPTION'], className='slider-desc'),
            dcc.Slider(
                id={'type': 'esc-profile-slider', 'profile_id': pid},
                min=0, max=100, step=1,
                value=round(row['CURRENT_PROPORTION_PCT'], 0),
                tooltip={'placement': 'bottom', 'always_visible': False},
                marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
            ),
        ], className='slider-row'))

    return html.Div([
        tab_header(
            '¿Qué escenario quieres simular?',
            'Ajusta la proporción de cada perfil para construir una composición hipotética.'
        ),
        html.Div([
            html.H3('¿Cómo funciona este simulador?', className='section-title'),
            html.Div([
                html.Div([
                    html.P('Los perfiles', className='explain-step-title'),
                    html.P(
                        'Los estudiantes fueron agrupados en 6 perfiles según sus '
                        'características de colegio y familia. Cada perfil tiene un '
                        'puntaje global esperado diferente.',
                        className='explain-step-body'
                    ),
                ], className='explain-step'),
                html.Div([
                    html.P('Las barras de proporción', className='explain-step-title'),
                    html.P(
                        'Cada barra indica qué porcentaje de la población hipotética '
                        'pertenecería a ese perfil. El valor inicial es la proporción '
                        'real en los datos actuales.',
                        className='explain-step-body'
                    ),
                ], className='explain-step'),
                html.Div([
                    html.P('El cálculo del resultado', className='explain-step-title'),
                    html.P(
                        'Puntaje simulado = Σ (proporción del perfil × puntaje promedio '
                        'del perfil). Si aumentas perfiles con puntaje alto, '
                        'el promedio esperado sube.',
                        className='explain-step-body'
                    ),
                ], className='explain-step'),
            ], className='explain-grid'),
        ], className='section'),
        html.Div([
            html.H3('Construcción del escenario hipotético', className='section-title'),
            html.P(
                'Si las proporciones no suman 100%, el simulador las normalizará automáticamente.',
                className='tab-subtitle'
            ),
            html.Div(sliders),
            html.Div(id='esc-scenario-warning', className='scenario-warning'),
            html.Div([
                html.Button('Simular escenario', id='esc-run-btn',
                            n_clicks=0, className='btn-primary'),
                html.Button('Restaurar escenario actual', id='esc-reset-btn',
                            n_clicks=0, className='btn-secondary'),
            ], className='btn-group'),
        ], className='section'),
    ])


def results_tab(scenario_data):
    if scenario_data is None:
        return html.Div([
            html.H3('Aún no has simulado un escenario', className='section-title'),
            html.P("Ve a la pestaña 'Construcción del escenario', ajusta las proporciones "
                   "y presiona 'Simular escenario'.", className='tab-subtitle'),
        ], className='section')

    _, ps, _ = build_profile_data(prediction=False)
    curr = scenario_data['current_score']
    sim  = scenario_data['scenario_score']
    unc  = scenario_data['uncertainty']
    sw   = np.array(scenario_data['scenario_weights'])
    cw   = np.array(scenario_data['current_weights'])
    cont = np.array(scenario_data['contributions'])

    def _layout(fig):
        fig.update_layout(paper_bgcolor='white', plot_bgcolor='#f9fafb',
                          margin=dict(t=50, b=30))
        return fig

    fig_comp = _layout(px.bar(
        pd.DataFrame({'Escenario': ['Actual', 'Simulado'], 'Puntaje': [curr, sim]}),
        x='Escenario', y='Puntaje', text=[round(curr, 1), round(sim, 1)],
        title='Puntaje esperado: actual vs simulado',
        color='Escenario',
        color_discrete_map={'Actual': '#2980b9', 'Simulado': '#27ae60'}))
    fig_comp.update_layout(showlegend=False)

    wf_df = ps.copy(); wf_df['CONTRIBUTION'] = cont
    fig_wf = go.Figure(go.Waterfall(
        name='Cambio', orientation='v',
        measure=['absolute'] + ['relative'] * len(wf_df) + ['total'],
        x=['Puntaje actual'] + wf_df['PROFILE_NAME'].tolist() + ['Puntaje simulado'],
        y=[curr] + wf_df['CONTRIBUTION'].tolist() + [0],
        text=[f'{curr:.1f}'] + [f'{x:+.1f}' for x in wf_df['CONTRIBUTION']] + [f'{sim:.1f}'],
        textposition='outside',
        connector={'line': {'color': '#bdc3c7'}},
        increasing={'marker': {'color': '#27ae60'}},
        decreasing={'marker': {'color': '#e74c3c'}},
        totals={'marker':    {'color': '#2980b9'}},
    ))
    fig_wf.update_layout(title='Explicación del cambio estimado por perfil',
                         yaxis_title='PUNT_GLOBAL',
                         paper_bgcolor='white', plot_bgcolor='#f9fafb',
                         margin=dict(t=50, b=30))

    scen_df = ps.copy()
    scen_df['CURRENT_PCT']  = cw * 100
    scen_df['SCENARIO_PCT'] = sw * 100
    scen_df['DELTA_PCT']    = scen_df['SCENARIO_PCT'] - scen_df['CURRENT_PCT']
    scen_df['CONTRIBUTION'] = cont

    fig_pc = _layout(px.bar(
        scen_df, x='PROFILE_NAME', y=['CURRENT_PCT', 'SCENARIO_PCT'],
        barmode='group', title='Composición actual vs simulada',
        labels={'value': 'Proporción (%)', 'PROFILE_NAME': 'Perfil',
                'variable': 'Escenario'}))
    fig_unc = _layout(px.histogram(
        pd.DataFrame({'Puntaje promedio simulado': unc['distribution']}),
        x='Puntaje promedio simulado', nbins=30,
        title='Distribución bootstrap del puntaje promedio simulado'))

    return html.Div([
        tab_header(
            '¿Qué resultados produciría el escenario simulado?',
            'Compara el puntaje actual vs el simulado e identifica qué perfiles explican el cambio.'
        ),
        html.Div([
            kpi_card('Puntaje esperado actual',   f'{curr:,.1f}'),
            kpi_card('Puntaje esperado simulado', f'{sim:,.1f}'),
            kpi_card('Cambio estimado',           f'{scenario_data["delta"]:+.1f} puntos'),
            kpi_card('Rango esperado',
                     f"{unc['p05']:.1f} – {unc['p95']:.1f}", 'Percentiles 5% y 95%'),
        ], className='kpi-grid'),
        html.Div([chart_card(fig_comp), chart_card(fig_pc),
                  chart_card(fig_wf),   chart_card(fig_unc)],
                 className='chart-grid'),
        html.Div([
            html.H3('Detalle del escenario por perfil', className='section-title'),
            dash_table.DataTable(
                columns=[
                    {'name': 'Perfil',           'id': 'PROFILE_NAME'},
                    {'name': 'Descripción',      'id': 'DESCRIPTION'},
                    {'name': 'Actual (%)',        'id': 'CURRENT_PCT',   'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Simulado (%)',      'id': 'SCENARIO_PCT',  'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Cambio p.p.',      'id': 'DELTA_PCT',     'type': 'numeric', 'format': {'specifier': '+.1f'}},
                    {'name': 'Puntaje esperado', 'id': 'PRED_MEAN_SCORE','type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Aporte al cambio', 'id': 'CONTRIBUTION',  'type': 'numeric', 'format': {'specifier': '+.2f'}},
                ],
                data=scen_df.to_dict('records'), page_size=10, **_table_style(),
            ),
        ], className='section'),
    ])


def segmentation_tab(scenario_data):
    df, ps, _ = build_profile_data(prediction=False)
    sw = (np.array(scenario_data['scenario_weights']) if scenario_data
          else ps['CURRENT_PROPORTION'].values)

    seg = ps.copy()
    seg['SCENARIO_PROPORTION_PCT'] = sw * 100
    seg['EXPECTED_STUDENTS']       = sw * len(df)
    seg['RISK_LEVEL'] = pd.qcut(seg['PRED_MEAN_SCORE'], q=3,
                                 labels=['Alto riesgo', 'Riesgo medio', 'Menor riesgo'])
    cmap = {'Alto riesgo': '#e74c3c', 'Riesgo medio': '#f39c12', 'Menor riesgo': '#27ae60'}

    fig_scatter = px.scatter(
        seg, x='EXPECTED_STUDENTS', y='PRED_MEAN_SCORE',
        size='SCENARIO_PROPORTION_PCT', color='RISK_LEVEL', color_discrete_map=cmap,
        hover_name='PROFILE_NAME',
        hover_data=['DESCRIPTION', 'SCENARIO_PROPORTION_PCT'],
        title='Mapa de priorización: tamaño del perfil vs puntaje esperado',
        labels={'EXPECTED_STUDENTS': 'Estudiantes esperados',
                'PRED_MEAN_SCORE': 'Puntaje esperado promedio',
                'RISK_LEVEL': 'Nivel de riesgo'},
    )
    fig_scatter.update_layout(paper_bgcolor='white', plot_bgcolor='#f9fafb',
                              margin=dict(t=50, b=30))

    seg_sorted = seg.sort_values('PRED_MEAN_SCORE')
    fig_rank = px.bar(
        seg_sorted, x='PRED_MEAN_SCORE', y='PROFILE_NAME', orientation='h',
        text=seg_sorted['PRED_MEAN_SCORE'].round(1),
        color='RISK_LEVEL', color_discrete_map=cmap,
        title='Ranking de perfiles por puntaje esperado',
        labels={'PRED_MEAN_SCORE': 'Puntaje esperado', 'PROFILE_NAME': 'Perfil'},
    )
    fig_rank.update_layout(paper_bgcolor='white', plot_bgcolor='#f9fafb',
                           showlegend=False, margin=dict(t=50, b=30))

    return html.Div([
        tab_header('¿Qué perfiles deberían priorizarse?',
                   'Identifica perfiles con bajo desempeño esperado y alto tamaño relativo.'),
        html.Div([chart_card(fig_scatter), chart_card(fig_rank)],
                 className='chart-grid'),
        html.Div([
            html.H3('Tabla de segmentos', className='section-title'),
            dash_table.DataTable(
                columns=[
                    {'name': 'Perfil',                   'id': 'PROFILE_NAME'},
                    {'name': 'Descripción',              'id': 'DESCRIPTION'},
                    {'name': 'Proporción escenario (%)', 'id': 'SCENARIO_PROPORTION_PCT', 'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Estudiantes esperados',    'id': 'EXPECTED_STUDENTS',       'type': 'numeric', 'format': {'specifier': '.0f'}},
                    {'name': 'Puntaje esperado',         'id': 'PRED_MEAN_SCORE',         'type': 'numeric', 'format': {'specifier': '.1f'}},
                    {'name': 'Nivel de riesgo',          'id': 'RISK_LEVEL'},
                ],
                data=seg.to_dict('records'), page_size=10, **_table_style(),
            ),
        ], className='section'),
    ])


# ── Layout ────────────────────────────────────────────────────────────────────
layout = html.Div([

    html.Div([
        html.Div([
            html.Span('Secretaría de Educación de Santander', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.Span('Modelo 3 — Simulación de Escenarios', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.A('← Panel principal', href='/', className='back-link'),
        ], className='header-badges'),
        html.H1('Simulación de Escenarios Educativos', className='header-title'),
        html.H3(
            '¿Qué pasaría si cambia la composición social de la población estudiantil?',
            className='header-subtitle'
        ),
        html.P(
            'Simula escenarios hipotéticos de desempeño usando perfiles poblacionales '
            'reales y un modelo predictivo entrenado sobre resultados históricos de '
            'las pruebas Saber 11 de Santander.',
            className='header-desc'
        ),
    ], className='header'),

    html.Div([
        html.Div(id='esc-main-content', children=[
            html.Div([
                html.Div([
                    html.H3('Simulador de escenarios', className='landing-card-title'),
                    html.P(
                        'Explora cómo cambiaría el puntaje global esperado si la '
                        'distribución de perfiles socioeconómicos de los estudiantes '
                        'fuera diferente.',
                        className='landing-card-desc'
                    ),
                    html.Div(
                        html.Button('Abrir simulador', id='esc-open-btn',
                                    n_clicks=0, className='btn-primary'),
                        className='btn-container'
                    ),
                ], className='landing-card'),
            ], className='landing-center'),
        ]),
    ], className='content-wrapper'),
])


def simulator_view():
    return html.Div([
        dcc.Store(id='esc-scenario-store'),
        dcc.Tabs(
            id='esc-tabs',
            value='tab-diagnosis',
            className='tabs-wrapper',
            children=[
                dcc.Tab(label='1. Diagnóstico actual',         value='tab-diagnosis',
                        className='tab', selected_className='tab-selected'),
                dcc.Tab(label='2. Construcción del escenario', value='tab-simulation',
                        className='tab', selected_className='tab-selected'),
                dcc.Tab(label='3. Resultados',                 value='tab-results',
                        className='tab', selected_className='tab-selected'),
                dcc.Tab(label='4. Segmentación',               value='tab-segmentation',
                        className='tab', selected_className='tab-selected'),
            ],
        ),
        html.Div(id='esc-tab-content', className='tab-content'),
    ])


# ── Callbacks ─────────────────────────────────────────────────────────────────
@callback(
    Output('esc-main-content', 'children'),
    Input('esc-open-btn', 'n_clicks'),
    prevent_initial_call=True,
)
def open_simulator(n):
    return simulator_view()


@callback(
    Output('esc-tab-content', 'children'),
    Input('esc-tabs', 'value'),
    State('esc-scenario-store', 'data'),
)
def render_tab(tab, scenario_data):
    if tab == 'tab-diagnosis':    return diagnosis_tab()
    if tab == 'tab-simulation':   return simulation_tab()
    if tab == 'tab-results':      return results_tab(scenario_data)
    if tab == 'tab-segmentation': return segmentation_tab(scenario_data)
    return diagnosis_tab()


@callback(
    Output('esc-scenario-warning', 'children'),
    Input({'type': 'esc-profile-slider', 'profile_id': dash.ALL}, 'value'),
    prevent_initial_call=True,
)
def update_warning(values):
    total = sum(values)
    if abs(total - 100) <= 0.5:
        return html.Span(f'La composición suma {total:.1f}%.', className='warning-ok')
    return html.Span(
        f'La composición suma {total:.1f}%. Se normalizará automáticamente al simular.',
        className='warning-alert'
    )


@callback(
    Output({'type': 'esc-profile-slider', 'profile_id': dash.ALL}, 'value'),
    Input('esc-reset-btn', 'n_clicks'),
    State({'type': 'esc-profile-slider', 'profile_id': dash.ALL}, 'id'),
    prevent_initial_call=True,
)
def reset_sliders(n, slider_ids):
    _, ps, _ = build_profile_data(prediction=False)
    pm = {int(r['PROFILE_ID']): round(float(r['CURRENT_PROPORTION_PCT']), 0)
          for _, r in ps.iterrows()}
    return [pm[int(s['profile_id'])] for s in slider_ids]


@callback(
    Output('esc-scenario-store', 'data'),
    Output('esc-tabs', 'value'),
    Input('esc-run-btn', 'n_clicks'),
    State({'type': 'esc-profile-slider', 'profile_id': dash.ALL}, 'value'),
    State({'type': 'esc-profile-slider', 'profile_id': dash.ALL}, 'id'),
    prevent_initial_call=True,
)
def run_simulation(n, values, slider_ids):
    df, ps, _ = build_profile_data(prediction=True)
    slider_df = pd.DataFrame({'PROFILE_ID': [int(x['profile_id']) for x in slider_ids],
                               'VALUE': values})
    ps_m   = ps.merge(slider_df, on='PROFILE_ID', how='left')
    sw     = normalize_weights(ps_m['VALUE'].values)
    result = simulate_scenario(ps_m, sw)
    unc    = bootstrap_uncertainty(df, ps_m, sw)
    return {
        'current_score':    float(result['current_score']),
        'scenario_score':   float(result['scenario_score']),
        'delta':            float(result['delta']),
        'current_weights':  result['current_weights'].tolist(),
        'scenario_weights': result['scenario_weights'].tolist(),
        'profile_scores':   result['profile_scores'].tolist(),
        'contributions':    result['contributions'].tolist(),
        'uncertainty':      unc,
    }, 'tab-results'
