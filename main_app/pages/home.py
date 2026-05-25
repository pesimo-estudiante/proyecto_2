import dash
from dash import html

dash.register_page(__name__, path='/', name='Panel Principal', order=0)


def metric_chip(label, value):
    return html.Div([
        html.Span(label, className='chip-label'),
        html.Span(value,  className='chip-value'),
    ], className='chip')


def model_card(badge, title, subtitle, description, chips, href, accent='#2980b9'):
    return html.Div([
        html.Div(
            html.Span(badge, className='model-badge'),
            className='model-card-top'
        ),
        html.H2(title,       className='model-card-title'),
        html.P(subtitle,     className='model-card-subtitle'),
        html.P(description,  className='model-card-desc'),
        html.Div(chips,      className='model-card-chips'),
        html.A('Abrir dashboard', href=href, className='btn-model'),
    ], className='model-card', style={'borderTopColor': accent})


layout = html.Div([

    html.Div([
        html.Div([
            html.Span('Secretaría de Educación de Santander', className='header-tag'),
            html.Span(' · ', className='header-sep'),
            html.Span('Analítica Computacional — Proyecto 2',  className='header-tag'),
        ], className='header-badges'),
        html.H1('Panel de Análisis Educativo Saber 11', className='header-title'),
        html.H3(
            'Modelos predictivos sobre resultados de estudiantes de Santander',
            className='header-subtitle'
        ),
        html.P(
            'Tres modelos de inteligencia artificial entrenados con datos históricos '
            'de las pruebas Saber 11 para analizar el desempeño estudiantil y simular '
            'escenarios de política educativa.',
            className='header-desc'
        ),
    ], className='header'),

    html.Div([

        html.Div([

            model_card(
                badge='Modelo 1 · Clasificación binaria',
                title='Desempeño en Inglés',
                subtitle='¿El estudiante supera el nivel A- en inglés?',
                description=(
                    'Red neuronal que predice si un estudiante logrará superar el nivel A- '
                    'en inglés en las pruebas Saber 11, a partir de sus características '
                    'socioeconómicas y de colegio.'
                ),
                chips=[
                    metric_chip('Accuracy', '68.4 %'),
                    metric_chip('ROC-AUC',  '75.5 %'),
                    metric_chip('F1-score', '70.7 %'),
                ],
                href='/ingles',
                accent='#2980b9',
            ),

            model_card(
                badge='Modelo 2 · Clasificación binaria',
                title='Rendimiento Global',
                subtitle='¿El estudiante tiene rendimiento Alto o Bajo?',
                description=(
                    'Red neuronal que clasifica el rendimiento global del estudiante en '
                    'Alto (puntaje ≥ 300) o Bajo usando variables del entorno familiar '
                    'y del colegio.'
                ),
                chips=[
                    metric_chip('Accuracy',      '71.6 %'),
                    metric_chip('ROC-AUC',        '77.9 %'),
                    metric_chip('Arquitectura',   '3 capas'),
                ],
                href='/rendimiento',
                accent='#8e44ad',
            ),

            model_card(
                badge='Modelo 3 · Simulación de escenarios',
                title='Simulación de Escenarios',
                subtitle='¿Qué pasaría si cambia la composición social?',
                description=(
                    'Simulador que explora cómo cambiaría el puntaje global esperado '
                    'si la distribución de perfiles socioeconómicos fuera diferente, '
                    'con estimación de incertidumbre.'
                ),
                chips=[
                    metric_chip('Perfiles',  '6 grupos'),
                    metric_chip('Modelo',    'Regresión'),
                    metric_chip('Variables', 'Socioecon.'),
                ],
                href='/escenarios',
                accent='#16a085',
            ),

        ], className='models-grid'),

        html.Div([
            html.P(
                'Selecciona un modelo para comenzar. Todos los modelos fueron entrenados '
                'con datos históricos de las pruebas Saber 11 del departamento de Santander.',
                className='disclaimer-text'
            ),
        ], className='disclaimer'),

    ], className='content-wrapper'),

])
