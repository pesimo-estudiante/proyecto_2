from pathlib import Path
import os
import numpy as np
import pandas as pd

from functools import lru_cache

import joblib
from tensorflow.keras.models import load_model

from sklearn.cluster import KMeans

import dash
from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURACIÓN GENERAL — compatible con Docker
# Override mediante variables de entorno:
#   MODELS_DIR → carpeta con modelo y preprocesador
#   DATA_DIR   → carpeta con data_samp.csv
# ============================================================
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get('MODELS_DIR', BASE_DIR))
DATA_DIR   = Path(os.environ.get('DATA_DIR',   BASE_DIR))

DATA_PATH         = DATA_DIR   / "data_samp.csv"
MODEL_PATH        = MODELS_DIR / "modelo_punt_global.keras"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

TARGET = "PUNT_GLOBAL"

N_PROFILES = 6
RANDOM_STATE = 42
N_BOOTSTRAPS = 300


# ============================================================
# CARGA DE DATOS Y MODELO
# ============================================================

@lru_cache(maxsize=1)
def load_assets():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET}' en la base.")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = load_model(MODEL_PATH)

    feature_cols = [col for col in df.columns if col != TARGET]

    return df, feature_cols, preprocessor, model


def transform_features(X, preprocessor):
    """
    Aplica el preprocesador y convierte a matriz densa si es necesario.
    """

    X_transformed = preprocessor.transform(X)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    return X_transformed


def predict_model(df_features, preprocessor, model):
    """
    Genera predicciones del modelo de red neuronal.
    """

    X_transformed = transform_features(df_features, preprocessor)
    y_pred = model.predict(X_transformed, verbose=0).ravel()

    return y_pred


# ============================================================
# CONSTRUCCIÓN DE PERFILES
# ============================================================

@lru_cache(maxsize=1)
def build_profile_data(prediction: bool= False):
    """
    Construye perfiles poblacionales a partir de las variables categóricas.
    """

    df_profiles, feature_cols, preprocessor, model = load_assets()
    X = df_profiles[feature_cols].copy()

    if prediction:
        y_pred = predict_model(X, preprocessor, model)
        df_profiles["PRED_PUNT_GLOBAL"] = y_pred
    else:
        df_profiles["PRED_PUNT_GLOBAL"] = df_profiles["PUNT_GLOBAL"]

    profile_summary = summarize_profiles(df_profiles, feature_cols)

    return df_profiles, profile_summary, feature_cols


def summarize_profiles(df_profiles, feature_cols):
    """
    Crea una tabla resumen de perfiles.
    """

    total_students = len(df_profiles)

    rows = []

    for profile_id, group in df_profiles.groupby("PROFILE_ID"):
        n = len(group)
        prop = n / total_students

        description = build_profile_description(group, feature_cols)

        rows.append({
            "PROFILE_ID": int(profile_id),
            "PROFILE_NAME": f"Perfil {int(profile_id) + 1}",
            "DESCRIPTION": description,
            "N_STUDENTS": n,
            "CURRENT_PROPORTION": prop,
            "CURRENT_PROPORTION_PCT": prop * 100,
            "ACTUAL_MEAN_SCORE": group[TARGET].mean(),
            "PRED_MEAN_SCORE": group["PRED_PUNT_GLOBAL"].mean(),
            "ACTUAL_STD_SCORE": group[TARGET].std()
        })

    profile_summary = pd.DataFrame(rows)

    profile_summary = profile_summary.sort_values(
        by="PRED_MEAN_SCORE",
        ascending=True
    ).reset_index(drop=True)

    profile_summary["PROFILE_ORDER"] = np.arange(1, len(profile_summary) + 1)

    return profile_summary


def build_profile_description(group, feature_cols, max_vars=4):
    """
    Construye una descripción simple del perfil usando las categorías más frecuentes.
    """

    parts = []

    for col in feature_cols[:max_vars]:
        mode_values = group[col].mode(dropna=True)

        if len(mode_values) > 0:
            parts.append(f"{col}: {mode_values.iloc[0]}")

    return " | ".join(parts)


# ============================================================
# SIMULACIÓN
# ============================================================

def normalize_weights(values):
    """
    Normaliza una lista de porcentajes para que sumen 1.
    """

    values = np.array(values, dtype=float)

    if values.sum() <= 0:
        values = np.ones_like(values)

    weights = values / values.sum()

    return weights


def simulate_scenario(profile_summary, scenario_weights):
    """
    Calcula el puntaje esperado bajo una nueva composición de perfiles.
    """

    current_weights = profile_summary["CURRENT_PROPORTION"].values
    profile_scores = profile_summary["PRED_MEAN_SCORE"].values

    current_score = np.sum(current_weights * profile_scores)
    scenario_score = np.sum(scenario_weights * profile_scores)

    delta = scenario_score - current_score

    contributions = (scenario_weights - current_weights) * profile_scores

    result = {
        "current_score": current_score,
        "scenario_score": scenario_score,
        "delta": delta,
        "current_weights": current_weights,
        "scenario_weights": scenario_weights,
        "profile_scores": profile_scores,
        "contributions": contributions
    }

    return result


def bootstrap_scenario_uncertainty(df_profiles, profile_summary, scenario_weights):
    """
    Estima incertidumbre mediante bootstrap.
    """

    n = len(df_profiles)

    stats_by_profile = (
        df_profiles
        .groupby("PROFILE_ID")["PRED_PUNT_GLOBAL"]
        .agg(["mean", "var", "count"])
        .reset_index()
    )

    stats_by_profile = profile_summary[["PROFILE_ID"]].merge(
        stats_by_profile,
        on="PROFILE_ID",
        how="left"
    )

    means = stats_by_profile["mean"].values
    variances = stats_by_profile["var"].fillna(0).values

    weights = np.array(scenario_weights, dtype=float)
    weights = weights / weights.sum()

    # Media esperada de la mezcla
    scenario_mean = np.sum(weights * means)

    # E[X^2] de cada perfil = Var(X) + E[X]^2
    expected_x2 = np.sum(weights * (variances + means ** 2))

    # Varianza total de la mezcla
    mixture_variance = expected_x2 - scenario_mean ** 2

    # Varianza de la media muestral
    standard_error = np.sqrt(mixture_variance / n)

    # Aproximación normal para percentiles 5, 50 y 95
    p05 = scenario_mean - 1.645 * standard_error
    p50 = scenario_mean
    p95 = scenario_mean + 1.645 * standard_error

    # Distribución simulada ligera para graficar histograma
    rng = np.random.default_rng(RANDOM_STATE)

    distribution = rng.normal(
        loc=scenario_mean,
        scale=standard_error,
        size=N_BOOTSTRAPS
    )

    return {
        "mean": float(scenario_mean),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "distribution": distribution.tolist()
    }


# ============================================================
# ESTILOS
# ============================================================

CARD_STYLE = {
    "padding": "18px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "14px",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.05)",
    "backgroundColor": "white"
}

PAGE_STYLE = {
    "fontFamily": "Arial, sans-serif",
    "backgroundColor": "#f9fafb",
    "padding": "24px"
}

SECTION_STYLE = {
    "marginTop": "24px",
    "marginBottom": "24px"
}

HEADER_CONTAINER_STYLE = {
    "textAlign": "center",
    "width": "100%",
    "maxWidth": "1400px",
    "margin": "0 auto 28px auto",
}

HEADER_TITLE_STYLE = {
    "fontSize": "30px",
    "fontWeight": "700",
    "marginBottom": "8px",
}

HEADER_SUBTITLE_STYLE = {
    "fontSize": "16px",
    "color": "#4b5563",
    "maxWidth": "1200px",
    "margin": "0 auto",
    "lineHeight": "1.4",
}

# ============================================================
# COMPONENTES VISUALES
# ============================================================


def page_header(title, subtitle=None):
    return html.Div(
        [
            html.H1(
                title,
                style=HEADER_TITLE_STYLE
            ),
            html.P(
                subtitle,
                style=HEADER_SUBTITLE_STYLE
            ) if subtitle else None
        ],
        style=HEADER_CONTAINER_STYLE
    )


def kpi_card(title, value, subtitle=None):
    return html.Div(
        [
            html.Div(title, style={"fontSize": "14px", "color": "#6b7280"}),
            html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "marginTop": "6px"}),
            html.Div(subtitle or "", style={"fontSize": "13px", "color": "#6b7280", "marginTop": "4px"})
        ],
        style=CARD_STYLE
    )


def landing_layout():
    return html.Div(
        [
            page_header(
                title="¿Qué pasaría si cambia la composición social de la población estudiantil?",
                subtitle=(
                    "Simula escenarios hipotéticos de desempeño educativo usando perfiles poblacionales reales "
                    "y un modelo predictivo entrenado sobre resultados históricos."
                )
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Simulador de escenarios"),
                            html.P(
                                "Explora cómo cambiaría el puntaje global esperado si cambia la proporción "
                                "de perfiles poblacionales.",
                                style={"color": "#4b5563"}
                            ),
                            html.Button(
                                "Abrir simulador",
                                id="open-simulator-btn",
                                n_clicks=0,
                                style={
                                    "padding": "12px 18px",
                                    "border": "none",
                                    "borderRadius": "10px",
                                    "backgroundColor": "#111827",
                                    "color": "white",
                                    "cursor": "pointer"
                                }
                            )
                        ],
                        style={**CARD_STYLE, "width": "360px"}
                    )
                ],
                style={
                    "display": "flex",
                    "marginTop": "28px",
                    "justifyContent": "center",
                    "alignItems": "center",
                    "width": "100%"
                }
            ),

            html.Div(id="main-content")
        ],
        style=PAGE_STYLE
    )


def simulator_layout():
    return html.Div(
        [
            page_header(
                title="¿Cómo cambiaría el desempeño esperado bajo un nuevo escenario?",
                subtitle=(
                    "El simulador modifica la composición de perfiles reales observados en la población. "
                    "Los resultados deben interpretarse como estimaciones predictivas, no como efectos causales directos."
                )
            ),

            dcc.Store(id="scenario-store"),

            dcc.Tabs(
                id="simulator-tabs",
                value="tab-diagnosis",
                children=[
                    dcc.Tab(label="1. Diagnóstico actual", value="tab-diagnosis"),
                    dcc.Tab(label="2. Construcción del escenario", value="tab-simulation"),
                    dcc.Tab(label="3. Resultados", value="tab-results"),
                    dcc.Tab(label="4. Segmentación", value="tab-segmentation"),
                ],
                style={"marginTop": "24px"}
            ),

            html.Div(
                id="simulator-tab-content",
                style={"marginTop": "24px"}
            )
        ]
    )


def diagnosis_tab():
    df_profiles, profile_summary, _ = build_profile_data(prediction= False)

    total_students = len(df_profiles)
    actual_mean = df_profiles[TARGET].mean()
    actual_std = df_profiles[TARGET].std()
    pred_mean = df_profiles["PRED_PUNT_GLOBAL"].mean()

    profile_summary_ordered = profile_summary.sort_values("PROFILE_ID")

    profile_order = profile_summary_ordered["PROFILE_NAME"].tolist()

    fig_hist = px.histogram(
        df_profiles,
        x=TARGET,
        nbins=40,
        title="Distribución actual del puntaje global"
    )

    fig_profiles = px.bar(
        profile_summary_ordered,
        x="PROFILE_NAME",
        y="CURRENT_PROPORTION_PCT",
        text=profile_summary_ordered["CURRENT_PROPORTION_PCT"].round(1),
        title="Composición actual por perfil poblacional",
        labels={
            "PROFILE_NAME": "Perfil",
            "CURRENT_PROPORTION_PCT": "Proporción actual (%)"
        },
        category_orders={
            "PROFILE_NAME": profile_order
        }
    )
    fig_profiles.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_profiles.update_layout(
        xaxis_title="Perfil",
        yaxis_title="Proporción actual (%)"
    )

    fig_profile_score = px.bar(
        profile_summary_ordered,
        x="PROFILE_NAME",
        y="PRED_MEAN_SCORE",
        text=profile_summary_ordered["PRED_MEAN_SCORE"].round(1),
        title="Puntaje predicho promedio por perfil",
        labels={
            "PROFILE_NAME": "Perfil",
            "PRED_MEAN_SCORE": "Puntaje predicho promedio"
        },
        category_orders={
            "PROFILE_NAME": profile_order
        }
    )

    fig_profile_score.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig_profile_score.update_layout(
        xaxis_title="Perfil",
        yaxis_title="Puntaje predicho promedio"
    )

    return html.Div(
        [
            
            page_header(
                title="¿Cómo está compuesta actualmente la población estudiantil?",
                subtitle=(
                    "Explora la distribución actual de perfiles poblacionales, el puntaje global observado "
                    "y el desempeño esperado por perfil."
                )
            ),

            html.Div(
                [
                    kpi_card("Estudiantes analizados", f"{total_students:,.0f}"),
                    kpi_card("Puntaje promedio real", f"{actual_mean:,.1f}"),
                    kpi_card("Desviación estándar", f"{actual_std:,.1f}"),
                    kpi_card("Puntaje promedio predicho", f"{pred_mean:,.1f}")
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "16px"
                }
            ),

            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_hist), style=CARD_STYLE),
                    html.Div(dcc.Graph(figure=fig_profiles), style=CARD_STYLE),
                    html.Div(dcc.Graph(figure=fig_profile_score), style=CARD_STYLE),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "20px",
                    "marginTop": "24px"
                }
            ),

            html.Div(
                [
                    html.H3("Resumen de perfiles"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Perfil", "id": "PROFILE_NAME"},
                            {"name": "Descripción dominante", "id": "DESCRIPTION"},
                            {"name": "Estudiantes", "id": "N_STUDENTS", "type": "numeric"},
                            {"name": "Proporción actual (%)", "id": "CURRENT_PROPORTION_PCT", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Puntaje real promedio", "id": "ACTUAL_MEAN_SCORE", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Puntaje predicho promedio", "id": "PRED_MEAN_SCORE", "type": "numeric", "format": {"specifier": ".1f"}},
                        ],
                        data=profile_summary_ordered.to_dict("records"),
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px",
                            "fontFamily": "Arial",
                            "fontSize": "13px"
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f3f4f6"
                        }
                    )
                ],
                style={**CARD_STYLE, "marginTop": "24px"}
            )
        ]
    )


def simulation_tab():
    _, profile_summary, _ = build_profile_data(prediction= False)

    slider_components = []

    for _, row in profile_summary.iterrows():
        profile_id = int(row["PROFILE_ID"])
        current_pct = float(row["CURRENT_PROPORTION_PCT"])

        slider_components.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Strong(row["PROFILE_NAME"]),
                            html.Span(
                                f" | Actual: {current_pct:.1f}% | Puntaje esperado: {row['PRED_MEAN_SCORE']:.1f}",
                                style={"color": "#6b7280", "marginLeft": "8px"}
                            )
                        ]
                    ),

                    html.Div(
                        row["DESCRIPTION"],
                        style={"fontSize": "13px", "color": "#6b7280", "marginTop": "4px"}
                    ),

                    dcc.Slider(
                        id={"type": "profile-slider", "profile_id": profile_id},
                        min=0,
                        max=100,
                        step=1,
                        value=round(current_pct, 0),
                        tooltip={"placement": "bottom", "always_visible": False},
                        marks={
                            0: "0%",
                            25: "25%",
                            50: "50%",
                            75: "75%",
                            100: "100%"
                        }
                    )
                ],
                style={
                    "padding": "16px",
                    "borderBottom": "1px solid #e5e7eb"
                }
            )
        )

    return html.Div(
        [
            page_header(
                title="¿Qué escenario quieres simular?",
                subtitle=(
                    "Ajusta la proporción de cada perfil poblacional para construir una composición hipotética "
                    "de la población estudiantil."
                )
            ),
            
            html.Div(
                [
                    html.H3("Construcción del escenario hipotético"),

                    html.P(
                        "Modifica la proporción de cada perfil poblacional. "
                        "Si las proporciones no suman 100%, el simulador las normalizará internamente "
                        "y mostrará el escenario equivalente.",
                        style={"color": "#4b5563"}
                    ),

                    html.Div(slider_components),

                    html.Div(
                        id="scenario-total-warning",
                        style={
                            "marginTop": "16px",
                            "fontWeight": "bold"
                        }
                    ),

                    html.Button(
                        "Simular escenario",
                        id="run-simulation-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "16px",
                            "padding": "12px 18px",
                            "border": "none",
                            "borderRadius": "10px",
                            "backgroundColor": "#111827",
                            "color": "white",
                            "cursor": "pointer"
                        }
                    ),

                    html.Button(
                        "Restaurar escenario actual",
                        id="reset-scenario-btn",
                        n_clicks=0,
                        style={
                            "marginTop": "16px",
                            "marginLeft": "12px",
                            "padding": "12px 18px",
                            "border": "1px solid #d1d5db",
                            "borderRadius": "10px",
                            "backgroundColor": "white",
                            "color": "#111827",
                            "cursor": "pointer"
                        }
                    )
                ],
                style=CARD_STYLE
            )
        ]
    )


def empty_results_message():
    return html.Div(
        [
            html.H3("Aún no has simulado un escenario"),
            html.P(
                "Ve a la pestaña 'Construcción del escenario', ajusta las proporciones "
                "y presiona 'Simular escenario'.",
                style={"color": "#6b7280"}
            )
        ],
        style=CARD_STYLE
    )


def results_tab(scenario_data):
    if scenario_data is None:
        return empty_results_message()

    _, profile_summary, _ = build_profile_data(prediction= False)

    current_score = scenario_data["current_score"]
    scenario_score = scenario_data["scenario_score"]
    delta = scenario_data["delta"]
    uncertainty = scenario_data["uncertainty"]

    scenario_weights = np.array(scenario_data["scenario_weights"])
    current_weights = np.array(scenario_data["current_weights"])
    contributions = np.array(scenario_data["contributions"])

    comparison_df = pd.DataFrame({
        "Escenario": ["Actual", "Simulado"],
        "Puntaje esperado": [current_score, scenario_score]
    })

    fig_comparison = px.bar(
        comparison_df,
        x="Escenario",
        y="Puntaje esperado",
        text=comparison_df["Puntaje esperado"].round(1),
        title="Puntaje esperado: actual vs simulado"
    )

    waterfall_df = profile_summary.copy()
    waterfall_df["CONTRIBUTION"] = contributions

    fig_waterfall = go.Figure(
        go.Waterfall(
            name="Cambio",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(waterfall_df) + ["total"],
            x=(
                ["Puntaje actual"]
                + waterfall_df["PROFILE_NAME"].tolist()
                + ["Puntaje simulado"]
            ),
            y=(
                [current_score]
                + waterfall_df["CONTRIBUTION"].tolist()
                + [0]
            ),
            text=[
                f"{current_score:.1f}"
            ] + [
                f"{x:+.1f}" for x in waterfall_df["CONTRIBUTION"]
            ] + [
                f"{scenario_score:.1f}"
            ],
            textposition="outside"
        )
    )

    fig_waterfall.update_layout(
        title="Explicación del cambio estimado por perfil",
        yaxis_title="Puntos de PUNT_GLOBAL"
    )

    distribution_df = pd.DataFrame({
        "Puntaje promedio simulado": uncertainty["distribution"]
    })

    fig_uncertainty = px.histogram(
        distribution_df,
        x="Puntaje promedio simulado",
        nbins=30,
        title="Distribución bootstrap del puntaje promedio simulado"
    )

    scenario_profile_df = profile_summary.copy()
    scenario_profile_df["CURRENT_PCT"] = current_weights * 100
    scenario_profile_df["SCENARIO_PCT"] = scenario_weights * 100
    scenario_profile_df["DELTA_PCT"] = scenario_profile_df["SCENARIO_PCT"] - scenario_profile_df["CURRENT_PCT"]

    fig_profile_change = px.bar(
        scenario_profile_df,
        x="PROFILE_NAME",
        y=["CURRENT_PCT", "SCENARIO_PCT"],
        barmode="group",
        title="Composición actual vs composición simulada",
        labels={
            "value": "Proporción (%)",
            "PROFILE_NAME": "Perfil",
            "variable": "Escenario"
        }
    )

    delta_label = f"{delta:+.1f} puntos"

    return html.Div(
        [
            page_header(
                title="¿Qué resultados produciría el escenario simulado?",
                subtitle=(
                    "Compara el puntaje esperado actual frente al escenario simulado e identifica qué perfiles "
                    "explican el cambio estimado."
                )
            ),
            
            html.Div(
                [
                    kpi_card("Puntaje esperado actual", f"{current_score:,.1f}"),
                    kpi_card("Puntaje esperado simulado", f"{scenario_score:,.1f}"),
                    kpi_card("Cambio estimado", delta_label),
                    kpi_card(
                        "Rango esperado",
                        f"{uncertainty['p05']:.1f} - {uncertainty['p95']:.1f}",
                        "Percentiles 5% y 95%"
                    )
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(4, 1fr)",
                    "gap": "16px"
                }
            ),

            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_comparison), style=CARD_STYLE),
                    html.Div(dcc.Graph(figure=fig_profile_change), style=CARD_STYLE),
                    html.Div(dcc.Graph(figure=fig_waterfall), style=CARD_STYLE),
                    html.Div(dcc.Graph(figure=fig_uncertainty), style=CARD_STYLE),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "20px",
                    "marginTop": "24px"
                }
            ),

            html.Div(
                [
                    html.H3("Detalle del escenario por perfil"),
                    dash_table.DataTable(
                        columns=[
                            {"name": "Perfil", "id": "PROFILE_NAME"},
                            {"name": "Descripción", "id": "DESCRIPTION"},
                            {"name": "Actual (%)", "id": "CURRENT_PCT", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Simulado (%)", "id": "SCENARIO_PCT", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Cambio p.p.", "id": "DELTA_PCT", "type": "numeric", "format": {"specifier": "+.1f"}},
                            {"name": "Puntaje esperado", "id": "PRED_MEAN_SCORE", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Aporte al cambio", "id": "CONTRIBUTION", "type": "numeric", "format": {"specifier": "+.2f"}},
                        ],
                        data=scenario_profile_df.to_dict("records"),
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px",
                            "fontFamily": "Arial",
                            "fontSize": "13px"
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f3f4f6"
                        }
                    )
                ],
                style={**CARD_STYLE, "marginTop": "24px"}
            )
        ]
    )


def segmentation_tab(scenario_data):
    df_profiles, profile_summary, _ = build_profile_data(prediction= False)

    if scenario_data is not None:
        scenario_weights = np.array(scenario_data["scenario_weights"])
    else:
        scenario_weights = profile_summary["CURRENT_PROPORTION"].values

    segment_df = profile_summary.copy()
    segment_df["SCENARIO_PROPORTION_PCT"] = scenario_weights * 100
    segment_df["EXPECTED_STUDENTS"] = scenario_weights * len(df_profiles)

    segment_df["RISK_LEVEL"] = pd.qcut(
        segment_df["PRED_MEAN_SCORE"],
        q=3,
        labels=["Alto riesgo", "Riesgo medio", "Menor riesgo"]
    )

    fig_risk = px.scatter(
        segment_df,
        x="EXPECTED_STUDENTS",
        y="PRED_MEAN_SCORE",
        size="SCENARIO_PROPORTION_PCT",
        color="RISK_LEVEL",
        hover_name="PROFILE_NAME",
        hover_data=["DESCRIPTION", "SCENARIO_PROPORTION_PCT"],
        title="Mapa de priorización: tamaño del perfil vs puntaje esperado",
        labels={
            "EXPECTED_STUDENTS": "Estudiantes esperados en el escenario",
            "PRED_MEAN_SCORE": "Puntaje esperado promedio",
            "RISK_LEVEL": "Nivel de riesgo"
        }
    )

    fig_ranking = px.bar(
        segment_df.sort_values("PRED_MEAN_SCORE", ascending=True),
        x="PRED_MEAN_SCORE",
        y="PROFILE_NAME",
        orientation="h",
        text=segment_df.sort_values("PRED_MEAN_SCORE", ascending=True)["PRED_MEAN_SCORE"].round(1),
        title="Ranking de perfiles por menor puntaje esperado",
        labels={
            "PRED_MEAN_SCORE": "Puntaje esperado promedio",
            "PROFILE_NAME": "Perfil"
        }
    )

    return html.Div(
        [
            page_header(
                title="¿Qué perfiles deberían priorizarse?",
                subtitle=(
                    "Identifica perfiles poblacionales con bajo desempeño esperado y alto tamaño relativo dentro "
                    "del escenario analizado."
                )
            ),
            
            html.Div(
                [
                    html.H3("Segmentación y priorización"),

                    html.P(
                        "Esta vista ayuda a identificar perfiles grandes con bajo puntaje esperado. "
                        "Puede usarse para orientar estrategias de diagnóstico, acompañamiento o focalización.",
                        style={"color": "#4b5563"}
                    )
                ],
                style=CARD_STYLE
            ),

            html.Div(
                [
                    html.Div(dcc.Graph(figure=fig_risk), style=CARD_STYLE),
                    html.Div(dcc.Graph(figure=fig_ranking), style=CARD_STYLE)
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr",
                    "gap": "20px",
                    "marginTop": "24px"
                }
            ),

            html.Div(
                [
                    html.H3("Tabla de segmentos"),

                    dash_table.DataTable(
                        columns=[
                            {"name": "Perfil", "id": "PROFILE_NAME"},
                            {"name": "Descripción", "id": "DESCRIPTION"},
                            {"name": "Proporción escenario (%)", "id": "SCENARIO_PROPORTION_PCT", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Estudiantes esperados", "id": "EXPECTED_STUDENTS", "type": "numeric", "format": {"specifier": ".0f"}},
                            {"name": "Puntaje esperado", "id": "PRED_MEAN_SCORE", "type": "numeric", "format": {"specifier": ".1f"}},
                            {"name": "Nivel de riesgo", "id": "RISK_LEVEL"},
                        ],
                        data=segment_df.to_dict("records"),
                        page_size=10,
                        style_table={"overflowX": "auto"},
                        style_cell={
                            "textAlign": "left",
                            "padding": "8px",
                            "fontFamily": "Arial",
                            "fontSize": "13px"
                        },
                        style_header={
                            "fontWeight": "bold",
                            "backgroundColor": "#f3f4f6"
                        }
                    )
                ],
                style={**CARD_STYLE, "marginTop": "24px"}
            )
        ]
    )


# ============================================================
# APP DASH
# ============================================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = landing_layout()


# ============================================================
# CALLBACK: ABRIR SIMULADOR
# ============================================================

@app.callback(
    Output("main-content", "children"),
    Input("open-simulator-btn", "n_clicks"),
    prevent_initial_call=True
)
def open_simulator(n_clicks):
    if n_clicks:
        return simulator_layout()

    return dash.no_update


# ============================================================
# CALLBACK: CONTENIDO DE TABS
# ============================================================

@app.callback(
    Output("simulator-tab-content", "children"),
    Input("simulator-tabs", "value"),
    State("scenario-store", "data"),
    # prevent_initial_call=True
)
def render_simulator_tab(tab, scenario_data):
    print(tab)
    if tab == "tab-diagnosis":
        return diagnosis_tab()

    if tab == "tab-simulation":
        return simulation_tab()

    if tab == "tab-results":
        return results_tab(scenario_data)

    if tab == "tab-segmentation":
        return segmentation_tab(scenario_data)

    return diagnosis_tab()


# ============================================================
# CALLBACK: AVISO DE SUMA DE PROPORCIONES
# ============================================================

@app.callback(
    Output("scenario-total-warning", "children"),
    Input({"type": "profile-slider", "profile_id": dash.ALL}, "value"),
    prevent_initial_call=True
)
def update_total_warning(values):
    total = sum(values)

    if abs(total - 100) <= 0.5:
        return html.Div(
            f"La composición suma {total:.1f}%.",
            style={"color": "#047857"}
        )

    return html.Div(
        f"La composición suma {total:.1f}%. Al simular, se normalizará para sumar 100%.",
        style={"color": "#b45309"}
    )


# ============================================================
# CALLBACK: RESTAURAR SLIDERS A COMPOSICIÓN ACTUAL
# ============================================================

@app.callback(
    Output({"type": "profile-slider", "profile_id": dash.ALL}, "value"),
    Input("reset-scenario-btn", "n_clicks"),
    State({"type": "profile-slider", "profile_id": dash.ALL}, "id"),
    prevent_initial_call=True
)
def reset_scenario(n_clicks, slider_ids):
    if not n_clicks:
        return dash.no_update

    _, profile_summary, _ = build_profile_data(prediction= False)

    profile_map = {
        int(row["PROFILE_ID"]): round(float(row["CURRENT_PROPORTION_PCT"]), 0)
        for _, row in profile_summary.iterrows()
    }

    return [
        profile_map[int(slider_id["profile_id"])]
        for slider_id in slider_ids
    ]


# ============================================================
# CALLBACK: EJECUTAR SIMULACIÓN
# ============================================================

@app.callback(
    Output("scenario-store", "data"),
    Output("simulator-tabs", "value"),
    Input("run-simulation-btn", "n_clicks"),
    State({"type": "profile-slider", "profile_id": dash.ALL}, "value"),
    State({"type": "profile-slider", "profile_id": dash.ALL}, "id"),
    prevent_initial_call=True
)
def run_simulation(n_clicks, values, slider_ids):
    if not n_clicks:
        return dash.no_update, dash.no_update

    df_profiles, profile_summary, _ = build_profile_data(prediction= True)

    slider_df = pd.DataFrame({
        "PROFILE_ID": [int(x["profile_id"]) for x in slider_ids],
        "VALUE": values
    })

    profile_summary_ordered = profile_summary.merge(
        slider_df,
        on="PROFILE_ID",
        how="left"
    )

    scenario_weights = normalize_weights(profile_summary_ordered["VALUE"].values)

    result = simulate_scenario(
        profile_summary=profile_summary_ordered,
        scenario_weights=scenario_weights
    )

    uncertainty = bootstrap_scenario_uncertainty(
        df_profiles=df_profiles,
        profile_summary=profile_summary_ordered,
        scenario_weights=scenario_weights
    )

    scenario_data = {
        "current_score": float(result["current_score"]),
        "scenario_score": float(result["scenario_score"]),
        "delta": float(result["delta"]),
        "current_weights": result["current_weights"].tolist(),
        "scenario_weights": result["scenario_weights"].tolist(),
        "profile_scores": result["profile_scores"].tolist(),
        "contributions": result["contributions"].tolist(),
        "uncertainty": uncertainty
    }

    return scenario_data, "tab-results"


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    port  = int(os.environ.get('PORT', 8052))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)