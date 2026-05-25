# Tablero de Predicción — Desempeño Mínimo en Inglés Saber 11

## Objetivo

Estimar si un estudiante del departamento de Santander superará o no el nivel
mínimo **A-** en inglés en las pruebas Saber 11, usando características
escolares, familiares y socioeconómicas.

**Variable objetivo:** `SUPERA_NIVEL_BAJO_INGLES`
- `0` → el estudiante permanece en nivel A-
- `1` → el estudiante supera el nivel A- (A1, A2, B1 o B+)

**Mejor modelo:** Red Neuronal Binaria — capas [256, 128, 64], ReLU,
dropout 0.3, lr 0.001

| Métrica   | Valor  |
|-----------|--------|
| Accuracy  | 0.6843 |
| Precision | 0.6994 |
| Recall    | 0.7151 |
| F1-score  | 0.7072 |
| ROC-AUC   | 0.7554 |

---

## Archivos necesarios

```
Proyecto_2/
├── models/
│   ├── modelo_binario_supera_nivel_bajo_ingles.keras   ← modelo entrenado
│   └── preprocessor_binario_ingles.pkl                 ← preprocesador
└── dashboard/
    ├── app.py
    ├── requirements.txt
    ├── Dockerfile
    ├── README.md
    └── assets/
        └── style.css
```

> El tablero carga los artefactos desde `../models/` relativo a `app.py`.
> No realiza ningún entrenamiento.

---

## Ejecución local

### 1. Instalar dependencias

```bash
cd dashboard
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación

```bash
python app.py
```

### 3. Abrir en el navegador

```
http://127.0.0.1:8050/
```

---

## Despliegue con Docker

El Dockerfile debe construirse desde la **raíz del proyecto** para incluir
la carpeta `models/`:

```bash
# Desde Proyecto_2/
docker build -f dashboard/Dockerfile -t saber11-ingles-dashboard .
docker run -p 8050:8050 saber11-ingles-dashboard
```

El contenedor queda disponible en `http://localhost:8050/`.

---

## Uso del tablero

1. Complete el formulario con las características del estudiante y su colegio.
2. Haga clic en **"Generar predicción"**.
3. El panel lateral mostrará:
   - Indicador gauge con la probabilidad estimada.
   - Resultado textual (supera / no supera).
   - Recomendación según el nivel de probabilidad.

> **Nota:** El resultado es una herramienta de apoyo para decisiones
> educativas, no una evaluación definitiva del estudiante.
