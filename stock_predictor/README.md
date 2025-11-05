# Stock Predictor - Predicción de Precios de Acciones con PyTorch

Sistema completo de predicción de precios de acciones utilizando Redes Neuronales Recurrentes (LSTM, GRU) y Transformers implementados en PyTorch.

## Características

- **Múltiples arquitecturas de modelos**: LSTM, GRU y Transformer
- **Indicadores técnicos**: Moving Averages, MACD, RSI, Bollinger Bands, y más
- **Entrenamiento robusto**: Early stopping, gradient clipping, y guardado de checkpoints
- **Predicción operativa**: Predicciones para el próximo día o múltiples días
- **Backtesting**: Evaluación del rendimiento del modelo en datos históricos
- **Visualizaciones**: Gráficas de entrenamiento, predicciones y backtesting

## Estructura del Proyecto

```
stock_predictor/
├── data/               # Datos descargados y procesados
├── models/             # Modelos entrenados y scalers
├── logs/               # Gráficas y logs de entrenamiento
├── notebooks/          # Jupyter notebooks para análisis
├── __init__.py         # Inicialización del paquete
├── data_loader.py      # Módulo de carga de datos del mercado
├── preprocessing.py    # Preprocesamiento y feature engineering
├── model.py            # Arquitecturas de redes neuronales
├── train.py            # Script de entrenamiento
├── predict.py          # Script de predicción operativa
├── config.py           # Configuración del proyecto
├── requirements.txt    # Dependencias
└── README.md           # Este archivo
```

## Instalación

### 1. Clonar o descargar el proyecto

```bash
cd stock_predictor
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### Requisitos

- Python 3.8+
- PyTorch 2.0+
- CUDA (opcional, para entrenamiento en GPU)

## Uso Rápido

### 1. Entrenar un modelo

```bash
python train.py
```

Por defecto, esto:
- Descarga datos de Apple (AAPL) de los últimos 5 años
- Entrena un modelo LSTM
- Guarda el mejor modelo en `models/best_model.pth`
- Guarda el scaler en `models/scaler.pkl`
- Genera gráficas en `logs/`

### 2. Hacer predicciones

```bash
python predict.py
```

Esto:
- Carga el modelo entrenado
- Predice el precio del próximo día
- Genera predicciones para los próximos 7 días
- Realiza backtesting con los últimos 30 días
- Crea visualizaciones

## Uso Avanzado

### Entrenar con diferentes configuraciones

```python
from train import StockTrainer

# Crear entrenador con configuración personalizada
trainer = StockTrainer(
    model_type="lstm",      # 'lstm', 'gru', 'transformer'
    hidden_size=256,        # Tamaño de capa oculta
    num_layers=3,           # Número de capas
    dropout=0.3,            # Tasa de dropout
    learning_rate=0.0005,   # Tasa de aprendizaje
    batch_size=64,          # Tamaño del batch
    sequence_length=90      # Días históricos para entrada
)

# Preparar datos para una acción específica
train_loader, val_loader = trainer.prepare_data(
    ticker="TSLA",          # Tesla
    period="10y",           # 10 años de datos
    train_ratio=0.8         # 80% entrenamiento, 20% validación
)

# Entrenar
history = trainer.train(
    train_loader,
    val_loader,
    epochs=200,
    patience=20,
    save_dir="models"
)

# Guardar preprocesador
trainer.preprocessor.save_scaler("models/scaler.pkl")

# Visualizar resultados
trainer.plot_training_history(save_path="logs/training_history.png")
```

### Predicción personalizada

```python
from predict import StockPredictor

# Cargar predictor
predictor = StockPredictor(
    model_path="models/best_model.pth",
    scaler_path="models/scaler.pkl"
)

# Predecir próximo día
price, data = predictor.predict_next_day("TSLA")
print(f"Precio predicho: ${price:.2f}")

# Predecir múltiples días
predictions, data = predictor.predict_multiple_days(
    ticker="TSLA",
    days_ahead=14,
    period="2y"
)

# Backtesting
metrics = predictor.backtest(
    ticker="TSLA",
    test_days=60,
    period="3y"
)

print(f"RMSE: ${metrics['rmse']:.2f}")
print(f"Precisión de dirección: {metrics['direction_accuracy']:.2f}%")

# Visualizar predicciones
predictor.plot_prediction("TSLA", days_ahead=7, save_path="logs/tsla_prediction.png")
predictor.plot_backtest("TSLA", test_days=30, save_path="logs/tsla_backtest.png")
```

### Usar configuraciones predefinidas

```python
from config import get_quick_test_config, get_production_config
from train import StockTrainer

# Para pruebas rápidas (menos épocas, menos datos)
config = get_quick_test_config()

# Para producción (más épocas, más datos, modelo más grande)
config = get_production_config()

# Imprimir configuración
config.print_config()

# Usar en el entrenador
trainer = StockTrainer(
    model_type=config.model.model_type,
    hidden_size=config.model.hidden_size,
    num_layers=config.model.num_layers,
    dropout=config.model.dropout,
    learning_rate=config.training.learning_rate,
    batch_size=config.training.batch_size,
    sequence_length=config.preprocessing.sequence_length
)
```

## Modelos Disponibles

### 1. LSTM (Long Short-Term Memory)
- **Mejor para**: Series temporales con dependencias a largo plazo
- **Ventajas**: Excelente para capturar patrones temporales complejos
- **Desventajas**: Más lento de entrenar que GRU

```python
trainer = StockTrainer(model_type="lstm")
```

### 2. GRU (Gated Recurrent Unit)
- **Mejor para**: Alternativa más rápida a LSTM
- **Ventajas**: Entrenamiento más rápido, menos parámetros
- **Desventajas**: Puede ser menos preciso en secuencias muy largas

```python
trainer = StockTrainer(model_type="gru")
```

### 3. Transformer
- **Mejor para**: Capturar dependencias complejas con mecanismos de atención
- **Ventajas**: Puede capturar relaciones más complejas
- **Desventajas**: Requiere más datos y más tiempo de entrenamiento

```python
trainer = StockTrainer(model_type="transformer")
```

## Indicadores Técnicos Incluidos

El preprocesador calcula automáticamente los siguientes indicadores técnicos:

- **Moving Averages**: MA-7, MA-21, MA-50
- **Exponential Moving Averages**: EMA-12, EMA-26
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index
- **Bollinger Bands**: Upper, Middle, Lower
- **Volatility**: Desviación estándar móvil
- **ROC**: Rate of Change
- **Volume indicators**: Volume MA y Volume Ratio
- **Price changes**: 1-day y 5-day price changes

## Métricas de Evaluación

El modelo se evalúa usando:

- **MSE** (Mean Squared Error): Error cuadrático medio
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático medio
- **MAE** (Mean Absolute Error): Error absoluto medio
- **MAPE** (Mean Absolute Percentage Error): Error porcentual absoluto medio
- **Direction Accuracy**: Precisión en predecir la dirección del movimiento

## Ejemplos de Acciones Populares

```python
# Tecnología
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "META"]

# Índices
tickers = ["^GSPC", "^DJI", "^IXIC"]  # S&P 500, Dow Jones, NASDAQ

# Criptomonedas (mediante Yahoo Finance)
tickers = ["BTC-USD", "ETH-USD"]
```

## Mejores Prácticas

### 1. Preparación de Datos
- Usa al menos 2-3 años de datos históricos
- Para modelos de producción, considera usar 5-10 años
- Ajusta `sequence_length` según la volatilidad de la acción

### 2. Entrenamiento
- Comienza con la configuración por defecto
- Usa early stopping para evitar overfitting
- Monitorea tanto train loss como validation loss
- Guarda múltiples checkpoints

### 3. Evaluación
- Siempre realiza backtesting antes de usar en producción
- No confíes solo en las métricas de error (RMSE, MAE)
- La precisión de dirección es crítica para trading
- Prueba el modelo en diferentes condiciones de mercado

### 4. Producción
- Actualiza los datos regularmente
- Re-entrena el modelo periódicamente (mensual/trimestral)
- Monitorea el rendimiento en tiempo real
- Ten un plan de fallback si el modelo falla

## Advertencias Importantes

**Este proyecto es solo para fines educativos y de investigación.**

- La predicción del mercado de valores es extremadamente difícil
- Los rendimientos pasados no garantizan resultados futuros
- No uses este modelo para tomar decisiones de inversión reales sin consultar a profesionales
- Los mercados son influenciados por muchos factores no capturados en datos históricos
- Siempre existe riesgo de pérdida de capital

## Solución de Problemas

### Error: "No se pudieron descargar datos"
- Verifica tu conexión a internet
- Asegúrate de que el ticker es válido
- Algunos tickers requieren sufijos (ej: ".MX" para México)

### Error: "CUDA out of memory"
- Reduce el `batch_size`
- Reduce el `hidden_size` o `num_layers`
- Usa un modelo más pequeño (GRU en lugar de LSTM)

### El modelo no converge
- Ajusta el `learning_rate` (prueba valores más pequeños como 0.0001)
- Aumenta las épocas de entrenamiento
- Verifica que los datos estén correctamente normalizados
- Prueba con un modelo diferente

### Predicciones pobres
- Aumenta la cantidad de datos de entrenamiento
- Ajusta `sequence_length`
- Prueba diferentes arquitecturas de modelo
- Añade más indicadores técnicos personalizados

## Próximas Mejoras

- [ ] Soporte para múltiples acciones simultáneas
- [ ] Análisis de sentimiento de noticias
- [ ] Estrategias de trading automatizadas
- [ ] Dashboard web interactivo
- [ ] Soporte para datos intraday
- [ ] Ensemble de modelos
- [ ] Análisis de riesgo y portfolio optimization

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## Licencia

Este proyecto está bajo licencia MIT. Ver archivo LICENSE para más detalles.

## Referencias

- [PyTorch Documentation](https://pytorch.org/docs/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)

## Contacto

Para preguntas, sugerencias o problemas, por favor abre un issue en el repositorio.

---

**Disclaimer**: Este software se proporciona "tal cual", sin garantías de ningún tipo. El uso de este software para trading real es bajo tu propio riesgo.
