# PROPUESTA DE MEJORA OPERATIVA SIN MODIFICAR LOS MODELOS BASE

**SÍ, es absolutamente posible mejorar la precisión del sistema sin alterar los 3 modelos base ModernBERT ni su cabezal de clasificación de 41 clases.**

A continuación se detalla la justificación técnica de cómo optimizar el rendimiento y la fiabilidad del detector modificando el resto del pipeline:

### 1. Activación y Entrenamiento de la Fusión Tardía (Late Fusion)
Actualmente, el sistema descarta toda la información de los plugins secundarios en la toma de decisiones finales. Si se entrena formalmente la clase [FusionClassifier](file:///Users/user/Documents/xplagiax_xota/app/engine/fusion.py#L328) en [fusion.py](file:///Users/user/Documents/xplagiax_xota/app/engine/fusion.py) (utilizando un modelo lineal regularizado, SVM o un árbol de decisión ligero como XGBoost), se pueden combinar las salidas probabilísticas de ModernBERT con los descriptores de los otros plugins. Las señales complementarias (como la densidad de razonamiento CoT, el índice de citas inventadas o la consistencia estilométrica) corregirán los errores del clasificador neuronal cuando se enfrente a textos ambiguos o fuera de distribución.

### 2. Calibración Estadística Post-Hoc
Las probabilidades del softmax de ModernBERT en [classify_text](file:///Users/user/Documents/xplagiax_xota/app/engine/detector_final.py#L127) sufren de sobreconfianza (overconfidence), un problema típico de las redes neuronales profundas. Aplicar técnicas de calibración como **Platt Scaling** o **Isotonic Regression** sobre la salida agregada del ensamble ajustará los scores a probabilidades empíricas reales. Esto no altera las predicciones relativas, pero reduce drásticamente los falsos positivos (FPs) en zonas de alta incertidumbre, mejorando la precisión en el umbral operativo deseado (p. ej., exigir un 99% de precisión para etiquetar un texto como IA).

### 3. Aprovechamiento de Señales Agnósticas al Modelo (Model-Agnostic Signals)
Aunque el cabezal de 41 clases de ModernBERT solo reconozca explícitamente modelos hasta 2023, los LLMs más nuevos (como Gemini 2.0 o DeepSeek-R1) siguen compartiendo rasgos estructurales universales con sus predecesores (baja burstiness, uso característico de marcadores lógicos, regularidad sintáctica). Un meta-clasificador entrenado con descriptores estilométricos avanzados y perplejidad real detectará estos textos nuevos como IA basándose en estas características lingüísticas transversales, superando la limitación del "mundo cerrado" del clasificador supervisado.

### 4. Implementación de un Mecanismo de Abstención (Rechazo Activo)
Una gran parte de las predicciones incorrectas ocurren en textos muy cortos (donde la varianza estadística de las métricas es inestable) o en casos donde las predicciones de las 3 semillas de ModernBERT divergen significativamente. Calcular la desviación estándar del ensemble (`ensemble_disagreement`) y definir un umbral de longitud mínima (p. ej., < 150 palabras) permitirá al sistema abstenerse de emitir veredictos en casos de baja confianza. Al no clasificar los casos difíciles/ruidosos, la precisión del sistema sobre los textos validados aumentará significativamente.
