# TinyGPT: Implementación de Estrategias de Decodificación y Mixture of Experts

**Autor:** Joaquín González  
**Curso:** NLP II - CEIA

## Descripción del Proyecto

Este repositorio contiene la implementación de TinyGPT, un modelo transformer compacto diseñado para generación de texto. El proyecto se enfoca en dos áreas principales: estrategias de decodificación para inferencia y arquitectura Mixture of Experts (MoE).

## Contenido del Repositorio

### Archivos Principales
- `notebook.ipynb` - Notebook principal con implementación completa y análisis
- `trainer.py` - Clase para entrenamiento optimizado con AMP y logging
- `metrics_logger.py` - Sistema de logging y visualización de métricas
- `requirements.txt` - Dependencias del proyecto

### Características Implementadas

#### 1. Estrategias de Decodificación (`generateV2`)
- **Decodificación Greedy**: Selección determinística del token más probable
- **Muestreo por Temperatura**: Control de aleatoriedad en la generación (0.1-2.0)
- **Top-k Sampling**: Restricción a los k tokens más probables
- **Nucleus Sampling (Top-p)**: Muestreo dinámico basado en probabilidad acumulada
- **Combinaciones**: Soporte para usar múltiples técnicas simultáneamente

#### 2. Mixture of Experts (MoE)
- **Arquitectura**: 4 expertos especializados con routing inteligente
- **Eficiencia**: Solo 1 experto activo por token (25% de parámetros FFN utilizados)
- **Especialización**: Cada experto aprende patrones diferentes durante el entrenamiento
- **Integración**: Reemplazo modular de capas feed-forward estándar

#### 3. Optimizaciones
- **KV-Cache**: Inferencia eficiente para generación secuencial
- **Mixed Precision**: Entrenamiento con `torch.bfloat16` para mayor velocidad
- **CosineAnnealingLR**: Scheduler suave para mejor convergencia
- **Visualización de Atención**: Mapas de calor para análisis de comportamiento

## Resultados Observados

### Rendimiento del Modelo
- **TinyGPT Base**: Convergencia estable en ~15 épocas, tendencia al overfitting
- **TinyGPT MoE**: Ligera mejora en loss de validación, mayor diversidad en patrones de atención
- **Tiempo de Entrenamiento**: MoE requiere ~40% más tiempo debido a overhead de routing

### Calidad de Generación
Las diferentes estrategias de decodificación muestran características distintivas:

| Estrategia | Coherencia | Diversidad | Uso Recomendado |
|------------|------------|------------|-----------------|
| Greedy | Alta | Baja | Texto técnico, consistencia |
| Temperatura 0.5 | Alta | Media | Escritura formal |
| Temperatura 1.5 | Baja | Alta | Creatividad, brainstorming |
| Top-k (k=5) | Media | Media | Balance general |
| Nucleus (p=0.9) | Media | Alta | Narrativa creativa |
| Combinado | Alta | Alta | Mejor balance observado |

### Análisis de Atención
- **TinyGPT Base**: Patrones de atención más uniformes, enfoque en tokens adyacentes
- **TinyGPT MoE**: Mayor diversidad en patrones, especialización por cabeza de atención

## Arquitectura del Modelo

**Configuración Base:**
- Dimensiones de embedding: 64
- Cabezas de atención: 4
- Capas transformer: 2
- Tamaño de contexto: 32 tokens
- Vocabulario: ~65 caracteres únicos

**MoE Específico:**
- Expertos: 4 redes FFN especializadas
- Routing: 1 experto activo por token
- Capacidad: 4x parámetros FFN con costo computacional similar

## Uso

```python
# Generación básica
result = generate(model, "To be", max_new_tokens=50)

# Generación con estrategias específicas
result_greedy = generate_greedy(model, "To be", max_new_tokens=50)
result_creative = generate_nucleus(model, "To be", top_p=0.9, max_new_tokens=50)
result_balanced = generateV2(model, "To be", temperature=0.8, top_k=10, top_p=0.9)
```

## Conclusiones

1. **Estrategias de Decodificación**: La combinación de temperatura + top-k + top-p produce el mejor balance entre coherencia y creatividad
2. **MoE**: Mejora marginal en métricas con mayor costo computacional, pero patrones de atención más diversos
3. **Optimizaciones**: KV-cache y mixed precision son esenciales para escalabilidad
4. **Limitaciones**: El tamaño compacto del modelo limita la calidad absoluta del texto generado

## Requisitos

- Python 3.10+
- PyTorch 2.0+
- CUDA compatible (opcional, recomendado)

```bash
pip install -r requirements.txt
jupyter notebook
```

## Dataset

Entrenado en TinyShakespeare (100k caracteres) con tokenización a nivel de caracteres para simplificar el scope