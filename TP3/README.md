# CV1 — TP3: Detección del logo Coca-Cola por Template Matching

Detección del logotipo en las imágenes provistas a partir de un único template, con
matching multi-escala y **doble polaridad** (NCC), **gate de color por polaridad**
(prior del dominio: blanco-sobre-rojo / oscuro-sobre-blanco) y validación de candidatos por
**HOG + LBP** (SIFT evaluado como verificación adicional).

## Pipeline
1. **Candidatos:** `cv2.matchTemplate` (`TM_CCOEFF_NORMED`) sobre una pirámide de
   escalas del template, con el template original y su negativo (el logo aparece
   blanco-sobre-rojo en las etiquetas, polaridad invertida respecto del template).
2. **NMS** por IoU para fusionar detecciones solapadas.
3. **Gate de color por polaridad:** candidato invertido → fracción de rojo ≥ 0.20;
   candidato normal → fracción de fondo blanco ≥ 0.30.
4. **Validación:** similitud HOG (coseno) + LBP (intersección de histogramas)
   contra el template → confianza combinada `0.5·NCC + 0.3·HOG + 0.2·LBP`.
5. Umbral de confianza → detecciones finales con su score sobre el bounding box.

Resultado en `coca_multi.png`: 16 logos detectados, 0 falsos positivos.

## Estructura
```
TP3/
├── TP3.ipynb
├── images/      # imágenes provistas por la cátedra
└── template/    # template del logo
```

## Cómo correr
```
pip install opencv-python numpy matplotlib
```
Abrir `TP3.ipynb` y ejecutar todas las celdas (las rutas son relativas a la raíz del repo).
