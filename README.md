---
title: CalistenIA
emoji: 🤸
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
---

# CalistenIA

Reconoce progresiones de **planche** y **front lever** en fotos y vídeos, y puntúa la forma.

**Cómo funciona**
1. **Esqueleto:** RTMPose (rtmlib) detecta 133 puntos del cuerpo. En vídeo se sigue a la persona entre fotogramas para ir ~8 veces más rápido.
2. **Qué skill es:** un Random Forest sobre 15 rasgos geométricos del esqueleto (ángulos, distancias, apertura de piernas...), combinado con DINOv2 afinado sobre las imágenes.
3. **Nota de forma:** reglas geométricas (codos bloqueados, espalda horizontal, línea hombro-cadera-pies) y un clasificador de protracción entrenado con etiquetas propias.
4. **Vídeo:** detecta los holds, mide su duración y la forma segundo a segundo.

**Validación:** todas las métricas con validación cruzada agrupada (los fotogramas de un mismo vídeo nunca están a la vez en entrenamiento y en examen).

Los vídeos y fotos se procesan al momento y no se guardan.
