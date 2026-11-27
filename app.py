"""App de CalistenIA: sube un vídeo o una foto de planche / front lever y recibe el análisis."""
import os
import tempfile

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import calistenia as cal

st.set_page_config(page_title="CalistenIA", page_icon="🤸", layout="wide")

NOMBRES = {"tuck": "Tuck", "adv_tuck": "Advanced tuck", "straddle": "Straddle", "full": "Full",
           "planche": "planche", "fl": "front lever"}


def bonito(skill):
    if skill == "planche_lean":
        return "Planche lean"
    if skill and "_" in skill:
        piernas, familia = skill.rsplit("_", 1)
        if piernas in NOMBRES and familia in NOMBRES:
            return f"{NOMBRES[piernas]} {NOMBRES[familia]}"
    return (skill or "?").replace("_", " ").capitalize()


@st.cache_resource(show_spinner=False)
def iniciar():
    cal.cargar(os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelos"))
    return True


def cabecera_skill(r):
    if not r["seguro"]:
        st.warning(f"No reconozco esta skill con seguridad (lo más parecido: {bonito(r['skill'])}), así que no te pongo nota. "
                   "Prueba con un vídeo de lado, con el cuerpo entero y buena luz.")
    elif not r["soportada"]:
        st.info(f"Parece **{bonito(r['skill'])}**. De esta skill todavía no sé puntuar la forma.")


NOMBRE_CRITERIO = {
    "espalda no horizontal": "Espalda: inclinación",
    "línea hombro-cadera-pies rota": "Cuerpo: hombro-cadera-pies en línea",
    "codo doblado": "Codo: cuánto se dobla",
    "codo no a 90°": "Codo: distancia a 90°",
    "cabeza fuera de línea": "Cabeza: cuánto se sale de la línea",
    "rodilla doblada": "Rodilla: cuánto se dobla",
}

LEYENDA = """
**Cómo leer el dibujo** (cada número es un ángulo en grados):
- 🟢 **Verde, espalda** (hombro → cadera): debe ir **horizontal**. El número es su inclinación: **0° = perfecta**.
- 🟠 **Naranja, brazo** (hombro → codo → muñeca): el número es el ángulo del codo. **180° = brazo bloqueado** (en el 90 hold, 90°).
- 🔵 **Azul, líneas del cuerpo** (cabeza → hombro → cadera y hombro → cadera → pies): el número es el ángulo en cada articulación. **180° = todo en línea recta**.
"""

AYUDA = {
    "duracion": "Tiempo que aguantas quieto en la posición, sin contar la entrada y la salida.",
    "forma": "Nota de 0 a 100: cómo de cerca estás de la posición ideal en cada criterio (espalda, codo, línea del cuerpo, "
             "cabeza y, en planche, protracción), de media durante todo el hold.",
    "progresion": "Solo en advanced tuck: por dónde vas dentro de la progresión según lo que abres la cadera. "
                  "0 % = recién salido del tuck, 100 % = a punto de pasar a straddle.",
}

AYUDA_PROT = ("Si la parte alta de la espalda está redondeada (escápulas separadas). "
              "Se mide al principio, a la mitad y al final del hold; en las fotos de abajo, en ese momento concreto.")


def mostrar_consejos(lista):
    if lista:
        st.markdown("**Qué corregir**")
        for c in lista:
            st.markdown(f"- {c}")
    else:
        st.success("Forma limpia: nada importante que corregir.")


# ---------------------------------------------------------------------
st.title("🤸 CalistenIA")
st.caption("Sube un vídeo o una foto de tu **planche**, **front lever** o **90 hold**: te digo qué progresión es, "
           "cuánto aguantas y cómo de limpia es tu forma.")

with st.spinner("Cargando modelos (la primera vez tarda un poco)..."):
    iniciar()

with st.expander("📏 Cómo grabarte para que funcione bien"):
    st.markdown(
        "- **De lado**, con la cámara a la altura de la cadera.\n"
        "- **Cuerpo entero** dentro del encuadre, sin cortar manos ni pies.\n"
        "- **Buena luz**, sin contraluz.\n"
        f"- En vídeo se analizan los primeros **{cal.MAX_SEGUNDOS} segundos**.")

tab_video, tab_foto = st.tabs(["🎥 Vídeo", "📷 Foto"])

# ---------------------------------------------------------------------
with tab_video:
    archivo = st.file_uploader("Sube un vídeo", key="video")      # sin filtro: en Android abre el selector de Archivos
    if archivo and not archivo.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".3gp")):
        st.error("Eso no parece un vídeo. Sube un .mp4 o .mov.")
        archivo = None
    if archivo and st.button("Analizar vídeo", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(archivo.name)[1], delete=False) as tmp:
            tmp.write(archivo.read())
            ruta = tmp.name
        barra = st.progress(0.0, text="Analizando...")
        try:
            res = cal.analizar_video(ruta, progreso=lambda x: barra.progress(min(x, 1.0), text=f"Analizando... {x:.0%}"))
        finally:
            os.remove(ruta)                       # el vídeo no se guarda
        barra.empty()

        if not res["ok"]:
            st.error(res["mensaje"])
        elif not res["holds"]:
            st.warning("No encuentro ningún hold. Comprueba que se te vea entero y que aguantes al menos 1-2 segundos.")
        else:
            if res["recortado"]:
                st.info(f"El vídeo es largo: he analizado los primeros {cal.MAX_SEGUNDOS} segundos.")
            for n, r in enumerate(res["holds"], 1):
                st.divider()
                st.subheader(f"Hold {n}: {bonito(r['skill'])}")
                cabecera_skill(r)
                cols = st.columns(4)
                cols[0].metric("Duración", f"{r['duracion']:.1f} s", help=AYUDA["duracion"])
                cols[1].metric("Forma media", f"{r['forma_media']}/100" if r["forma_media"] is not None else "—", help=AYUDA["forma"])
                cols[2].metric("Progresión", f"{r['progresion']}%" if r["progresion"] is not None else "—", help=AYUDA["progresion"])
                cols[3].metric("Protracción", r["protraccion"] or "—", help=AYUDA_PROT)
                st.caption(f"Del segundo {r['inicio']:.1f} al {r['fin']:.1f}")
                if r["mejor"] is None:
                    continue

                c1, c2, c3 = st.columns([1, 1, 1.4])
                for col, clave, nombre in [(c1, "mejor", "Mejor"), (c2, "peor", "Peor")]:
                    m = r[clave]
                    texto = f"{nombre} momento: {m['forma']:.1f}/100 (segundo {m['t']:.1f})"
                    if m["protraccion"]:
                        texto += f" · protracción: {m['protraccion'].lower()}"
                    col.image(m["imagen"], caption=texto)
                with c3:
                    st.markdown("**Forma segundo a segundo**")
                    st.caption("Tu nota de forma (0-100) en cada momento del hold. Si cae al final, "
                               "estás perdiendo la posición por cansancio.")
                    datos = pd.DataFrame({"segundo": r["serie_t"], "forma": r["serie_forma"]})
                    grafica = alt.Chart(datos).mark_line(point=True).encode(
                        x=alt.X("segundo:Q", title="segundo del vídeo", scale=alt.Scale(zero=False)),
                        y=alt.Y("forma:Q", title="forma (0-100)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[alt.Tooltip("segundo:Q", format=".1f"), alt.Tooltip("forma:Q", format=".1f")],
                    )                                   # sin .interactive(): no se mueve al hacer scroll en el móvil
                    st.altair_chart(grafica, width="stretch")
                    st.markdown("**Cuánto te alejas de la posición ideal**")
                    st.caption("En grados: **0° = perfecto**. Media de todo el hold, y en tu mejor y tu peor momento.")
                    tabla = pd.DataFrame(r["desviaciones"], index=["Media", "Mejor momento", "Peor momento"]).T.round(1)
                    tabla.index = [NOMBRE_CRITERIO.get(c, c) for c in tabla.index]
                    st.dataframe(tabla, width="stretch")
                with st.expander("📐 Cómo leer el dibujo"):
                    st.markdown(LEYENDA)
                mostrar_consejos(r["consejos"])

# ---------------------------------------------------------------------
with tab_foto:
    archivo = st.file_uploader("Sube una foto", key="foto")
    if archivo and not archivo.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        st.error("Eso no parece una foto. Sube un .jpg o .png.")
        archivo = None
    if archivo:
        img = cv2.imdecode(np.frombuffer(archivo.read(), np.uint8), cv2.IMREAD_COLOR)
        with st.spinner("Analizando..."):
            r = cal.analizar_foto(img)
        if not r["ok"]:
            st.error(r["mensaje"])
        else:
            c1, c2 = st.columns([1.2, 1])
            c1.image(r["imagen"])
            with c2:
                st.subheader(bonito(r["skill"]))
                cabecera_skill(r)
                if r["forma"] is not None:
                    st.metric("Forma", f"{r['forma']}/100", help=AYUDA["forma"])
                    if r["progresion"] is not None:
                        st.metric("Progresión", f"{r['progresion']}%", help=AYUDA["progresion"])
                    if r["protraccion"] is not None:
                        st.metric("Protracción", r["protraccion"], help="Si la parte alta de la espalda está redondeada (escápulas separadas).")
                    mostrar_consejos(r["consejos"])
                    with st.expander("📐 Cómo leer el dibujo"):
                        st.markdown(LEYENDA)

# ---------------------------------------------------------------------
st.divider()
st.caption("Los vídeos y fotos se procesan al momento y **no se guardan**. "
           "Proyecto personal: modelo de pose (RTMPose) + clasificador + DINOv2 afinado. "
           "Es una herramienta de apoyo; la nota es orientativa.")
