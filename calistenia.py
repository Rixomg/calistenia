"""
CalistenIA: reconoce skills de calistenia (planche y front lever) y puntúa la forma.

Uso:
    import calistenia as cal
    cal.cargar("modelos")                     # una vez, al arrancar
    res = cal.analizar_foto(img_bgr)          # foto (array de OpenCV)
    res = cal.analizar_video("video.mp4")     # vídeo

Todo lo que se ajusta está en el bloque PARÁMETROS.
"""
import gc
import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rtmlib import Wholebody
from transformers import AutoImageProcessor, AutoModel

# =====================================================================
# PARÁMETROS
# =====================================================================
# --- Clasificación ---
PESO_IMG = 0.5            # peso del modelo de imagen al combinar con la pose (0 = solo pose)
UMBRAL_CONF = 0.30        # por debajo de esta probabilidad, "no estoy seguro de qué skill es"

# --- Vídeo ---
MAX_SEGUNDOS = 30         # solo se analizan los primeros N segundos
FPS_ANALISIS = 3          # fotogramas analizados por segundo
DET_CADA = 10             # buscar a la persona (lento) solo 1 de cada N fotogramas; en el resto se la sigue
UMBRAL_MOV = 0.08         # cuánto se puede mover el cuerpo (en torsos) entre fotogramas y seguir siendo hold
MIN_HOLD = 1.0            # segundos mínimos para contar como hold (después de recortar)
RECORTE = 0.4             # segundos que se quitan al principio y al final de cada hold
FOTOS_GIRO = 5            # fotogramas usados para decidir la orientación del vídeo
VENTAJA_DERECHO = 0.10    # para girar el vídeo, la otra orientación tiene que ganar por al menos esto
FOTOS_LADO = 2            # fotogramas que mira el modelo de imagen para saber hacia qué lado girar
N_IMG = 5                 # fotogramas por hold que pasan por el modelo de imagen
N_PROT = 3                # fotogramas por hold que pasan por el modelo de protracción

# --- Nota de forma ---   tolerancia (°), peso
TOL_HORIZ,  PESO_HORIZ  = 1, 3      # espalda horizontal (hombro-cadera)
TOL_LINEA,  PESO_LINEA  = 1, 3      # línea hombro-cadera-pies
TOL_CODO,   PESO_CODO   = 1, 3      # codo recto
TOL_CODO90, PESO_CODO90 = 3, 3      # codo a 90° (90 hold)
TOL_CABEZA, PESO_CABEZA = 20, 1     # cabeza alineada: mucho margen y poco peso
TOL_RODILLA, PESO_RODILLA = 10, 3   # rodilla estirada (straddle, full, 90 hold, lean)
MAX_EXTRA = 25            # a partir de tolerancia + esto, ese criterio vale 0
DECIMALES = 1
PROG_0, PROG_100 = 90, 150          # progresión adv tuck: ángulo de cadera al 0 % y al 100 %
PESO_PROT = 2             # peso de la protracción en la nota de forma (planches)
UMBRAL_PROT_SI = 65       # probabilidad de protracción a partir de la cual se dice "Sí"
UMBRAL_PROT_NO = 35       # por debajo, "No"; entre medias, "Dudosa"

# --- Consejos que se muestran cuando falla cada criterio ---
CONSEJOS = {
    "cadera alta": "Baja la cadera hasta la altura de los hombros: aprieta abdomen y glúteo.",
    "cadera baja": "Sube la cadera hasta la altura de los hombros: aprieta abdomen y glúteo.",
    "línea hombro-cadera-pies rota": "Busca una línea recta de hombros a pies: glúteo y piernas apretados.",
    "codo doblado": "Bloquea los codos: brazos totalmente rectos.",
    "codo no a 90°": "Codos a 90°: ni más abierto ni más cerrado.",
    "cabeza fuera de línea": "Cabeza neutra, en línea con la espalda.",
    "rodilla doblada": "Estira las piernas del todo: rodillas bloqueadas y puntas estiradas.",
    "sin protracción": "Empuja el suelo y separa las escápulas (protracción): la espalda alta debe redondearse.",
}

DEVICE = "cpu"

# =====================================================================
# CARGA DE MODELOS
# =====================================================================
body = processor = dino = general = mod_img = orden_img = modelo_prot = None


class ModeloImagen(nn.Module):
    # DINOv2 con las últimas capas reajustadas (fine-tuning en Kaggle) y una capa final por skill
    def __init__(self, nombre, n_clases):
        super().__init__()
        self.dino = AutoModel.from_pretrained(nombre)
        d = self.dino.config.hidden_size
        self.cabeza = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * d, n_clases))

    def forward(self, x):
        h = self.dino(pixel_values=x).last_hidden_state
        return self.cabeza(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1))


def cargar(carpeta="modelos"):
    """Carga detector, DINOv2 y los tres modelos entrenados (pose, imagen y protracción)."""
    global body, processor, dino, general, mod_img, orden_img, modelo_prot
    body = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino = AutoModel.from_pretrained("facebook/dinov2-base").to(DEVICE).eval()

    general = joblib.load(os.path.join(carpeta, "calistenia_modelo_v2.joblib"))["general"]

    ruta_img = os.path.join(carpeta, "calistenia_imagen.pt")
    if not os.path.exists(ruta_img):
        from huggingface_hub import hf_hub_download
        ruta_img = hf_hub_download(repo_id="Rixomg/calistenia-modelos", filename="calistenia_imagen.pt")
    if os.path.exists(ruta_img):
        ck = torch.load(ruta_img, map_location=DEVICE)
        clases_img = list(ck["clases"])
        if set(clases_img) == set(general.classes_):
            mod_img = ModeloImagen(ck["nombre"], len(clases_img)).to(DEVICE)
            mod_img.load_state_dict(ck["estado"])
            mod_img.eval()
            mod_img = torch.quantization.quantize_dynamic(mod_img, {nn.Linear}, dtype=torch.qint8)
            orden_img = [clases_img.index(c) for c in general.classes_]

    ruta_prot = os.path.join(carpeta, "calistenia_protraccion.joblib")
    if os.path.exists(ruta_prot):
        modelo_prot = joblib.load(ruta_prot)


# =====================================================================
# DETECCIÓN DEL ESQUELETO
# =====================================================================
def _detectar(img_bgr):
    kps, scores = body(img_bgr)
    if len(kps) == 0:
        return None, None, 0.0
    i = np.argmax(scores[:, 5:17].mean(1))
    return kps[i], scores[i], scores[i, 5:17].mean()


GIROS = {"0": 0, "cw": 1, "180": 2, "ccw": 3}          # en cuartos de vuelta, sentido horario
_CV = {"cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE}


def girar(img, giro):
    return img if giro == "0" else cv2.rotate(img, _CV[giro])


def girar_puntos(p, giro, H, W):
    # Dónde caen los puntos de una foto H x W cuando la foto se gira con girar()
    if giro == "0":
        return p
    x, y = p[:, 0], p[:, 1]
    if giro == "cw":
        return np.stack([H - 1 - y, x], 1)
    if giro == "ccw":
        return np.stack([y, W - 1 - x], 1)
    return np.stack([W - 1 - x, H - 1 - y], 1)


def desgirar(p, giro, H, W):
    # Devuelve los puntos de la imagen girada a la foto original (H x W)
    if giro == "0":
        return p
    inverso = {"cw": "ccw", "ccw": "cw", "180": "180"}[giro]
    Hg, Wg = (W, H) if giro in ("cw", "ccw") else (H, W)    # tamaño de la imagen girada
    return girar_puntos(p, inverso, Hg, Wg)


def restar_giros(a, b):
    # Giro que queda por aplicar a una imagen ya girada b para llegar al giro total a
    inv = {v: k for k, v in GIROS.items()}
    return inv[(GIROS[a] - GIROS[b]) % 4]


def orientacion_escena(muestras, H, W, fotos=()):
    """Si el vídeo/foto está grabado de lado, hacia dónde hay que girarlo. Es MUY prudente:
    un vídeo normal nunca se gira. Solo se gira si se cumplen las tres cosas:
    1) Derecho, la pose no reconoce ninguna skill.
    2) Girado, la pose sí la reconoce, y claramente mejor.
    3) El modelo de imagen también lo ve mejor girado que derecho. Además decide HACIA DÓNDE,
       porque una planche boca abajo tiene la misma geometría que un front lever."""
    muestras = [(p, s) for p, s in muestras if p is not None]
    if not muestras:
        return "0"
    otras = [i for i, c in enumerate(general.classes_) if c != "nada"]
    puntos = {}
    for g in ("0", "cw", "ccw"):
        X = [rasgos_pose(girar_puntos(p, g, H, W), s) for p, s in muestras]
        puntos[g] = float(general.predict_proba(X)[:, otras].max(1).mean())
    if puntos["0"] >= UMBRAL_CONF:                       # derecho ya se reconoce: no se toca
        return "0"
    if max(puntos["cw"], puntos["ccw"]) <= puntos["0"] + VENTAJA_DERECHO:
        return "0"                                       # girado no mejora: no está de lado
    fotos = [f for f in fotos if f is not None][:FOTOS_LADO]
    if mod_img is None or not fotos:
        return max(("cw", "ccw"), key=puntos.get)
    img = {g: float(np.mean([prob_imagen(girar(f, g))[otras].max() for f in fotos])) for g in ("0", "cw", "ccw")}
    mejor = max(("cw", "ccw"), key=img.get)
    return mejor if img[mejor] > img["0"] else "0"     # la imagen también tiene que verlo mejor girado


def detectar_con_giro(img, giro):
    H, W = img.shape[:2]
    p, s, c = _detectar(girar(img, giro))
    if p is None:
        return None, None, 0.0
    return desgirar(p, giro, H, W), s, c


def persona_principal(img_bgr):
    # Foto: prueba tal cual y girada 90° a cada lado (el detector va mejor con cuerpos verticales)
    mejor = (None, None, 0.0)
    for g in ("0", "cw", "ccw"):
        r = detectar_con_giro(img_bgr, g)
        if r[0] is not None and r[2] > mejor[2]:
            mejor = r
    return mejor[0], mejor[1]


# =====================================================================
# RASGOS DE POSE (lo que ve el clasificador de pose)
# =====================================================================
NOMBRES_POSE = ["detectado", "ang_cadera", "ang_rodilla", "ang_codo", "inclinacion",
                "rodilla_hombro", "apertura_piernas", "munecas_vs_hombros",
                "largo_piernas", "dif_piernas", "conf_tobillos",
                "pies_perp", "pies_rango", "asim_pies", "muslo_vs_caderas"]
PUNTOS_PIES = [15, 16, 17, 18, 19, 20, 21, 22]
IZQ = [5, 7, 9, 11, 13, 15]       # hombro, codo, muñeca, cadera, rodilla, tobillo
DER = [6, 8, 10, 12, 14, 16]


def angulo(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rasgos_pose(p, s):
    if p is None:
        return np.zeros(len(NOMBRES_POSE))
    lado = IZQ if s[IZQ].mean() >= s[DER].mean() else DER     # solo el lado que se ve
    hombro, codo, muneca, cadera, rodilla, tobillo = (p[k] for k in lado)
    torso = np.linalg.norm(hombro - cadera) + 1e-6
    dx, dy = cadera - hombro
    pierna_izq = np.linalg.norm(p[11] - p[13]) + np.linalg.norm(p[13] - p[15])
    pierna_der = np.linalg.norm(p[12] - p[14]) + np.linalg.norm(p[14] - p[16])

    pies = np.array([p[k] for k in PUNTOS_PIES if s[k] > 0.3])
    if len(pies) >= 3:
        eje = tobillo - cadera
        eje = eje / (np.linalg.norm(eje) + 1e-6)
        perp = np.array([-eje[1], eje[0]])
        pies_perp = np.ptp(pies @ perp) / torso
        pies_rango = np.linalg.norm(pies[:, None] - pies[None], axis=-1).max() / torso
    else:
        pies_perp, pies_rango = 0.0, 0.0

    asim_pies = abs(s[[15, 17, 18, 19]].mean() - s[[16, 20, 21, 22]].mean())
    muslo_vs_caderas = max(abs(angulo(p[12], p[11], p[13]) - 90),
                           abs(angulo(p[11], p[12], p[14]) - 90))
    return np.array([
        1,
        angulo(hombro, cadera, rodilla),
        angulo(cadera, rodilla, tobillo),
        angulo(hombro, codo, muneca),
        np.degrees(np.arctan2(abs(dy), abs(dx))),
        np.linalg.norm(rodilla - hombro) / torso,
        np.linalg.norm(p[15] - p[16]) / torso,
        (muneca[1] - hombro[1]) / torso,
        (pierna_izq + pierna_der) / 2 / torso,
        abs(pierna_izq - pierna_der) / torso,
        min(s[15], s[16]),
        pies_perp, pies_rango, asim_pies, muslo_vs_caderas,
    ])


# =====================================================================
# MODELOS DE IMAGEN (skill y protracción)
# =====================================================================
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def embedding(img_bgr):
    # DINOv2 sin reajustar: convierte una imagen en números (para el modelo de protracción)
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = dino(**inputs).last_hidden_state[0]
    return torch.cat([out[0], out[1:].mean(0)]).cpu().numpy()


def prob_imagen(img_bgr):
    # Probabilidad de cada skill según el modelo de imagen (mismo preprocesado que en Kaggle)
    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    l = max(w, h)
    lienzo = Image.new("RGB", (l, l), (124, 116, 104))
    lienzo.paste(img, ((l - w) // 2, (l - h) // 2))
    x = (np.asarray(lienzo.resize((224, 224), Image.BICUBIC)) / 255.0 - MEAN) / STD
    x = torch.tensor(x.transpose(2, 0, 1)[None], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        p = mod_img(x).softmax(1) + mod_img(torch.flip(x, [3])).softmax(1)
    return (p / 2)[0].cpu().numpy()[orden_img]


def combinar(p_pose, p_img):
    return p_pose if p_img is None else PESO_IMG * p_img + (1 - PESO_IMG) * p_pose


# =====================================================================
# NOTA DE FORMA
# =====================================================================
CABEZA = {"izq": 3, "der": 4}       # oreja del lado visible


def puntos_forma(p, s):
    izq = s[IZQ].mean() >= s[DER].mean()
    lado = IZQ if izq else DER
    oreja = CABEZA["izq" if izq else "der"]
    cabeza = p[oreja] if s[oreja] > 0.3 else p[0]
    hombro, codo, muneca, cadera, rodilla, tobillo = (p[k] for k in lado)
    return dict(cabeza=cabeza, hombro=hombro, codo=codo, muneca=muneca,
                cadera=cadera, rodilla=rodilla, tobillo=tobillo)


def desviacion_linea(pt, nombres):
    # Peor ángulo en los puntos intermedios: 0° = línea perfecta
    return max(180 - angulo(pt[a], pt[b], pt[c]) for a, b, c in zip(nombres, nombres[1:], nombres[2:]))


# Cada criterio: (tipo, puntos, tolerancia, peso, nombre)
HORIZ = ("horizontal", ["hombro", "cadera"], TOL_HORIZ, PESO_HORIZ, "espalda no horizontal")
LINEA = ("linea", ["hombro", "cadera", "tobillo"], TOL_LINEA, PESO_LINEA, "línea hombro-cadera-pies rota")
CODO = ("codo", ["hombro", "codo", "muneca"], TOL_CODO, PESO_CODO, "codo doblado")
CODO90 = ("codo90", ["hombro", "codo", "muneca"], TOL_CODO90, PESO_CODO90, "codo no a 90°")
CABEZA_ = ("linea", ["cabeza", "hombro", "cadera"], TOL_CABEZA, PESO_CABEZA, "cabeza fuera de línea")
RODILLA = ("linea", ["cadera", "rodilla", "tobillo"], TOL_RODILLA, PESO_RODILLA, "rodilla doblada")


def criterios(skill):
    if skill == "hold_90":
        return [CODO90, HORIZ, LINEA, RODILLA, CABEZA_]
    if skill.startswith("planche_lean"):
        return [CODO, LINEA, RODILLA, CABEZA_]
    if "_" not in skill:
        return []
    piernas, familia = skill.rsplit("_", 1)
    if familia not in ("planche", "fl") or piernas not in ("tuck", "adv_tuck", "straddle", "full"):
        return []
    c = [HORIZ]
    if piernas not in ("tuck", "adv_tuck"):
        c += [LINEA, RODILLA]
    if familia == "planche":
        c += [CODO, CABEZA_]
    return c


def desviaciones(skill, p, s):
    """Grados de desviación de cada criterio, y si la cadera está por encima de los hombros."""
    pt = puntos_forma(p, s)
    out = {}
    for tipo, puntos, _, _, nombre in criterios(skill):
        if tipo == "horizontal":
            dx, dy = pt["cadera"] - pt["hombro"]
            out[nombre] = np.degrees(np.arctan2(abs(dy), abs(dx)))
        elif tipo == "codo90":
            out[nombre] = abs(angulo(pt["hombro"], pt["codo"], pt["muneca"]) - 90)
        else:
            out[nombre] = desviacion_linea(pt, puntos)
    cadera_alta = pt["cadera"][1] < pt["hombro"][1]       # en la imagen, y crece hacia abajo
    return out, cadera_alta


def evaluar_forma(skill, p, s):
    """Nota 0-100, criterios que fallan y progresión (solo adv tuck)."""
    desv, cadera_alta = desviaciones(skill, p, s)
    total, pesos, fallos = 0, 0, {}
    for _, _, tol, peso, nombre in criterios(skill):
        nota = max(0, 1 - max(0, desv[nombre] - tol) / MAX_EXTRA)
        total += peso * nota
        pesos += peso
        if nota < 1:
            fallos[nombre] = desv[nombre]
    forma = round(100 * total / pesos, DECIMALES) if pesos else None
    prog = None
    if skill.startswith("adv_tuck"):
        pt = puntos_forma(p, s)
        a = angulo(pt["hombro"], pt["cadera"], pt["rodilla"])
        prog = round(100 * np.clip((a - PROG_0) / (PROG_100 - PROG_0), 0, 1), DECIMALES)
    return forma, fallos, prog, cadera_alta


COLORES = {"horizontal": (0, 200, 0), "linea": (255, 120, 0), "codo": (0, 140, 255), "codo90": (0, 140, 255)}


def dibujar_forma(img, pt, skill):
    img = img.copy()
    grosor = max(2, img.shape[1] // 250)
    P = lambda k: tuple(int(v) for v in pt[k])
    orden = sorted(criterios(skill), key=lambda c: c[0] == "horizontal")   # la espalda (verde), encima
    for tipo, puntos, _, _, _ in orden:
        color = COLORES[tipo]
        for a, b in zip(puntos, puntos[1:]):
            cv2.line(img, P(a), P(b), color, grosor)
        for k in puntos:
            cv2.circle(img, P(k), grosor * 3, color, -1)
        for a, b, c in zip(puntos, puntos[1:], puntos[2:]):
            g = angulo(pt[a], pt[b], pt[c])
            cv2.putText(img, f"{g:.0f}", (P(b)[0] + 10, P(b)[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, grosor * 0.35, color, max(1, grosor // 2))
        if tipo == "horizontal":                 # inclinación de la espalda, en el centro del tramo
            dx, dy = pt["cadera"] - pt["hombro"]
            inc = np.degrees(np.arctan2(abs(dy), abs(dx)))
            mx, my = ((pt["hombro"] + pt["cadera"]) / 2).astype(int)
            cv2.putText(img, f"{inc:.0f}", (mx, my + 12 * grosor),
                        cv2.FONT_HERSHEY_SIMPLEX, grosor * 0.35, color, max(1, grosor // 2))
    return img


# =====================================================================
# PROTRACCIÓN (modelo aprendido de etiquetas)
# =====================================================================
LADO_RECORTE = 224


def es_planche(skill):
    return skill.endswith("_planche") or skill.startswith("planche_lean")


def recorte_espalda(img_bgr, p, s):
    # Recorte de la parte alta de la espalda, girado: línea hombro→cadera horizontal y espalda arriba
    pt = puntos_forma(p, s)
    a, b = pt["hombro"], pt["cadera"]
    eje = b - a
    L = np.linalg.norm(eje) + 1e-6
    u = eje / L
    n = np.array([-u[1], u[0]])
    if np.dot(pt["codo"] - a, n) > 0:
        n = -n
    centro = a + 0.35 * eje + 0.15 * L * n
    R = np.array([u, -n])
    esc = LADO_RECORTE / L
    M = np.hstack([esc * R, (LADO_RECORTE / 2 - esc * R @ centro)[:, None]])
    return cv2.warpAffine(img_bgr, M, (LADO_RECORTE, LADO_RECORTE), borderValue=(124, 116, 104))


def predecir_protraccion(img_bgr, p, s):
    """Probabilidad (0-100) de que haya protracción, o None si no hay modelo."""
    if modelo_prot is None:
        return None
    rec = recorte_espalda(img_bgr, p, s)
    x = (embedding(rec) + embedding(cv2.flip(rec, 1))) / 2
    i = list(modelo_prot.classes_).index("protraccion")
    return 100 * modelo_prot.predict_proba([x])[0][i]


def etiqueta_protraccion(prot):
    """Convierte la probabilidad (0-100) en Sí / No / Dudosa."""
    if prot is None:
        return None
    return "Sí" if prot >= UMBRAL_PROT_SI else ("No" if prot < UMBRAL_PROT_NO else "Dudosa")


def forma_con_protraccion(skill, forma, prot):
    pesos = sum(c[3] for c in criterios(skill))
    return round((forma * pesos + prot * PESO_PROT) / (pesos + PESO_PROT), DECIMALES)


# =====================================================================
# CONSEJOS
# =====================================================================
def consejos(fallos, cadera_alta, prot):
    out = []
    for nombre in fallos:
        if nombre == "espalda no horizontal":
            out.append(CONSEJOS["cadera alta" if cadera_alta else "cadera baja"])
        elif nombre in CONSEJOS:
            out.append(CONSEJOS[nombre])
    if prot is not None and prot < 50:
        out.append(CONSEJOS["sin protracción"])
    return out


def soportada(skill):
    return skill is not None and bool(criterios(skill))


def top3(probs):
    return [(general.classes_[i], float(probs[i])) for i in np.argsort(probs)[::-1][:3]]


# =====================================================================
# FOTO
# =====================================================================
def analizar_foto(img_bgr, orientacion=None):
    """orientacion: None = automática; "0", "cw" o "ccw" para forzarla."""
    p, s = persona_principal(img_bgr)
    if p is None:
        return dict(ok=False, mensaje="No encuentro a nadie en la foto.")
    escena = orientacion if orientacion is not None else orientacion_escena([(p, s)], *img_bgr.shape[:2], fotos=[img_bgr])
    if escena != "0":
        p = girar_puntos(p, escena, *img_bgr.shape[:2])
        img_bgr = girar(img_bgr, escena)
    pp = general.predict_proba([rasgos_pose(p, s)])[0]
    pc = combinar(pp, prob_imagen(img_bgr) if mod_img is not None else None)
    skill = general.classes_[pc.argmax()]
    res = dict(ok=True, skill=skill, confianza=float(pc.max()), top3=top3(pc),
               seguro=pc.max() >= UMBRAL_CONF, soportada=soportada(skill),
               forma=None, progresion=None, protraccion=None, fallos={}, consejos=[],
               imagen=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    if res["seguro"] and res["soportada"]:
        forma, fallos, prog, cadera_alta = evaluar_forma(skill, p, s)
        prot = predecir_protraccion(img_bgr, p, s) if es_planche(skill) else None
        if prot is not None:
            forma = forma_con_protraccion(skill, forma, prot)
        tolerancias = {c[4]: c[2] for c in criterios(skill)}
        claros = {c: v for c, v in fallos.items() if v > tolerancias[c] + 3}   # fallo claro, no ruido
        res.update(forma=forma, progresion=prog, protraccion=etiqueta_protraccion(prot), fallos=fallos,
                   consejos=consejos(claros, cadera_alta, prot),
                   imagen=cv2.cvtColor(dibujar_forma(img_bgr, puntos_forma(p, s), skill), cv2.COLOR_BGR2RGB))
    return res


# =====================================================================
# VÍDEO
# =====================================================================
def leer_fotograma(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def caja_de_puntos(p, s, forma_img):
    pts = p[:23][s[:23] > 0.3]
    if len(pts) < 5:
        return None
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
    h, w = forma_img[:2]
    return np.array([max(0, x0 - mx), max(0, y0 - my), min(w - 1, x1 + mx), min(h - 1, y1 + my)])


def detectar_seguido(fr, giro, caja):
    # Sin caja: busca a la persona (lento). Con caja: solo pone el esqueleto (rápido)
    H, W = fr.shape[:2]
    rot = girar(fr, giro)
    if caja is None:
        cajas = body.det_model(rot)
        if len(cajas) == 0:
            return None, None, None
        caja = max(cajas, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    kps, scores = body.pose_model(rot, bboxes=[caja])
    p, s = kps[0], scores[0]
    if s[5:17].mean() < 0.4:
        return desgirar(p, giro, H, W), s, None
    return desgirar(p, giro, H, W), s, caja_de_puntos(p, s, rot.shape)


def _holds(datos):
    holds, actual, prev = [], [], None
    for d in datos:
        p, s = d["p"], d["s"]
        if p is None or s[5:17].mean() < 0.4:
            if actual:
                holds.append(actual)
            actual, prev = [], None
            continue
        if prev is not None:
            torso = np.linalg.norm((p[5] + p[6]) / 2 - (p[11] + p[12]) / 2) + 1e-6
            mov = np.median(np.linalg.norm(p[5:17] - prev[5:17], axis=1)) / torso
            if mov > UMBRAL_MOV and actual:
                holds.append(actual)
                actual = []
        actual.append(d)
        prev = p
    if actual:
        holds.append(actual)
    holds = [[d for d in h if h[0]["t"] + RECORTE <= d["t"] <= h[-1]["t"] - RECORTE] for h in holds]
    return [h for h in holds if h and h[-1]["t"] - h[0]["t"] >= MIN_HOLD]


def _repartidos(h, n):
    return [h[i] for i in np.linspace(0, len(h) - 1, min(n, len(h))).astype(int)]


def analizar_video(ruta, progreso=None, orientacion=None):
    """Analiza un vídeo. progreso: función opcional (0 a 1). orientacion: None = automática; "0", "cw" o "ccw" para forzarla."""
    cap = cv2.VideoCapture(ruta)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fotogramas = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(fotogramas, int(MAX_SEGUNDOS * fps))
    idxs = list(range(0, total, max(1, round(fps / FPS_ANALISIS))))
    if not idxs:
        cap.release()
        return dict(ok=False, mensaje="No puedo leer el vídeo.")

    # 1. Orientación, una vez para todo el vídeo
    #    a) giro con el que el detector ve mejor el cuerpo
    muestras_fr = [leer_fotograma(cap, i) for i in idxs[::max(1, len(idxs) // FOTOS_GIRO)][:FOTOS_GIRO]]
    muestras_fr = [fr for fr in muestras_fr if fr is not None]
    conf = {"0": 0.0, "cw": 0.0, "ccw": 0.0}
    detecciones = {g: [] for g in conf}
    for fr in muestras_fr:
        for g in conf:
            p, s, c = detectar_con_giro(fr, g)
            conf[g] += c
            detecciones[g].append((p, s))
    giro_det = max(conf, key=conf.get)
    #    b) de qué lado está el suelo (vídeos grabados de lado): el vídeo se endereza
    H, W = muestras_fr[0].shape[:2] if muestras_fr else (0, 0)
    if orientacion is not None:
        escena = orientacion
    else:
        escena = orientacion_escena(detecciones[giro_det], H, W, fotos=muestras_fr) if muestras_fr else "0"
    giro = restar_giros(giro_det, escena)          # giro del detector sobre el vídeo ya enderezado

    def leer(idx):
        fr = leer_fotograma(cap, idx)
        return None if fr is None else girar(fr, escena)

    # 2. Esqueleto de cada fotograma, siguiendo a la persona
    datos, caja = [], None
    for n, idx in enumerate(idxs):
        fr = leer(idx)
        if fr is None:
            break
        if n % DET_CADA == 0:
            caja = None
        p, s, caja = detectar_seguido(fr, giro, caja)
        datos.append(dict(idx=idx, t=idx / fps, p=p, s=s))
        if progreso:
            progreso(0.8 * (n + 1) / len(idxs))

    # 3. Holds y su resumen
    holds = _holds(datos)
    resultado = []
    for k, h in enumerate(holds):
        pp = general.predict_proba(np.array([rasgos_pose(d["p"], d["s"]) for d in h])).mean(0)
        pi = None
        if mod_img is not None:
            frs = [leer(d["idx"]) for d in _repartidos(h, N_IMG)]
            pi = np.mean([prob_imagen(fr) for fr in frs if fr is not None], axis=0)
        pc = combinar(pp, pi)
        skill = general.classes_[pc.argmax()]
        r = dict(skill=skill, confianza=float(pc.max()), top3=top3(pc), seguro=pc.max() >= UMBRAL_CONF,
                 soportada=soportada(skill), inicio=h[0]["t"], fin=h[-1]["t"], duracion=h[-1]["t"] - h[0]["t"],
                 forma_media=None, progresion=None, protraccion=None, desviaciones={}, consejos=[],
                 serie_t=[], serie_forma=[], mejor=None, peor=None)

        if r["seguro"] and r["soportada"]:
            prot = None
            if es_planche(skill) and modelo_prot is not None:
                vals = [predecir_protraccion(leer(d["idx"]), d["p"], d["s"]) for d in _repartidos(h, N_PROT)]
                prot = float(np.mean(vals))
            formas, progs, desv_todas = [], [], []
            for d in h:
                forma, _, prog, d["cadera_alta"] = evaluar_forma(skill, d["p"], d["s"])
                d["forma"] = forma_con_protraccion(skill, forma, prot) if prot is not None else forma
                formas.append(d["forma"])
                desv_todas.append(desviaciones(skill, d["p"], d["s"])[0])
                if prog is not None:
                    progs.append(prog)
            mejor = max(h, key=lambda d: d["forma"])
            peor = min(h, key=lambda d: d["forma"])
            i_m, i_p = h.index(mejor), h.index(peor)
            tolerancias = {c[4]: c[2] for c in criterios(skill)}
            medias = {c: float(np.mean([dd[c] for dd in desv_todas])) for c in tolerancias}
            fallan = {c: v for c, v in medias.items() if v > tolerancias[c] + 3}   # fallo claro, no ruido
            cadera_alta = np.mean([d["cadera_alta"] for d in h]) > 0.5

            def foto(d):
                fr = leer(d["idx"])
                prot_d = etiqueta_protraccion(predecir_protraccion(fr, d["p"], d["s"])) if prot is not None else None
                img = cv2.cvtColor(dibujar_forma(fr, puntos_forma(d["p"], d["s"]), skill), cv2.COLOR_BGR2RGB)
                return img, prot_d

            img_m, prot_m = foto(mejor)
            img_p, prot_p = foto(peor)

            r.update(forma_media=round(float(np.mean(formas)), DECIMALES),
                     progresion=round(float(np.median(progs)), DECIMALES) if progs else None,
                     protraccion=etiqueta_protraccion(prot),
                     desviaciones={c: (medias[c], desv_todas[i_m][c], desv_todas[i_p][c]) for c in medias},
                     consejos=consejos(fallan, cadera_alta, prot),
                     serie_t=[d["t"] for d in h], serie_forma=formas,
                     mejor=dict(t=mejor["t"], forma=mejor["forma"], imagen=img_m, protraccion=prot_m),
                     peor=dict(t=peor["t"], forma=peor["forma"], imagen=img_p, protraccion=prot_p))
        resultado.append(r)
        if progreso:
            progreso(0.8 + 0.2 * (k + 1) / len(holds))

    cap.release()
    gc.collect()                                  # libera memoria después de cada vídeo
    return dict(ok=True, holds=resultado, giro=giro, enderezado=escena != "0",
                segundos=idxs[-1] / fps, recortado=fotogramas > total)
