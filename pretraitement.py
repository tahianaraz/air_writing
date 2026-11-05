"""
Air Writing — prétraitement commun (entraînement ET application)
-----------------------------------------------------------------
Un tracé = une liste de traits, chaque trait = une suite de points (x, y).
Le même pipeline sert partout, sinon le modèle ne voit pas la même chose
à l'entraînement et en direct :

    traits  ->  image 28x28 (lettre centrée, épaisseur fixe)  ->  features (HOG + pixels)

Contient aussi les modèles de lettres (A-Z en majuscules) et l'augmentation
qui génère des tracés synthétiques "façon doigt dans l'air".
"""
import cv2
import numpy as np

TAILLE = 28            # taille de l'image vue par le modèle
MARGE = 4              # marge autour de la lettre (en pixels sur 28)
SUR_ECHANTILLON = 4    # on dessine en 112x112 puis on réduit (traits plus propres)
EPAISSEUR = 0.09       # épaisseur du trait, relative à la taille de l'image

LETTRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# =====================================================================
# 1) Tracé -> image
# =====================================================================
def traits_vers_image(traits, epaisseur=EPAISSEUR):
    """Transforme une liste de traits en image 28x28 (uint8, blanc sur noir).

    La lettre est mise à l'échelle en gardant ses proportions, puis centrée.
    L'épaisseur est fixée APRÈS la mise à l'échelle : une lettre dessinée
    petite ou grande donne la même image.
    """
    traits = [np.asarray(t, dtype=np.float32).reshape(-1, 2) for t in traits]
    traits = [t for t in traits if len(t) > 0]
    grand = TAILLE * SUR_ECHANTILLON
    img = np.zeros((grand, grand), np.uint8)
    if not traits:
        return cv2.resize(img, (TAILLE, TAILLE), interpolation=cv2.INTER_AREA)

    tous = np.concatenate(traits)
    mini, maxi = tous.min(axis=0), tous.max(axis=0)
    dim = max(float((maxi - mini).max()), 1e-6)       # plus grand côté
    utile = (TAILLE - 2 * MARGE) * SUR_ECHANTILLON
    echelle = utile / dim
    centre = (mini + maxi) / 2
    ep = max(1, int(round(epaisseur * grand)))

    for t in traits:
        pts = (t - centre) * echelle + grand / 2
        pts = np.round(pts).astype(np.int32)
        if len(pts) == 1:
            cv2.circle(img, tuple(int(v) for v in pts[0]), ep // 2, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, 255, ep, cv2.LINE_AA)

    return cv2.resize(img, (TAILLE, TAILLE), interpolation=cv2.INTER_AREA)


# =====================================================================
# 2) Image -> features
# =====================================================================
_HOG = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)


def image_vers_features(img):
    """HOG (forme des contours) + pixels réduits en 14x14 (forme globale)."""
    hog = _HOG.compute(img).ravel()
    pixels = cv2.resize(img, (14, 14), interpolation=cv2.INTER_AREA).ravel() / 255.0
    return np.concatenate([hog, pixels]).astype(np.float32)


def traits_vers_features(traits, epaisseur=EPAISSEUR):
    return image_vers_features(traits_vers_image(traits, epaisseur))


# =====================================================================
# 3) Modèles de lettres (coordonnées dans un carré 0..1, y vers le bas)
# =====================================================================
def _arc(cx, cy, rx, ry, a0, a1, n=24):
    """Arc d'ellipse. Angles en degrés : 0 = droite, 90 = bas, -90 = haut."""
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], axis=1)


def _L(*pts):
    return np.array(pts, dtype=np.float32)


def _cat(*morceaux):
    return np.concatenate([np.asarray(m, np.float32).reshape(-1, 2) for m in morceaux])


_O = _arc(0.5, 0.5, 0.5, 0.5, -90, -450, 40)
_P_BOUCLE = _cat(_L((0, 1), (0, 0), (0.6, 0)), _arc(0.6, 0.27, 0.4, 0.27, -90, 90), _L((0, 0.54)))

# Chaque lettre a une ou plusieurs variantes ; chaque variante = liste de traits.
MODELES = {
    "A": [[_L((0, 1), (0.5, 0), (1, 1)), _L((0.22, 0.6), (0.78, 0.6))]],
    "B": [[_L((0, 0), (0, 1)),
           _cat(_L((0, 0), (0.6, 0)), _arc(0.6, 0.25, 0.3, 0.25, -90, 90), _L((0, 0.5), (0.65, 0.5)),
                _arc(0.65, 0.75, 0.35, 0.25, -90, 90), _L((0, 1)))],
          [_cat(_L((0, 1), (0, 0), (0.6, 0)), _arc(0.6, 0.25, 0.3, 0.25, -90, 90), _L((0.1, 0.5), (0.65, 0.5)),
                _arc(0.65, 0.75, 0.35, 0.25, -90, 90), _L((0, 1)))]],
    "C": [[_arc(0.5, 0.5, 0.5, 0.5, -45, -315, 36)]],
    "D": [[_L((0, 0), (0, 1)), _cat(_L((0, 0), (0.4, 0)), _arc(0.4, 0.5, 0.6, 0.5, -90, 90), _L((0, 1)))],
          [_cat(_L((0, 1), (0, 0), (0.4, 0)), _arc(0.4, 0.5, 0.6, 0.5, -90, 90), _L((0, 1)))]],
    "E": [[_L((1, 0), (0, 0), (0, 1), (1, 1)), _L((0, 0.5), (0.75, 0.5))]],
    "F": [[_L((1, 0), (0, 0), (0, 1)), _L((0, 0.5), (0.75, 0.5))]],
    "G": [[_cat(_arc(0.5, 0.5, 0.5, 0.5, -45, -360, 40), _L((0.55, 0.5)))],
          [_cat(_arc(0.5, 0.5, 0.5, 0.5, -45, -340, 40), _L((1, 0.5), (0.55, 0.5)))]],
    "H": [[_L((0, 0), (0, 1)), _L((1, 0), (1, 1)), _L((0, 0.5), (1, 0.5))]],
    "I": [[_L((0.5, 0), (0.5, 1))],
          [_L((0.2, 0), (0.8, 0)), _L((0.5, 0), (0.5, 1)), _L((0.2, 1), (0.8, 1))]],
    "J": [[_cat(_L((0.8, 0), (0.8, 0.7)), _arc(0.45, 0.7, 0.35, 0.3, 0, 180))],
          [_L((0.3, 0), (1, 0)), _cat(_L((0.8, 0), (0.8, 0.7)), _arc(0.45, 0.7, 0.35, 0.3, 0, 180))]],
    "K": [[_L((0, 0), (0, 1)), _L((0.9, 0), (0, 0.55), (0.9, 1))],
          [_L((0, 0), (0, 1)), _L((0.9, 0), (0, 0.55)), _L((0.3, 0.4), (0.9, 1))]],
    "L": [[_L((0, 0), (0, 1), (0.8, 1))]],
    "M": [[_L((0, 1), (0, 0), (0.5, 0.6), (1, 0), (1, 1))]],
    "N": [[_L((0, 1), (0, 0), (1, 1), (1, 0))]],
    "O": [[_O]],
    "P": [[_P_BOUCLE]],
    "Q": [[_O, _L((0.6, 0.7), (1, 1.05))]],
    "R": [[_cat(_P_BOUCLE, _L((1, 1)))],
          [_P_BOUCLE, _L((0.4, 0.54), (1, 1))]],
    "S": [[_cat(_arc(0.5, 0.25, 0.45, 0.25, -30, -270), _arc(0.5, 0.75, 0.45, 0.25, -90, 150))]],
    "T": [[_L((0, 0), (1, 0)), _L((0.5, 0), (0.5, 1))]],
    "U": [[_cat(_L((0, 0), (0, 0.6)), _arc(0.5, 0.6, 0.5, 0.4, 180, 0), _L((1, 0)))]],
    "V": [[_L((0, 0), (0.5, 1), (1, 0))]],
    "W": [[_L((0, 0), (0.25, 1), (0.5, 0.35), (0.75, 1), (1, 0))]],
    "X": [[_L((0, 0), (1, 1)), _L((1, 0), (0, 1))]],
    "Y": [[_L((0, 0), (0.5, 0.5)), _L((1, 0), (0.5, 0.5), (0.5, 1))],
          [_L((0, 0), (0.5, 0.5), (1, 0)), _L((0.5, 0.5), (0.5, 1))]],
    "Z": [[_L((0, 0), (1, 0), (0, 1), (1, 1))]],
}


# =====================================================================
# 4) Augmentation : imiter un doigt qui écrit dans l'air
# =====================================================================
def _reechantillonner(t, pas):
    """Points régulièrement espacés le long du trait (comme une webcam à fps fixe)."""
    if len(t) < 2:
        return t.copy()
    seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    if cum[-1] < 1e-9:
        return t[:1].copy()
    n = max(2, int(cum[-1] / pas) + 1)
    s = np.linspace(0, cum[-1], n)
    return np.stack([np.interp(s, cum, t[:, 0]), np.interp(s, cum, t[:, 1])], axis=1)


def _lisser(t, alpha):
    """Moyenne glissante exponentielle, comme dans l'application."""
    out = t.copy()
    for i in range(1, len(out)):
        out[i] = alpha * t[i] + (1 - alpha) * out[i - 1]
    return out


def augmenter(traits, rng, force=1.0):
    """Renvoie une version déformée du tracé. force=1 pour les modèles,
    plus faible pour augmenter de vrais gestes déjà variés."""
    traits = [np.asarray(t, np.float32).reshape(-1, 2) for t in traits]
    tous = np.concatenate(traits)
    mini, maxi = tous.min(0), tous.max(0)
    dim = max(float((maxi - mini).max()), 1e-6)
    centre = (mini + maxi) / 2
    traits = [(t - centre) / dim for t in traits]     # carré unité centré en 0

    # Déformation globale : rotation, italique, proportions
    ang = np.radians(rng.uniform(-12, 12) * force)
    cis = rng.uniform(-0.3, 0.3) * force
    sx, sy = np.exp(rng.uniform(-0.25, 0.25, 2) * force)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    M = rot @ np.array([[1, cis], [0, 1]]) @ np.diag([sx, sy])

    # Ondulation douce (main qui ne trace pas droit)
    amp = rng.uniform(0, 0.05) * force
    fx, fy = rng.uniform(1, 3, 2)
    px, py = rng.uniform(0, 2 * np.pi, 2)

    # Bout de trait "parasite" quand on lève/baisse le doigt (latence du geste)
    p_liaison = 0.35 * force

    sortie = []
    for i, t in enumerate(traits):
        t = t @ M.T
        # dépassement / raccourcissement aux extrémités
        if len(t) >= 2:
            d0, d1 = t[1] - t[0], t[-1] - t[-2]
            t = np.vstack([t[0] - d0 * rng.uniform(-0.1, 0.15) * force, t[1:-1],
                           t[-1] + d1 * rng.uniform(-0.1, 0.15) * force])
        t = _reechantillonner(t, rng.uniform(0.015, 0.035))
        t = t + amp * np.stack([np.sin(fx * 2 * np.pi * t[:, 1] + px),
                                np.sin(fy * 2 * np.pi * t[:, 0] + py)], axis=1)
        t = t + rng.normal(0, 0.006 * force + 0.002, t.shape)   # tremblement
        t = _lisser(t, rng.uniform(0.35, 0.8))
        sortie.append(t)

        if i + 1 < len(traits) and rng.random() < p_liaison:
            fin = t[-1]
            debut = (traits[i + 1] @ M.T)[0]
            k = rng.uniform(0.05, 0.25)
            if rng.random() < 0.5:
                sortie[-1] = np.vstack([t, fin + (debut - fin) * k])
            else:
                sortie.append(np.array([debut - (debut - fin) * k, debut]))
    return sortie


def generer_synthetique(n_par_lettre, graine=0, lettres=LETTRES):
    """Jeu de données synthétique : (features, labels, images)."""
    rng = np.random.default_rng(graine)
    X, y, imgs = [], [], []
    for lettre in lettres:
        variantes = MODELES[lettre]
        for _ in range(n_par_lettre):
            traits = augmenter(variantes[rng.integers(len(variantes))], rng)
            img = traits_vers_image(traits, epaisseur=rng.uniform(0.07, 0.11))
            X.append(image_vers_features(img))
            y.append(lettre)
            imgs.append(img)
    return np.array(X), np.array(y), np.array(imgs)
