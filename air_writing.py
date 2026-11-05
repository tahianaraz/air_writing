"""
Air Writing — écrire dans l'air avec son index et une webcam
-------------------------------------------------------------
MediaPipe suit la main, le bout de l'index devient un stylo, et un modèle
scikit-learn reconnaît la lettre dessinée.

Gestes :
  index seul levé                     -> écrire
  index + majeur levés                -> lever le stylo (se déplacer sans écrire, ex. barre du A)
  index + majeur + annulaire (maintenus) -> espace (le pouce ne compte pas)
  main ouverte (maintenue)            -> effacer le tracé
  pause d'environ 1 seconde           -> la lettre est reconnue et ajoutée au texte

Touches (mode écriture) :
  espace = reconnaître tout de suite   c = effacer le tracé
  x = supprimer la dernière lettre     q ou Échap = quitter

Mode collecte (enregistrer tes vrais gestes pour évaluer / améliorer le modèle) :
    python air_writing.py --collecte
  espace = enregistrer le tracé   c = recommencer   n = passer à une autre lettre   q = quitter
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request

import cv2
import joblib
import numpy as np
from mediapipe import Image as MpImage
from mediapipe import ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from pretraitement import LETTRES, image_vers_features, traits_vers_image

# ---------------- Paramètres ----------------
DOSSIER = os.path.dirname(os.path.abspath(__file__))
FICHIER_MODELE = os.path.join(DOSSIER, "modele_lettres.joblib")
DOSSIER_GESTES = os.path.join(DOSSIER, "mes_gestes")
MODELE_MAIN = os.path.join(DOSSIER, "hand_landmarker.task")
URL_MODELE_MAIN = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")

CAMERA_ID = 0            # essaie 1 si la webcam ne s'ouvre pas
LARGEUR, HAUTEUR = 1280, 720
PAUSE_RECO = 1.0         # secondes sans écrire avant la reconnaissance automatique
LISSAGE = 0.5            # 0..1 : plus petit = tracé plus lisse mais plus de retard
FRAMES_ENTREE = 3        # images consécutives pour valider "écrire" (évite les faux départs)
FRAMES_SORTIE = 2        # images consécutives pour valider les autres gestes
FRAMES_EFFACER = 10      # main ouverte maintenue ~1/3 s pour effacer
FRAMES_ESPACE = 8        # 3 doigts maintenus ~1/4 s pour un espace
LONGUEUR_MIN = 40        # tracé plus court (en pixels) = ignoré
DUREE_RESULTAT = 2.5     # secondes d'affichage du dernier résultat

horloge = time.monotonic  # remplaçable pour les tests

FONT = cv2.FONT_HERSHEY_SIMPLEX   # OpenCV n'affiche pas les accents
CYAN, VERT, JAUNE, ROUGE, BLANC, GRIS = ((255, 220, 0), (80, 220, 80), (0, 215, 255),
                                         (60, 60, 255), (255, 255, 255), (170, 170, 170))
VIOLET = (255, 110, 200)
CONNEXIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]


# =====================================================================
# Main et gestes
# =====================================================================
def preparer_modele_main():
    if os.path.exists(MODELE_MAIN):
        return
    print("Téléchargement du modèle de main MediaPipe (quelques Mo, une seule fois)...")
    try:
        urllib.request.urlretrieve(URL_MODELE_MAIN, MODELE_MAIN + ".part")
        os.replace(MODELE_MAIN + ".part", MODELE_MAIN)
    except Exception as e:
        sys.exit(f"Téléchargement impossible ({e}).\nTélécharge ce fichier à la main :\n  {URL_MODELE_MAIN}\n"
                 f"et place-le ici :\n  {MODELE_MAIN}")


def creer_detecteur_main():
    preparer_modele_main()
    with open(MODELE_MAIN, "rb") as f:   # buffer : évite les soucis de chemins avec accents
        tampon = f.read()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=tampon),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def doigts_tendus(p):
    """[index, majeur, annulaire, auriculaire] tendus ?  (p = 21 points en pixels)
    Un doigt est tendu si son bout est nettement plus loin du poignet que son articulation
    du milieu. Ça marche quelle que soit l'orientation de la main."""
    poignet = p[0]
    return [np.linalg.norm(p[bout] - poignet) > 1.15 * np.linalg.norm(p[art] - poignet)
            for bout, art in ((8, 6), (12, 10), (16, 14), (20, 18))]


def geste_brut(p):
    if p is None:
        return "aucun"
    d = doigts_tendus(p)
    n = sum(d)
    if d[0] and n == 1:
        return "ecrire"
    if n >= 4:
        return "ouverte"
    if d[0] and d[1] and d[2]:          # index + majeur + annulaire, auriculaire plié
        return "trois"
    return "lever"


class Gestes:
    """Stabilise le geste : il doit être vu plusieurs images de suite."""

    def __init__(self):
        self.stable, self.candidat, self.n = "aucun", "aucun", 0
        self.effacer = self.espace = False

    def maj(self, p):
        brut = geste_brut(p)
        if brut == self.candidat:
            self.n += 1
        else:
            self.candidat, self.n = brut, 1
        if self.n >= (FRAMES_ENTREE if brut == "ecrire" else FRAMES_SORTIE):
            self.stable = brut
        # événements déclenchés une seule fois par geste maintenu
        self.effacer = brut == "ouverte" and self.n == FRAMES_EFFACER
        self.espace = brut == "trois" and self.n == FRAMES_ESPACE
        return self.stable

    def progression(self):
        """Avancement du geste maintenu (pour l'anneau affiché autour de la main)."""
        if self.candidat == "ouverte":
            return min(self.n / FRAMES_EFFACER, 1.0)
        if self.candidat == "trois":
            return min(self.n / FRAMES_ESPACE, 1.0)
        return 0.0


# =====================================================================
# Tracé
# =====================================================================
class Trace:
    def __init__(self):
        self.effacer()

    def effacer(self):
        self.traits, self.en_cours, self.t_dernier = [], False, 0.0

    def ajouter(self, point, t, saut_max):
        x, y = float(point[0]), float(point[1])
        if self.en_cours and self.traits[-1]:
            px, py = self.traits[-1][-1]
            d = np.hypot(x - px, y - py)
            if d < 2:                       # le doigt n'a pas bougé
                self.t_dernier = t
                return
            if d > saut_max:                # saut aberrant du suivi : nouveau trait
                self.en_cours = False
        if not self.en_cours:
            self.traits.append([])
            self.en_cours = True
        self.traits[-1].append((x, y))
        self.t_dernier = t

    def lever(self):
        self.en_cours = False

    def vide(self):
        return not any(self.traits)

    def longueur(self):
        return sum(float(np.linalg.norm(np.diff(np.array(t), axis=0), axis=1).sum())
                   for t in self.traits if len(t) > 1)


# =====================================================================
# Reconnaissance
# =====================================================================
def charger_modele(obligatoire):
    if not os.path.exists(FICHIER_MODELE):
        if obligatoire:
            sys.exit("Pas de modèle de lettres. Lance d'abord :  python entrainer.py")
        print("(Pas encore de modèle : la collecte marche quand même, sans prédiction.)")
        return None
    return joblib.load(FICHIER_MODELE)["pipeline"]


def reconnaitre(modele, traits):
    """Renvoie (image 28x28, [(lettre, proba) x3])."""
    img = traits_vers_image(traits)
    if modele is None:
        return img, []
    proba = modele.predict_proba(image_vers_features(img)[None])[0]
    ordre = np.argsort(proba)[::-1][:3]
    return img, [(str(modele.classes_[i]), float(proba[i])) for i in ordre]


def lettre_finie(modele, trace, ecrit, maintenant):
    """Reconnaît le tracé, l'ajoute au texte et vide le tracé."""
    img28, top3 = reconnaitre(modele, trace.traits)
    trace.effacer()
    return ecrit + top3[0][0], (img28, top3, maintenant)


# =====================================================================
# Collecte de vrais gestes
# =====================================================================
def compter_gestes(lettres):
    return {l: len(glob.glob(os.path.join(DOSSIER_GESTES, l, "*.json"))) for l in lettres}


def prochaine_lettre(lettres, compte, objectif, actuelle, forcer_autre=False):
    """Alterne les lettres (A, B, C... puis on recommence) plutôt que 10 A d'affilée :
    les gestes sont plus variés, donc plus réalistes."""
    restantes = [l for l in lettres if compte[l] < objectif]
    if not restantes:
        return None
    mini = min(compte[l] for l in restantes)
    candidates = restantes if forcer_autre else [l for l in restantes if compte[l] == mini]
    i0 = lettres.index(actuelle) + 1 if actuelle in lettres else 0
    for k in range(len(lettres)):
        l = lettres[(i0 + k) % len(lettres)]
        if l in candidates and not (forcer_autre and l == actuelle and len(candidates) > 1):
            return l
    return candidates[0]


def enregistrer_geste(lettre, traits, taille, img):
    dossier = os.path.join(DOSSIER_GESTES, lettre)
    os.makedirs(dossier, exist_ok=True)
    nom = f"{lettre}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
    donnees = {"lettre": lettre, "traits": [[[round(x, 1), round(y, 1)] for x, y in t] for t in traits if t],
               "taille_image": list(taille), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(dossier, nom + ".json"), "w", encoding="utf-8") as f:
        json.dump(donnees, f)
    cv2.imwrite(os.path.join(dossier, nom + ".png"), img)


# =====================================================================
# Affichage
# =====================================================================
def texte(img, txt, org, echelle=0.7, couleur=BLANC, ep=2, fond=False):
    """Texte lisible ; fond=True ajoute un rectangle sombre derrière."""
    if fond:
        (tw, th), base = cv2.getTextSize(txt, FONT, echelle, ep)
        x, y = org
        assombrir(img, max(x - 8, 0), max(y - th - 8, 0),
                  min(x + tw + 8, img.shape[1]), min(y + base + 6, img.shape[0]), 0.6)
    cv2.putText(img, txt, org, FONT, echelle, couleur, ep, cv2.LINE_AA)


def assombrir(img, x0, y0, x1, y1, alpha=0.55):
    zone = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = (zone * (1 - alpha)).astype(np.uint8)


def dessiner_traits(img, traits):
    for t in traits:
        if len(t) >= 2:
            p = np.array(t, np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [p], False, (90, 60, 0), 16, cv2.LINE_AA)
            cv2.polylines(img, [p], False, CYAN, 7, cv2.LINE_AA)
        elif len(t) == 1:
            cv2.circle(img, tuple(int(v) for v in t[0]), 5, CYAN, -1, cv2.LINE_AA)


def dessiner_main(img, p, geste, progression):
    for a, b in CONNEXIONS:
        cv2.line(img, tuple(p[a].astype(int)), tuple(p[b].astype(int)), GRIS, 1, cv2.LINE_AA)
    couleur = {"ecrire": VERT, "ouverte": ROUGE, "trois": VIOLET}.get(geste, JAUNE)
    bout = tuple(p[8].astype(int))
    cv2.circle(img, bout, 10, couleur, -1 if geste == "ecrire" else 2, cv2.LINE_AA)
    if progression > 0:
        cv2.ellipse(img, tuple(p[9].astype(int)), (40, 40), -90, 0, 360 * progression, couleur, 4, cv2.LINE_AA)


def dessiner_resultat(img, apercu, top3, x0, y0):
    """Ce que voit le modèle (28x28 agrandi) + les 3 lettres les plus probables."""
    taille = 140
    assombrir(img, x0 - 10, y0 - 30, x0 + taille + 10, y0 + taille + 110, 0.65)
    texte(img, "vu par le modele", (x0, y0 - 10), 0.5, GRIS, 1)
    vignette = cv2.resize(apercu, (taille, taille), interpolation=cv2.INTER_NEAREST)
    img[y0:y0 + taille, x0:x0 + taille] = cv2.cvtColor(vignette, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (x0, y0), (x0 + taille, y0 + taille), GRIS, 1)
    for k, (lettre, p) in enumerate(top3):
        y = y0 + taille + 28 + 26 * k
        couleur = VERT if k == 0 else GRIS
        texte(img, f"{lettre}", (x0, y), 0.7, couleur, 2)
        cv2.rectangle(img, (x0 + 30, y - 14), (x0 + 30 + int(80 * p), y - 2), couleur, -1)
        texte(img, f"{100 * p:.0f}%", (x0 + 115, y - 2), 0.45, couleur, 1)


# =====================================================================
# Boucle principale
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Air Writing : écrire dans l'air avec son index")
    ap.add_argument("--collecte", action="store_true", help="enregistrer tes vrais gestes")
    ap.add_argument("--lettres", default=LETTRES, help="lettres à collecter (ex. AEIOU)")
    ap.add_argument("--par-lettre", type=int, default=10, help="objectif de gestes par lettre")
    ap.add_argument("--camera", type=int, default=CAMERA_ID)
    ap.add_argument("--video", help="fichier vidéo à la place de la webcam")
    args = ap.parse_args(argv)
    lettres = [l for l in args.lettres.upper() if l in LETTRES]

    modele = charger_modele(obligatoire=not args.collecte)
    detecteur = creer_detecteur_main()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not args.video:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGEUR)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HAUTEUR)
    if not cap.isOpened():
        sys.exit("Impossible d'ouvrir la webcam. Essaie : python air_writing.py --camera 1")

    gestes, trace = Gestes(), Trace()
    doigt = None
    ecrit = ""
    resultat = None            # (image, top3, instant)
    t_espace = -10.0           # instant du dernier espace (pour l'afficher)
    attente = None             # collecte : (image, top3) du tracé terminé
    compte = compter_gestes(lettres) if args.collecte else {}
    cible = prochaine_lettre(lettres, compte, args.par_lettre, None) if args.collecte else None
    ts_prec, t_prec, fps = -1, horloge(), 0.0
    fenetre = "Air Writing"
    cv2.namedWindow(fenetre, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                 # effet miroir : plus naturel
        h, w = frame.shape[:2]
        maintenant = horloge()
        fps = 0.9 * fps + 0.1 / max(maintenant - t_prec, 1e-3)
        t_prec = maintenant

        # --- 1. Main ---
        ts = max(ts_prec + 1, int(maintenant * 1000))
        ts_prec = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detecteur.detect_for_video(MpImage(image_format=ImageFormat.SRGB, data=rgb), ts)
        pts = None
        if res.hand_landmarks:
            pts = np.array([[lm.x * w, lm.y * h] for lm in res.hand_landmarks[0]], np.float32)
        geste = gestes.maj(pts)

        # --- 2. Tracé ---
        if pts is not None:
            doigt = pts[8] if doigt is None else LISSAGE * pts[8] + (1 - LISSAGE) * doigt
        else:
            doigt = None
        if geste == "ecrire" and doigt is not None:
            trace.ajouter(doigt, maintenant, saut_max=0.25 * min(w, h))
            attente = None
        else:
            trace.lever()
        if gestes.effacer:
            trace.effacer()
            attente = None

        # --- 3. Espace (3 doigts) : termine la lettre en cours puis ajoute un espace ---
        if gestes.espace and not args.collecte:
            if not trace.vide() and trace.longueur() >= LONGUEUR_MIN:
                ecrit, resultat = lettre_finie(modele, trace, ecrit, maintenant)
            trace.effacer()
            if ecrit and not ecrit.endswith(" "):     # pas d'espace au début ni en double
                ecrit += " "
                t_espace = maintenant

        # --- 4. Fin de lettre (pause) ---
        fini = (not trace.vide() and geste != "ecrire"
                and maintenant - trace.t_dernier >= PAUSE_RECO)
        if fini and trace.longueur() < LONGUEUR_MIN:
            trace.effacer()
        elif fini and not args.collecte:
            ecrit, resultat = lettre_finie(modele, trace, ecrit, maintenant)
        elif fini and args.collecte and attente is None:
            attente = reconnaitre(modele, trace.traits)

        # --- 5. Affichage ---
        affichage = frame.copy()
        dessiner_traits(affichage, trace.traits)
        if pts is not None:
            maintenu = gestes.candidat in ("ouverte", "trois")
            dessiner_main(affichage, pts, gestes.candidat if maintenu else geste, gestes.progression())
        assombrir(affichage, 0, 0, w, 70)
        if args.collecte:
            total = sum(compte.values())
            if cible:
                texte(affichage, f"COLLECTE   Dessine : {cible}", (20, 48), 1.1, JAUNE, 3)
                texte(affichage, f"{compte[cible]}/{args.par_lettre} pour {cible}   total : {total}",
                      (w - 380, 45), 0.7, BLANC, 2)
            else:
                texte(affichage, f"Collecte terminee ! ({total} gestes)  q = quitter", (20, 48), 1.0, VERT, 3)
            if attente is not None:
                dessiner_resultat(affichage, attente[0], attente[1], w - 180, 110)
                texte(affichage, "ESPACE = enregistrer    C = recommencer", (20, 115), 0.9, VERT, 2, fond=True)
            aide = "index = ecrire | index+majeur = lever | main ouverte = effacer | n = autre lettre | q = quitter"
        else:
            texte(affichage, "Texte : " + ecrit + ("_" if int(maintenant * 2) % 2 else ""), (20, 50), 1.3, BLANC, 3)
            if resultat and maintenant - resultat[2] < DUREE_RESULTAT:
                dessiner_resultat(affichage, resultat[0], resultat[1], w - 180, 110)
            if maintenant - t_espace < 0.8:
                texte(affichage, "[ espace ]", (20, 115), 0.9, VIOLET, 2, fond=True)
            aide = ("index = ecrire | index+majeur = lever | 3 doigts = espace | main ouverte = effacer"
                    " | x = suppr. | q = quitter")
        assombrir(affichage, 0, h - 36, w, h)
        texte(affichage, aide, (12, h - 12), 0.5, GRIS, 1)
        texte(affichage, f"{fps:.0f} FPS", (w - 90, h - 12), 0.5, GRIS, 1)
        cv2.imshow(fenetre, affichage)

        # --- 6. Clavier ---
        touche = cv2.waitKey(1) & 0xFF
        if touche in (ord("q"), 27):
            break
        if touche == ord("c"):
            trace.effacer()
            attente = None
        if args.collecte:
            if touche == ord(" ") and cible and not trace.vide():
                img28 = attente[0] if attente is not None else traits_vers_image(trace.traits)
                enregistrer_geste(cible, trace.traits, (w, h), img28)
                compte[cible] += 1
                trace.effacer()
                attente = None
                cible = prochaine_lettre(lettres, compte, args.par_lettre, cible)
            elif touche == ord("n") and cible:
                cible = prochaine_lettre(lettres, compte, args.par_lettre, cible, forcer_autre=True)
        else:
            if touche == ord(" ") and not trace.vide() and trace.longueur() >= LONGUEUR_MIN:
                ecrit, resultat = lettre_finie(modele, trace, ecrit, maintenant)
            elif touche == ord("x"):
                ecrit = ecrit[:-1]

    cap.release()
    cv2.destroyAllWindows()
    if args.collecte:
        print(f"{sum(compte.values())} gestes enregistrés dans {DOSSIER_GESTES}/")
        print("Réentraîne le modèle avec :  python entrainer.py")
    elif ecrit:
        print("Texte écrit :", ecrit)
    return ecrit


if __name__ == "__main__":
    main()
