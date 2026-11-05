"""
Air Writing — entraînement du classifieur de lettres (scikit-learn)
--------------------------------------------------------------------
Données :
  1. Tracés SYNTHÉTIQUES générés à partir de modèles de lettres + déformations
     (rotation, italique, tremblement, liaisons parasites...). Aucun téléchargement.
  2. Tes VRAIS gestes enregistrés avec :  python air_writing.py --collecte
     (dossier mes_gestes/). Ce sont eux qui disent si le modèle marche vraiment.

Évaluation :
  - précision sur des tracés synthétiques jamais vus (optimiste) ;
  - si tu as des vrais gestes : précision du modèle "synthétique seul" sur tes gestes,
    puis précision "synthétique + tes gestes" sur une partie de tes gestes gardée de côté.

Utilisation :
    python entrainer.py
    python entrainer.py --n-par-lettre 600 --modele mlp
"""
import argparse
import glob
import json
import os
import time
from collections import Counter

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from pretraitement import (LETTRES, augmenter, generer_synthetique,
                           image_vers_features, traits_vers_image)

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DOSSIER_GESTES = os.path.join(DOSSIER, "mes_gestes")
FICHIER_MODELE = os.path.join(DOSSIER, "modele_lettres.joblib")


def creer_classifieur(nom):
    if nom == "svm":
        # SVM + calibration pour obtenir des probabilités (top-3 affiché dans l'appli)
        clf = CalibratedClassifierCV(SVC(kernel="rbf", C=10, gamma="scale"), ensemble=False, cv=3)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-3, max_iter=300,
                            early_stopping=True, random_state=0)
    return make_pipeline(StandardScaler(), clf)


def charger_vrais_gestes(dossier):
    """Lit les fichiers JSON enregistrés par le mode collecte."""
    gestes = []
    for f in sorted(glob.glob(os.path.join(dossier, "*", "*.json"))):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("lettre") in LETTRES and d.get("traits"):
            gestes.append((d["traits"], d["lettre"]))
    return gestes


def features_vrais(gestes, rng=None, n_aug=0):
    """Features des vrais gestes (+ copies légèrement déformées pour l'entraînement)."""
    X, y = [], []
    for traits, lettre in gestes:
        X.append(image_vers_features(traits_vers_image(traits)))
        y.append(lettre)
        for _ in range(n_aug):
            X.append(image_vers_features(traits_vers_image(augmenter(traits, rng, force=0.4))))
            y.append(lettre)
    return np.array(X), np.array(y)


def afficher_confusions(y_vrai, y_pred, titre, top=8):
    cm = confusion_matrix(y_vrai, y_pred, labels=list(LETTRES))
    np.fill_diagonal(cm, 0)
    paires = [(cm[i, j], LETTRES[i], LETTRES[j]) for i in range(26) for j in range(26) if cm[i, j]]
    paires.sort(reverse=True)
    if paires:
        print(f"  {titre} — confusions les plus fréquentes (vraie -> prédite) :")
        print("   " + ", ".join(f"{a}->{b} ({n})" for n, a, b in paires[:top]))


def precision_par_lettre(y_vrai, y_pred):
    lignes = []
    for L in LETTRES:
        m = y_vrai == L
        if m.any():
            lignes.append(f"{L}:{100 * (y_pred[m] == L).mean():.0f}%")
    print("  par lettre : " + "  ".join(lignes))


def main():
    p = argparse.ArgumentParser(description="Entraîne le classifieur de lettres Air Writing")
    p.add_argument("--n-par-lettre", type=int, default=400, help="tracés synthétiques par lettre")
    p.add_argument("--modele", choices=["svm", "mlp"], default="svm")
    p.add_argument("--aug-vrais", type=int, default=5,
                   help="copies déformées de chaque vrai geste ajoutées à l'entraînement")
    p.add_argument("--gestes", default=DOSSIER_GESTES)
    p.add_argument("--sortie", default=FICHIER_MODELE)
    args = p.parse_args()
    rng = np.random.default_rng(42)

    # ---------- 1. Données synthétiques ----------
    t0 = time.time()
    print(f"Génération de {args.n_par_lettre} tracés synthétiques par lettre...")
    X_syn, y_syn, _ = generer_synthetique(args.n_par_lettre, graine=0)
    X_syn_test, y_syn_test, _ = generer_synthetique(60, graine=123)   # autre graine = jamais vus
    print(f"  {len(y_syn)} exemples d'entraînement, {len(y_syn_test)} de test ({time.time() - t0:.0f} s)")

    # ---------- 2. Modèle synthétique seul ----------
    print(f"\nEntraînement ({args.modele.upper()}) sur le synthétique...")
    t0 = time.time()
    modele = creer_classifieur(args.modele).fit(X_syn, y_syn)
    pred = modele.predict(X_syn_test)
    print(f"  terminé en {time.time() - t0:.0f} s")
    print(f"  précision sur synthétique jamais vu : {100 * accuracy_score(y_syn_test, pred):.1f} %"
          "  (optimiste : même type de données)")
    afficher_confusions(y_syn_test, pred, "synthétique")

    # ---------- 3. Vrais gestes ----------
    gestes = charger_vrais_gestes(args.gestes)
    if not gestes:
        print(f"\nAucun vrai geste dans '{args.gestes}/'. Pour en enregistrer :"
              "\n    python air_writing.py --collecte")
    else:
        compte = Counter(l for _, l in gestes)
        print(f"\n{len(gestes)} vrais gestes trouvés pour {len(compte)} lettres.")
        X_vrai, y_vrai = features_vrais(gestes)

        pred = modele.predict(X_vrai)
        print(f"  Modèle synthétique seul, testé sur TES gestes : "
              f"{100 * accuracy_score(y_vrai, pred):.1f} %")
        precision_par_lettre(y_vrai, pred)
        afficher_confusions(y_vrai, pred, "tes gestes")

        # Évaluation honnête : on garde 30 % de tes gestes de côté
        stratif = y_vrai if min(compte.values()) >= 2 else None
        idx = np.arange(len(gestes))
        if len(gestes) >= 10:
            i_train, i_test = train_test_split(idx, test_size=0.3, random_state=0, stratify=stratif)
            X_g, y_g = features_vrais([gestes[i] for i in i_train], rng, args.aug_vrais)
            m2 = creer_classifieur(args.modele).fit(np.vstack([X_syn, X_g]), np.concatenate([y_syn, y_g]))
            pred = m2.predict(X_vrai[i_test])
            print(f"  Modèle synthétique + 70 % de tes gestes, testé sur les 30 % restants : "
                  f"{100 * accuracy_score(y_vrai[i_test], pred):.1f} %  ({len(i_test)} gestes)")
            afficher_confusions(y_vrai[i_test], pred, "après ajout de tes gestes")

        # Modèle final : tout le synthétique + tous tes gestes
        print("\nEntraînement du modèle final (synthétique + tous tes gestes)...")
        X_g, y_g = features_vrais(gestes, rng, args.aug_vrais)
        modele = creer_classifieur(args.modele).fit(np.vstack([X_syn, X_g]), np.concatenate([y_syn, y_g]))

    joblib.dump({"pipeline": modele, "lettres": list(modele.classes_),
                 "n_vrais_gestes": len(gestes)}, args.sortie)
    print(f"\nModèle enregistré : {args.sortie}")
    print("Lance maintenant :  python air_writing.py")


if __name__ == "__main__":
    main()
