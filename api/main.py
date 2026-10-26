"""API d'entity resolution : vérifie si une entreprise existe déjà dans le CRM."""
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, "/opt/project/src")
import pipeline as p  # noqa: E402

app = FastAPI(title="Entity Resolution API",
              description="Détecte si une entreprise existe déjà dans le CRM.")


# ---------- Format des données reçues ----------
class Entreprise(BaseModel):
    name: str
    address: str = ""
    city: str = ""
    phone: str = ""


# ---------- 1. L'API est-elle vivante ? ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- 2. Quelques chiffres sur le dernier run du pipeline ----------
@app.get("/stats")
def stats():
    eng = p.get_engine()
    compter = lambda table: int(pd.read_sql(f"SELECT COUNT(*) FROM {table}", eng).iloc[0, 0])  # noqa: E731
    return {
        "fiches_crm": compter("raw_crm"),
        "fiches_erp": compter("raw_erp"),
        "fiches_rattachees": compter("matches"),
    }


# ---------- 3. Cette entreprise existe-t-elle déjà dans le CRM ? ----------
@app.post("/match")
def match(entreprise: Entreprise):
    fiche = entreprise.model_dump()

    # a) Blocking : les 5 fiches CRM les plus proches, via ChromaDB
    texte = p.normalize(f"{fiche['name']} {fiche['address']} {fiche['city']}")
    res = p.get_collection().query(query_texts=[texte], n_results=p.TOP_K, include=["distances"])
    if not res["ids"][0]:
        raise HTTPException(status_code=503, detail="Index vide : lancez d'abord le pipeline Airflow.")

    # b) Features fuzzy + probabilité du modèle, pour chaque candidat
    crm = pd.read_sql("SELECT * FROM raw_crm", p.get_engine()).set_index("id")
    clf = joblib.load(p.MODEL_PATH)

    candidats = []
    for cid, dist in zip(res["ids"][0], res["distances"][0]):
        features = p.pair_features(fiche, crm.loc[cid], dist)
        proba = clf.predict_proba(pd.DataFrame([features])[p.FEATURES])[0, 1]
        candidats.append({
            "crm_id": cid,
            "name": crm.loc[cid, "name"],
            "city": crm.loc[cid, "city"],
            "proba": round(float(proba), 3),
        })

    # c) Le meilleur candidat, et la décision
    candidats.sort(key=lambda c: c["proba"], reverse=True)
    meilleur = candidats[0]
    return {
        "doublon": meilleur["proba"] >= p.SEUIL,
        "meilleur_candidat": meilleur,
        "candidats": candidats,
    }