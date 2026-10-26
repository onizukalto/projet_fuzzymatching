"""Étapes 2 à 6 du pipeline d'entity resolution. Chaque fonction = une tâche Airflow."""
import os
import re
import unicodedata

import joblib
import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from rapidfuzz import fuzz
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

# ---------- Configuration (variables d'environnement, avec valeurs par défaut pour ton PC) ----------
DATA_DIR = os.getenv("DATA_DIR", "data")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
PG_URL = os.getenv("PG_URL", "postgresql+psycopg2://er:er@localhost:5432/er")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
BATCH = 2000
QUERY_BATCH = 500

TOP_K = 20
SEUIL = 0.5
FEATURES = ["name_score", "addr_score", "city_score", "phone_eq", "phone_diff", "emb_dist"]

ABBR_NORM = {
    r"\bste\b": "societe",
    r"\bcie\b": "compagnie",
    r"\bets\b": "etablissements",
    r"&": " et ",
    # Abréviations de rue
    r"\bbd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\br\b": "rue",
    r"\bpl\b": "place",
    r"\bch\b": "chemin",
    r"\bimp\b": "impasse",
    r"\ball\b": "allee",
}

# ---------- Fonctions utilitaires (reprises du notebook) ----------
def normalize(text):
    text = str(text or "").lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    for pattern, remplacement in ABBR_NORM.items():
        text = re.sub(pattern, remplacement, text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def digits(s):
    return re.sub(r"\D", "", str(s or ""))[-9:]


def pair_features(rec, cand, dist):
    tel_erp, tel_crm = digits(rec["phone"]), digits(cand["phone"])
    return {
        "name_score": fuzz.token_sort_ratio(normalize(rec["name"]), normalize(cand["name"])) / 100,
        "addr_score": fuzz.token_set_ratio(normalize(rec["address"]), normalize(cand["address"])) / 100,
        "city_score": fuzz.ratio(normalize(rec["city"]), normalize(cand["city"])) / 100,
        "phone_eq":   float(tel_erp != "" and tel_erp == tel_crm),                    # même numéro
        "phone_diff": float(tel_erp != "" and tel_crm != "" and tel_erp != tel_crm),  # deux numéros différents
        "emb_dist":   dist,
    }


def get_engine():
    return create_engine(PG_URL)


def get_collection(reset=False):
    client = chromadb.PersistentClient(path=f"{DATA_DIR}/chroma")
    if reset:
        try:
            client.delete_collection("crm")
        except Exception:
            pass
    return client.get_or_create_collection(
        "crm", embedding_function=DefaultEmbeddingFunction(), metadata={"hnsw:space": "cosine"}
    )


# ---------- Étape 2 : chargement dans PostgreSQL ----------
def load_raw():
    eng = get_engine()
    for source in ["crm", "erp"]:
        df = pd.read_csv(f"{DATA_DIR}/source_{source}.csv", dtype=str).fillna("")
        df["text"] = (df["name"] + " " + df["address"] + " " + df["city"]).map(normalize)
        df.to_sql(f"raw_{source}", eng, if_exists="replace", index=False)
        print(f"raw_{source} : {len(df)} lignes")


def index_embeddings():
    crm = pd.read_sql("SELECT id, text FROM raw_crm", get_engine())
    col = get_collection(reset=True)
    for start in range(0, len(crm), BATCH):
        lot = crm.iloc[start:start + BATCH]
        col.add(ids=lot["id"].tolist(), documents=lot["text"].tolist())
        print(f"  {min(start + BATCH, len(crm))}/{len(crm)} fiches indexées")
    print(f"{col.count()} fiches CRM indexées")


# ---------- Étape 4 : blocking + features fuzzy ----------
def build_candidates():
    eng = get_engine()
    crm = pd.read_sql("SELECT * FROM raw_crm", eng).set_index("id")
    erp = pd.read_sql("SELECT * FROM raw_erp", eng)
    
    col = get_collection()
    res = {"ids": [], "distances": []}
    for start in range(0, len(erp), QUERY_BATCH):
        lot = erp["text"].iloc[start:start + QUERY_BATCH].tolist()
        r = col.query(query_texts=lot, n_results=TOP_K, include=["distances"])
        res["ids"] += r["ids"]
        res["distances"] += r["distances"]
        print(f"  {min(start + QUERY_BATCH, len(erp))}/{len(erp)} fiches ERP interrogées")

    rows = []
    for i, fiche in erp.iterrows():
        if i % 1000 == 0:
            print(f"  features : {i}/{len(erp)} fiches ERP traitées")
        for cid, dist in zip(res["ids"][i], res["distances"][i]):
            cand = crm.loc[cid]
            rows.append({
                "erp_id": fiche["id"],
                "crm_id": cid,
                "label": int(fiche["true_id"] == cand["true_id"]),
                **pair_features(fiche, cand, dist),
            })
    pairs = pd.DataFrame(rows)
    pairs.to_sql("candidate_pairs", eng, if_exists="replace", index=False)

    # Recall du blocking, enregistré pour être loggé dans MLflow à l'étape suivante
    connues = erp[erp.true_id != "-1"]
    recall = pairs[pairs.label == 1].erp_id.nunique() / len(connues)
    pd.DataFrame([{"blocking_recall": recall}]).to_sql("blocking_stats", eng, if_exists="replace", index=False)
    print(f"{len(pairs)} paires candidates, {pairs.label.sum()} vrais matchs, recall du blocking = {recall:.1%}")


# ---------- Étape 5 : entraînement + MLflow ----------
def train_model():
    eng = get_engine()
    pairs = pd.read_sql("SELECT * FROM candidate_pairs", eng)
    blocking_recall = float(pd.read_sql("SELECT blocking_recall FROM blocking_stats", eng).iloc[0, 0])

    X_train, X_test, y_train, y_test = train_test_split(
        pairs[FEATURES], pairs["label"], test_size=0.3, stratify=pairs["label"], random_state=0
    )
    
    params = {"n_estimators": 200, "max_depth": 6, "top_k": TOP_K,
              "difficulty": os.getenv("DIFFICULTY", "easy"),
              "features": ",".join(FEATURES),
              "street_abbr": True,
              "n_entities": int(os.getenv("N_ENTITIES", "500"))}

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("entity-resolution")
    with mlflow.start_run(run_name="pipeline"):
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                     max_depth=params["max_depth"], random_state=0)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        metrics = {
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "blocking_recall": blocking_recall,
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
                # Recall de bout en bout : part des doublons retrouvés par l'ensemble du pipeline
        mlflow.log_metric("e2e_recall", blocking_recall * cv_metrics["cv_recall_mean"])
                # Validation croisée à 5 plis, groupée par fiche ERP
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_validate(
            RandomForestClassifier(n_estimators=params["n_estimators"],
                                   max_depth=params["max_depth"], random_state=0),
            pairs[FEATURES], pairs["label"],
            groups=pairs["erp_id"], cv=cv,
            scoring=["precision", "recall", "f1"],
        )
        cv_metrics = {}
        for m in ["precision", "recall", "f1"]:
            cv_metrics[f"cv_{m}_mean"] = scores[f"test_{m}"].mean()
            cv_metrics[f"cv_{m}_std"] = scores[f"test_{m}"].std()
        mlflow.log_metrics(cv_metrics)
        print("Validation croisée :", {k: round(v, 3) for k, v in cv_metrics.items()})
        mlflow.sklearn.log_model(clf, name="model", serialization_format="cloudpickle")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print("Métriques :", metrics)


# ---------- Étape 6 : résolution finale ----------
def resolve():
    eng = get_engine()
    pairs = pd.read_sql("SELECT * FROM candidate_pairs", eng)
    clf = joblib.load(MODEL_PATH)

    pairs["proba"] = clf.predict_proba(pairs[FEATURES])[:, 1]
    best = pairs.sort_values("proba", ascending=False).drop_duplicates("erp_id")
    matches = best[best["proba"] >= SEUIL][["erp_id", "crm_id", "proba"]]
    matches.to_sql("matches", eng, if_exists="replace", index=False)
    print(f"{len(matches)} fiches ERP rattachées au CRM")


# ---------- Tout le pipeline d'un coup (pour tester sans Airflow) ----------
def run_all():
    load_raw()
    index_embeddings()
    build_candidates()
    train_model()
    resolve()


if __name__ == "__main__":
    run_all()