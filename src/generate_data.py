"""Étape 1 : génère deux sources (CRM, ERP) décrivant les mêmes entreprises.

Deux niveaux de difficulté :
- easy : le bruit simple du début (abréviations, une faute, majuscules)
- hard : fautes multiples, abréviations de rue, villes abîmées, déménagements,
         homonymes et entreprises inconnues au même format
"""
import os
import re
import random
import pandas as pd
from faker import Faker

DATA_DIR = os.getenv("DATA_DIR", "data")
DIFFICULTY = os.getenv("DIFFICULTY", "easy")
N_ENTITIES = int(os.getenv("N_ENTITIES", "500"))

ABBR = {"Société": "Ste", "Compagnie": "Cie", "Etablissements": "Ets", "et": "&"}
STREET_ABBR = {"boulevard": "bd", "avenue": "av", "rue": "r", "place": "pl",
               "chemin": "ch", "impasse": "imp", "allée": "all"}
PREFIXES = ["Société", "Compagnie", "Etablissements", ""]


# ---------- Les différentes erreurs de saisie ----------
def typo(text):
    """Une faute de frappe : inversion, lettre supprimée ou lettre remplacée."""
    if len(text) < 5:
        return text
    i = random.randint(1, len(text) - 2)
    kind = random.choice(["swap", "delete", "replace"])
    if kind == "swap":
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if kind == "delete":
        return text[:i] + text[i + 1:]
    return text[:i] + random.choice("abcdefghijklmnopqrstuvwxyz") + text[i + 1:]


def noisy_easy(text):
    """Le bruit du début."""
    for long, court in ABBR.items():
        if random.random() < 0.5:
            text = text.replace(long, court)
    if random.random() < 0.3:
        text = typo(text)
    if random.random() < 0.3:
        text = text.upper()
    return text


def noisy_hard(text):
    """Un bruit plus réaliste."""
    for long, court in ABBR.items():
        if random.random() < 0.5:
            text = re.sub(rf"\b{long}\b", court, text)   # cette fois, mots entiers seulement
    for long, court in STREET_ABBR.items():
        if random.random() < 0.5:
            text = re.sub(rf"\b{long}\b", court, text, flags=re.IGNORECASE)
    words = text.split()
    if len(words) > 2 and random.random() < 0.1:       # un mot oublié
        words.pop(random.randrange(len(words)))
        text = " ".join(words)
    for _ in range(random.choice([0, 1, 1, 2])):          # 0 à 2 fautes de frappe
        text = typo(text)
    if random.random() < 0.3:
        text = text.upper()
    return text


def company_name(fake):
    prefix = random.choice(PREFIXES)
    return f"{prefix} {fake.last_name()} et {fake.last_name()}".strip()


# ---------- Génération ----------
def main(n=None, n_unknown=None, difficulty=None):
    n = n or N_ENTITIES
    n_unknown = n_unknown or max(100, n // 5)
    difficulty = difficulty or DIFFICULTY
    hard = difficulty == "hard"
    noisy = noisy_hard if hard else noisy_easy

    fake = Faker("fr_FR")
    random.seed(42)
    Faker.seed(42)

    # 1. Source CRM
    entities = []
    for i in range(n):
        entities.append({
            "true_id": i,
            "name": company_name(fake),
            "address": fake.street_address(),
            "city": fake.city(),
            "phone": fake.phone_number(),
        })

    # 2. Homonymes (hard) : une "jumelle" avec le même nom et la même ville, mais une autre entreprise
    if hard:
        for e in random.sample(entities, int(n * 0.1)):
            entities.append({
                "true_id": len(entities),
                "name": e["name"],
                "address": fake.street_address(),
                "city": e["city"],
                "phone": fake.phone_number(),
            })

    crm = pd.DataFrame(entities)
    crm.insert(0, "id", [f"A{i}" for i in range(len(crm))])

    # 3. Source ERP : 70 % des entreprises du CRM, abîmées
    rows = []
    for e in random.sample(entities, int(len(entities) * 0.7)):
        address, city = e["address"], e["city"]
        if hard:
            if random.random() < 0.15:
                address = fake.street_address()          # déménagement
            else:
                address = noisy(address)
            r = random.random()
            if r < 0.10:
                city = ""                                # ville non saisie
            elif r < 0.25:
                city = typo(city)                        # ville mal orthographiée
        else:
            address = noisy(address)
            city = city if random.random() > 0.1 else city.lower()

        rows.append({
            "true_id": e["true_id"],
            "name": noisy(e["name"]),
            "address": address,
            "city": city,
            "phone": e["phone"].replace(" ", "") if random.random() < 0.5 else "",
        })

    # 4. Entreprises inconnues du CRM
    known_cities = [e["city"] for e in entities]
    for _ in range(n_unknown):
        rows.append({
            "true_id": -1,
            "name": company_name(fake) if hard else fake.company(),
            "address": fake.street_address(),
            "city": random.choice(known_cities) if hard else fake.city(),
            "phone": fake.phone_number(),
        })

    erp = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
    erp.insert(0, "id", [f"B{i}" for i in range(len(erp))])

    os.makedirs(DATA_DIR, exist_ok=True)
    crm.to_csv(f"{DATA_DIR}/source_crm.csv", index=False)
    erp.to_csv(f"{DATA_DIR}/source_erp.csv", index=False)
    print(f"[{difficulty}] CRM : {len(crm)} lignes, ERP : {len(erp)} lignes -> {DATA_DIR}/")


if __name__ == "__main__":
    main()