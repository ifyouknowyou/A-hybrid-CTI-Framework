#!/usr/bin/env python3
"""
A Hybrid Cyber Threat Intelligence Framework Integrating OSINT and Structured Threat Data
==========================================================================================
"""

from __future__ import annotations

import os
import json
import time
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Prevent tokenizers deadlocks
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import requests
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, 
    precision_recall_curve, auc
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler  # Note: These are deprecated; using torch.amp below

from transformers import (
    AutoTokenizer, AutoModel, BertTokenizerFast,
    BertForSequenceClassification, get_linear_schedule_with_warmup
)

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Optional packages
try:
    from imblearn.over_sampling import SMOTE
    IMBL_AVAILABLE = True
except Exception:
    IMBL_AVAILABLE = False

# NOTE: SHAP often pulls in TensorFlow, which can be incompatible with Kaggle's
# protobuf/TensorFlow stack and spam errors like:
#   AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
# To keep the framework robust, we lazy-import SHAP only when explicitly enabled.
SHAP_AVAILABLE = False

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except Exception:
    NEO4J_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


# =====================================================================
# CONFIGURATION
# =====================================================================
OUTPUT_DIR = Path("./output_hybrid_cti")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

CONFIG = {
    # OSINT Data Sources
    "osint_data_paths": [
        "/kaggle/input/big-twitter-dataset/raw_tweets_text.csv",  # Example: Twitter OSINT
        "/kaggle/input/cyber-bert/CyberBERT.csv"  # Example: Cyber-specific OSINT
    ],
    # Structured Threat Data
    "structured_data_paths": [
        "/kaggle/input/text-based-cyber-threat-detection/cyber-threat-intelligence_all.csv"  # Example: CSV with IOCs, labels
    ],
    "sample_n": 100000,
    "random_seed": 42,
    
    # Text cleaning
    "keep_hashtag_word": True,
    
    # IOC Enrichment (optional, requires API key)
    "enrich_iocs": False,
    "virustotal_api_key": None,  # Set for IOC enrichment
    
    # TF-IDF configuration
    "tfidf_max_features": 20000,
    "tfidf_ngram_range": (1, 2),
    
    # CNN configuration
    "cnn_max_len": 100,
    "cnn_embed_dim": 128,
    "cnn_num_filters": 128,
    "cnn_filter_sizes": [3, 4, 5],
    "cnn_epochs": 3,
    "cnn_batch_size": 256,
    "cnn_dropout": 0.3,
    
    # BERT configuration
    "bert_model_name": "bert-base-uncased",
    "bert_max_len": 128,
    "bert_epochs": 3,
    "bert_batch_size": 32,
    "bert_accumulation_steps": 1,
    "bert_early_stopping_patience": 2,
    
    # Embeddings configuration
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    
    # Training toggles
    "do_baseline": False,
    "do_embeddings_lr": False,
    "do_cnn": True,
    "do_bert": True,
    # Explainability
    "do_shap": False,
    
    # Threat Detection
    "threat_detection_threshold": 0.7,  # Probability threshold for alerts
    "ioc_weight": 0.3,  # Weight for IOCs in threat score
    "graph_centrality_weight": 0.2,  # Weight for graph centrality
    
    # Data balancing
    "balance_method": "undersample",  # "undersample", "oversample", or None
    "test_size": 0.2,
    
    # Feature extraction
    "do_ioc_features": True,
    "do_mitre_mapping": True,
    "mitre_keywords_path": "/kaggle/input/mitre-attack-technique/mitre_keywords.csv",
    
    # Knowledge graph
    "do_build_kg": True,
    "push_to_neo4j": False,
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password",
    
    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "log_level": logging.INFO
}


# =====================================================================
# LOGGING SETUP
# =====================================================================
def setup_logging():
    """Configure logging for the framework."""
    logger = logging.getLogger("hybrid_cti_framework")
    logger.setLevel(CONFIG["log_level"])
    
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(message)s'
        ))
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(OUTPUT_DIR / "framework.log")
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(message)s'
        ))
        logger.addHandler(fh)
    
    return logger

logger = setup_logging()


# =====================================================================
# UTILITIES
# =====================================================================
def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json_serializable(obj, path):
    """Save object to JSON with numpy/pandas type conversion."""
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, pd.Series):
            return x.tolist()
        if isinstance(x, pd.Timestamp):
            return str(x)
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        return x
    
    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2)

def get_global_split(df):
    """
    Returns global stratified train/test indices.
    """
    from sklearn.model_selection import train_test_split

    y = df["label"].values
    indices = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        indices,               # split indices
        test_size=CONFIG["test_size"],
        stratify=y,
        random_state=CONFIG["random_seed"]
    )

    return train_idx, test_idx


# =====================================================================
# IOC ENRICHMENT (STRUCTURED THREAT DATA INTEGRATION)
# =====================================================================
def enrich_ioc(ioc: str, ioc_type: str) -> Dict[str, Union[str, int]]:
    """Enrich IOC using external APIs (e.g., VirusTotal)."""
    if not CONFIG["enrich_iocs"] or not CONFIG["virustotal_api_key"]:
        return {"enriched": False}
    
    try:
        if ioc_type == "hashes":
            url = f"https://www.virustotal.com/api/v3/files/{ioc}"
        elif ioc_type == "ips":
            url = f"https://www.virustotal.com/api/v3/ip_addresses/{ioc}"
        elif ioc_type == "domains":
            url = f"https://www.virustotal.com/api/v3/domains/{ioc}"
        else:
            return {"enriched": False}
        
        headers = {"x-apikey": CONFIG["virustotal_api_key"]}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            malicious = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {}).get("malicious", 0)
            return {"enriched": True, "malicious_score": malicious}
        else:
            return {"enriched": False, "error": response.status_code}
    except Exception as e:
        logger.debug(f"IOC enrichment failed for {ioc}: {e}")
        return {"enriched": False, "error": str(e)}


# =====================================================================
# TEXT PREPROCESSING
# =====================================================================
URL_PATTERN = re.compile(r"http\S+|www\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_ALPHANUM = re.compile(r"[^A-Za-z0-9\s\.\-/_']")

CYBER_KEYWORDS = [
    "malware", "ransomware", "phishing", "cve", "exploit", "vulnerability",
    "hacker", "breach", "payload", "botnet", "ioc", "zero-day", "trojan",
    "virus", "worm", "ddos", "cyberattack", "cyber security", "firewall",
    "encryption", "backdoor", "rootkit", "spyware", "adware", "keylogger",
    "mitm", "sql injection", "xss", "csrf", "buffer overflow", "patch",
    "update", "threat", "alert", "incident", "forensic", "pentest",
    "red team", "blue team", "soc", "siem", "ids", "ips"
]

IOC_PATTERNS = {
    "ips": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
    "domains": r"\b(?:[a-zA-Z0-9-]+\.)+(?:com|net|org|io|ru|cn|info|biz|co|xyz|gov|edu)\b",
    "hashes": r"\b[a-fA-F0-9]{32,64}\b",
    "cves": r"\bCVE-\d{4}-\d{4,7}\b"
}

def clean_text(text: str, keep_hashtag_word: bool = True) -> str:
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""
    
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)
    
    if keep_hashtag_word:
        text = HASHTAG_PATTERN.sub(r"\1", text)
    else:
        text = HASHTAG_PATTERN.sub(" ", text)
    
    text = NON_ALPHANUM.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def robust_label(text: str, keywords: List[str] = CYBER_KEYWORDS) -> int:
    """Label text using word-boundary matching to reduce false positives."""
    if not text or not isinstance(text, str):
        return 0
    
    txt = " " + text + " "
    for k in keywords:
        pat = r"\b" + re.escape(k) + r"\b"
        if re.search(pat, txt, flags=re.IGNORECASE):
            return 1
    return 0

def extract_iocs(text: str) -> Dict[str, List[str]]:
    """Extract IOCs from text."""
    result = {}
    for k, pat in IOC_PATTERNS.items():
        flags = re.IGNORECASE if k == "cves" else 0
        result[k] = re.findall(pat, text, flags=flags)
    return result

def add_ioc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add IOC count features to dataframe."""
    iocs = df['clean_text'].apply(extract_iocs)
    df["num_ips"] = iocs.map(lambda x: len(x["ips"]))
    df["num_domains"] = iocs.map(lambda x: len(x["domains"]))
    df["num_hashes"] = iocs.map(lambda x: len(x["hashes"]))
    df["num_cves"] = iocs.map(lambda x: len(x["cves"]))
    df["cyber_keyword_count"] = df["clean_text"].apply(
        lambda x: sum(1 for k in CYBER_KEYWORDS 
                     if re.search(r"\b" + re.escape(k) + r"\b", x, flags=re.IGNORECASE))
    )
    
    # Enrich IOCs if enabled
    if CONFIG["enrich_iocs"]:
        enriched = []
        for _, row in df.iterrows():
            row_enrich = {}
            for ioc_type, ioc_list in extract_iocs(row['clean_text']).items():
                for ioc in ioc_list:
                    enrich_data = enrich_ioc(ioc, ioc_type)
                    row_enrich[f"{ioc_type}_{ioc}"] = enrich_data.get("malicious_score", 0)
            enriched.append(row_enrich)
        enrich_df = pd.DataFrame(enriched).fillna(0)
        df = pd.concat([df, enrich_df], axis=1)
    
    return df


# =====================================================================
# MITRE ATT&CK MAPPING
# =====================================================================
def load_mitre_keywords(path: str) -> List[Dict]:
    """Load MITRE ATT&CK keywords from CSV."""
    if not os.path.exists(path):
        logger.warning(f"MITRE keywords file not found: {path}")
        return []
    
    try:
        df = pd.read_csv(path)
        required = {"technique_id", "technique_name", "keywords"}
        
        if not required.issubset(set(df.columns)):
            logger.error(f"MITRE file missing required columns: {required}")
            return []
        
        out = []
        for _, r in df.iterrows():
            kws = [k.strip().lower() for k in str(r["keywords"]).split(";") 
                   if k.strip()]
            out.append({
                "id": r["technique_id"],
                "name": r["technique_name"],
                "keywords": kws
            })
        
        logger.info(f"Loaded {len(out)} MITRE techniques")
        return out
    
    except Exception as e:
        logger.error(f"Failed to load MITRE keywords: {e}")
        return []

def map_to_mitre(text: str, mitre_list: List[Dict]) -> List[str]:
    """Map text to MITRE techniques using keyword lookup."""
    if not mitre_list:
        return []
    
    text_l = text.lower()
    matches = []
    
    for entry in mitre_list:
        for kw in entry["keywords"]:
            if kw in text_l:
                matches.append(entry["id"])
                break
    
    return matches


# =====================================================================
# DATA LOADING & PREPARATION
# =====================================================================
def load_and_merge_datasets(osint_paths: List[str], structured_paths: List[str]) -> pd.DataFrame:
    """Load and merge OSINT and structured threat datasets with robust text auto-detection."""
    
    dfs = []

    # -------------------------
    # Load OSINT data
    # -------------------------
    for p in osint_paths:
        p = str(p)
        if not os.path.exists(p):
            logger.warning(f"OSINT file not found: {p}")
            continue

        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            logger.warning(f"Failed to read OSINT {p}: {e}")
            continue

        # Auto-detect OSINT text column (case-insensitive)
        lower_cols = {c.lower(): c for c in df.columns}
        text_col = None

        for cand in [
            "text", "description", "summary", "details",
            "attack_description", "event_description", "content"
        ]:
            if cand in lower_cols:
                text_col = lower_cols[cand]
                break

        if text_col is None:
            logger.warning(f"No suitable text column in OSINT {p}")
            continue

        # Ensure text is string and fill missing values
        df_sub = df[[text_col]].rename(columns={text_col: "text"}).astype(str).fillna("")

        if "label" in df.columns:
            df_sub["label"] = df["label"]

        dfs.append(df_sub)
        logger.info(
            f"Loaded {len(df_sub)} rows from OSINT {p} "
            f"(mapped '{text_col}' → 'text')"
        )

    # -------------------------
    # Load structured data
    # -------------------------
    for p in structured_paths:
        p = str(p)
        if not os.path.exists(p):
            logger.warning(f"Structured file not found: {p}")
            continue

        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as e:
            logger.warning(f"Failed to read structured {p}: {e}")
            continue

        # Auto-detect structured text column (case-insensitive)
        lower_cols = {c.lower(): c for c in df.columns}
        text_col = None

        for cand in [
            "text", "description", "summary", "details",
            "attack_description", "event_description", "content"
        ]:
            if cand in lower_cols:
                text_col = lower_cols[cand]
                break

        # Optional: fallback to keyword-based column search
        if text_col is None:
            for c in df.columns:
                c_lower = c.lower()
                if any(k in c_lower for k in ["desc", "detail", "narrat", "report", "content", "text"]):
                    text_col = c
                    break

        if text_col is None:
            logger.warning(f"Structured data {p} skipped (no usable text column)")
            continue

        # Ensure text is string and fill missing values
        df_sub = df[[text_col]].rename(columns={text_col: "text"}).astype(str).fillna("")

        if "label" in df.columns:
            df_sub["label"] = df["label"]

        dfs.append(df_sub)
        logger.info(
            f"Loaded {len(df_sub)} rows from structured {p} "
            f"(mapped '{text_col}' → 'text')"
        )

    # -------------------------
    # Final merge
    # -------------------------
    if not dfs:
        raise ValueError("No valid datasets found")

    merged = pd.concat(dfs, ignore_index=True)
    merged["text"] = merged["text"].astype(str).fillna("")

    logger.info(f"Merged dataset size: {len(merged)} rows")
    return merged


def prepare_data(df: pd.DataFrame, sample_n: Optional[int] = None) -> pd.DataFrame:
    """Prepare and clean dataset."""
    logger.info("Preparing data...")
    
    # Clean text
    df['clean_text'] = df['text'].map(lambda x: clean_text(x, CONFIG["keep_hashtag_word"]))
    
    # Remove empty and duplicates
    df = df[df['clean_text'] != ""].drop_duplicates(subset=["clean_text"])
    logger.info(f"After cleaning & dedup: {len(df)} rows")
    
    # Sample if needed
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=CONFIG["random_seed"]).reset_index(drop=True)
        logger.info(f"Sampled to {len(df)} rows")
    
    # Label
    df['label'] = df['clean_text'].apply(lambda x: robust_label(x, CYBER_KEYWORDS))
    logger.info(f"Labels: {df['label'].sum()} positive, {len(df) - df['label'].sum()} negative")
    
    # IOC features (directly update df)
    if CONFIG["do_ioc_features"]:
        df = add_ioc_features(df)
    
    # MITRE mapping
    if CONFIG["do_mitre_mapping"]:
        mitre_list = load_mitre_keywords(CONFIG["mitre_keywords_path"])
        df["mitre_matches"] = df["clean_text"].map(lambda t: map_to_mitre(t, mitre_list))
    
    return df.reset_index(drop=True)

def balance_dataset(df: pd.DataFrame, method: Optional[str] = "undersample") -> pd.DataFrame:
    """Balance dataset using specified method."""
    if method is None:
        return df
    
    class_counts = df["label"].value_counts().to_dict()
    logger.info(f"Class counts before balancing: {class_counts}")
    
    if method == "undersample":
        min_n = df["label"].value_counts().min()
        dfs = [df[df["label"] == l].sample(min_n, random_state=CONFIG["random_seed"]) 
               for l in df["label"].unique()]
        out = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=CONFIG["random_seed"]).reset_index(drop=True)
    
    elif method == "oversample" and IMBL_AVAILABLE:
        max_n = df["label"].value_counts().max()
        dfs = []
        for l in df["label"].unique():
            dlab = df[df["label"] == l]
            if len(dlab) < max_n:
                dlab = dlab.sample(max_n, replace=True, random_state=CONFIG["random_seed"])
            dfs.append(dlab)
        out = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=CONFIG["random_seed"]).reset_index(drop=True)
    
    else:
        out = df
    
    logger.info(f"Class counts after balancing: {out['label'].value_counts().to_dict()}")
    return out


# =====================================================================
# KNOWLEDGE GRAPH BUILDING
# =====================================================================
# Initialize spaCy
nlp = None
if SPACY_AVAILABLE:
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy model loaded successfully")
    except Exception as e:
        logger.warning(f"spaCy model not found: {e}")
        try:
            logger.info("Attempting to download spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception as e2:
            logger.warning(f"Failed to load spaCy: {e2}")
            nlp = None
else:
    logger.warning("spaCy not installed. NER functionality disabled.")

def extract_ner_entities(text: str) -> List[Tuple[str, str]]:
    """Extract NER entities from text."""
    if nlp is None or not text:
        return []
    
    try:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents 
                if ent.label_ in {"ORG", "PRODUCT", "PERSON", "GPE", "NORP", "LOC"}]
    except Exception as e:
        logger.debug(f"NER extraction failed: {e}")
        return []

def build_kg(df: pd.DataFrame, batch_size: int = 10000) -> nx.Graph:
    """Build knowledge graph from dataset with MITRE ATT&CK integration."""
    if not CONFIG["do_build_kg"]:
        return nx.Graph()
    
    logger.info("Building knowledge graph...")
    G = nx.Graph()
    
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        
        for _, row in batch.iterrows():
            text = row.get("clean_text", "")
            if not text:
                continue
            
            ents = extract_ner_entities(text)
            iocs = extract_iocs(text)
            mitres = row.get("mitre_matches", [])  # 🔹 MITRE
            
            # =====================
            # ENTITY NODES
            # =====================
            for ent_text, ent_label in ents:
                key = f"ENT::{ent_text}"
                if not G.has_node(key):
                    G.add_node(key, type="entity", entity_label=ent_label)
            
            # =====================
            # IOC NODES + EDGES
            # =====================
            for ip in iocs["ips"]:
                ipk = f"IP::{ip}"
                if not G.has_node(ipk):
                    G.add_node(ipk, type="ip")
                for ent_text, _ in ents:
                    G.add_edge(f"ENT::{ent_text}", ipk, relation="mentions_ip")
            
            for d in iocs["domains"]:
                dk = f"DOM::{d}"
                if not G.has_node(dk):
                    G.add_node(dk, type="domain")
                for ent_text, _ in ents:
                    G.add_edge(f"ENT::{ent_text}", dk, relation="mentions_domain")
            
            for cve in iocs["cves"]:
                ck = f"CVE::{cve}"
                if not G.has_node(ck):
                    G.add_node(ck, type="cve")
                for ent_text, _ in ents:
                    G.add_edge(f"ENT::{ent_text}", ck, relation="mentions_cve")
            
            # =====================
            # 🔥 MITRE ATT&CK NODES + EDGES (NEW)
            # =====================
            for tech in mitres:
                tech_node = f"MITRE::{tech}"
                
                if not G.has_node(tech_node):
                    G.add_node(tech_node, type="mitre_technique")
                
                # Entity → MITRE technique
                for ent_text, _ in ents:
                    G.add_edge(
                        f"ENT::{ent_text}",
                        tech_node,
                        relation="uses_technique"
                    )
            
            # =====================
            # MITRE CO-OCCURRENCE (OPTIONAL BUT IMPORTANT)
            # =====================
            for i in range(len(mitres)):
                for j in range(i + 1, len(mitres)):
                    G.add_edge(
                        f"MITRE::{mitres[i]}",
                        f"MITRE::{mitres[j]}",
                        relation="co_occurs"
                    )
    
    logger.info(f"KG: nodes={len(G.nodes())}, edges={len(G.edges())}")
    return G


def push_to_neo4j(G: nx.Graph):
    """Push knowledge graph to Neo4j."""
    if not NEO4J_AVAILABLE or not CONFIG["push_to_neo4j"]:
        logger.warning("Neo4j push disabled or unavailable")
        return
    
    try:
        driver = GraphDatabase.driver(
            CONFIG["neo4j_uri"],
            auth=(CONFIG["neo4j_user"], CONFIG["neo4j_password"])
        )
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        return
    
    def tx_func(tx, batch_nodes, batch_rels):
        for node, attrs in batch_nodes:
            tx.run("MERGE (n:Node {id:$id}) SET n += $attrs",
                   id=str(node), attrs=attrs)
        for u, v, attrs in batch_rels:
            tx.run("""
                MATCH (a:Node {id:$u}), (b:Node {id:$v})
                MERGE (a)-[r:RELATED {type:$t}]->(b)
            """, u=str(u), v=str(v), t=attrs.get("relation", "related_to"))
    
    nodes = list(G.nodes(data=True))
    edges = list(G.edges(data=True))
    batch = 500
    
    with driver.session() as session:
        for i in range(0, len(nodes), batch):
            b = nodes[i:i + batch]
            session.write_transaction(tx_func, b, [])
        
        for i in range(0, len(edges), batch):
            b = edges[i:i + batch]
            session.write_transaction(tx_func, [], b)
    
    logger.info("Knowledge graph pushed to Neo4j")


# =====================================================================
# THREAT DETECTION
# =====================================================================
def compute_threat_score(df: pd.DataFrame, model_probs: np.ndarray, G: nx.Graph) -> pd.DataFrame:
    """Compute threat scores for entities based on model predictions, IOCs, and graph centrality."""
    logger.info("Computing threat scores...")
    
    df = df.copy()
    df["model_prob"] = model_probs
    
    # IOC score (check if columns exist)
    if all(col in df.columns for col in ["num_ips", "num_domains", "num_hashes", "num_cves"]):
        df["ioc_score"] = (df["num_ips"] + df["num_domains"] + df["num_hashes"] + df["num_cves"]) * CONFIG["ioc_weight"]
    else:
        logger.warning("IOC features not found; setting ioc_score to 0")
        df["ioc_score"] = 0.0
    
    # Graph centrality score (if KG is built)
    centrality_scores = {}
    if G and len(G.nodes()) > 0:
        centrality = nx.degree_centrality(G)
        for node in G.nodes():
            if node.startswith("ENT::"):
                ent = node[5:]  # Remove "ENT::" prefix
                centrality_scores[ent] = centrality.get(node, 0)
    
    df["centrality_score"] = df["clean_text"].apply(
        lambda text: max([centrality_scores.get(ent, 0) for ent, _ in extract_ner_entities(text)], default=0)
    ) * CONFIG["graph_centrality_weight"]
    
    # Combined threat score
    df["threat_score"] = df["model_prob"] + df["ioc_score"] + df["centrality_score"]
    
    # Alerts
    df["alert"] = df["threat_score"] > CONFIG["threat_detection_threshold"]
    
    logger.info(f"Threat detection: {df['alert'].sum()} alerts generated")
    return df


# =====================================================================
# EVALUATION METRICS
# =====================================================================
def compute_metrics(y_true, y_pred, y_score=None, prefix="model"):
    res = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None:
        res["roc_auc"] = float(roc_auc_score(y_true, y_score))
        # OPTIONAL (scalar only)
        # res["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        res["roc_auc"] = None

    logger.info(
        f"[{prefix}] acc={res['accuracy']:.4f} "
        f"f1={res['f1']:.4f} roc_auc={res['roc_auc']}"
    )

    return res

###

# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================

def plot_roc_pr_curves(results: Dict, out_dir: str):
    """Plot ROC and PR curves for all models."""
    ensure_dir(out_dir)
    
    for model_name, metrics in results.items():
        if not metrics or "roc_curve" not in metrics or not metrics.get("roc_curve"):
            continue
        
        # ROC curve
        roc = metrics.get("roc_curve")
        if roc:
            #fpr = np.array(roc["fpr"])
            #tpr = np.array(roc["tpr"])
            auc_score = metrics.get("roc_auc", 0)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, lw=2, label=f"AUC = {auc_score:.3f}")
            plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            roc_path = os.path.join(out_dir, f"{model_name}_roc.png")
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved ROC curve to {roc_path}")
            plt.close()
        
        # PR curve
        pr = metrics.get("pr_curve")
        if pr:
            precision = np.array(pr["precision"])
            recall = np.array(pr["recall"])
            pr_auc = pr.get("pr_auc", 0)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, lw=2, label=f"PR-AUC = {pr_auc:.3f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve - {model_name}")
            plt.legend(loc="best")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            pr_path = os.path.join(out_dir, f"{model_name}_pr.png")
            plt.savefig(pr_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved PR curve to {pr_path}")
            plt.close()

def plot_model_comparison(results: Dict, out_dir: str):
    """Plot comparison of all models."""
    ensure_dir(out_dir)
    
    metrics_names = ["accuracy", "precision", "recall", "f1"]
    model_names = list(results.keys())
    
    data = {metric: [] for metric in metrics_names}
    
    for model_name in model_names:
        if model_name not in results or not results[model_name]:
            continue
        
        for metric in metrics_names:
            data[metric].append(results[model_name].get(metric, 0))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Comparison", fontsize=16, fontweight='bold')
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics_names)):
        if data[metric]:
            bars = ax.bar(model_names, data[metric], color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f"{metric.capitalize()} Comparison")
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    comp_path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(comp_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved model comparison to {comp_path}")
    plt.close()


####
def plot_accuracy_comparison(results: Dict, out_dir: str):
    """
    Paper-ready accuracy comparison bar chart.
    """
    ensure_dir(out_dir)

    model_names = []
    accuracies = []

    for model_name, metrics in results.items():
        if metrics and "accuracy" in metrics:
            model_names.append(model_name)
            accuracies.append(metrics["accuracy"])

    if not model_names:
        logger.warning("No accuracy data available for comparison plot")
        return

    plt.figure(figsize=(8, 6))
    bars = plt.bar(
        model_names,
        accuracies,
        edgecolor="black",
        alpha=0.85
    )

    plt.ylim(0, 1)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    plt.tight_layout()
    out_path = os.path.join(out_dir, "accuracy_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved accuracy comparison plot to {out_path}")

####


def plot_combined_roc(results: Dict, out_dir: str):
    ensure_dir(out_dir)

    plt.figure(figsize=(6, 5))

    for model_name, m in results.items():
        if not m or not m.get("roc_curve"):
            continue
        #fpr = np.array(m["roc_curve"]["fpr"])
        #tpr = np.array(m["roc_curve"]["tpr"])
        auc_val = m.get("roc_auc", None)
        label = f"{model_name} (AUC={auc_val:.3f})" if auc_val else model_name
        plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig1_combined_roc.pdf"))
    plt.close()



def plot_accuracy_ieee(results: Dict, out_dir: str):
    ensure_dir(out_dir)

    models, accs = [], []
    for k, v in results.items():
        if v and "accuracy" in v:
            models.append(k)
            accs.append(v["accuracy"])

    plt.figure(figsize=(6, 4))
    bars = plt.bar(models, accs, edgecolor="black")

    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison Across Models", fontsize=14, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}",
                 ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_accuracy_comparison.pdf"))
    plt.close()


# =====================================================================
# CNN WITH BATCH NORMALIZATION
# =====================================================================
class ImprovedTextCNN(nn.Module):
    """CNN model with batch normalization and dropout."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 num_filters: int = 128, filter_sizes: List[int] = [3, 4, 5],
                 dropout: float = 0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                     kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in filter_sizes
        ])
        
        self.fc = nn.Linear(num_filters * len(filter_sizes), 64)
        self.out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        emb = self.embedding(x)  # (B, L, E)
        emb = emb.permute(0, 2, 1)  # (B, E, L)
        
        conv_outs = []
        for conv, bn in zip(self.convs, self.bns):
            c = torch.relu(conv(emb))
            c = bn(c)
            pooled = torch.max(c, dim=2)[0]
            conv_outs.append(pooled)
        
        cat = torch.cat(conv_outs, dim=1)
        x = self.dropout(torch.relu(self.fc(cat)))
        x = self.out(x)
        
        return x.squeeze(1)

def train_cnn(
    df: pd.DataFrame,
    output_dir: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray) -> Dict:
    """
    Train a CNN model on text data using tokenization via BERT tokenizer.
    Supports logging, GPU/CPU, balanced BCE loss, and saves the model + tokenizer meta.
    Returns evaluation metrics dictionary.
    """
    logger.info("Starting CNN training...")

    # Set device
    device = torch.device(CONFIG["device"])
    set_random_seeds(CONFIG["random_seed"])

    # Tokenization
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["bert_model_name"])
    enc = tokenizer(
        df["clean_text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=CONFIG["cnn_max_len"],
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].numpy()
    labels = df["label"].values.astype(int)

    # Train/test split using explicit indices
    X_train = input_ids[train_idx]
    X_test  = input_ids[test_idx]
    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    # Create TensorDatasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.long),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["cnn_batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=CONFIG["cnn_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize CNN model
    vocab_size = tokenizer.vocab_size + 2
    model = ImprovedTextCNN(
        vocab_size=vocab_size,
        embed_dim=CONFIG["cnn_embed_dim"],
        num_filters=CONFIG["cnn_num_filters"],
        filter_sizes=CONFIG["cnn_filter_sizes"],
        dropout=CONFIG["cnn_dropout"]
    ).to(device)

    # BCEWithLogitsLoss with pos_weight for imbalance
    pos = max(1, (labels == 1).sum())
    neg = max(1, (labels == 0).sum())
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    max_grad_norm = 1.0  # optional gradient clipping

    # Training loop with early stopping
    best_loss = float("inf")
    early_stop_counter = 0
    patience = 3

    for epoch in range(CONFIG["cnn_epochs"]):
        model.train()
        epoch_loss = 0.0

        for i, (Xb, yb) in enumerate(train_loader):
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()

            if i % 50 == 0:
                logger.info(
                    f"CNN epoch {epoch+1}/{CONFIG['cnn_epochs']} "
                    f"batch {i}/{len(train_loader)} loss={loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"CNN epoch {epoch+1} average loss={avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_counter = 0
            ensure_dir(output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, "cnn_model.pt"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Evaluation
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.numpy().astype(int))

    # Compute metrics
    metrics = compute_metrics(all_labels, all_preds, y_score=np.array(all_probs), prefix="cnn")
    logger.info(
    f"CNN metrics | acc={metrics['accuracy']:.4f}, "
    f"prec={metrics['precision']:.4f}, "
    f"rec={metrics['recall']:.4f}, "
    f"f1={metrics['f1']:.4f}, "
    f"roc_auc={metrics['roc_auc']}"
    )


    # Save tokenizer meta
    save_json_serializable(
        {"bert_model_name": CONFIG["bert_model_name"], "cnn_max_len": CONFIG["cnn_max_len"]},
        os.path.join(output_dir, "cnn_tokenizer_meta.json")
    )

    return metrics



def predict_cnn_probs(model: nn.Module,
                      input_ids: np.ndarray,
                      batch_size: int,
                      device: torch.device) -> np.ndarray:
    """Predict P(label=1) for the CNN model on an array of token ids."""
    model.eval()
    probs = []
    ds = TensorDataset(torch.tensor(input_ids, dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for (Xb,) in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            p1 = torch.sigmoid(logits)
            probs.extend(p1.detach().cpu().numpy().tolist())
    return np.array(probs, dtype=float)


# =====================================================================
# BERT FINE-TUNING WITH GRADIENT ACCUMULATION AND EARLY STOPPING
# =====================================================================
class OnTheFlyBertDataset(Dataset):
    """On-the-fly tokenization dataset for memory efficiency."""
    
    def __init__(self, texts: List[str], labels: List[int],
                 tokenizer: BertTokenizerFast, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        txt = str(self.texts[idx])
        encoding = self.tokenizer(txt, truncation=True, padding='max_length',
                                 max_length=self.max_len, return_tensors=None)
        
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item

def collate_fn(batch):
    """Custom collate function for batch processing."""
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out

def predict_bert_probs(model: BertForSequenceClassification,
                       loader: DataLoader,
                       device: torch.device) -> np.ndarray:
    """Predict P(label=1) for a dataloader of BERT batches."""
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            # HuggingFace expects keyword args, not a dict as one positional arg.
            outputs = model(**batch)
            logits = outputs.logits  # (B, 2)
            p1 = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p1.detach().cpu().numpy().tolist())
    return np.array(probs, dtype=float)

def train_bert(df: pd.DataFrame, output_dir: str, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict:
    """
    Train BERT with gradient accumulation, early stopping, and global stratified split.
    Saves best model and tokenizer, returns evaluation metrics.
    """
    logger.info("Starting BERT training...")

    device = torch.device(CONFIG["device"])
    set_random_seeds(CONFIG["random_seed"])

    # Extract train/val using global indices
    X_train = df.loc[train_idx, "clean_text"].values
    y_train = df.loc[train_idx, "label"].values.astype(int)

    X_val = df.loc[test_idx, "clean_text"].values
    y_val = df.loc[test_idx, "label"].values.astype(int)

    # Tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(CONFIG["bert_model_name"])

    train_ds = OnTheFlyBertDataset(list(X_train), list(y_train), tokenizer, CONFIG["bert_max_len"])
    val_ds = OnTheFlyBertDataset(list(X_val), list(y_val), tokenizer, CONFIG["bert_max_len"])

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["bert_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["bert_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # Model
    model = BertForSequenceClassification.from_pretrained(CONFIG["bert_model_name"], num_labels=2)
    model.to(device)

    # Optimizer & Scheduler
    total_steps = len(train_loader) * CONFIG["bert_epochs"] // CONFIG["bert_accumulation_steps"]
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )

    # Mixed precision scaler
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    # Early stopping
    best_f1 = -1.0
    no_improve = 0
    patience = CONFIG["bert_early_stopping_patience"]

    # Training loop
    for epoch in range(CONFIG["bert_epochs"]):
        logger.info(f"BERT epoch {epoch+1}/{CONFIG['bert_epochs']}")
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            # Mixed precision
            if scaler is not None:
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(**batch)
                    loss = outputs.loss / CONFIG["bert_accumulation_steps"]

                scaler.scale(loss).backward()

                if (step + 1) % CONFIG["bert_accumulation_steps"] == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss / CONFIG["bert_accumulation_steps"]
                loss.backward()

                if (step + 1) % CONFIG["bert_accumulation_steps"] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            total_loss += loss.item() * CONFIG["bert_accumulation_steps"]

            if step % 20 == 0:
                logger.info(f"Step {step}/{len(train_loader)} loss={loss.item() * CONFIG['bert_accumulation_steps']:.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"BERT epoch {epoch+1} average loss={avg_loss:.6f}")

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(batch["labels"].cpu().numpy())

        metrics = compute_metrics(trues, preds, prefix=f"bert_epoch_{epoch+1}")

        # Early stopping
        if metrics["f1"] > best_f1 + 1e-4:
            best_f1 = metrics["f1"]
            no_improve = 0

            # Save best model + tokenizer
            ensure_dir(output_dir)
            model.save_pretrained(os.path.join(output_dir, "best_bert"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best_bert_tokenizer"))
            logger.info(f"Model saved at epoch {epoch+1} with F1={best_f1:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Final evaluation on validation set
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(batch["labels"].cpu().numpy())

    final_metrics = compute_metrics(trues, preds, prefix="bert_final")
    logger.info(f"Final BERT metrics: {final_metrics}")

    return final_metrics



# =====================================================================
# MAIN PIPELINE
# =====================================================================
def run_pipeline():
    """
    Run complete Hybrid Cyber Threat Intelligence pipeline.
    Supports:
        - Data loading, preparation, balancing
        - Global stratified train/test split
        - CNN and BERT training with early stopping
        - Knowledge graph construction (MITRE + OSINT)
        - Threat detection with best model
        - IEEE-ready evaluation figures (ROC, confusion matrices, accuracy)
        - Alerts report
    """
    logger.info("="*70)
    logger.info("Starting Hybrid Cyber Threat Intelligence Framework")
    logger.info("="*70)

    start_time = time.time()
    set_random_seeds(CONFIG["random_seed"])
    ensure_dir(OUTPUT_DIR)

    # ----------------------------
    # Load and prepare data
    # ----------------------------
    try:
        df = load_and_merge_datasets(CONFIG["osint_data_paths"], CONFIG["structured_data_paths"])
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    df = prepare_data(df, sample_n=CONFIG["sample_n"])
    df = balance_dataset(df, CONFIG["balance_method"])

    # ----------------------------
    # Global Stratified Split (IEEE-compliant)
    # ----------------------------
    train_idx, test_idx = get_global_split(df)

    # ----------------------------
    # Knowledge Graph (MITRE + OSINT)
    # ----------------------------
    if CONFIG["do_build_kg"]:
        try:
            G = build_kg(df)  # build MITRE + OSINT knowledge graph
            nx.write_gexf(G, OUTPUT_DIR / "cti_graph.gexf")
            logger.info(f"Saved knowledge graph to {OUTPUT_DIR / 'cti_graph.gexf'}")
            
            if CONFIG["push_to_neo4j"]:
                push_to_neo4j(G)
        except Exception as e:
            logger.error(f"KG building failed: {e}")
            G = None
    else:
        G = None

    results = {}
    artifacts = {}

    # ----------------------------
    # Train CNN
    # ----------------------------
    if CONFIG["do_cnn"]:
        try:
            metrics = train_cnn(df, str(OUTPUT_DIR), train_idx=train_idx, test_idx=test_idx)
            results["CNN"] = metrics
            artifacts["cnn_trained"] = True
        except Exception as e:
            logger.error(f"CNN training failed: {e}")
            results["CNN"] = {"error": str(e)}
            artifacts["cnn_trained"] = False

    # ----------------------------
    # Train BERT
    # ----------------------------
    if CONFIG["do_bert"]:
        try:
            metrics = train_bert(df, str(OUTPUT_DIR / "bert"), train_idx=train_idx, test_idx=test_idx)
            results["BERT"] = metrics
            artifacts["bert_trained"] = True
        except Exception as e:
            logger.error(f"BERT training failed: {e}")
            results["BERT"] = {"error": str(e)}
            artifacts["bert_trained"] = False

    # ----------------------------
    # Threat Detection (Choose best model)
    # ----------------------------
    if results:
        best_model_key = None
        if artifacts.get("bert_trained"):
            best_model_key = "BERT"
        elif artifacts.get("cnn_trained"):
            best_model_key = "CNN"
        else:
            best_model_key = list(results.keys())[0]

        logger.info(f"Using {best_model_key} for threat detection")
        model_probs = None
        device = torch.device(CONFIG["device"])

        # ---------- BERT Inference ----------
        if best_model_key == "BERT" and artifacts.get("bert_trained"):
            try:
                bert_dir = OUTPUT_DIR / "bert" / "best_bert"
                tok_dir = OUTPUT_DIR / "bert" / "best_bert_tokenizer"
                model = BertForSequenceClassification.from_pretrained(str(bert_dir))
                tokenizer = BertTokenizerFast.from_pretrained(str(tok_dir))
                model.to(device)
                model.eval()

                infer_ds = OnTheFlyBertDataset(
                    texts=df["clean_text"].tolist(),
                    labels=[0]*len(df),
                    tokenizer=tokenizer,
                    max_len=CONFIG["bert_max_len"]
                )
                infer_loader = DataLoader(
                    infer_ds,
                    batch_size=CONFIG["bert_batch_size"],
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0
                )

                model_probs = predict_bert_probs(model, infer_loader, device)

            except Exception as e:
                logger.warning(f"BERT inference failed, falling back to CNN: {e}")
                best_model_key = "CNN"

        # ---------- CNN Inference ----------
        if model_probs is None and best_model_key == "CNN" and artifacts.get("cnn_trained"):
            try:
                tokenizer = BertTokenizerFast.from_pretrained(CONFIG["bert_model_name"])
                enc = tokenizer(df["clean_text"].tolist(), truncation=True,
                                padding="max_length", max_length=CONFIG["cnn_max_len"],
                                return_tensors="pt")
                input_ids = enc["input_ids"].numpy()
                vocab_size = tokenizer.vocab_size + 2

                model = ImprovedTextCNN(
                    vocab_size=vocab_size,
                    embed_dim=CONFIG["cnn_embed_dim"],
                    num_filters=CONFIG["cnn_num_filters"],
                    filter_sizes=CONFIG["cnn_filter_sizes"],
                    dropout=CONFIG["cnn_dropout"]
                )
                model.load_state_dict(torch.load(OUTPUT_DIR / "cnn_model.pt", map_location=device))
                model.to(device)
                model.eval()

                model_probs = predict_cnn_probs(model, input_ids, batch_size=CONFIG["cnn_batch_size"], device=device)
            except Exception as e:
                logger.warning(f"CNN inference failed, using random probabilities: {e}")
                model_probs = np.random.rand(len(df))

        if model_probs is None:
            logger.warning("No model probabilities available; using random probabilities.")
            model_probs = np.random.rand(len(df))

        # ----------------------------
        # Compute Threat Scores
        # ----------------------------
        df_with_scores = compute_threat_score(df, model_probs, G)

        # Save results
        save_json_serializable(results, OUTPUT_DIR / "model_results.json")
        df_with_scores.to_csv(OUTPUT_DIR / "threat_detection_results.csv", index=False)

        # ----------------------------
        # Generate IEEE-Ready Evaluation Figures
        # ----------------------------
        fig_dir = OUTPUT_DIR / "figures"
        ensure_dir(fig_dir)

        plot_combined_roc(results, fig_dir)
        
        plot_accuracy_ieee(results, fig_dir)
        logger.info("IEEE-ready evaluation figures generated")

        # ----------------------------
        # Alerts Summary
        # ----------------------------
        alerts = df_with_scores[df_with_scores["alert"] == True]
        logger.info(f"Generated {len(alerts)} threat alerts")

        if len(alerts) > 0:
            alerts_summary = {
                "total_alerts": len(alerts),
                "avg_threat_score": float(alerts["threat_score"].mean()),
                "max_threat_score": float(alerts["threat_score"].max()),
                "alert_examples": alerts[["clean_text", "threat_score", "alert"]].head(10).to_dict("records")
            }
            save_json_serializable(alerts_summary, OUTPUT_DIR / "alerts_summary.json")

    elapsed_time = time.time() - start_time
    logger.info("="*70)
    logger.info("Pipeline completed successfully!")
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info(f"Models evaluated: {list(results.keys())}")
    logger.info("="*70)


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    run_pipeline()
