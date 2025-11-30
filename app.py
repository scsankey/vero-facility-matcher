"""
vero_engine.py
VERO Entity Resolution Engine
Core matching logic for facility entity resolution
"""

import pandas as pd
import numpy as np
import networkx as nx
from rapidfuzz import fuzz, distance
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

FEATURE_COLS = [
    "jw_name", "lev_name", "token_sim", "starts_with", "ends_with", "nickname",
    "jw_alt_a", "jw_alt_b", "jw_alt_both",
    "cos_embed",
    "district_exact", "district_fuzzy",
    "phone_exact", "phone_partial", "phone_both_missing",
]

HIGH_CONFIDENCE_THRESHOLD = 0.90
MEDIUM_CONFIDENCE_THRESHOLD = 0.75
NAME_SIMILARITY_THRESHOLD = 0.85
DISTRICT_BLOCKING_THRESHOLD = 0.75

def prepare_unified_dataset(gov_df, ngo_df, whatsapp_df):
    """Combine all datasets into unified format"""
    gov_prepared = pd.DataFrame({
        'RecordID': gov_df['RecordID'],
        'Name': gov_df['OfficialFacilityName'],
        'AltName': gov_df.get('AltName', None),
        'District': gov_df['District'],
        'Phone': None,
        'Source': 'Gov'
    })
    
    ngo_prepared = pd.DataFrame({
        'RecordID': ngo_df['RecordID'],
        'Name': ngo_df['FacilityName'],
        'AltName': None,
        'District': ngo_df['District'],
        'Phone': ngo_df.get('Phone', None),
        'Source': 'NGO'
    })
    
    whatsapp_prepared = pd.DataFrame({
        'RecordID': whatsapp_df['RecordID'],
        'Name': whatsapp_df['RelatedFacility'],
        'AltName': whatsapp_df.get('LocationNickname', None),
        'District': whatsapp_df['DistrictNote'],
        'Phone': whatsapp_df.get('Phone', None),
        'Source': 'WhatsApp'
    })
    
    unified = pd.concat([gov_prepared, ngo_prepared, whatsapp_prepared], ignore_index=True)
    
    unified["RecordID"] = unified["RecordID"].astype(str)
    unified["Source"] = unified["Source"].astype(str)
    unified["name_clean"] = (
        unified["Name"].fillna("") + " " + unified["AltName"].fillna("")
    ).str.strip().str.lower()
    unified["alt_a_clean"] = unified["Name"].fillna("").str.lower()
    unified["alt_b_clean"] = unified["AltName"].fillna("").str.lower()
    unified["district_clean"] = unified["District"].fillna("").str.lower().str.strip()
    unified["phone_clean"] = unified["Phone"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    
    return unified

def extract_facility_type(name: str) -> str:
    """Extract facility type from name"""
    name_lower = str(name).lower()
    
    if 'district hospital' in name_lower:
        return 'district_hospital'
    elif 'hospital' in name_lower:
        return 'hospital'
    elif 'rural health centre' in name_lower or 'rhc' in name_lower:
        return 'rural_health_centre'
    elif 'health centre' in name_lower or 'health center' in name_lower or ' hc' in name_lower:
        return 'health_centre'
    elif 'clinic' in name_lower:
        return 'clinic'
    elif 'health post' in name_lower:
        return 'health_post'
    else:
        return 'unknown'

def nickname_score(name_a: str, name_b: str) -> int:
    """Simple nickname heuristic"""
    tokens_a = name_a.split()
    tokens_b = name_b.split()
    if not tokens_a or not tokens_b:
        return 0
    if tokens_a[0] == tokens_b[0]:
        return 1
    if tokens_a[0] and tokens_b[0] and tokens_a[0][0] == tokens_b[0][0]:
        return 1
    return 0

def compute_pair_features(row_a, row_b, id_to_emb):
    """Compute all similarity features for a pair"""
    name_a = row_a["name_clean"]
    name_b = row_b["name_clean"]
    
    jw_name = distance.JaroWinkler.normalized_similarity(name_a, name_b)
    lev_name = distance.Levenshtein.normalized_similarity(name_a, name_b)
    token_sim = fuzz.token_set_ratio(name_a, name_b) / 100.0
    
    tokens_a = name_a.split()
    tokens_b = name_b.split()
    starts_with = int(len(tokens_a) > 0 and len(tokens_b) > 0 and tokens_a[0] == tokens_b[0])
    ends_with = int(len(tokens_a) > 0 and len(tokens_b) > 0 and tokens_a[-1] == tokens_b[-1])
    nickname = nickname_score(name_a, name_b)
    
    alt_a_a = row_a["alt_a_clean"]
    alt_a_b = row_b["alt_a_clean"]
    alt_b_a = row_a["alt_b_clean"]
    alt_b_b = row_b["alt_b_clean"]
    
    jw_alt_a = distance.JaroWinkler.normalized_similarity(alt_a_a, alt_a_b)
    jw_alt_b = distance.JaroWinkler.normalized_similarity(alt_b_a, alt_b_b)
    jw_alt_both = distance.JaroWinkler.normalized_similarity(
        alt_a_a + " " + alt_b_a, alt_a_b + " " + alt_b_b
    )
    
    district_a = row_a["district_clean"]
    district_b = row_b["district_clean"]
    district_exact = int(district_a == district_b and district_a != "")
    district_fuzzy = fuzz.token_set_ratio(district_a, district_b) / 100.0
    
    phone_a = row_a["phone_clean"]
    phone_b = row_b["phone_clean"]
    phone_exact = int(phone_a != "" and phone_a == phone_b)
    
    last4_a = phone_a[-4:] if len(phone_a) >= 4 else ""
    last4_b = phone_b[-4:] if len(phone_b) >= 4 else ""
    phone_partial = int(last4_a != "" and last4_a == last4_b)
    phone_both_missing = int(phone_a == "" and phone_b == "")
    
    emb_a = id_to_emb.get(row_a.name)
    emb_b = id_to_emb.get(row_b.name)
    cos_embed = float(util.cos_sim(emb_a, emb_b).cpu().numpy()[0][0]) if emb_a is not None and emb_b is not None else 0.0
    
    return {
        "jw_name": jw_name,
        "lev_name": lev_name,
        "token_sim": token_sim,
        "starts_with": starts_with,
        "ends_with": ends_with,
        "nickname": nickname,
        "jw_alt_a": jw_alt_a,
        "jw_alt_b": jw_alt_b,
        "jw_alt_both": jw_alt_both,
        "cos_embed": cos_embed,
        "district_exact": district_exact,
        "district_fuzzy": district_fuzzy,
        "phone_exact": phone_exact,
        "phone_partial": phone_partial,
        "phone_both_missing": phone_both_missing,
    }

def should_compare(rid_a, rid_b, row_a, row_b, district_threshold=DISTRICT_BLOCKING_THRESHOLD):
    """Tightened blocking logic - requires at least 2 conditions"""
    if rid_a == rid_b:
        return False
    
    conditions_met = 0
    
    d_a = row_a["district_clean"]
    d_b = row_b["district_clean"]
    if d_a and d_b:
        district_score = fuzz.token_set_ratio(d_a, d_b) / 100.0
        if district_score >= district_threshold:
            conditions_met += 1
    
    p_a = row_a["phone_clean"]
    p_b = row_b["phone_clean"]
    last4_a = p_a[-4:] if len(p_a) >= 4 else ""
    last4_b = p_b[-4:] if len(p_b) >= 4 else ""
    if last4_a and last4_a == last4_b:
        conditions_met += 1
    
    common_words = {'health', 'centre', 'center', 'hospital', 'district', 
                   'rural', 'clinic', 'post', 'community', 'area'}
    tokens_a = set(row_a["name_clean"].split()) - common_words
    tokens_b = set(row_b["name_clean"].split()) - common_words
    
    if len(tokens_a & tokens_b) > 0:
        conditions_met += 1
    
    type_a = extract_facility_type(row_a["Name"])
    type_b = extract_facility_type(row_b["Name"])
    if type_a != 'unknown' and type_b != 'unknown':
        if type_a != type_b:
            return False
    
    return conditions_met >= 2

def generate_candidate_pairs(unified_df):
    """Generate candidate pairs using blocking"""
    u_idx = unified_df.set_index("RecordID")
    
    ngo_ids = unified_df[unified_df["Source"] == "NGO"]["RecordID"].tolist()
    gov_ids = unified_df[unified_df["Source"] == "Gov"]["RecordID"].tolist()
    wa_ids = unified_df[unified_df["Source"] == "WhatsApp"]["RecordID"].tolist()
    
    candidate_pairs = []
    
    for rid_ng in ngo_ids:
        row_ng = u_idx.loc[rid_ng]
        for rid_g in gov_ids:
            row_g = u_idx.loc[rid_g]
            if should_compare(rid_ng, rid_g, row_ng, row_g):
                candidate_pairs.append((rid_ng, rid_g))
    
    for rid_ng in ngo_ids:
        row_ng = u_idx.loc[rid_ng]
        for rid_w in wa_ids:
            row_w = u_idx.loc[rid_w]
            if should_compare(rid_ng, rid_w, row_ng, row_w):
                candidate_pairs.append((rid_ng, rid_w))
    
    for rid_g in gov_ids:
        row_g = u_idx.loc[rid_g]
        for rid_w in wa_ids:
            row_w = u_idx.loc[rid_w]
            if should_compare(rid_g, rid_w, row_g, row_w):
                candidate_pairs.append((rid_g, rid_w))
    
    return list(set(candidate_pairs))

def train_model(similarity_features_df):
    """Train logistic regression model"""
    X = similarity_features_df[FEATURE_COLS].fillna(0)
    y = similarity_features_df['SameEntity']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight='balanced',
        C=1.0
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    
    return model, scaler, metrics

def build_clusters(matched_pairs_df, unified_df):
    """Build entity clusters using graph connected components"""
    G = nx.Graph()
    for _, row in matched_pairs_df.iterrows():
        G.add_edge(row["record_A"], row["record_B"])
    
    clusters = list(nx.connected_components(G))
    
    u_idx = unified_df.set_index("RecordID")
    cluster_rows = []
    
    for i, comp in enumerate(clusters, start=1):
        cluster_id = f"Cluster_{i}"
        for rid in comp:
            row = u_idx.loc[rid]
            cluster_rows.append({
                "ClusterID": cluster_id,
                "RecordID": rid,
                "Source": row["Source"],
                "Name": row["Name"],
                "AltName": row["AltName"],
                "District": row["District"],
                "Phone": row["Phone"],
            })
    
    all_matched_ids = set(matched_pairs_df["record_A"]) | set(matched_pairs_df["record_B"])
    singleton_ids = set(unified_df["RecordID"]) - all_matched_ids
    
    for rid in singleton_ids:
        row = u_idx.loc[rid]
        cluster_rows.append({
            "ClusterID": f"Singleton_{rid}",
            "RecordID": rid,
            "Source": row["Source"],
            "Name": row["Name"],
            "AltName": row["AltName"],
            "District": row["District"],
            "Phone": row["Phone"],
        })
    
    return pd.DataFrame(cluster_rows)

def build_golden_tables(clusters_df: pd.DataFrame) -> dict:
    """Build golden tables for each entity type from clusters"""
    if clusters_df is None or len(clusters_df) == 0:
        return {}

    clusters_df = clusters_df.copy()

    if "EntityType" not in clusters_df.columns:
        clusters_df["EntityType"] = "facility"

    golden_tables = {}

    for entity_type, group in clusters_df.groupby("EntityType"):
        rows = []
        
        for cluster_id, cluster_rows in group.groupby("ClusterID"):
            canonical_name = None
            if "Name" in cluster_rows.columns:
                non_null = cluster_rows["Name"].dropna().astype(str)
                if len(non_null) > 0:
                    canonical_name = non_null.value_counts().idxmax()

            alias_values = []
            for col in ["Name", "AltName"]:
                if col in cluster_rows.columns:
                    alias_values.append(cluster_rows[col].dropna().astype(str))
            
            aliases_str = ""
            if alias_values:
                aliases = pd.unique(pd.concat(alias_values, ignore_index=True))
                aliases_str = "; ".join(sorted(aliases))

            main_district = None
            if "District" in cluster_rows.columns:
                non_null_dist = cluster_rows["District"].dropna().astype(str)
                if len(non_null_dist) > 0:
                    main_district = non_null_dist.value_counts().idxmax()

            sources_str = ""
            if "Source" in cluster_rows.columns:
                srcs = cluster_rows["Source"].dropna().unique().tolist()
                sources_str = ", ".join(sorted(map(str, srcs)))

            rows.append({
                "GoldenID": f"{entity_type[:3].upper()}_{cluster_id}",
                "ClusterID": cluster_id,
                "EntityType": entity_type,
                "CanonicalName": canonical_name,
                "Aliases": aliases_str,
                "MainDistrict": main_district,
                "SourcesRepresented": sources_str,
                "RecordCount": len(cluster_rows),
                "LLM_Summary": None,
            })

        golden_df = pd.DataFrame(rows)

        if entity_type == "facility":
            key = "golden_facilities"
        elif entity_type == "person":
            key = "golden_persons"
        elif entity_type == "district":
            key = "golden_districts"
        else:
            key = f"golden_{entity_type}s"

        golden_tables[key] = golden_df

    return golden_tables

def build_canonical_entities_table(golden_tables: dict) -> pd.DataFrame:
    """Merge all golden tables into single canonical entity table"""
    dfs = []
    for key, df in golden_tables.items():
        if df is None or len(df) == 0:
            continue
        
        tmp = df.copy()
        for col in ["GoldenID", "EntityType", "CanonicalName", "Aliases",
                    "MainDistrict", "SourcesRepresented", "RecordCount", "LLM_Summary", "ClusterID"]:
            if col not in tmp.columns:
                tmp[col] = None
        
        dfs.append(tmp[[
            "GoldenID", "ClusterID", "EntityType", "CanonicalName", "Aliases",
            "MainDistrict", "SourcesRepresented", "RecordCount", "LLM_Summary"
        ]])

    if not dfs:
        return pd.DataFrame(columns=[
            "GoldenID", "ClusterID", "EntityType", "CanonicalName", "Aliases",
            "MainDistrict", "SourcesRepresented", "RecordCount", "LLM_Summary"
        ])

    return pd.concat(dfs, ignore_index=True)

def run_vero_pipeline(gov_df, ngo_df, whatsapp_df, ground_truth_df=None, use_pretrained=False, pretrained_model=None, pretrained_scaler=None):
    """Main VERO pipeline - from raw data to golden records"""
    
    print("="*70)
    print("VERO PIPELINE STARTING")
    print("="*70)
    
    print("\n[1/6] Preparing unified dataset...")
    unified = prepare_unified_dataset(gov_df, ngo_df, whatsapp_df)
    print(f"✅ Unified dataset: {len(unified)} records")
    
    print("\n[2/6] Computing embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(unified["name_clean"].tolist(), convert_to_tensor=True)
    id_to_emb = {rid: emb for rid, emb in zip(unified["RecordID"], embeddings)}
    print(f"✅ Embeddings computed")
    
    if use_pretrained and pretrained_model and pretrained_scaler:
        print("\n[3/6] Using pre-trained model...")
        model = pretrained_model
        scaler = pretrained_scaler
        metrics = {"note": "Using pre-trained model"}
    elif ground_truth_df is not None:
        print("\n[3/6] Training model from ground truth...")
        u_idx = unified.set_index("RecordID")
        
        features = []
        for _, row in ground_truth_df.iterrows():
            rec_a_id = str(row['Record_ID A'])
            rec_b_id = str(row['Record_ID B'])
            same_entity = 1 if str(row['Same Entity']).lower() == 'yes' else 0
            
            if rec_a_id in u_idx.index and rec_b_id in u_idx.index:
                rec_a = u_idx.loc[rec_a_id]
                rec_b = u_idx.loc[rec_b_id]
                
                feats = compute_pair_features(rec_a, rec_b, id_to_emb)
                feats['SameEntity'] = same_entity
                features.append(feats)
        
        similarity_features = pd.DataFrame(features)
        model, scaler, metrics = train_model(similarity_features)
        print(f"✅ Model trained - ROC-AUC: {metrics['roc_auc']:.3f}")
    else:
        raise ValueError("Must provide either ground_truth_df or use_pretrained=True with model/scaler")
    
    print("\n[4/6] Generating candidate pairs with blocking...")
    candidate_pairs = generate_candidate_pairs(unified)
    print(f"✅ Generated {len(candidate_pairs)} candidate pairs")
    
    print("\n[5/6] Computing features and predicting matches...")
    u_idx = unified.set_index("RecordID")
    
    rows = []
    for rid_a, rid_b in candidate_pairs:
        row_a = u_idx.loc[rid_a]
        row_b = u_idx.loc[rid_b]
        feats = compute_pair_features(row_a, row_b, id_to_emb)
        feats["record_A"] = rid_a
        feats["record_B"] = rid_b
        feats["source_A"] = row_a["Source"]
        feats["source_B"] = row_b["Source"]
        feats["name_A"] = row_a["Name"]
        feats["name_B"] = row_b["Name"]
        rows.append(feats)
    
    cand_features = pd.DataFrame(rows)
    
    X_cand = cand_features[FEATURE_COLS].values
    X_cand_scaled = scaler.transform(X_cand)
    probs = model.predict_proba(X_cand_scaled)[:, 1]
    cand_features["match_prob"] = probs
    
    high_conf = cand_features[cand_features["match_prob"] >= HIGH_CONFIDENCE_THRESHOLD]
    medium_conf = cand_features[
        (cand_features["match_prob"] >= MEDIUM_CONFIDENCE_THRESHOLD) &
        (cand_features["match_prob"] < HIGH_CONFIDENCE_THRESHOLD) &
        (cand_features["jw_name"] >= NAME_SIMILARITY_THRESHOLD)
    ]
    matched_pairs = pd.concat([high_conf, medium_conf], ignore_index=True)
    matched_pairs = matched_pairs.sort_values("match_prob", ascending=False)
    
    print(f"✅ Found {len(matched_pairs)} strong matches")
    
    print("\n[6/6] Building clusters and golden records...")
    clusters = build_clusters(matched_pairs, unified)
    
    golden_tables = build_golden_tables(clusters)
    canonical_entities = build_canonical_entities_table(golden_tables)
    
    golden = golden_tables.get("golden_facilities", pd.DataFrame())
    
    print(f"✅ Created {len(golden)} golden facility records")
    print(f"✅ Created {len(canonical_entities)} canonical entities")
    
    print("\n" + "="*70)
    print("VERO PIPELINE COMPLETE")
    print("="*70)
    
    return {
        "unified": unified,
        "clusters": clusters,
        "golden": golden,
        "matched_pairs": matched_pairs,
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "golden_facilities": golden,
        "golden_persons": golden_tables.get("golden_persons", pd.DataFrame()),
        "golden_districts": golden_tables.get("golden_districts", pd.DataFrame()),
        "canonical_entities": canonical_entities,
    }
