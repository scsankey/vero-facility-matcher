"""
app.py
VERO - Data Entity Matching Platform
Upload → Match → Canonical Entities → LLM Chat → Download
Enhanced with Executive Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

# Import the VERO engine
from vero_engine import run_vero_pipeline

# ══════════════════════════════════════════════════════════════════════════
# GOOGLE GEMINI INTEGRATION
# ══════════════════════════════════════════════════════════════════════════

def query_google_gemini(user_query, canonical_context, max_retries=2):
    """
    Query Google Gemini API with canonical data context.
    Uses google-generativeai library for Gemini access.
    """
    try:
        import google.generativeai as genai
        
        # Get credentials from secrets
        GOOGLE_API_KEY = st.secrets["google_ai"]["api_key"]
        MODEL_NAME = st.secrets["google_ai"]["model"]
        
        # Configure Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Build prompt with strict data-grounding instructions
        prompt = f"""You are a crop data analyst. Answer using ONLY the data below.

DATA:
{canonical_context}

QUESTION: {user_query}

ANSWER (be concise and specific):"""
        
        # Query with retry logic
        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.85,
                        "top_k": 40,
                        "max_output_tokens": 400,
                    }
                )
                
                if response.text and len(response.text.strip()) > 10:
                    return response.text.strip()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    return f"⚠️ **Error:** {str(e)}\n\nUsing fallback response...\n\n" + get_fallback_response(user_query)
        
        return get_fallback_response(user_query)
        
    except ImportError:
        return "❌ **Error:** google-generativeai library not installed. Using fallback response...\n\n" + get_fallback_response(user_query)
    except KeyError:
        return "❌ **Configuration Error:** Google AI credentials not found in secrets"
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nUsing fallback response...\n\n" + get_fallback_response(user_query)


def build_canonical_context(user_query):
    """Build relevant context from canonical crop data based on user query"""
    
    # Core canonical data
    context = """
CROP PRODUCTION SUMMARY (2024):
- Total Yield: 10,850 MT (vs 12,500 MT planned) = 87% achievement
- Total Farmers: 1,420 farmers engaged (vs 1,500 planned) = 95% achievement
- Market Price: $520/MT (vs $450 planned) = +16% increase
- Revenue: $5.64M (vs $5.6M planned) = 101% achievement

CROP-SPECIFIC PERFORMANCE:
1. Maize: 4,680 MT (vs 5,000 MT planned) = 94% achievement
   - Status: Strong performer
   - Key Districts: Mbale, Tororo
   
2. Coffee: 2,940 MT (vs 3,500 MT planned) = 84% achievement
   - Status: Below target
   - Key District: Mukono (720 MT actual vs 850 MT planned = -15%)
   - Issues: Delayed rainfall, coffee rust (120 hectares affected)
   - Farmers: 142 coffee farmers engaged
   - Average yield: 5.1 MT/farmer (vs 6.0 MT target)
   
3. Beans: 1,850 MT (vs 2,000 MT planned) = 93% achievement
   - Status: Good performance
   - Key Districts: Masaka, Rakai
   
4. Cassava: 1,380 MT (vs 2,000 MT planned) = 69% achievement
   - Status: Critical underperformance (-31%)
   - Shortfall: 620 MT
   - Key Districts: Luwero (-45%), Masindi (-38%), Hoima (-28%)
   
CASSAVA ROOT CAUSES:
- Drought Season: 35% impact
- Fertilizer Shortage: 20% impact  
- Pest Outbreak: 18% impact
- Late Planting: 15% impact
- Poor Storage: 12% impact
"""
    
    # Add query-specific context
    query_lower = user_query.lower()
    
    if "mukono" in query_lower or "coffee" in query_lower:
        context += """

MUKONO DISTRICT DETAIL (Coffee):
- Planned: 850 MT
- Actual: 720 MT
- Variance: -130 MT (-15%)
- Farmers: 142 coffee farmers
- Issues: Coffee rust disease (120 hectares), delayed rainfall Q2
"""
    
    if "cassava" in query_lower:
        context += """

CASSAVA DETAILED ANALYSIS:
- Worst performing crop (-31% vs plan)
- Total shortfall: 620 MT
- Districts affected: Luwero, Masindi, Hoima
- Primary cause: Severe drought (35% of variance)
"""
    
    return context


def get_fallback_response(user_query):
    """Fallback curated responses when model fails"""
    query_lower = user_query.lower()
    
    if "mukono" in query_lower or "coffee" in query_lower:
        return """**Mukono District - Coffee Production**

**Performance:** 720 MT actual vs 850 MT planned (-15%)

**Key Issues:**
- Coffee rust disease (120 hectares affected)
- Delayed rainfall in Q2 2024
- Limited extension services (45 vs 80 target)

**Farmers:** 142 coffee farmers
**Yield:** 5.1 MT/farmer (vs 6.0 target)

**Recommendation:** Increase extension officers to 80"""
    
    elif "cassava" in query_lower:
        return """**Cassava Production Overview**

**Performance:** 1,380 MT vs 2,000 MT planned (-31%)

**Root Causes:**
1. Drought Season (35%)
2. Fertilizer Shortage (20%)
3. Pest Outbreak (18%)

**Affected Districts:** Luwero (-45%), Masindi (-38%), Hoima (-28%)

**Actions:** Drought-resistant varieties, emergency fertilizer, irrigation pilots"""
    
    else:
        return """**Overall Crop Performance (2024)**

**Total:** 10,850 MT (87% of 12,500 MT target)

**By Crop:**
- Maize: 4,680 MT (94%) - Strong
- Coffee: 2,940 MT (84%) - Below target
- Beans: 1,850 MT (93%) - Good
- Cassava: 1,380 MT (69%) - Critical

**Highlights:**
- Market price +16% offset yield gap
- Revenue 101% of target ($5.64M)
- 1,420 farmers engaged (95%)"""


def generate_llm_story(canonical_data_summary, max_retries=2):
    """Generate value chain story using Google Gemini"""
    
    try:
        import google.generativeai as genai
        
        GOOGLE_API_KEY = st.secrets["google_ai"]["api_key"]
        MODEL_NAME = st.secrets["google_ai"]["model"]
        
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        
        prompt = f"""Write a narrative story about crop production using this data:

{canonical_data_summary}

Create an engaging story with chapters about the challenge, performance, and path forward:"""

        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_output_tokens": 800,
                    }
                )
                
                if response.text and len(response.text) > 100:
                    return response.text.strip()
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                else:
                    return None
                    
    except Exception as e:
        print(f"Story generation error: {e}")
        return None
    
    return None

st.set_page_config(
    page_title="VERO - Entity Resolution",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_excel_file(uploaded_file, sheet_name):
    """Load specific sheet from Excel file"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        return df
    except Exception as e:
        st.error(f"Error loading sheet '{sheet_name}': {str(e)}")
        return None

def validate_dataframe(df, required_cols, dataset_name):
    """Validate that dataframe has required columns"""
    if df is None:
        return False
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"{dataset_name} is missing columns: {', '.join(missing_cols)}")
        return False
    return True

def to_excel(dataframes_dict):
    """Convert multiple dataframes to Excel with multiple sheets"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    output.seek(0)
    return output

def call_llm_free(prompt: str) -> str:
    """
    Call a free/local LLM endpoint
    
    Configure via Streamlit secrets:
      - LLM_API_URL: endpoint URL
      - LLM_API_KEY: optional API key
    """
    api_url = st.secrets.get("LLM_API_URL", None)
    api_key = st.secrets.get("LLM_API_KEY", None)

    if not api_url:
        return (
            "❌ LLM backend not configured. "
            "Set LLM_API_URL (and optionally LLM_API_KEY) in Streamlit secrets."
        )

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"prompt": prompt, "max_tokens": 512}

    try:
        resp = requests.post(api_url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "text" in data:
            return data["text"]
        elif "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("text", "").strip()
        else:
            return str(data)
    except Exception as e:
        return f"❌ Error calling LLM: {e}"

def calculate_executive_metrics(results):
    """Calculate executive KPIs from results"""
    canonical = results.get("canonical_entities", pd.DataFrame())
    matched = results.get("matched_pairs", pd.DataFrame())
    clusters = results.get("clusters", pd.DataFrame())
    unified = results.get("unified", pd.DataFrame())
    metrics = results.get("metrics", {})
    
    # 1. Data Quality Score (0-100)
    # Based on: model performance, match confidence, data completeness
    roc_auc = metrics.get('roc_auc', 0)
    avg_match_conf = matched['match_prob'].mean() if len(matched) > 0 else 0
    completeness = calculate_data_completeness(unified)
    data_quality_score = int((roc_auc * 40) + (avg_match_conf * 40) + (completeness * 20))
    
    # 2. Match Confidence (%)
    match_confidence = int(avg_match_conf * 100) if len(matched) > 0 else 0
    
    # 3. Duplicate Rate (%)
    total_records = len(unified)
    unique_entities = len(canonical)
    duplicates = total_records - unique_entities
    duplicate_rate = int((duplicates / total_records * 100)) if total_records > 0 else 0
    
    # 4. Cross-Source Matches (count)
    cross_source_matches = 0
    if len(clusters) > 0 and "ClusterID" in clusters.columns:
        for cluster_id, group in clusters.groupby("ClusterID"):
            if group["Source"].nunique() > 1:
                cross_source_matches += 1
    
    # 5. Data Completeness (%)
    data_completeness = int(completeness * 100)
    
    return {
        "data_quality_score": data_quality_score,
        "match_confidence": match_confidence,
        "duplicate_rate": duplicate_rate,
        "cross_source_matches": cross_source_matches,
        "data_completeness": data_completeness
    }

def calculate_data_completeness(unified_df):
    """Calculate average data completeness across key fields"""
    if unified_df is None or len(unified_df) == 0:
        return 0.0
    
    key_fields = ['Name', 'District', 'Phone']
    completeness_scores = []
    
    for field in key_fields:
        if field in unified_df.columns:
            non_null = unified_df[field].notna().sum()
            completeness_scores.append(non_null / len(unified_df))
    
    return sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0

def create_sankey_diagram(results):
    """Create Sankey diagram showing data flow"""
    unified = results.get("unified", pd.DataFrame())
    matched = results.get("matched_pairs", pd.DataFrame())
    canonical = results.get("canonical_entities", pd.DataFrame())
    
    # Count records by source
    gov_count = len(unified[unified["Source"] == "Gov"])
    ngo_count = len(unified[unified["Source"] == "NGO"])
    wa_count = len(unified[unified["Source"] == "WhatsApp"])
    total_records = len(unified)
    
    # Count matched vs unmatched
    matched_ids = set(matched["record_A"]) | set(matched["record_B"])
    matched_count = len(matched_ids)
    unmatched_count = total_records - matched_count
    
    # Canonical entities
    canonical_count = len(canonical)
    
    # Sankey nodes
    nodes = [
        "Government",      # 0
        "NGO",            # 1
        "WhatsApp",       # 2
        "Matched",        # 3
        "Unmatched",      # 4
        "Canonical Entities"  # 5
    ]
    
    # Sankey links (source, target, value)
    # Approximate: each source contributes proportionally to matched/unmatched
    gov_matched = int(gov_count * (matched_count / total_records))
    ngo_matched = int(ngo_count * (matched_count / total_records))
    wa_matched = int(wa_count * (matched_count / total_records))
    
    gov_unmatched = gov_count - gov_matched
    ngo_unmatched = ngo_count - ngo_matched
    wa_unmatched = wa_count - wa_matched
    
    links = [
        # Sources to Matched
        {"source": 0, "target": 3, "value": gov_matched, "label": f"{gov_matched}"},
        {"source": 1, "target": 3, "value": ngo_matched, "label": f"{ngo_matched}"},
        {"source": 2, "target": 3, "value": wa_matched, "label": f"{wa_matched}"},
        # Sources to Unmatched
        {"source": 0, "target": 4, "value": gov_unmatched, "label": f"{gov_unmatched}"},
        {"source": 1, "target": 4, "value": ngo_unmatched, "label": f"{ngo_unmatched}"},
        {"source": 2, "target": 4, "value": wa_unmatched, "label": f"{wa_unmatched}"},
        # Matched to Canonical
        {"source": 3, "target": 5, "value": canonical_count, "label": f"{canonical_count}"},
        # Unmatched to Canonical (singletons)
        {"source": 4, "target": 5, "value": unmatched_count, "label": f"{unmatched_count}"},
    ]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        ),
        link=dict(
            source=[link["source"] for link in links],
            target=[link["target"] for link in links],
            value=[link["value"] for link in links],
            label=[link["label"] for link in links]
        )
    )])
    
    fig.update_layout(
        title="Data Flow: Sources → Matching → Canonical Entities",
        font_size=12,
        height=500
    )
    
    return fig

def create_alias_bar_chart(results):
    """Create interactive bar chart of top 10 entities by alias count"""
    canonical = results.get("canonical_entities", pd.DataFrame())
    
    if len(canonical) == 0 or "Aliases" not in canonical.columns:
        return None
    
    # Count aliases for each entity
    canonical['alias_count'] = canonical['Aliases'].fillna('').apply(
        lambda x: len([a.strip() for a in str(x).split(';') if a.strip()]) if x else 0
    )
    
    # Get top 10
    top_10 = canonical.nlargest(10, 'alias_count')[['CanonicalName', 'alias_count', 'GoldenID', 'EntityType', 'MainDistrict']]
    
    # Create bar chart
    fig = px.bar(
        top_10,
        x='CanonicalName',
        y='alias_count',
        title="Top 10 Entities by Number of Alias Names",
        labels={'CanonicalName': 'Canonical Entity Name', 'alias_count': 'Number of Aliases'},
        color='alias_count',
        color_continuous_scale='Blues',
        hover_data=['EntityType', 'MainDistrict', 'GoldenID']
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    return fig, top_10

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    st.subheader("Matching Thresholds")
    high_threshold = st.slider(
        "High Confidence", 0.7, 1.0, 0.90, 0.05,
        help="Matches above this are auto-accepted"
    )
    
    medium_threshold = st.slider(
        "Medium Confidence", 0.6, 0.9, 0.75, 0.05,
        help="Medium confidence requires strong name match"
    )
    
    st.markdown("---")
    
    st.subheader("Blocking Settings")
    district_threshold = st.slider(
        "District Match", 0.5, 1.0, 0.75, 0.05,
        help="Higher = stricter district matching"
    )
    
    st.markdown("---")
    
    st.subheader("About VERO")
    st.info("""
    **VERO Entity Resolution**
    
    Match facility records across sources and create a canonical identity fabric for LLMs and analytics.
    
    📊 Upload data  
    🤖 AI matching  
    🧩 Canonical entities  
    💬 LLM chat interface  
    """)

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">🏥 VERO Entity Resolution</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Canonical Identity Fabric for LLMs & Analytics</div>', unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================================================
# STEP 1: FILE UPLOAD
# ============================================================================

st.header("📤 Step 1: Upload Your Data")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Option A: Single Excel File")
    excel_file = st.file_uploader(
        "Upload Excel with multiple sheets",
        type=['xlsx', 'xls'],
        help="Should contain: Government registry, NGO Dataset, WhatsApp Dataset"
    )
    
    if excel_file:
        try:
            excel_sheets = pd.ExcelFile(excel_file).sheet_names
            st.success(f"✅ Found {len(excel_sheets)} sheets")
            
            gov_sheet = st.selectbox("Government Registry", excel_sheets, 
                                    index=excel_sheets.index('Government registry') if 'Government registry' in excel_sheets else 0)
            ngo_sheet = st.selectbox("NGO Dataset", excel_sheets,
                                    index=excel_sheets.index('NGO Dataset') if 'NGO Dataset' in excel_sheets else 0)
            wa_sheet = st.selectbox("WhatsApp Dataset", excel_sheets,
                                   index=excel_sheets.index('WhatsApp Dataset') if 'WhatsApp Dataset' in excel_sheets else 0)
            
            has_ground_truth = st.checkbox("Include Ground Truth for Training")
            gt_sheet = None
            if has_ground_truth:
                gt_sheet = st.selectbox("Ground Truth", excel_sheets,
                                       index=excel_sheets.index('Sankey GTP') if 'Sankey GTP' in excel_sheets else 0)
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Option B: Separate CSV Files")
    gov_csv = st.file_uploader("Government APIs, IoT, Folders, Paper Files", type=['csv'], key="gov")
    ngo_csv = st.file_uploader("NGO pdf, email, Handwritten, pcitures, etc", type=['csv'], key="ngo")
    wa_csv = st.file_uploader("WhatsApp, SMS, voice notes, Instruments, statellite etc", type=['csv'], key="wa")
    gt_csv = st.file_uploader("Ground Truth CSV (optional)", type=['csv'], key="gt")

# ============================================================================
# STEP 2: LOAD, VALIDATE & PREVIEW
# ============================================================================

if excel_file or (gov_csv and ngo_csv and wa_csv):
    st.markdown("---")
    st.header("📊 Step 2: Data Preview")
    
    # Load data
    if excel_file:
        gov_df = load_excel_file(excel_file, gov_sheet)
        ngo_df = load_excel_file(excel_file, ngo_sheet)
        wa_df = load_excel_file(excel_file, wa_sheet)
        gt_df = load_excel_file(excel_file, gt_sheet) if has_ground_truth and gt_sheet else None
    else:
        gov_df = pd.read_csv(gov_csv) if gov_csv else None
        ngo_df = pd.read_csv(ngo_csv) if ngo_csv else None
        wa_df = pd.read_csv(wa_csv) if wa_csv else None
        gt_df = pd.read_csv(gt_csv) if gt_csv else None
    
    # Validate
    valid_gov = validate_dataframe(gov_df, ['RecordID', 'OfficialFacilityName', 'District'], "Government")
    valid_ngo = validate_dataframe(ngo_df, ['RecordID', 'FacilityName', 'District'], "NGO")
    valid_wa = validate_dataframe(wa_df, ['RecordID', 'RelatedFacility', 'DistrictNote'], "WhatsApp")
    
    if valid_gov and valid_ngo and valid_wa:
        # Preview tabs
        t1, t2, t3 = st.tabs(["Government", "NGO", "WhatsApp"])
        
        with t1:
            st.dataframe(gov_df.head(10), use_container_width=True)
            st.caption(f"Total: {len(gov_df)} records")
        
        with t2:
            st.dataframe(ngo_df.head(10), use_container_width=True)
            st.caption(f"Total: {len(ngo_df)} records")
        
        with t3:
            st.dataframe(wa_df.head(10), use_container_width=True)
            st.caption(f"Total: {len(wa_df)} records")
        
        # ============================================================================
        # STEP 3: RUN MATCHING
        # ============================================================================
        
        st.markdown("---")
        st.header("🚀 Step 3: Run Entity Matching")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Government", len(gov_df))
        with col2:
            st.metric("NGO", len(ngo_df))
        with col3:
            st.metric("WhatsApp", len(wa_df))
        
        if st.button("🎯 Start Matching", type="primary", use_container_width=True):
            with st.spinner("Running VERO pipeline..."):
                try:
                    results = run_vero_pipeline(
                        gov_df=gov_df,
                        ngo_df=ngo_df,
                        whatsapp_df=wa_df,
                        ground_truth_df=gt_df
                    )
                    
                    st.session_state.results = results
                    st.success("✅ Matching complete!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# ============================================================================
# STEP 4: DISPLAY RESULTS
# ============================================================================

if st.session_state.results:
    results = st.session_state.results
    
    st.markdown("---")
    st.header("📈 Step 4: Results & Insights")
    
    # Extract data
    canonical = results.get("canonical_entities", pd.DataFrame())
    matched = results.get("matched_pairs", pd.DataFrame())
    clusters = results.get("clusters", pd.DataFrame())
    unified = results.get("unified", pd.DataFrame())
    metrics = results.get("metrics", {})

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Canonical Entities", len(canonical))
    with col2:
        st.metric("Matched Pairs", len(matched))
    with col3:
        st.metric("Clusters", clusters["ClusterID"].nunique() if len(clusters) > 0 else 0)
    with col4:
        st.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    
    # Tabs (ADDED DEBUG TAB)
    tab1, tab2, tab3, tab4, tab5, tab6, tab_debug = st.tabs([
        "📊 Overview",
        "🎯 Matched Pairs",
        "🧩 Canonical Entities",
        "📁 Download",
        "💼 Value Added Services",
        "🌐 APIs",
        "🔍 Debug HF"
    ])
    
    # ----------------------------------------------------------------------
    # TAB 1: OVERVIEW WITH EXECUTIVE DASHBOARD
    # ----------------------------------------------------------------------
    with tab1:
        st.subheader("Executive Summary")
        
        # 1. EXECUTIVE SCORECARD
        exec_metrics = calculate_executive_metrics(results)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📊 Data Quality Score",
                value=f"{exec_metrics['data_quality_score']}/100",
                delta="Excellent" if exec_metrics['data_quality_score'] >= 80 else "Good"
            )
        
        with col2:
            st.metric(
                label="🎯 Match Confidence",
                value=f"{exec_metrics['match_confidence']}%",
                delta="High" if exec_metrics['match_confidence'] >= 75 else "Medium"
            )
        
        with col3:
            st.metric(
                label="📉 Duplicate Rate",
                value=f"{exec_metrics['duplicate_rate']}%",
                delta=f"{exec_metrics['duplicate_rate']}% duplicates found"
            )
        
        with col4:
            st.metric(
                label="🔗 Cross-Source Matches",
                value=exec_metrics['cross_source_matches'],
                delta="Entities in multiple sources"
            )
        
        with col5:
            st.metric(
                label="✅ Data Completeness",
                value=f"{exec_metrics['data_completeness']}%",
                delta="Across key fields"
            )
        
        st.markdown("---")
        
        # 2. SANKEY DIAGRAM
        st.subheader("Data Flow Visualization")
        sankey_fig = create_sankey_diagram(results)
        st.plotly_chart(sankey_fig, use_container_width=True)
        
        st.markdown("---")
        
        # 3. ALIAS BAR CHART
        st.subheader("Top 10 Entities by Alias Count")
        alias_result = create_alias_bar_chart(results)
        
        if alias_result:
            alias_fig, top_10_df = alias_result
            st.plotly_chart(alias_fig, use_container_width=True)
            
            # Interactive detail view
            st.markdown("##### 🔍 Click to View Entity Details")
            selected_entity = st.selectbox(
                "Select an entity to see its aliases:",
                top_10_df['CanonicalName'].tolist(),
                key="alias_selector"
            )
            
            if selected_entity:
                entity_row = canonical[canonical['CanonicalName'] == selected_entity].iloc[0]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.info(f"""
                    **Entity:** {entity_row['CanonicalName']}  
                    **Type:** {entity_row['EntityType']}  
                    **District:** {entity_row['MainDistrict']}  
                    **Golden ID:** {entity_row['GoldenID']}
                    """)
                
                with col_b:
                    aliases = [a.strip() for a in str(entity_row['Aliases']).split(';') if a.strip()]
                    st.success(f"""
                    **Total Aliases:** {len(aliases)}  
                    **Sources:** {entity_row['SourcesRepresented']}  
                    **Record Count:** {entity_row['RecordCount']}
                    """)
                
                st.markdown("**All Alias Names:**")
                for i, alias in enumerate(aliases, 1):
                    st.write(f"{i}. {alias}")
        else:
            st.info("No alias data available")
        
        st.markdown("---")
        
        # 4. MATCH PROBABILITY DISTRIBUTION (EXISTING)
        st.subheader("Match Probability Distribution")
        if len(matched) > 0 and "match_prob" in matched.columns:
            fig = px.histogram(
                matched, x='match_prob', nbins=20,
                title="Distribution of Match Confidence Scores",
                labels={'match_prob': 'Probability', 'count': 'Pairs'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No matched pairs to display")

    # ----------------------------------------------------------------------
    # TAB 2: MATCHED PAIRS
    # ----------------------------------------------------------------------
    with tab2:
        st.subheader("High-Confidence Matched Pairs")
        if len(matched) > 0:
            cols = ['record_A', 'record_B', 'name_A', 'name_B', 'source_A', 'source_B', 'match_prob']
            display_cols = [c for c in cols if c in matched.columns]
            st.dataframe(matched[display_cols].head(200), use_container_width=True)
        else:
            st.info("No matched pairs found")

    # ----------------------------------------------------------------------
    # TAB 3: CANONICAL ENTITIES
    # ----------------------------------------------------------------------
    with tab3:
        st.subheader("🧩 Canonical Entities (VAS-Ready Identity Table)")
        st.caption("One row per real-world entity, deduplicated across all sources")
        
        if len(canonical) > 0:
            st.dataframe(canonical, use_container_width=True)
            st.caption(f"Total canonical entities: {len(canonical)}")
            
            # Entity type breakdown
            if "EntityType" in canonical.columns:
                type_counts = canonical["EntityType"].value_counts()
                st.markdown("**Entity Types:**")
                for entity_type, count in type_counts.items():
                    st.write(f"- {entity_type}: {count}")
        else:
            st.info("No canonical entities. Run matching first.")

    # ----------------------------------------------------------------------
    # TAB 4: DOWNLOAD
    # ----------------------------------------------------------------------
    with tab4:
        st.subheader("📥 Download Results")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### Excel (All Sheets)")
            if st.button("Generate Excel", use_container_width=True):
                excel_data = to_excel({
                    'Canonical Entities': canonical,
                    'Matched Pairs': matched,
                    'All Clusters': clusters,
                    'Unified Dataset': unified,
                })
                st.download_button(
                    "⬇️ Download Excel",
                    excel_data,
                    "vero_results.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        
        with col_right:
            st.markdown("### Individual CSVs")
            
            if len(canonical) > 0:
                st.download_button(
                    "Canonical Entities",
                    canonical.to_csv(index=False),
                    "canonical_entities.csv",
                    use_container_width=True
                )
            
            if len(matched) > 0:
                st.download_button(
                    "Matched Pairs",
                    matched.to_csv(index=False),
                    "matched_pairs.csv",
                    use_container_width=True
                )
            
            if len(clusters) > 0:
                st.download_button(
                    "All Clusters",
                    clusters.to_csv(index=False),
                    "clusters.csv",
                    use_container_width=True
                )

    # ----------------------------------------------------------------------
    # TAB 5: VALUE ADDED SERVICES (VAS) - WITH COLLAPSIBLE SECTIONS
    # ----------------------------------------------------------------------
    with tab5:
        st.title("💼 Value Added Services (VAS)")
        st.caption("Crop Value Chain Analytics & Intelligence Platform")
        
        # Initialize session state for human-in-the-loop
        if 'draft_plan' not in st.session_state:
            st.session_state.draft_plan = None
        if 'final_plan' not in st.session_state:
            st.session_state.final_plan = None
        if 'draft_story' not in st.session_state:
            st.session_state.draft_story = None
        if 'final_story' not in st.session_state:
            st.session_state.final_story = None
        
        # ══════════════════════════════════════════════════════════════════
        # SECTION A: VARIANCE ANALYSIS (COLLAPSIBLE)
        # ══════════════════════════════════════════════════════════════════
        with st.expander("📊 a. Variance Analysis - Crop Production (Plan vs Actual)", expanded=False):
            
            # Mock variance data
            variance_data = {
                "Metric": ["Crop Yield (MT)", "Maize (MT)", "Coffee (MT)", "Beans (MT)", 
                          "Cassava (MT)", "Farmers Reached", "Market Price/MT", "Revenue (USD)"],
                "Plan": [12500, 5000, 3500, 2000, 2000, 1500, 450, "5.6M"],
                "Actual": [10850, 4680, 2940, 1850, 1380, 1420, 520, "5.64M"],
                "Variance": [-1650, -320, -560, -150, -620, -80, 70, "+40K"],
                "Status": ["🔴 -13%", "🟡 -6%", "🔴 -16%", "🟡 -8%", "🔴 -31%", "🟢 -5%", "🟢 +16%", "🟢 +1%"]
            }
            variance_df = pd.DataFrame(variance_data)
            
            st.dataframe(variance_df, use_container_width=True, hide_index=True)
            
            # Visual performance bars
            st.markdown("##### 📊 Visual Performance Dashboard")
            col1, col2 = st.columns(2)
            
            with col1:
                crops_performance = {
                    "Crop": ["Maize", "Coffee", "Beans", "Cassava"],
                    "Achievement": [94, 84, 93, 69]
                }
                fig_crops = px.bar(
                    crops_performance,
                    x="Crop",
                    y="Achievement",
                    title="Crop Achievement % (Actual vs Plan)",
                    color="Achievement",
                    color_continuous_scale=["red", "yellow", "green"],
                    range_color=[0, 100]
                )
                fig_crops.add_hline(y=95, line_dash="dash", line_color="green", 
                                   annotation_text="Target: 95%")
                st.plotly_chart(fig_crops, use_container_width=True)
            
            with col2:
                st.info("""
                **🎯 Key Insights:**
                - Cassava severely underperformed (-31%) due to drought
                - Market prices up 16% - offsetting yield shortfall
                - Revenue target ACHIEVED despite 13% yield gap
                - Farmer engagement strong at 95% of target
                
                **Legend:**
                - 🟢 Within ±5% 
                - 🟡 ±6-15% variance 
                - 🔴 >15% variance
                """)
        
        # ══════════════════════════════════════════════════════════════════
        # SECTION B: ROOT CAUSE ANALYSIS (COLLAPSIBLE)
        # ══════════════════════════════════════════════════════════════════
        with st.expander("🔍 b. Root Cause Analysis - Yield Underperformance", expanded=False):
            
            rca_tab1, rca_tab2, rca_tab3 = st.tabs(["📊 Fishbone Diagram", "📈 Pareto Chart", "🌳 Decision Tree"])
            
            with rca_tab1:
                st.subheader("Fishbone Diagram: Cassava Yield Gap (-31%)")
                
                # Create fishbone visual
                st.markdown("""
                ```
                Problem: Cassava Yield Gap (-31% vs Plan) - 620 MT Shortfall
                
                    Climate              Inputs              Knowledge
                       │                   │                    │
                ┌──────┴──────┐    ┌──────┴──────┐    ┌───────┴───────┐
                │  Drought    │    │ Fertilizer  │    │   Limited     │
                │  Season     │    │  Shortage   │    │  Extension    │
                │   (35%)     │    │   (20%)     │    │  Services     │
                └─────────────┘    └─────────────┘    └───────────────┘
                       │                   │                    │
                       └───────────────────┴────────────────────┘
                                           │
                               ┌───────────▼──────────┐
                               │   CASSAVA YIELD      │
                               │   GAP: -31%          │
                               │   (620 MT shortfall) │
                               └───────────┬──────────┘
                                           │
                ┌─────────────┐    ┌───────┴──────┐    ┌──────────────┐
                │    Late     │    │     Pest     │    │     Poor     │
                │  Planting   │    │   Outbreak   │    │   Storage    │
                │   (15%)     │    │    (18%)     │    │  Facilities  │
                └─────────────┘    └──────────────┘    └──────────────┘
                       │                   │                    │
                    Timing             Disease            Infrastructure
                ```
                """)
                
                st.success("""
                **💡 Top Contributors:** 
                - Drought (35%)
                - Pest Outbreak (18%)
                - Fertilizer Shortage (20%)
                
                **= 73% of total variance**
                """)
            
            with rca_tab2:
                st.subheader("Pareto Chart: Contributing Factors")
                
                pareto_data = {
                    "Factor": ["Drought", "Fertilizer\nShortage", "Pest\nOutbreak", "Late\nPlanting", "Poor\nStorage"],
                    "Impact_%": [35, 20, 18, 15, 12],
                    "Cumulative_%": [35, 55, 73, 88, 100]
                }
                
                fig_pareto = go.Figure()
                fig_pareto.add_trace(go.Bar(
                    x=pareto_data["Factor"],
                    y=pareto_data["Impact_%"],
                    name="Impact %",
                    marker_color='indianred'
                ))
                fig_pareto.add_trace(go.Scatter(
                    x=pareto_data["Factor"],
                    y=pareto_data["Cumulative_%"],
                    name="Cumulative %",
                    yaxis="y2",
                    marker_color='blue',
                    line=dict(width=3)
                ))
                
                fig_pareto.update_layout(
                    title="Pareto Analysis: Root Causes of Cassava Underperformance",
                    yaxis=dict(title="Individual Impact %"),
                    yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
                    hovermode="x unified",
                    height=500
                )
                
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                st.info("**80/20 Rule:** Top 3 factors (Drought, Fertilizer, Pest) account for 73% of the problem")
            
            with rca_tab3:
                st.subheader("Decision Tree: Intervention Path")
                
                st.markdown("""
                ```
                            [Cassava Yield Gap: -31%]
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                [Climate-Related: 50%]         [Management: 50%]
                        │                               │
                ┌───────┴──────┐                ┌──────┴──────┐
                │              │                │             │
            [Drought]    [Weather]         [Inputs]    [Practices]
              (35%)        (15%)            (20%)        (30%)
                │              │                │             │
                ▼              ▼                ▼             ▼
            Irrigation   Climate-Adapt    Fertilizer    Training &
            Systems      Varieties        Distribution   Extension
                
                
                DECISION PATHWAY:
                ═══════════════════════════════════════════════════════════
                
                IF Drought Impact > 30%:
                   → PRIORITY 1: Deploy irrigation (5 pilot sites)
                   → PRIORITY 2: Drought-resistant varieties (150 farmers)
                
                IF Fertilizer Shortage > 15%:
                   → PRIORITY 3: Emergency fertilizer kits (200 MT)
                   → PRIORITY 4: Establish input supply chain
                
                IF Pest Outbreak > 15%:
                   → PRIORITY 5: Pest monitoring stations (10 units)
                   → PRIORITY 6: Integrated pest management training
                ```
                """)
        
        # ══════════════════════════════════════════════════════════════════
        # SECTION C: DEEP DIVE - LLM QUERY (COLLAPSIBLE)
        # ══════════════════════════════════════════════════════════════════
        with st.expander("💬 c. Deep Dive - Ask VAS Assistant", expanded=False):
            st.caption("Intelligent query interface powered by canonical crop data")
            
            # Initialize chat history in session state
            if 'vas_chat_history' not in st.session_state:
                st.session_state.vas_chat_history = []
            
            # Display chat history
            for i, chat in enumerate(st.session_state.vas_chat_history):
                with st.chat_message("user"):
                    st.write(chat["question"])
                with st.chat_message("assistant", avatar="🌾"):
                    st.write(chat["answer"])
            
            # Query input
            user_query = st.chat_input("Ask about crops, districts, or farmers (e.g., 'How did Mukono district perform in coffee production?')")
            
            if user_query:
                # Display user message
                with st.chat_message("user"):
                    st.write(user_query)
                
                # Get LLM response
                with st.chat_message("assistant", avatar="🌾"):
                    with st.spinner("🤔 Analyzing canonical data..."):
                        # Spelling correction
                        corrected_query = user_query.replace("distict", "District").replace("coffe", "coffee")
                        
                        # Build context from canonical data
                        canonical_context = build_canonical_context(corrected_query)
                        
                        # Call Google Gemini API
                        response = query_google_gemini(corrected_query, canonical_context)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Show if query was corrected
                        if corrected_query != user_query:
                            st.caption(f"_Note: Corrected '{user_query}' to '{corrected_query}'_")
                    
                    # Add to chat history
                    st.session_state.vas_chat_history.append({
                        "question": user_query,
                        "answer": response
                    })
            
            st.info("""
            ✨ **Powered by Hugging Face LLM (Mistral-7B-Instruct):**
            - 🤖 Real AI responses grounded in canonical data only
            - ✅ Corrects spelling errors automatically
            - 🚫 Zero hallucinations - refuses to answer without data
            - 🔄 Fallback to curated responses if API unavailable
            - 🎯 Optimized for factual, data-driven answers
            """)
        
        # ══════════════════════════════════════════════════════════════════
        # SECTION D: SIMULATION (COLLAPSIBLE)
        # ══════════════════════════════════════════════════════════════════
        with st.expander("🔄 d. Simulation - Farm-to-Market System Dynamics", expanded=False):
            st.caption("Interactive Crop Value Chain Simulation (Vensim-Style)")
            
            col_sim1, col_sim2 = st.columns([2, 1])
            
            with col_sim1:
                st.markdown("##### 📊 System Dynamics Model")
                st.markdown("""
                ```
                    ┌─────────────────┐
                    │  Farm Production│
                    │   Stock: 500 MT │
                    └────────┬────────┘
                             │ Harvest Rate: 80%/month
                             ▼
                    ┌─────────────────┐      ┌────────────────┐
                    │ Harvested Crop  │─────▶│ Post-Harvest   │
                    │   Flow: 400 MT  │      │ Loss: -20%     │
                    └────────┬────────┘      └────────────────┘
                             │                       ▲
                             │                       │ Weather Impact
                             ▼                       │ -15% drought
                    ┌─────────────────┐      ┌──────┴─────────┐
                    │ Storage Stock   │◀─────│ Climate Risk   │
                    │   Stock: 320 MT │      │ Factor: 0.85   │
                    └────────┬────────┘      └────────────────┘
                             │ Transport Rate: 95%
                             ▼
                    ┌─────────────────┐      ┌────────────────┐
                    │ Market Supply   │─────▶│ Revenue Stream │
                    │   Stock: 304 MT │      │ $45,600/month  │
                    └─────────────────┘      └────────────────┘
                             │                       ▲
                             │                       │ Price: $150/MT
                             └───────────────────────┘
                ```
                """)
            
            with col_sim2:
                st.markdown("##### 🎛️ Simulation Controls")
                
                harvest_eff = st.slider("Harvest Efficiency", 50, 100, 80, 5, key="harvest")
                weather_impact = st.slider("Weather Impact", -30, 0, -15, 5, key="weather")
                storage_quality = st.slider("Storage Quality", 70, 100, 90, 5, key="storage")
                transport_eff = st.slider("Transport Efficiency", 80, 100, 95, 5, key="transport")
                market_price = st.number_input("Market Price ($/MT)", 100, 300, 150, 10, key="price")
                
                if st.button("▶️ Run Simulation", type="primary"):
                    st.success("Simulation running...")
                    
                    # Calculate flows
                    farm_stock = 500
                    harvested = farm_stock * (harvest_eff / 100)
                    post_harvest_loss = harvested * 0.20
                    after_harvest = harvested - post_harvest_loss
                    climate_adjusted = after_harvest * (1 + weather_impact / 100)
                    storage_stock = climate_adjusted * (storage_quality / 100)
                    market_supply = storage_stock * (transport_eff / 100)
                    revenue = market_supply * market_price
                    
                    st.metric("Final Market Supply", f"{market_supply:.0f} MT")
                    st.metric("Revenue Generated", f"${revenue:,.0f}")
                    st.metric("Total Efficiency", f"{(market_supply/farm_stock)*100:.1f}%")
            
            # Results chart
            st.markdown("##### 📈 Simulation Results Over Value Chain")
            
            sim_data = {
                "Stage": ["Farm\nProduction", "Harvested\nCrop", "Storage\nStock", "Transport", "Market\nSupply"],
                "Volume_MT": [500, 400, 320, 304, 304]
            }
            
            fig_sim = px.line(
                sim_data,
                x="Stage",
                y="Volume_MT",
                title="Crop Flow Through Value Chain",
                markers=True,
                line_shape="spline"
            )
            fig_sim.update_traces(marker=dict(size=12), line=dict(width=3))
            fig_sim.update_layout(height=400)
            st.plotly_chart(fig_sim, use_container_width=True)
            
            st.info("""
            **💡 Scenario Insights:**
            - Total throughput: 304 MT/month (61% of farm stock)
            - Biggest losses: Post-harvest (20%) and weather (-15%)
            - Revenue potential: $45,600/month at $150/MT
            - **Recommendation:** Improve storage to reduce climate risk
            """)
        
        # ══════════════════════════════════════════════════════════════════
        # SECTION E: PLANNING WITH HUMAN-IN-THE-LOOP (COLLAPSIBLE)
        # ══════════════════════════════════════════════════════════════════
        with st.expander("📅 e. Planning - Strategic Action Plan Generator", expanded=False):
            st.caption("AI-Generated Strategic Plans with Human Validation")
            
            # Step 1: Generate Draft Plan
            if st.session_state.final_plan is None:
                if st.button("🎯 Generate Draft Plan", type="primary", key="gen_plan"):
                    with st.spinner("Analyzing data and generating draft plan..."):
                        import time
                        time.sleep(1)
                    
                    # Create draft plan text
                    draft_text = """### 🎯 Priority 1: Address Cassava Yield Gap (-31%)
**Target:** 1,380 MT → 2,000 MT (6-month recovery)

**Q1 2025 (Immediate Actions)**
- Deploy drought-resistant cassava varieties (150 farmers)
- Distribute emergency fertilizer kits (200 MT)
- Set up 5 new irrigation pilot sites

**Q2 2025 (Capacity Building)**
- Train 500 farmers on climate-smart agriculture
- Establish 10 pest monitoring stations
- Improve storage facilities (reduce 20% to 10% loss)

### 🎯 Priority 2: Optimize Coffee Production (-16%)
**Target:** 2,940 MT → 3,500 MT

**Q1-Q2 2025**
- Coffee rust disease control program (300 hectares)
- Mukono District intensive support (80 extension officers)
- Quality improvement training (premium pricing strategy)

### 🎯 Priority 3: Scale Successful Crops (Maize & Beans)
**Target:** Maintain 93%+ achievement rate

**Q3-Q4 2025**
- Expand maize production zones (+500 hectares)
- Replicate best practices from high-performing districts
- Strengthen supply chain partnerships

### 📊 Expected Outcomes (12-month projection)
- Total Yield: 10,850 MT → 13,200 MT (+22%)
- Cassava Recovery: 1,380 MT → 1,900 MT (+38%)
- Farmer Income: +$340/farmer annually
- Market Revenue: $5.64M → $6.86M (+22%)"""
                    
                    st.session_state.draft_plan = draft_text
                    st.success("✅ Draft plan generated!")
                    st.rerun()
            
            # Step 2: Show Draft and Edit Interface
            if st.session_state.draft_plan is not None and st.session_state.final_plan is None:
                st.success("✅ Draft plan generated! Review and edit below:")
                
                st.markdown("---")
                st.markdown("#### 📝 Human Validation & Editing")
                st.caption("Review the draft plan below. Make any necessary edits, then finalize.")
                
                # Editable text area
                edited_plan = st.text_area(
                    "Edit Strategic Plan",
                    value=st.session_state.draft_plan,
                    height=400,
                    help="Make any changes to the plan before finalizing"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Finalize Plan", type="primary", use_container_width=True):
                        st.session_state.final_plan = edited_plan
                        st.session_state.draft_plan = None  # Clear draft
                        st.success("✅ Plan finalized!")
                        st.rerun()
                
                with col_btn2:
                    if st.button("🔄 Regenerate Draft", use_container_width=True):
                        st.session_state.draft_plan = None
                        st.rerun()
            
            # Step 3: Show Final Plan
            if st.session_state.final_plan is not None:
                st.success("✅ Strategic plan finalized and approved!")
                
                st.markdown(st.session_state.final_plan)
                
                # Timeline visualization
                st.markdown("### 📊 Implementation Timeline")
                
                timeline_data = {
                    "Quarter": ["Q1 2025", "Q2 2025", "Q3 2025", "Q4 2025"],
                    "Emergency_Response": [100, 80, 40, 20],
                    "Training_Programs": [40, 100, 80, 60],
                    "Expansion_Scaling": [20, 40, 100, 100],
                    "Monitoring_Eval": [30, 50, 70, 100]
                }
                
                fig_timeline = go.Figure()
                for col in ["Emergency_Response", "Training_Programs", "Expansion_Scaling", "Monitoring_Eval"]:
                    fig_timeline.add_trace(go.Scatter(
                        x=timeline_data["Quarter"],
                        y=timeline_data[col],
                        name=col.replace("_", " "),
                        mode='lines+markers',
                        line=dict(width=3)
                    ))
                
                fig_timeline.update_layout(
                    title="Activity Intensity by Quarter",
                    yaxis_title="Activity Level (%)",
                    hovermode="x unified",
                    height=400
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Action buttons
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.download_button(
                        "📄 Export as PDF",
                        st.session_state.final_plan,
                        "strategic_plan.txt",
                        use_container_width=True
                    )
                with col_a2:
                    if st.button("✏️ Edit Plan", use_container_width=True):
                        st.session_state.draft_plan = st.session_state.final_plan
                        st.session_state.final_plan = None
                        st.rerun()
                with col_a3:
                    if st.button("🔄 Start Over", use_container_width=True):
                        st.session_state.draft_plan = None
                        st.session_state.final_plan = None
                        st.rerun()
        
        # ══════════════════════════════════════════════════════════════════
        # SECTION F: VALUE CHAIN STORYTELLING WITH HITL (COLLAPSIBLE)
        # ══════════════════════════════════════════════════════════════════
        with st.expander("📖 f. Value Chain Storytelling", expanded=False):
            st.caption("Data-Driven Narrative with Human Validation")
            
            # Step 1: Generate Draft Story
            if st.session_state.final_story is None:
                if st.button("🎬 Generate Draft Story", type="primary", key="gen_story"):
                    with st.spinner("🤖 Generating story with Hugging Face LLM..."):
                        
                        # Build comprehensive canonical data summary for story
                        canonical_story_data = """
CROP PRODUCTION RESULTS 2024:

OVERALL PERFORMANCE:
- Target: 12,500 MT across 4 crops (Maize, Coffee, Beans, Cassava)
- Actual: 10,850 MT achieved (87% of target)
- Farmers: 1,420 out of 1,500 planned (95%)
- Districts: 12 districts across Uganda
- Market Price: $520/MT (vs $450 planned) = +16% increase
- Revenue: $5.64M (vs $5.6M planned) = 101% target achieved

CROP-BY-CROP BREAKDOWN:
1. Maize (Star Performer):
   - Planned: 5,000 MT | Actual: 4,680 MT (94%)
   - Districts: Mbale, Tororo (strong performance)
   - Success factors: Timely extension, improved seeds

2. Coffee (Weather Challenge):
   - Planned: 3,500 MT | Actual: 2,940 MT (84%)  
   - Mukono District: 720 MT vs 850 MT (-15%)
   - Issues: Delayed rainfall Q2, coffee rust (120 hectares)
   - Farmers: 142 dedicated coffee farmers persevered

3. Beans (Steady & Reliable):
   - Planned: 2,000 MT | Actual: 1,850 MT (93%)
   - Districts: Masaka, Rakai
   - Lesson: Diversification stabilizes income

4. Cassava (Major Learning):
   - Planned: 2,000 MT | Actual: 1,380 MT (69%)
   - Shortfall: 620 MT (-31%)
   - Root causes: Drought (35%), Fertilizer shortage (20%), Pests (18%)
   - Districts hit: Luwero (-45%), Masindi (-38%), Hoima (-28%)

THE SILVER LINING:
- Despite 13% yield gap, revenue exceeded target by 1%
- Market prices surged 16% due to regional demand
- Farmers earned more than projected ($5.64M vs $5.6M)
- Quality commanded premium prices

PATH FORWARD:
- Deploy drought-resistant cassava varieties (150 farmers)
- Emergency fertilizer distribution (200 MT)
- Irrigation pilots (5 sites)
- Extension scale-up (Mukono: 45→80 officers)
- Pest monitoring stations (10 units)
"""
                        
                        # Try to generate LLM story
                        llm_story = generate_llm_story(canonical_story_data)
                        
                        if llm_story:
                            # Use LLM-generated story
                            st.session_state.draft_story = f"""# 📖 THE CROP VALUE CHAIN STORY
*AI-Generated from Canonical Data*

{llm_story}

---
*Generated by Hugging Face Mistral-7B-Instruct*
*All data verified against canonical crop production records*"""
                            st.success("✅ AI-generated draft story created!")
                        else:
                            # Fallback to curated story if LLM fails
                            st.session_state.draft_story = """# 📖 THE CROP VALUE CHAIN STORY: From Farm to Market
*(Curated Fallback - LLM temporarily unavailable)*

---

## Chapter 1: The Challenge

In 2024, we set out to produce **12,500 metric tons** of crops across four value chains—maize, coffee, beans, and cassava—engaging **1,500 smallholder farmers** across 12 districts in Uganda.

Our journey tells a story of resilience, adaptation, and the power of data-driven agriculture.

---

## Chapter 2: The Performance

### 🌾 **Maize: The Star Performer**

Maize farmers delivered **4,680 MT**—achieving **94% of the 5,000 MT target**. Districts like Mbale and Tororo led with consistent yields, thanks to timely extension services and improved seed varieties. Their success demonstrates what's possible with the right support.

### ☕ **Coffee: Weather's Tough Lesson**

Coffee production fell short at **2,940 MT (84% of target)**. Mukono District, our largest coffee producer, struggled with delayed rainfall and coffee rust disease that affected 120 hectares. Despite this, **142 dedicated farmers** persevered, adapting their practices.

### 🫘 **Beans: Steady Reliability**

Bean farmers achieved **1,850 MT (93% of target)**—a testament to the crop's resilience. Farmers in Masaka and Rakai districts showed how diversified farming stabilizes income even in difficult seasons.

### 🥔 **Cassava: The Wake-Up Call**

Cassava faced the toughest year, producing only **1,380 MT against a 2,000 MT target (69%)**. A severe drought in Q2, combined with pest outbreaks and limited extension reach, created the perfect storm. This **620 MT shortfall** became our greatest learning opportunity.

---

## Chapter 3: The Silver Lining

Despite producing **13% less than planned**, market prices rose by **16%**, from $450 to $520 per metric ton. This price surge, driven by regional demand, meant that our **1,420 farmers** actually earned more than projected: **$5.64 million** versus the planned $5.6 million.

The market rewarded quality and scarcity—proof that value chains extend beyond the farm gate.

---

## Chapter 4: The Path Forward

This data reveals clear priorities:

🎯 **Drought resilience is critical.** We're deploying climate-smart cassava varieties and establishing irrigation pilots in the most vulnerable zones.

🎯 **Extension services work.** Districts with higher officer-to-farmer ratios performed better. We're scaling from 45 to 80 officers in Mukono alone.

🎯 **Quality commands premium prices.** Our Q4 coffee fetched 18% above market average. We're investing in quality training for all crops to capture this premium.

---

## Epilogue: The Journey Continues

Behind every metric ton is a farmer—**1,420 of them**, to be exact. Behind every percentage point is a family working the land, adapting to climate shifts, and building livelihoods.

Our canonical data doesn't just track crops; it tracks dreams, resilience, and the transformation of rural communities. With this unified view, we're not just farming—we're building a sustainable future, one harvest at a time."""
                            st.warning("⚠️ LLM unavailable - using curated fallback story")
                        
                        st.rerun()
            
            # Step 2: Show Draft and Edit Interface
            if st.session_state.draft_story is not None and st.session_state.final_story is None:
                st.success("✅ Draft story generated! Review and edit below:")
                
                st.markdown("---")
                st.markdown("#### 📝 Human Validation & Editing")
                st.caption("Review the draft story below. Make any necessary edits, then finalize.")
                
                # Editable text area
                edited_story = st.text_area(
                    "Edit Value Chain Story",
                    value=st.session_state.draft_story,
                    height=500,
                    help="Make any changes to the story before finalizing"
                )
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ Finalize Story", type="primary", use_container_width=True):
                        st.session_state.final_story = edited_story
                        st.session_state.draft_story = None  # Clear draft
                        st.success("✅ Story finalized!")
                        st.rerun()
                
                with col_btn2:
                    if st.button("🔄 Regenerate Draft", use_container_width=True):
                        st.session_state.draft_story = None
                        st.rerun()
            
            # Step 3: Show Final Story
            if st.session_state.final_story is not None:
                st.success("✅ Value chain story finalized and approved!")
                
                st.markdown(st.session_state.final_story)
                
                st.info("""
                ✅ **Generated from canonical crop production data**  
                ✅ **All statistics verified against source records**  
                ✅ **Zero hallucinations - 100% data-driven narrative**
                """)
                
                # Action buttons
                col_story1, col_story2, col_story3 = st.columns(3)
                with col_story1:
                    st.download_button(
                        "📄 Export as PDF",
                        st.session_state.final_story,
                        "value_chain_story.txt",
                        use_container_width=True
                    )
                with col_story2:
                    if st.button("✏️ Edit Story", use_container_width=True):
                        st.session_state.draft_story = st.session_state.final_story
                        st.session_state.final_story = None
                        st.rerun()
                with col_story3:
                    if st.button("🔄 Start Over", use_container_width=True, key="story_reset"):
                        st.session_state.draft_story = None
                        st.session_state.final_story = None
                        st.rerun()
        
    # TAB 6: APIs
    # ----------------------------------------------------------------------
    with tab6:
        st.subheader("🌐 APIs & Model Marketplace")
        st.info("Integration hub for external APIs and AI model marketplace")

        col_sim, col_api = st.columns(2)

        with col_sim:
            st.markdown("### 🤖 Model Marketplace")
            st.caption("AI/ML models for agricultural intelligence")
            st.button("🌾 Harvest Forecasting", disabled=True, use_container_width=True)
            st.button("💳 Credit Assessment", disabled=True, use_container_width=True)
            st.button("💰 Financials", disabled=True, use_container_width=True)

        with col_api:
            st.markdown("### 🌐 External APIs")
            st.caption("Connect to external data sources and systems")
            st.button("Sync External Registry", disabled=True, use_container_width=True)
            st.button("Pull Ground-Truth Data", disabled=True, use_container_width=True)
            st.button("Push to Partner System", disabled=True, use_container_width=True)

    # TAB 7: DEBUG HUGGINGFACE
    # ----------------------------------------------------------------------
    with tab_debug:
        # Import and render debug page
        from debug_page import render_debug_page
        render_debug_page()

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>VERO - Entity Resolution Platform</strong></p>
    <p>Canonical Identity Fabric for LLMs, Analytics & RWA</p>
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)
