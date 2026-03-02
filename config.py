import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ══════════════════════════════════════════════════════
# PALETTE NASA / MISSION CONTROL
# ══════════════════════════════════════════════════════
C_BG        = "#020408"
C_SURFACE   = "#040d18"
C_PANEL     = "#061225"
C_BORDER    = "#0a2040"
C_BORDER2   = "#0d3060"
C_CYAN      = "#00e5ff"
C_CYAN_DIM  = "#005577"
C_AMBER     = "#ffb300"
C_AMBER_DIM = "#4d3500"
C_RED       = "#ff1744"
C_RED_DIM   = "#4d0010"
C_GREEN     = "#00e676"
C_GREEN_DIM = "#003d1a"
C_TEXT      = "#c8dff0"
C_MUTED     = "#3a6080"
C_WHITE     = "#e8f4ff"

PORT_NAMES = {
    20:"FTP-DATA",21:"FTP",22:"SSH",23:"TELNET",25:"SMTP",
    53:"DNS",67:"DHCP",80:"HTTP",110:"POP3",123:"NTP",
    143:"IMAP",161:"SNMP",443:"HTTPS",445:"SMB",514:"SYSLOG",
    1433:"MSSQL",3306:"MYSQL",3389:"RDP",5432:"PGSQL",8080:"HTTP-ALT",
    8443:"HTTPS-ALT",27017:"MONGODB",6379:"REDIS",5900:"VNC",
}

TCP_PORTS = {20,21,22,23,25,80,110,143,443,445,1433,3306,3389,5432,8080,8443}
UDP_PORTS = {53,67,68,69,123,161,162,514,1194,4500,5060}

JOUR_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
JOUR_FR    = {"Monday":"Lundi","Tuesday":"Mardi","Wednesday":"Mercredi",
              "Thursday":"Jeudi","Friday":"Vendredi","Saturday":"Samedi","Sunday":"Dimanche"}


# ══════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════
def plotly_layout(height=360, title=""):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(4,13,24,0.6)",
        font=dict(family="'Share Tech Mono', monospace", color=C_TEXT, size=11),
        title=dict(text=title, font=dict(size=13, color=C_CYAN), x=0.02, xanchor="left"),
        margin=dict(l=16, r=16, t=44 if title else 16, b=16),
        height=height,
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER2,
                   zerolinecolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER2,
                   zerolinecolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)),
        legend=dict(bgcolor="rgba(4,13,24,0.9)", bordercolor=C_BORDER2,
                    borderwidth=1, font=dict(size=10)),
        coloraxis=dict(colorbar=dict(
            tickfont=dict(color=C_MUTED, size=9),
            title=dict(font=dict(color=C_MUTED, size=10)),
            outlinecolor=C_BORDER, outlinewidth=1,
        )),
    )


# ══════════════════════════════════════════════════════
# GLOBAL CSS — NASA MISSION CONTROL
# ══════════════════════════════════════════════════════
NASA_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:wght@200;300;400;700;900&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:       #020408;
    --surface:  #040d18;
    --panel:    #061225;
    --b1:       #0a2040;
    --b2:       #0d3060;
    --cyan:     #00e5ff;
    --cyan-d:   #005577;
    --amber:    #ffb300;
    --amber-d:  #4d3500;
    --red:      #ff1744;
    --red-d:    #4d0010;
    --green:    #00e676;
    --green-d:  #003d1a;
    --text:     #c8dff0;
    --muted:    #3a6080;
    --white:    #e8f4ff;
}

/* ── App Shell ── */
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0,100,180,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(0,50,120,0.06) 0%, transparent 60%),
        repeating-linear-gradient(0deg, transparent, transparent 39px, rgba(0,229,255,0.02) 39px, rgba(0,229,255,0.02) 40px),
        repeating-linear-gradient(90deg, transparent, transparent 39px, rgba(0,229,255,0.02) 39px, rgba(0,229,255,0.02) 40px) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--b2) !important;
    box-shadow: 4px 0 20px rgba(0,229,255,0.04) !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label { color: var(--muted) !important; font-size: 0.72rem !important; letter-spacing: 2px; text-transform: uppercase; font-family: 'Share Tech Mono', monospace !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--b1) !important;
    border-radius: 4px !important;
    padding: 14px 18px !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), transparent);
}
[data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.8rem !important;
    letter-spacing: 2px;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 3px;
    text-transform: uppercase;
}
[data-testid="stMetricDelta"] { font-family: 'Share Tech Mono', monospace !important; font-size: 0.75rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 2px !important;
    border: 1px solid var(--b1) !important;
    gap: 2px; padding: 3px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 8px 16px !important;
    border: none !important;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--cyan) !important; background: rgba(0,229,255,0.05) !important; }
.stTabs [aria-selected="true"] {
    background: rgba(0,229,255,0.1) !important;
    color: var(--cyan) !important;
    border: 1px solid var(--b2) !important;
    box-shadow: 0 0 12px rgba(0,229,255,0.15), inset 0 0 8px rgba(0,229,255,0.05) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--b2) !important;
    border-radius: 3px !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}
.stSlider > div { color: var(--text) !important; }
[data-baseweb="slider"] > div:first-child { background: var(--b1) !important; }

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid var(--b1) !important;
    border-radius: 3px !important;
    overflow: hidden;
}
.stDataFrame table { font-family: 'Share Tech Mono', monospace !important; font-size: 0.78rem !important; }
.stDataFrame th { background: var(--panel) !important; color: var(--cyan) !important; letter-spacing: 1px; border-bottom: 1px solid var(--b2) !important; }
.stDataFrame td { color: var(--text) !important; border-bottom: 1px solid var(--b1) !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: var(--panel) !important;
    border: 1px solid var(--b1) !important;
    border-radius: 3px !important;
    color: var(--muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 2px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--b2); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan-d); }

/* ── Custom Components ── */
.nasa-panel {
    background: var(--panel);
    border: 1px solid var(--b1);
    border-radius: 4px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}
.nasa-panel::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 60px; height: 60px;
    border-right: 2px solid var(--b2);
    border-bottom: 2px solid var(--b2);
    border-radius: 0 0 4px 0;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: var(--cyan);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--b1);
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::before {
    content: '▶';
    color: var(--cyan);
    font-size: 0.5rem;
}

.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,229,255,0.06);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 2px;
    padding: 4px 12px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: var(--cyan);
    margin: 3px;
}
.stat-chip.red   { background:rgba(255,23,68,0.06); border-color:rgba(255,23,68,0.3); color:var(--red); }
.stat-chip.green { background:rgba(0,230,118,0.06); border-color:rgba(0,230,118,0.3); color:var(--green); }
.stat-chip.amber { background:rgba(255,179,0,0.06); border-color:rgba(255,179,0,0.3); color:var(--amber); }

.alert-bar {
    background: linear-gradient(90deg, rgba(255,23,68,0.15), rgba(255,23,68,0.05));
    border: 1px solid rgba(255,23,68,0.3);
    border-left: 3px solid var(--red);
    border-radius: 3px;
    padding: 10px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--red);
    letter-spacing: 1px;
    margin: 6px 0;
    animation: pulse-red 3s infinite;
}
@keyframes pulse-red {
    0%, 100% { border-left-color: var(--red); box-shadow: none; }
    50% { border-left-color: #ff6090; box-shadow: 0 0 8px rgba(255,23,68,0.2); }
}

.terminal-box {
    background: rgba(0,0,0,0.4);
    border: 1px solid var(--b2);
    border-radius: 3px;
    padding: 12px 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--green);
    letter-spacing: 1px;
    line-height: 1.8;
}

hr { border: none !important; border-top: 1px solid var(--b1) !important; }

/* ── Sidebar logo block ── */
.sidebar-logo {
    padding: 20px 0 16px;
    text-align: center;
    border-bottom: 1px solid var(--b1);
    margin-bottom: 16px;
}
.logo-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 4px;
    filter: drop-shadow(0 0 8px rgba(0,229,255,0.6));
}
.logo-title {
    font-family: 'Exo 2', sans-serif;
    font-weight: 900;
    font-size: 1rem;
    color: var(--cyan);
    letter-spacing: 4px;
    text-transform: uppercase;
}
.logo-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.6rem;
    color: var(--muted);
    letter-spacing: 2px;
    margin-top: 2px;
}

/* ── Blinking cursor ── */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
.cursor { animation: blink 1s infinite; }

/* ── Scanline overlay ── */
.scanlines {
    position: fixed; top:0; left:0; right:0; bottom:0;
    pointer-events: none; z-index: 9999;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
    );
}
</style>
"""


# ══════════════════════════════════════════════════════
# UTILS — Anti LargeUtf8 / Arrow types
# ══════════════════════════════════════════════════════
def _force_no_pyarrow_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit peut planter sur certains dtypes pyarrow (LargeUtf8).
    On convertit les colonnes string[pyarrow] en strings Python (object).
    """
    df = df.copy()

    # Convert dtypes backend -> numpy (enlève beaucoup de types pyarrow)
    try:
        df = df.convert_dtypes(dtype_backend="numpy_nullable")
    except Exception:
        pass

    # Convertir explicitement les colonnes string en object(str)
    for c in df.columns:
        s = df[c]
        try:
            if pd.api.types.is_string_dtype(s) or s.dtype == "object":
                # astype(str) force python string => évite LargeUtf8
                df[c] = s.astype(str)
        except Exception:
            # fallback safe
            df[c] = s.astype(str)

    # datetimes -> garder datetime côté calcul, mais date_jour en string pour affichage
    if "date_jour" in df.columns:
        df["date_jour"] = df["date_jour"].astype(str)

    return df


# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    data_path = "data/logs_export.csv"

    def _read_logs_dataframe(path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        head = p.read_bytes()[:4]
        if head == b"PAR1":
            return pd.read_parquet(path)

        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_parquet(path)

    try:
        try:
            df = _read_logs_dataframe(data_path)
        except Exception:
            return _demo_data(), True

        # ✅ IMPORTANT : supprime LargeUtf8 tout de suite
        df = _force_no_pyarrow_strings(df)

        col_map = {}
        cols_l = {c.lower().strip(): c for c in df.columns}
        mapping = {
            "timestamp":     ["timestamp","date","datetime","time"],
            "src_ip":        ["src_ip","src","source_ip","saddr"],
            "dst_ip":        ["dst_ip","dst","dest_ip","daddr"],
            "protocole":     ["protocole","protocol","proto"],
            "dport":         ["dport","dst_port","dest_port","dpt"],
            "action":        ["action","verdict"],
            "rule":          ["rule","ruleid","rule_id","policyid"],
            "interface_in":  ["interface_in","in","iface_in","interface"],
            "interface_out": ["interface_out","out","iface_out"],
        }
        for key, candidates in mapping.items():
            for c in candidates:
                if c in cols_l and cols_l[c] not in col_map:
                    col_map[cols_l[c]] = key
                    break
        df = df.rename(columns=col_map)

        # Nettoyer colonnes parasites (FW=6, \n)
        for col in list(df.columns):
            if col not in list(mapping.keys()):
                try:
                    uniq = set(df[col].astype(str).str.strip().unique())
                    if uniq.issubset({"6","\\n","\n","","nan"}):
                        df.drop(columns=[col], inplace=True)
                except Exception:
                    pass

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[(df["timestamp"] >= "2025-11-01") & (df["timestamp"] < "2026-03-01")]
            df["date_jour"]    = df["timestamp"].dt.date.astype(str)  # ✅ string
            df["heure"]        = df["timestamp"].dt.hour
            df["mois"]         = df["timestamp"].dt.to_period("M").astype(str)
            df["jour_semaine"] = df["timestamp"].dt.day_name()

        if "dport" in df.columns:
            df["dport"]    = pd.to_numeric(df["dport"], errors="coerce")
            df["port_nom"] = df["dport"].map(PORT_NAMES).fillna(df["dport"].astype(str))

        def deduce_proto(row):
            p = str(row.get("protocole","")).upper().strip()
            if p in ("TCP","6"):  return "TCP"
            if p in ("UDP","17"): return "UDP"
            if p in ("ICMP","1"): return "ICMP"
            port = row.get("dport")
            if pd.notna(port):
                try:
                    port = int(port)
                    if port in TCP_PORTS: return "TCP"
                    if port in UDP_PORTS: return "UDP"
                except Exception:
                    pass
            return "TCP"

        df["PROTO"] = df.apply(deduce_proto, axis=1)

        # ✅ Re-check anti LargeUtf8 après transformations
        df = _force_no_pyarrow_strings(df)

        return df, False  # (data, is_demo)

    except FileNotFoundError:
        return _demo_data(), True


def _demo_data():
    np.random.seed(42)
    n = 50_000
    src_ips = (
        [f"77.90.{np.random.randint(0,255)}.{np.random.randint(1,255)}" for _ in range(8)] +
        [f"94.102.{np.random.randint(0,60)}.{np.random.randint(1,255)}" for _ in range(6)] +
        [f"176.111.{np.random.randint(0,200)}.{np.random.randint(1,255)}" for _ in range(6)] +
        [f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,100)}" for _ in range(10)] +
        [f"10.0.{np.random.randint(0,3)}.{np.random.randint(1,50)}" for _ in range(6)] +
        ["89.89.56.2","28.12.15.20","172.5.2.8","47.128.20.252","79.124.60.150","23.22.35.162"]
    )
    ports  = [80,443,22,53,3306,8080,21,23,3389,123,161,445,25,110,514]
    port_p = [.18,.16,.12,.10,.08,.07,.06,.05,.04,.04,.03,.03,.02,.01,.01]
    rules  = [431,999,153,283,512,77,202]
    rule_p = [.24,.30,.15,.12,.08,.06,.05]

    dports  = np.random.choice(ports, n, p=port_p)
    protos  = ["TCP" if p in TCP_PORTS else "UDP" for p in dports]
    actions = np.random.choice(["DENY","PERMIT"], n, p=[.62,.38])

    dates = pd.date_range("2025-11-01","2026-02-28", periods=n)
    dates = pd.Series(dates)

    df = pd.DataFrame({
        "timestamp":     dates,
        "src_ip":        np.random.choice(src_ips, n),
        "dst_ip":        "159.84.146.99",
        "protocole":     protos,
        "dport":         dports,
        "action":        actions,
        "rule":          np.random.choice(rules, n, p=rule_p),
        "interface_in":  "eth0",
        "interface_out": "eth1",
    })

    df["timestamp"]     = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date_jour"]     = df["timestamp"].dt.date.astype(str)  # ✅ string
    df["heure"]         = df["timestamp"].dt.hour
    df["mois"]          = df["timestamp"].dt.to_period("M").astype(str)
    df["jour_semaine"]  = df["timestamp"].dt.day_name()
    df["dport"]         = pd.to_numeric(df["dport"], errors="coerce")
    df["port_nom"]      = df["dport"].map(PORT_NAMES).fillna(df["dport"].astype(str))
    df["PROTO"]         = protos

    # ✅ anti LargeUtf8
    df = _force_no_pyarrow_strings(df)
    return df


def render_sidebar(active="accueil"):
    """Sidebar commune."""
    with st.sidebar:
        page = active
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">FILTRES OPÉRATIONNELS</div>', unsafe_allow_html=True)

        df_raw, is_demo = load_data()

        mois_dispo = sorted(df_raw["mois"].dropna().unique()) if "mois" in df_raw.columns else []
        mois_sel   = st.multiselect("PÉRIODE", mois_dispo, default=[], key=f"f_mois_{active}")

        action_dispo = df_raw["action"].dropna().unique().tolist() if "action" in df_raw.columns else []
        action_sel   = st.multiselect("ACTION", action_dispo, default=[], key=f"f_action_{active}")

        proto_dispo = df_raw["PROTO"].dropna().unique().tolist() if "PROTO" in df_raw.columns else []
        proto_sel   = st.multiselect("PROTOCOLE", proto_dispo, default=[], key=f"f_proto_{active}")

        port_range = st.slider("", 0, 65535, (0, 65535), key=f"f_port_{active}", label_visibility="collapsed")

        rules_dispo = sorted(df_raw["rule"].dropna().unique().tolist()) if "rule" in df_raw.columns else []
        rules_sel   = st.multiselect("RÈGLES ACTIVES", rules_dispo, default=[], key=f"f_rules_{active}")

        top_n = st.slider("TOP N", 5, 30, 15, key=f"f_topn_{active}", label_visibility="collapsed")

        return page, mois_sel, action_sel, proto_sel, port_range, rules_sel, top_n


def apply_filters(df_raw, mois_sel, action_sel, proto_sel, port_range, rules_sel):
    df = df_raw.copy()

    if mois_sel:
        df = df[df["mois"].isin(mois_sel)]
    if action_sel:
        df = df[df["action"].isin(action_sel)]
    if proto_sel:
        df = df[df["PROTO"].isin(proto_sel)]
    if port_range and "dport" in df.columns:
        df = df[(df["dport"] >= port_range[0]) & (df["dport"] <= port_range[1])]
    if rules_sel:
        df = df[df["rule"].isin(rules_sel)]

    return df