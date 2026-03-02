import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config import (
    NASA_CSS, plotly_layout,
    C_CYAN, C_CYAN_DIM, C_RED, C_GREEN, C_MUTED, C_BORDER2,
    PORT_NAMES,
    load_data, apply_filters, render_sidebar
)

st.set_page_config(page_title="Analyses · OPSIE", layout="wide")

DAY_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# ══════════════════════════════════════════════════════════════════════════════
# GÉOLOCALISATION
# Priorité 1 : geoip2 + GeoLite2-Country.mmdb (offline, si dispo)
# Priorité 2 : ip-api.com batch (gratuit, sans clé, 100 IPs/req)
# ══════════════════════════════════════════════════════════════════════════════

def _try_geoip2_mmdb(ips: list) -> dict:
    """Tente geoip2 local — retourne {} si mmdb absent ou geoip2 non installé."""
    candidates = [
        "GeoLite2-Country.mmdb",
        "/usr/share/GeoIP/GeoLite2-Country.mmdb",
        os.path.expanduser("~/GeoLite2-Country.mmdb"),
    ]
    mmdb = next((p for p in candidates if os.path.exists(p)), None)
    if not mmdb:
        return {}
    try:
        import geoip2.database  # type: ignore
        result = {}
        with geoip2.database.Reader(mmdb) as reader:
            for ip in ips:
                try:
                    rec = reader.country(ip)
                    result[ip] = {
                        "country": rec.country.name or "Unknown",
                        "iso":     rec.country.iso_code or "XX",
                        "lat":     float(getattr(rec.location, "latitude",  0) or 0),
                        "lon":     float(getattr(rec.location, "longitude", 0) or 0),
                    }
                except Exception:
                    pass
        return result
    except ImportError:
        return {}


@st.cache_data(show_spinner=False, ttl=3600)
def _geo_batch_ipapi(ips: tuple) -> dict:
    """Batch ip-api.com — max 100 IPs / requête, résultat mis en cache 1 h."""
    result = {}
    for i in range(0, len(ips), 100):
        chunk = list(ips[i:i + 100])
        try:
            resp = requests.post(
                "http://ip-api.com/batch",
                json=[{"query": ip,
                       "fields": "query,country,countryCode,lat,lon,status"}
                      for ip in chunk],
                timeout=10,
            )
            if resp.status_code == 200:
                for item in resp.json():
                    if item.get("status") == "success":
                        result[item["query"]] = {
                            "country": item["country"],
                            "iso":     item["countryCode"],
                            "lat":     float(item["lat"]),
                            "lon":     float(item["lon"]),
                        }
        except Exception:
            pass
    return result


_PRIVATE_PREFIXES = ("10.", "172.16.", "172.17.", "172.18.", "172.19.",
                     "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                     "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                     "172.30.", "172.31.", "192.168.", "127.", "0.", "::1")


@st.cache_data(show_spinner=False, ttl=3600)
def geolocate_ips_smart(ips_tuple: tuple) -> pd.DataFrame:
    """
    Géolocalise une liste d'IPs uniques (tuple pour le cache Streamlit).
    Retourne un DataFrame : ip | country | iso | lat | lon
    """
    ips = list(ips_tuple)

    # 1. geoip2 local (offline, rapide)
    geo_map = _try_geoip2_mmdb(ips)

    # 2. ip-api.com pour les IPs publiques restantes
    missing_public = [
        ip for ip in ips
        if ip not in geo_map
        and not any(ip.startswith(p) for p in _PRIVATE_PREFIXES)
    ]
    if missing_public:
        geo_map.update(_geo_batch_ipapi(tuple(missing_public)))

    # 3. Assemblage
    rows = []
    for ip in ips:
        g = geo_map.get(ip, {"country": "Unknown", "iso": "XX", "lat": 0.0, "lon": 0.0})
        rows.append({**g, "ip": ip})

    df = pd.DataFrame(rows)
    return df[df["iso"] != "XX"].copy()


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRE IA  — chaque section dépose son contexte ici
# ══════════════════════════════════════════════════════════════════════════════

_IA_REGISTRY: dict = {}


def register_chart(name: str, ctx: dict):
    """Enregistre le contexte d'un graphique pour le panneau IA sidebar."""
    _IA_REGISTRY[name] = ctx


# ══════════════════════════════════════════════════════════════════════════════
# MISTRAL
# ══════════════════════════════════════════════════════════════════════════════

def _get_api_key() -> str:
    try:
        k = st.secrets.get("MISTRAL_API_KEY", "")
    except Exception:
        k = ""
    return k or os.getenv("MISTRAL_API_KEY", "")


def _call_mistral(api_key: str, prompt: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": "mistral-small-latest",
            "messages": [
                {"role": "system", "content": (
                    "Tu es analyste SOC senior spécialisé en sécurité réseau. "
                    "Tu n'inventes jamais de chiffres. "
                    "Analyses structurées, concises, actionnables. "
                    "Base-toi uniquement sur les données JSON fournies."
                )},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 900,
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# PANNEAU IA SIDEBAR
# Affiché après le rendu de tous les graphiques.
# Sélecteur de graphique + mode + bouton → résultat dans la sidebar.
# ══════════════════════════════════════════════════════════════════════════════

def render_ia_sidebar():
    api_key = _get_api_key()
    if not api_key:
        return  # Silencieux si pas de clé

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""<div style='font-family:"Share Tech Mono",monospace;
                        font-size:0.65rem;color:{C_CYAN};
                        letter-spacing:2px;margin-bottom:6px;'>
            🤖 INTERPRÉTATION IA · MISTRAL
        </div>""",
        unsafe_allow_html=True,
    )

    if not _IA_REGISTRY:
        st.sidebar.caption("Aucun graphique enregistré.")
        return

    chart_names = list(_IA_REGISTRY.keys())

    selected = st.sidebar.selectbox(
        "📊 Graphique",
        chart_names,
        key="ia_chart_select",
    )

    mode = st.sidebar.radio(
        "🎯 Mode d'analyse",
        ["Résumé · 3 points clés", "Analyse complète", "Recommandations sécurité"],
        key="ia_mode_radio",
    )

    run = st.sidebar.button(
        "🚀 Lancer l'analyse",
        use_container_width=True,
        key="ia_run_btn",
        type="primary",
    )

    if run:
        ctx     = _IA_REGISTRY[selected]
        ctx_str = json.dumps(ctx, ensure_ascii=False, indent=2, default=str)

        INSTRUCTIONS = {
            "Résumé · 3 points clés": (
                f"Graphique analysé : '{selected}'. "
                "Réponds UNIQUEMENT avec 3 bullet points (•) courts et actionnables. "
                "Pas d'introduction ni de conclusion."
            ),
            "Recommandations sécurité": (
                f"Graphique analysé : '{selected}'. "
                "Donne exactement 4 recommandations opérationnelles numérotées. "
                "Chaque recommandation doit citer un chiffre précis issu des données."
            ),
            "Analyse complète": (
                f"Graphique analysé : '{selected}'. "
                "Structure ta réponse en 3 sections titrées :\n"
                "**1. Ce que montre ce graphique**\n"
                "**2. Points d'attention & anomalies**\n"
                "**3. Recommandations prioritaires**"
            ),
        }

        prompt = f"{INSTRUCTIONS[mode]}\n\nDONNÉES (JSON) :\n{ctx_str}"

        with st.sidebar:
            with st.spinner("Analyse en cours…"):
                try:
                    out = _call_mistral(api_key, prompt)

                    st.sidebar.markdown(
                        f"<div style='font-size:0.7rem;color:{C_MUTED};"
                        f"letter-spacing:1px;margin:8px 0 4px;'>"
                        f"📋 RÉSULTAT · {selected.upper()}</div>",
                        unsafe_allow_html=True,
                    )
                    # Affichage scrollable
                    st.sidebar.markdown(
                        f"""<div style='
                            background:rgba(0,229,255,0.04);
                            border:1px solid {C_BORDER2};
                            border-radius:6px;
                            padding:10px 12px;
                            font-size:0.78rem;
                            line-height:1.55;
                            max-height:420px;
                            overflow-y:auto;
                        '>{out.replace(chr(10),'<br>')}</div>""",
                        unsafe_allow_html=True,
                    )
                    st.sidebar.download_button(
                        "⬇️ Télécharger l'analyse",
                        data=out.encode("utf-8"),
                        file_name=f"ia_{selected.lower().replace(' ','_')[:40]}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        key="ia_dl_btn",
                    )
                except Exception as e:
                    st.sidebar.error(f"Erreur Mistral : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS VISUELS
# ══════════════════════════════════════════════════════════════════════════════

def _cbar(title: str) -> dict:
    """Colorbar compatible Plotly 5+  (titlefont → title.font)."""
    return dict(
        title=dict(text=title, font=dict(color=C_MUTED, size=11)),
        tickfont=dict(color=C_MUTED, size=10),
        bgcolor="rgba(0,0,0,0)",
        bordercolor=C_BORDER2,
        borderwidth=1,
    )


def _geo_base() -> dict:
    return dict(
        showframe=False,
        showcoastlines=True, coastlinecolor=C_BORDER2,
        showland=True,       landcolor="#0a1220",
        showocean=True,      oceancolor="#060d1a",
        showlakes=False,
        bgcolor="rgba(0,0,0,0)",
    )


def _map_layout(height: int = 460, projection: str = "natural earth") -> dict:
    geo = _geo_base()
    geo["projection_type"] = projection
    return dict(
        geo=geo,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_MUTED, family="Share Tech Mono, monospace"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# PREP & DISPLAY UTILS
# ══════════════════════════════════════════════════════════════════════════════

def prep_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "PROTO" not in df.columns and "protocole" in df.columns:
        df["PROTO"] = df["protocole"].astype(str).str.upper()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "date_jour"   not in df.columns:
            df["date_jour"]   = df["timestamp"].dt.date.astype(str)
        if "heure"        not in df.columns:
            df["heure"]       = df["timestamp"].dt.hour.astype("Int64")
        if "jour_semaine" not in df.columns:
            df["jour_semaine"]= df["timestamp"].dt.dayofweek.astype("Int64")
        if "mois"         not in df.columns:
            df["mois"]        = df["timestamp"].dt.to_period("M").astype(str)
    return df


def df_for_display(df: pd.DataFrame, max_rows: int = 5000) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.tail(max_rows).copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            out[c] = s.astype(str)
        elif pd.api.types.is_string_dtype(s) or s.dtype == "object":
            out[c] = s.astype(str)
    return out.astype(object).reset_index(drop=True)


def header():
    st.markdown(NASA_CSS, unsafe_allow_html=True)
    st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="border-bottom:1px solid {C_BORDER2};
                    padding:18px 4px 16px;margin-bottom:18px;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;
                      color:{C_MUTED};letter-spacing:3px;">
            SISE-OPSIE · NASA UI · FIREWALL INTELLIGENCE
          </div>
          <div style="font-family:'Exo 2',sans-serif;font-weight:900;
                      font-size:2rem;color:{C_CYAN};letter-spacing:3px;">
            ANALYSES OPÉRATIONNELLES
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTIONS GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════════

def section_timeline_proto(df: pd.DataFrame, top_n: int, n_total: int):
    colA, colB = st.columns([3, 1])

    # ── Timeline ─────────────────────────────────────────────────────────
    with colA:
        st.markdown('<div class="section-label">TIMELINE · DENY vs PERMIT</div>',
                    unsafe_allow_html=True)
        if set(["date_jour", "action"]).issubset(df.columns) and n_total:
            daily    = df.groupby(["date_jour", "action"]).size().reset_index(name="count")
            deny_d   = daily[daily["action"] == "DENY"]
            permit_d = daily[daily["action"] == "PERMIT"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=deny_d["date_jour"], y=deny_d["count"],
                mode="lines", name="DENY",
                line=dict(color=C_RED, width=1.5),
                fill="tozeroy", fillcolor="rgba(255,23,68,0.08)"))
            fig.add_trace(go.Scatter(
                x=permit_d["date_jour"], y=permit_d["count"],
                mode="lines", name="PERMIT",
                line=dict(color=C_GREEN, width=1.5),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.06)"))

            lo = plotly_layout(height=280)
            lo["xaxis"]["showgrid"] = False
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True)

            register_chart("Timeline DENY vs PERMIT", {
                "periode":       f"{daily['date_jour'].min()} → {daily['date_jour'].max()}",
                "total_events":  n_total,
                "deny_pic":      int(deny_d["count"].max())   if len(deny_d)   else 0,
                "permit_pic":    int(permit_d["count"].max()) if len(permit_d) else 0,
                "deny_moy_jour": round(float(deny_d["count"].mean()), 1) if len(deny_d) else 0,
                "top5_deny_days": deny_d.nlargest(5, "count")[["date_jour","count"]]
                                        .to_dict(orient="records"),
            })
        else:
            st.info("Colonnes nécessaires : date_jour + action")

    # ── Top protocoles ───────────────────────────────────────────────────
    with colB:
        st.markdown('<div class="section-label">TOP PROTO</div>', unsafe_allow_html=True)
        if "PROTO" in df.columns and n_total:
            vc     = df["PROTO"].value_counts()
            colors = [C_CYAN, C_CYAN_DIM, "#0077aa", "#005577", "#003344"]
            fig = go.Figure(go.Bar(
                x=vc.index, y=vc.values,
                marker=dict(color=colors[:len(vc)]),
                text=[f"{v:,}" for v in vc.values],
                textposition="outside"))
            fig.update_layout(**plotly_layout(height=280))
            st.plotly_chart(fig, use_container_width=True)

            register_chart("Distribution des Protocoles", {
                "protocoles":     {str(k): int(v) for k, v in vc.items()},
                "proto_dominant": str(vc.index[0]) if len(vc) else "—",
                "pct_dominant":   round(float(vc.iloc[0] / vc.sum() * 100), 1) if len(vc) else 0,
            })
        else:
            st.info("Colonne PROTO manquante.")


def section_top_ip_ports(df: pd.DataFrame, top_n: int, n_total: int):
    c1, c2 = st.columns(2)

    # ── Top IP bloquées ──────────────────────────────────────────────────
    with c1:
        st.markdown('<div class="section-label">TOP IP SOURCES · BLOQUÉES</div>',
                    unsafe_allow_html=True)
        if set(["src_ip", "action"]).issubset(df.columns) and n_total:
            top_ip = (df[df["action"] == "DENY"]["src_ip"]
                      .value_counts().head(int(top_n)).reset_index())
            top_ip.columns = ["IP", "Blocages"]

            fig = go.Figure(go.Bar(
                x=top_ip["Blocages"], y=top_ip["IP"], orientation="h",
                marker=dict(color=top_ip["Blocages"],
                            colorscale=[[0,"rgba(255,23,68,0.2)"],[1,C_RED]])))
            lo = plotly_layout(height=360)
            lo["yaxis"]["autorange"] = "reversed"
            lo["xaxis"]["showgrid"]  = False
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True)

            register_chart("Top IP Sources Bloquées", {
                "top_ip":      top_ip.to_dict(orient="records"),
                "ip_top":      str(top_ip.iloc[0]["IP"])      if len(top_ip) else "—",
                "blocages_top":int(top_ip.iloc[0]["Blocages"]) if len(top_ip) else 0,
                "total_deny":  int((df["action"] == "DENY").sum()),
            })

    # ── Top ports destination ────────────────────────────────────────────
    with c2:
        st.markdown('<div class="section-label">TOP PORTS DESTINATION</div>',
                    unsafe_allow_html=True)
        if "dport" in df.columns and n_total:
            tp = df["dport"].value_counts().head(int(top_n)).reset_index()
            tp.columns = ["Port", "Occ"]
            tp["Label"] = (tp["Port"].astype(str) + " · "
                           + tp["Port"].map(PORT_NAMES).fillna("UNKNOWN"))

            fig = go.Figure(go.Bar(
                x=tp["Occ"], y=tp["Label"], orientation="h",
                marker=dict(color=tp["Occ"],
                            colorscale=[[0,"rgba(0,229,255,0.18)"],[1,C_CYAN]])))
            lo = plotly_layout(height=360)
            lo["yaxis"]["autorange"] = "reversed"
            lo["xaxis"]["showgrid"]  = False
            fig.update_layout(**lo)
            st.plotly_chart(fig, use_container_width=True)

            register_chart("Top Ports Destination", {
                "top_ports":       tp[["Port","Label","Occ"]].to_dict(orient="records"),
                "port_dominant":   int(tp.iloc[0]["Port"])   if len(tp) else 0,
                "service_dominant":str(tp.iloc[0]["Label"])  if len(tp) else "—",
            })


def section_heatmap_heure_jour(df: pd.DataFrame, n_total: int):
    st.markdown('<div class="section-label">HEATMAP · ACTIVITÉ PAR HEURE & JOUR</div>',
                unsafe_allow_html=True)
    if not set(["heure","jour_semaine"]).issubset(df.columns) or not n_total:
        st.info("Colonnes heure + jour_semaine nécessaires.")
        return

    tmp = df.copy()

    # ── Cast explicite en int natif Python (évite Int64 nullable qui casse .between)
    tmp["heure"]    = pd.to_numeric(tmp["heure"],        errors="coerce").astype("float").astype("Int64")
    jours_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
 }  

    # Applique le mappage pour convertir les jours en numéros
    tmp["jour_num"] = tmp["jour_semaine"].map(jours_map)


    # Retire les NaN et force en int64 standard pour groupby + pivot
      
    tmp = tmp.dropna(subset=["heure","jour_num"])   
    tmp["heure"]    = tmp["heure"].astype(int)
    tmp["jour_num"] = tmp["jour_num"].astype(int)
    tmp = tmp[tmp["jour_num"].between(0, 6) & tmp["heure"].between(0, 23)]
    if tmp.empty:
        st.info("Aucune donnée temporelle valide après nettoyage.")
        return

    heat  = tmp.groupby(["jour_num","heure"]).size().reset_index(name="count")
    pivot = (heat.pivot(index="jour_num", columns="heure", values="count")
                 .fillna(0).sort_index())
    pivot.index = [DAY_FR[int(i)] for i in pivot.index]

    # Colonnes : toutes les heures 0-23 même si absentes (grille complète)
    all_hours = list(range(24))
    for h in all_hours:
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot[sorted(pivot.columns)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}h" for h in pivot.columns],
        y=list(pivot.index),
        colorscale=[[0, "#0a1628"], [0.4, "#003355"], [1, "#4a90e2"]],
        hoverongaps=False,
        hovertemplate="Jour: %{y}<br>Heure: %{x}<br>Événements: %{z:,}<extra></extra>"))
    fig.update_layout(**plotly_layout(height=300))
    st.plotly_chart(fig, use_container_width=True)

    # Stats pour le registre IA — même cast int natif
    heure_int = tmp["heure"]  
    jour_int  = tmp["jour_num"]

    peak = heat.sort_values("count", ascending=False).iloc[0] if len(heat) else None
    register_chart("Heatmap Heure × Jour", {
        "pic": {
            "jour":   DAY_FR[int(peak["jour_num"])] if peak is not None else "—",
            "heure":  int(peak["heure"])             if peak is not None else 0,
            "events": int(peak["count"])             if peak is not None else 0,
        },
        "nuit_00_06":   int((heure_int.between(0,  6)).sum()),
        "bureau_08_18": int((heure_int.between(8, 18)).sum()),
        "pct_weekend":  round(float(jour_int.isin([5,6]).sum() / n_total * 100), 1),
        "total": n_total,
    })


def section_heatmap_port_proto(df: pd.DataFrame, top_n: int, n_total: int):
    st.markdown('<div class="section-label">HEATMAP · PORTS × PROTOCOLES</div>',
                unsafe_allow_html=True)
    if not set(["dport","PROTO"]).issubset(df.columns) or not n_total:
        st.info("Colonnes dport + PROTO nécessaires.")
        return

    top_ports  = df["dport"].value_counts().head(int(top_n)).index
    top_protos = df["PROTO"].value_counts().head(6).index
    sub = df[df["dport"].isin(top_ports) & df["PROTO"].isin(top_protos)]
    pivot = sub.groupby(["PROTO","dport"]).size().unstack(fill_value=0)
    pivot.columns = [
        f"{int(c)} · {PORT_NAMES.get(int(c),'UNKNOWN')}" for c in pivot.columns
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[[0,"#0a1628"],[0.4,"#003355"],[0.75,C_CYAN_DIM],[1,C_CYAN]],
        hovertemplate="Proto: %{y}<br>Port: %{x}<br>Événements: %{z:,}<extra></extra>"))
    fig.update_layout(**plotly_layout(height=320))
    st.plotly_chart(fig, use_container_width=True)

    register_chart("Heatmap Ports × Protocoles", {
        "top_ports":  [str(p) for p in top_ports.tolist()],
        "top_protos": [str(p) for p in top_protos.tolist()],
        "top5_combos": (
            sub.groupby(["PROTO","dport"]).size()
               .reset_index(name="count")
               .sort_values("count", ascending=False)
               .head(5).to_dict(orient="records")
        ),
    })


def section_deny_par_regle(df: pd.DataFrame, top_n: int, n_total: int):
    st.markdown('<div class="section-label">DENY · PAR RÈGLE FIREWALL</div>',
                unsafe_allow_html=True)
    if not set(["rule","action"]).issubset(df.columns) or not n_total:
        st.info("Colonnes rule + action nécessaires.")
        return

    rules = (df[df["action"] == "DENY"]["rule"]
             .value_counts().head(int(top_n)).reset_index())
    rules.columns = ["Règle","DENY"]

    fig = go.Figure(go.Bar(
        x=rules["Règle"].astype(str), y=rules["DENY"],
        marker=dict(color=rules["DENY"],
                    colorscale=[[0,"rgba(255,23,68,0.3)"],[1,C_RED]]),
        text=[f"{v:,}" for v in rules["DENY"]],
        textposition="outside"))
    lo = plotly_layout(height=320)
    lo["xaxis"]["showgrid"] = False
    fig.update_layout(**lo)
    st.plotly_chart(fig, use_container_width=True)

    total_deny = int((df["action"] == "DENY").sum())
    register_chart("DENY par Règle Firewall", {
        "top_regles": rules.to_dict(orient="records"),
        "regle_top":  str(rules.iloc[0]["Règle"])  if len(rules) else "—",
        "deny_top":   int(rules.iloc[0]["DENY"])   if len(rules) else 0,
        "pct_top": round(
            float(rules.iloc[0]["DENY"] / total_deny * 100), 1
        ) if len(rules) and total_deny else 0,
        "total_deny": total_deny,
    })


# ══════════════════════════════════════════════════════════════════════════════
# CARTES GÉOGRAPHIQUES (4 tabs)
# ══════════════════════════════════════════════════════════════════════════════

def section_cartes(df: pd.DataFrame, top_n: int, n_total: int):
    st.markdown(
        '<div class="section-label">🌍 CARTOGRAPHIE GÉOGRAPHIQUE · TRAFIC FIREWALL</div>',
        unsafe_allow_html=True)

    if "src_ip" not in df.columns or n_total == 0:
        st.info("Colonne src_ip nécessaire pour les cartes.")
        return

    # ── Géolocalisation (cachée) ─────────────────────────────────────────
    unique_ips = tuple(df["src_ip"].dropna().unique().tolist())
    with st.spinner("Géolocalisation des IP en cours…"):
        geo_df = geolocate_ips_smart(unique_ips)

    if len(geo_df) == 0:
        st.warning(
            "Aucune IP publique géolocalisée. "
            "Vérifiez la connectivité vers ip-api.com ou placez un fichier "
            "GeoLite2-Country.mmdb dans le répertoire de l'application.")
        return

    # ── Merge avec les comptages action ─────────────────────────────────
    if "action" in df.columns:
        cnt = (df.groupby(["src_ip","action"]).size()
                 .reset_index(name="n"))
        cnt.columns = ["ip","action","n"]
        geo_df = geo_df.merge(cnt, on="ip", how="left")
        geo_df["action"] = geo_df["action"].fillna("UNKNOWN")
        geo_df["n"]      = geo_df["n"].fillna(0).astype(int)
    else:
        geo_df["action"] = "UNKNOWN"
        geo_df["n"] = 1

    # ── Agrégation par pays ──────────────────────────────────────────────
    # Important: total et deny doivent être calculés au même niveau de groupement
    # pour éviter les %DENY > 100 liés à des duplications lors d'un merge par ISO.
    geo_df = geo_df.copy()
    geo_df["deny_n"] = np.where(geo_df["action"] == "DENY", geo_df["n"], 0)

    by_country = (
        geo_df.groupby(["country", "iso"], as_index=False)
              .agg(
                  total=("n", "sum"),
                  deny=("deny_n", "sum"),
                  lat=("lat", "mean"),
                  lon=("lon", "mean"),
              )
    )
    by_country["total"] = by_country["total"].astype(int)
    by_country["deny"] = by_country["deny"].astype(int)
    by_country["deny_pct"] = np.where(
        by_country["total"] > 0,
        (by_country["deny"] / by_country["total"] * 100).round(1),
        0.0,
    )

    # ── Top IPs DENY pour bubble ─────────────────────────────────────────
    top_ip_geo = (
        geo_df[geo_df["action"] == "DENY"]
        .groupby(["ip","country","iso","lat","lon"])["n"]
        .sum().reset_index(name="events")
        .sort_values("events", ascending=False)
        .head(int(top_n) * 4)
    ).copy()
    rng = np.random.default_rng(42)
    top_ip_geo["lat_j"] = top_ip_geo["lat"] + rng.uniform(-1.5, 1.5, len(top_ip_geo))
    top_ip_geo["lon_j"] = top_ip_geo["lon"] + rng.uniform(-1.5, 1.5, len(top_ip_geo))

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🗺️  Volume par Pays",
        "🔴  Bubble Map · IP Bloquées",
        "🔥  Taux DENY par Pays"
    ])

    # ── Tab 1 : Choroplèthe volume ────────────────────────────────────────
    with tab1:
        st.caption("Volume total d'événements (DENY + PERMIT) par pays d'origine des IP sources") 
        test_data = by_country.copy()
        fig = go.Figure(go.Choropleth(
            locations=test_data["country"],
            locationmode="country names",
            z=test_data["total"],
            text=test_data["country"],
            colorscale="Viridis",
            zmin=0,
            zmax=test_data["total"].max(),
            autocolorscale=False,
            marker_line_color="#1a2a4a",
            marker_line_width=0.5,
            colorbar=_cbar("Événements"),
            hovertemplate="<b>%{text}</b><br>Événements: %{z:,}<extra></extra>"
        ))
        fig.update_layout(**_map_layout())
        st.plotly_chart(fig, use_container_width=True, key="map_vol")

        top10 = by_country.sort_values("total", ascending=False).head(10)
        register_chart("Carte · Volume par Pays", {
            "nb_pays":   int(by_country["country"].nunique()),
            "top10":     top10[["country","total"]].to_dict(orient="records"),
            "pays_top1": str(top10.iloc[0]["country"]) if len(top10) else "—",
            "vol_top1":  int(top10.iloc[0]["total"])   if len(top10) else 0,
            "pct_top1":  round(
                float(top10.iloc[0]["total"] / by_country["total"].sum() * 100), 1
            ) if len(top10) else 0,
        })

    # ── Tab 2 : Bubble map IP bloquées ───────────────────────────────────
    with tab2:
        st.caption(
            "Top IP sources bloquées géolocalisées — "
            "taille = volume DENY · couleur = intensité · survolez pour détails")
        if len(top_ip_geo) > 0:
            fig = go.Figure(go.Scattergeo(
                lat=top_ip_geo["lat_j"],
                lon=top_ip_geo["lon_j"],
                text="<b>" + top_ip_geo["ip"] + "</b><br>" + top_ip_geo["country"],
                mode="markers",
                marker=dict(
                    size=np.clip(np.log1p(top_ip_geo["events"]) * 5.5, 7, 32),
                    color=top_ip_geo["events"],
                    colorscale=[
                        [0,"rgba(255,23,68,0.3)"],
                        [0.5,"rgba(255,80,0,0.7)"],
                        [1,C_RED]],
                    showscale=True,
                    colorbar=_cbar("Blocages"),
                    line=dict(color=C_RED, width=0.5),
                    opacity=0.88),
                hovertemplate="%{text}<br>Blocages: %{marker.color:,}<extra></extra>"))
            fig.update_layout(**_map_layout())
            st.plotly_chart(fig, use_container_width=True, key="map_bubble")

            register_chart("Carte · Bubble Map IP Bloquées", {
                "nb_ip_tracees": len(top_ip_geo),
                "top15": top_ip_geo.head(15)[["ip","country","events"]]
                                   .to_dict(orient="records"),
                "ip_top":   str(top_ip_geo.iloc[0]["ip"])      if len(top_ip_geo) else "—",
                "pays_top": str(top_ip_geo.iloc[0]["country"]) if len(top_ip_geo) else "—",
                "events_top": int(top_ip_geo.iloc[0]["events"]) if len(top_ip_geo) else 0,
            })
        else:
            st.info("Pas de données DENY géolocalisées.")

    # ── Tab 3 : Choroplèthe DENY% ─────────────────────────────────────────
    with tab3:
        st.caption(
            "Taux de blocage (DENY%) par pays — "
            "rouge foncé = trafic quasi-intégralement bloqué")
        deny_map = by_country[by_country["deny"] > 0].copy()
        if len(deny_map) > 0:
            fig = go.Figure(go.Choropleth(
                locations=deny_map["country"],
                locationmode="country names",
                z=deny_map["deny_pct"],
                text=deny_map["country"],
                customdata=np.stack(
                    [deny_map["deny"], deny_map["total"]], axis=-1),
                colorscale=[
                    [0,"#0a1628"],[0.25,"#220011"],
                    [0.6,"#880022"],[1,C_RED]],
                autocolorscale=False,
                zmin=0, zmax=100,
                marker_line_color="#1a2a4a",
                marker_line_width=0.5,
                colorbar=_cbar("DENY %"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "DENY: %{z:.1f}%<br>"
                    "Bloqués: %{customdata[0]:,}<br>"
                    "Total: %{customdata[1]:,}<extra></extra>")))
            fig.update_layout(**_map_layout())
            st.plotly_chart(fig, use_container_width=True, key="map_deny")

            top_deny = deny_map.sort_values("deny_pct", ascending=False)
            register_chart("Carte · Taux DENY par Pays", {
                "top10":       top_deny.head(10)[["country","deny_pct","deny","total"]]
                                       .to_dict(orient="records"),
                "pays_100pct": deny_map[deny_map["deny_pct"] == 100]["country"].tolist(),
                "pays_gt80":   deny_map[deny_map["deny_pct"] > 80]["country"].tolist(),
                "nb_pays_partiels": int((deny_map["deny_pct"] < 100).sum()),
            })
        else:
            st.info("Pas de données DENY disponibles.")

    


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    header()

    df_raw, _is_demo = load_data()
    df_raw = prep_df(df_raw)

    _page, mois_sel, action_sel, proto_sel, port_range, rules_sel, top_n = \
        render_sidebar("analyses")
    df = apply_filters(df_raw, mois_sel, action_sel, proto_sel, port_range, rules_sel)

    n_total  = int(len(df))
    n_deny   = int((df["action"] == "DENY").sum())   if "action" in df.columns else 0
    n_permit = int((df["action"] == "PERMIT").sum()) if "action" in df.columns else 0
    pct_deny = (n_deny / n_total * 100) if n_total else 0.0

    # ── KPIs ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOTAL",           f"{n_total:,}")
    c2.metric("DENY",            f"{n_deny:,}",
              f"{pct_deny:.1f}%", delta_color="inverse")
    c3.metric("PERMIT",          f"{n_permit:,}",
              f"{(n_permit/n_total*100):.1f}%" if n_total else "0%")
    c4.metric("IP SRC UNIQUES",
              f"{df['src_ip'].nunique():,}" if "src_ip" in df.columns else "—")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Graphiques ───────────────────────────────────────────────────────
    section_timeline_proto(df, top_n, n_total)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    section_top_ip_ports(df, top_n, n_total)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    section_heatmap_heure_jour(df, n_total)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    section_heatmap_port_proto(df, top_n, n_total)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    section_deny_par_regle(df, top_n, n_total)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Cartes ───────────────────────────────────────────────────────────
    section_cartes(df, top_n, n_total)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Table ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">ÉVÉNEMENTS · TABLE</div>',
                unsafe_allow_html=True)
    st.dataframe(df_for_display(df, 5000), use_container_width=True, height=320)
    st.download_button(
        "⬇️ Télécharger les données FILTRÉES (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="opsie_filtre.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Panneau IA sidebar ───────────────────────────────────────────────
    # Appelé EN DERNIER : tous les register_chart() ont été exécutés
    render_ia_sidebar()


main()   