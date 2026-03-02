import json
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (
    NASA_CSS, plotly_layout,
    C_CYAN, C_CYAN_DIM, C_MUTED, C_BORDER2,
    PORT_NAMES,
    load_data, apply_filters, render_sidebar
)

st.set_page_config(page_title="ML · OPSIE", layout="wide")


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
def header():
    st.markdown(NASA_CSS, unsafe_allow_html=True)
    st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="border-bottom:1px solid {C_BORDER2};padding:18px 4px 16px;margin-bottom:18px;">
          <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:{C_MUTED};letter-spacing:3px;">
            SISE-OPSIE · NASA UI · MACHINE LEARNING
          </div>
          <div style="font-family:'Exo 2',sans-serif;font-weight:900;font-size:2rem;color:{C_CYAN};letter-spacing:3px;">
            ACP + KMEANS · TYPOLOGIE DU TRAFIC
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _safe_cols(df, cols):
    return [c for c in cols if c in df.columns]


def _df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_timedelta64_dtype(s):
            out[c] = s.astype(str)
        elif pd.api.types.is_string_dtype(s) or s.dtype == "object":
            out[c] = s.astype(str)
    return out.astype(object).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Profiling clusters
# ──────────────────────────────────────────────────────────────────────────────
def cluster_profile_table(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    rows = []
    for cl, g in df.groupby("cluster"):
        n = len(g)
        deny_pct = float((g["action"] == "DENY").mean() * 100) if "action" in g.columns else np.nan

        top_ports = g["dport"].value_counts().head(top_n).index.tolist() if "dport" in g.columns else []
        top_ports_lbl = ", ".join([f"{int(p)}({PORT_NAMES.get(int(p),'?')})" for p in top_ports if pd.notna(p)])

        top_rules = g["rule"].value_counts().head(top_n).index.tolist() if "rule" in g.columns else []
        top_rules_lbl = ", ".join([str(r) for r in top_rules])

        top_proto = g["PROTO"].value_counts().head(top_n).index.tolist() if "PROTO" in g.columns else []
        top_proto_lbl = ", ".join([str(p) for p in top_proto])

        rows.append({
            "cluster": int(cl),
            "n_events": int(n),
            "deny_%": round(deny_pct, 1) if not np.isnan(deny_pct) else None,
            "top_ports": top_ports_lbl,
            "top_rules": top_rules_lbl,
            "top_PROTO": top_proto_lbl,
        })

    return pd.DataFrame(rows).sort_values("deny_%", ascending=False, na_position="last")


# ──────────────────────────────────────────────────────────────────────────────
# LLM (Mistral)
# ──────────────────────────────────────────────────────────────────────────────
def build_llm_context(df_ml: pd.DataFrame, prof_df: pd.DataFrame, k: int, sil, var_exp: float) -> str:
    ctx = {
        "ml_summary": {
            "k_clusters": int(k),
            "n_sample": int(len(df_ml)),
            "variance_pc1_pc2_pct": round(float(var_exp) * 100, 1),
            "silhouette": float(sil) if sil is not None else None,
        },
        "clusters_profile": prof_df.to_dict(orient="records"),
        "reading_hint": (
            "Un cluster est considéré à risque s’il combine DENY% élevé + ports sensibles "
            "(22/3389/445/23/3306...) + éventuelle signature temporelle (nuit)."
        ),
        "constraints": [
            "Ne pas inventer de chiffres absents du contexte.",
            "Rester prudent : données simulées, pas de causalité."
        ],
    }
    return json.dumps(ctx, ensure_ascii=False, indent=2)


def call_mistral(api_key: str, user_prompt: str, model: str = "mistral-small-latest") -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Tu es analyste SOC. Tu n’inventes jamais de chiffres. Analyse structurée et actionnable."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _get_api_key() -> str:
    try:
        secret_key = st.secrets.get("MISTRAL_API_KEY", "")
    except Exception:
        secret_key = ""
    return secret_key or os.getenv("MISTRAL_API_KEY", "")


def render_llm_panel(df_ml: pd.DataFrame, prof: pd.DataFrame, k: int, sil, var_exp: float):
    st.markdown('<div class="section-label">IA GÉNÉRATIVE · INTERPRÉTATION (MISTRAL)</div>', unsafe_allow_html=True)

    with st.expander("🧠 Générer un commentaire automatique des résultats ML", expanded=False):
        api_key = _get_api_key()
        if not api_key:
            st.warning('Ajoute ta clé dans `.streamlit/secrets.toml` : `MISTRAL_API_KEY = "..."` ou passe `-e MISTRAL_API_KEY=...` en docker.')
            return

        mode = st.selectbox("Type de sortie", ["Résumé exécutif", "Analyse complète", "Recommandations sécurité"], index=1)
        ctx = build_llm_context(df_ml, prof, k, sil, var_exp)

        if mode == "Résumé exécutif":
            instruction = "Résumé exécutif (8 lignes max) : qualité (variance/silhouette), profils, risque principal, recommandation prioritaire."
        elif mode == "Recommandations sécurité":
            instruction = "6 recommandations opérationnelles basées sur les clusters (ports/règles/protocoles). Justifie avec le contexte."
        else:
            instruction = (
                "Analyse structurée en 4 sections : "
                "1) Résumé exécutif, 2) Qualité (silhouette/variance) et lecture, "
                "3) Lecture cluster par cluster (scénarios plausibles), "
                "4) Recommandations + limites (données simulées)."
            )

        prompt = f"{instruction}\n\nCONTEXTE (JSON):\n{ctx}"

        if st.button("🚀 Générer", use_container_width=True):
            with st.spinner("Analyse IA en cours..."):
                try:
                    out = call_mistral(api_key, prompt)
                    st.markdown(out)
                    st.download_button("⬇️ Télécharger le rapport (TXT)", out.encode("utf-8"), "rapport_ml_llm.txt", "text/plain", use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur appel Mistral : {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    header()

    df_raw, _is_demo = load_data()
    _page, mois_sel, action_sel, proto_sel, port_range, rules_sel, top_n = render_sidebar("ml")
    df = apply_filters(df_raw, mois_sel, action_sel, proto_sel, port_range, rules_sel)

    st.markdown('<div class="section-label">OBJECTIF</div>', unsafe_allow_html=True)
    st.write(
        "Créer une **typologie automatique** des événements firewall (profils homogènes). "
        "ACP = projection 2D (lecture visuelle), KMeans = segmentation. "
        "**action** sert uniquement à interpréter le niveau de risque des clusters."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        k = st.slider("Nombre de clusters (KMeans)", 2, 8, 4, 1)
    with c2:
        max_up = min(50000, len(df)) if len(df) else 5000
        sample_n = st.slider("Taille échantillon (perf)", 5000, max_up, min(20000, max_up), 5000)
    with c3:
        random_state = st.number_input("Random state", value=42, step=1)

    if len(df) == 0:
        st.warning("Dataset vide après filtres.")
        return

    df_ml = df.sample(n=min(int(sample_n), len(df)), random_state=int(random_state)).copy()

    num_cols = _safe_cols(df_ml, ["dport", "heure", "rule"])
    cat_cols = _safe_cols(df_ml, ["PROTO", "jour_semaine"])

    if not num_cols and not cat_cols:
        st.error("Aucune colonne exploitable (dport/heure/rule/PROTO/jour_semaine).")
        return

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols) if num_cols else ("num", "drop", []),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) if cat_cols else ("cat", "drop", []),
        ],
        remainder="drop",
    )

    X = pre.fit_transform(df_ml)

    pca = PCA(n_components=2, random_state=int(random_state))
    X2 = pca.fit_transform(X)

    km = KMeans(n_clusters=int(k), n_init="auto", random_state=int(random_state))
    labels = km.fit_predict(X)

    df_ml["cluster"] = labels
    df_ml["pc1"] = X2[:, 0]
    df_ml["pc2"] = X2[:, 1]

    df_ml = df_ml.dropna(subset=["pc1", "pc2"])

    sil = None
    try:
        if int(k) >= 2 and len(df_ml) > int(k):
            sil = silhouette_score(X, labels)
    except Exception:
        sil = None

    st.markdown('<div class="section-label">RÉSULTATS</div>', unsafe_allow_html=True)
    cA, cB, cC = st.columns(3)
    cA.metric("ÉCHANTILLON", f"{len(df_ml):,}")
    var_exp = float(pca.explained_variance_ratio_.sum())
    cB.metric("VAR EXPLIQUÉE (PC1+PC2)", f"{(var_exp*100):.1f}%")
    cC.metric("SILHOUETTE", f"{sil:.3f}" if sil is not None else "—")

    st.markdown('<div class="section-label">ACP · PROJECTION 2D + CLUSTERS</div>', unsafe_allow_html=True)
    fig = go.Figure()

    # ✅ FIX Docker: Scatter (pas Scattergl)
    for cl in sorted(df_ml["cluster"].unique()):
        g = df_ml[df_ml["cluster"] == cl]
        fig.add_trace(go.Scatter(
            x=g["pc1"], y=g["pc2"],
            mode="markers",
            name=f"Cluster {cl}",
            marker=dict(size=5, opacity=0.7),
            text=[cl] * len(g),
            hovertemplate="Cluster: %{text}<br>PC1=%{x:.2f}<br>PC2=%{y:.2f}<extra></extra>",
        ))

    layout = plotly_layout(height=520)
    layout["xaxis"]["title"] = "PC1"
    layout["yaxis"]["title"] = "PC2"
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">INTERPRÉTATION · PROFIL DES CLUSTERS</div>', unsafe_allow_html=True)
    prof = cluster_profile_table(df_ml, top_n=int(top_n))
    st.dataframe(_df_for_display(prof), use_container_width=True, height=260)

    st.info(
        "Lecture recommandée : cluster **à risque** si **DENY% élevé** + **ports sensibles** "
        "(22/3389/445/23/3306...) + éventuellement une signature temporelle (nuit)."
    )

    render_llm_panel(df_ml=df_ml, prof=prof, k=int(k), sil=sil, var_exp=var_exp)

    st.download_button(
        "⬇️ Télécharger l’échantillon clusterisé (CSV)",
        data=df_ml.drop(columns=["pc1", "pc2"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="ml_clusters_sample.csv",
        mime="text/csv",
        use_container_width=True,
    )


main()  