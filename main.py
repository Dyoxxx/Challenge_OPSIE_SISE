import streamlit as st

from config import NASA_CSS, C_CYAN, C_MUTED, C_BORDER2, load_data

st.set_page_config(page_title="OPSIE · Accueil", layout="wide")


def _header():
    st.markdown(NASA_CSS, unsafe_allow_html=True)
    st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="border-bottom:1px solid {C_BORDER2};padding:18px 4px 16px;margin-bottom:18px;">
            <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;color:{C_MUTED};letter-spacing:3px;">
                SISE-OPSIE · NASA UI · FIREWALL INTELLIGENCE
            </div>
            <div style="font-family:'Exo 2',sans-serif;font-weight:900;font-size:2rem;color:{C_CYAN};letter-spacing:3px;">
                ACCUEIL
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    _header()

    df, is_demo = load_data()
    total = int(len(df))
    deny = int((df["action"] == "DENY").sum()) if "action" in df.columns else 0
    permit = int((df["action"] == "PERMIT").sum()) if "action" in df.columns else 0

    st.markdown(
        """
        ### Plateforme d'analyse des logs firewall
        Cette application propose deux volets :
        - **Analyses opérationnelles** (tendances, heatmaps, cartes, top IP/ports)
        - **ML & interprétation** (ACP, clustering et aide à l'interprétation)
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{total:,}")
    c2.metric("DENY", f"{deny:,}")
    c3.metric("PERMIT", f"{permit:,}")

    if is_demo:
        st.warning("Mode démo actif : aucun fichier de logs valide détecté dans data/logs_export.csv")

    st.info("Utilise le menu latéral Streamlit pour ouvrir Analyses et ML & interprétation.")


main()
