import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from config import (
    C_CYAN_DIM, load_data, apply_filters, render_sidebar, NASA_CSS,
    plotly_layout, PORT_NAMES, JOUR_ORDER, JOUR_FR,
    C_CYAN, C_RED, C_GREEN, C_AMBER, C_TEXT, C_MUTED,
    C_BORDER, C_BORDER2, C_PANEL, C_BG, C_SURFACE,
)

def show():
    st.markdown(NASA_CSS, unsafe_allow_html=True)
    st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)

    df_raw, is_demo = load_data()
    page, mois_sel, action_sel, proto_sel, port_range, rules_sel, top_n = render_sidebar("accueil")

    if "ANALYSES" in page:
        from pages import analyse
        analyse.show()
        return

    df = apply_filters(df_raw, mois_sel, action_sel, proto_sel, port_range, rules_sel)

    n_total  = len(df)
    n_deny   = len(df[df["action"]=="DENY"])   if "action" in df.columns else 0
    n_permit = len(df[df["action"]=="PERMIT"]) if "action" in df.columns else 0
    n_rules  = df["rule"].nunique()            if "rule"   in df.columns else 0
    n_ips    = df["src_ip"].nunique()          if "src_ip" in df.columns else 0
    pct_deny = n_deny/n_total*100 if n_total > 0 else 0

    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ══════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════
   

    # ══════════════════════════════════════════════════
    # ALERTE si fort taux DENY
    # ══════════════════════════════════════════════════
    if pct_deny > 60:
        st.markdown(f"""
        <div class="alert-bar">
            ⚠ ALERT · TAUX DE BLOCAGE ÉLEVÉ : {pct_deny:.1f}% DES CONNEXIONS REJETÉES
            · {n_deny:,} ÉVÉNEMENTS DENY DÉTECTÉS · ANALYSE EN COURS
            <span class="cursor">█</span>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # KPIs — Ligne 1
    # ══════════════════════════════════════════════════
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("TOTAL EVENTS",    f"{n_total:,}")
    c2.metric("⛔ DENY",          f"{n_deny:,}",   f"{pct_deny:.1f}%", delta_color="inverse")
    c3.metric("✅ PERMIT",        f"{n_permit:,}", f"{n_permit/n_total*100:.1f}%" if n_total else "0%")
    c4.metric("RÈGLES ACTIVES",  f"{n_rules}")
    c5.metric("IPS UNIQUES",     f"{n_ips:,}")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # LIGNE 2 : Timeline principale + Donut
    # ══════════════════════════════════════════════════
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown('<div class="section-label">TIMELINE OPÉRATIONNELLE · FLUX JOURNALIERS</div>', unsafe_allow_html=True)
        if "date_jour" in df.columns and "action" in df.columns:
            daily = df.groupby(["date_jour","action"]).size().reset_index(name="count")

            fig = go.Figure()
            deny_d   = daily[daily["action"]=="DENY"]
            permit_d = daily[daily["action"]=="PERMIT"]

            # Zone remplie DENY
            fig.add_trace(go.Scatter(
                x=deny_d["date_jour"], y=deny_d["count"],
                mode="lines", name="DENY",
                line=dict(color=C_RED, width=1.5),
                fill="tozeroy", fillcolor="rgba(255,23,68,0.08)",
            ))
            # Zone remplie PERMIT
            fig.add_trace(go.Scatter(
                x=permit_d["date_jour"], y=permit_d["count"],
                mode="lines", name="PERMIT",
                line=dict(color=C_GREEN, width=1.5),
                fill="tozeroy", fillcolor="rgba(0,230,118,0.06)",
            ))

            layout = plotly_layout(height=280, title="")
            layout["xaxis"]["showgrid"] = False
            layout["yaxis"]["title"] = dict(text="CONNEXIONS/JOUR", font=dict(size=9, color=C_MUTED))
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    with col_side:
        st.markdown('<div class="section-label">STATUT GLOBAL</div>', unsafe_allow_html=True)
        if "action" in df.columns:
            ac = df["action"].value_counts()
            fig = go.Figure(go.Pie(
                values=ac.values, labels=ac.index,
                hole=0.7,
                marker=dict(
                    colors=[C_RED if l=="DENY" else C_GREEN for l in ac.index],
                    line=dict(color=C_BG, width=3)
                ),
                textinfo="none",
                hovertemplate="%{label}: %{value:,}<extra></extra>"
            ))
            fig.add_annotation(
                text=f"<b>{pct_deny:.0f}%</b><br><span style='font-size:10px'>DENY</span>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color=C_RED, size=22, family="Share Tech Mono"),
                align="center"
            )
            layout = plotly_layout(height=280)
            layout["margin"] = dict(l=10,r=10,t=10,b=10)
            layout["showlegend"] = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════
    # LIGNE 3 : Top IP + Top Ports + Protocoles
    # ══════════════════════════════════════════════════
    c1, c2, c3 = st.columns([1.2, 1.2, 0.8])

    with c1:
        st.markdown('<div class="section-label">TOP IP SOURCES · BLOQUÉES</div>', unsafe_allow_html=True)
        if "src_ip" in df.columns and "action" in df.columns:
            top_ip = (df[df["action"]=="DENY"]["src_ip"]
                      .value_counts().head(top_n).reset_index())
            top_ip.columns = ["IP","Blocages"]

            fig = go.Figure(go.Bar(
                x=top_ip["Blocages"], y=top_ip["IP"],
                orientation="h",
                marker=dict(
                    color=top_ip["Blocages"],
                    colorscale=[[0, "rgba(255,23,68,0.3)"], [1, C_RED]],
                    line=dict(color="rgba(255,23,68,0.5)", width=0.5)
                ),
                text=top_ip["Blocages"].apply(lambda x: f"{x:,}"),
                textposition="outside",
                textfont=dict(size=9, color=C_MUTED, family="Share Tech Mono"),
                hovertemplate="<b>%{y}</b><br>Blocages: %{x:,}<extra></extra>"
            ))
            layout = plotly_layout(height=320)
            layout["yaxis"]["autorange"] = "reversed"
            layout["xaxis"]["showgrid"]  = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-label">TOP PORTS CIBLÉS · DESTINATION</div>', unsafe_allow_html=True)
        if "dport" in df.columns:
            top_p = df["dport"].value_counts().head(top_n).reset_index()
            top_p.columns = ["Port","Occ"]
            top_p["Label"] = (top_p["Port"].astype(int).astype(str) + " · " +
                              top_p["Port"].map(PORT_NAMES).fillna("UNKNOWN"))

            fig = go.Figure(go.Bar(
                x=top_p["Occ"], y=top_p["Label"],
                orientation="h",
                marker=dict(
                    color=top_p["Occ"],
                    colorscale=[[0,"rgba(0,229,255,0.2)"],[1,C_CYAN]],
                    line=dict(color="rgba(0,229,255,0.4)", width=0.5)
                ),
                text=top_p["Occ"].apply(lambda x: f"{x:,}"),
                textposition="outside",
                textfont=dict(size=9, color=C_MUTED, family="Share Tech Mono"),
                hovertemplate="<b>%{y}</b><br>Occurrences: %{x:,}<extra></extra>"
            ))
            layout = plotly_layout(height=320)
            layout["yaxis"]["autorange"] = "reversed"
            layout["xaxis"]["showgrid"]  = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    with c3:
        st.markdown('<div class="section-label">PROTOCOLES</div>', unsafe_allow_html=True)
        pc = df["PROTO"].value_counts()
        proto_colors = {"TCP": C_CYAN, "UDP": C_AMBER, "ICMP": C_GREEN}
        fig = go.Figure(go.Bar(
            x=pc.index, y=pc.values,
            marker=dict(
                color=[proto_colors.get(p, C_MUTED) for p in pc.index],
                line=dict(color=C_BG, width=2)
            ),
            text=pc.values,
            textposition="outside",
            textfont=dict(size=10, color=C_TEXT, family="Share Tech Mono"),
        ))
        layout = plotly_layout(height=320)
        layout["xaxis"]["showgrid"] = False
        layout["yaxis"]["showgrid"] = True
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════
    # LIGNE 4 : Heatmap + Terminal
    # ══════════════════════════════════════════════════
    col_heat, col_term = st.columns([2.5, 1])

    with col_heat:
        st.markdown('<div class="section-label">HEATMAP · ACTIVITÉ TEMPORELLE · HEURE × JOUR</div>', unsafe_allow_html=True)
        if "heure" in df.columns and "jour_semaine" in df.columns:
            ht = df.groupby(["jour_semaine","heure"]).size().reset_index(name="count")
            htp = ht.pivot(index="jour_semaine", columns="heure", values="count").fillna(0)
            htp = htp.reindex([j for j in JOUR_ORDER if j in htp.index])
            htp.index = [JOUR_FR.get(j, j) for j in htp.index]

            fig = go.Figure(go.Heatmap(
                z=htp.values,
                x=[f"{h:02d}h" for h in htp.columns],
                y=htp.index,  
                colorscale=[[0,"#020408"],[0.3,"#0d2040"],[0.6,C_CYAN_DIM],[1,C_CYAN]],
                showscale=True,
                hovertemplate="Jour: %{y}<br>Heure: %{x}<br>Connexions: %{z:,}<extra></extra>",
                colorbar=dict(
                    tickfont=dict(color=C_MUTED, size=9, family="Share Tech Mono"),
                    thickness=10, len=0.8,
                )
            ))
            layout = plotly_layout(height=240)
            layout["margin"] = dict(l=16,r=60,t=20,b=20)
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    with col_term:
        st.markdown('<div class="section-label">LOG TERMINAL</div>', unsafe_allow_html=True)
        if len(df) > 0:
            sample = df.tail(8)[["timestamp","src_ip","action","dport"]].copy()
            lines = []
            for _, row in sample.iterrows():
                ts  = str(row.get("timestamp",""))[:19]
                ip  = str(row.get("src_ip",""))
                act = str(row.get("action",""))
                prt = str(int(row.get("dport",0))) if row.get("dport") is not None else "?"
                color = C_RED if act=="DENY" else C_GREEN
                lines.append(f'<span style="color:{C_MUTED}">{ts}</span> '
                              f'<span style="color:{color}">[{act}]</span> '
                              f'<span style="color:{C_TEXT}">{ip}</span>:{prt}')
            st.markdown(f"""
            <div class="terminal-box" style="height:200px;overflow-y:auto;font-size:0.68rem;">
                > FIREWALL LOG STREAM<br>
                {'<br>'.join(lines)}<br>
                <span class="cursor">█</span>
            </div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="margin-top:40px;padding:16px 4px;border-top:1px solid {C_BORDER};
                display:flex;justify-content:space-between;align-items:center;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:{C_MUTED};">
            PROJET SISE-OPSIE 2026 · R.RAKOTOMALALA & D.PIERROT · UNIVERSITÉ LYON 2
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:{C_MUTED};">
            DONNÉES : {'DEMO MODE' if is_demo else 'IPTABLES CLOUD'} · {n_total:,} ÉVÉNEMENTS ANALYSÉS
        </div>
    </div>
    """, unsafe_allow_html=True)    


show()       