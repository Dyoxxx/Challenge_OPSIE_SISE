import streamlit as st
import plotly.express as px    
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from config import (
    load_data, apply_filters, render_sidebar, NASA_CSS,
    plotly_layout, PORT_NAMES, JOUR_ORDER, JOUR_FR,
    C_CYAN, C_RED, C_GREEN, C_AMBER, C_TEXT, C_MUTED,
    C_BORDER, C_BORDER2, C_PANEL, C_BG, C_SURFACE,
    TCP_PORTS, UDP_PORTS,
)

def show():
    st.markdown(NASA_CSS, unsafe_allow_html=True)
    st.markdown('<div class="scanlines"></div>', unsafe_allow_html=True)

    df_raw, is_demo = load_data()
    page, mois_sel, action_sel, proto_sel, port_range, rules_sel, top_n = render_sidebar("analyse")

    if "ACCUEIL" in page:
        from pages import accueil
        accueil.show()
        return

    df = apply_filters(df_raw, mois_sel, action_sel, proto_sel, port_range, rules_sel)
    n_total = len(df)

    # ══════════════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════════════
    st.markdown(f"""
    <div style="border-bottom:1px solid {C_BORDER2};padding:20px 4px 18px;margin-bottom:20px;
                position:relative;">
        <div style="position:absolute;top:0;left:0;width:30px;height:30px;
                    border-top:2px solid {C_AMBER};border-left:2px solid {C_AMBER};"></div>
        <div style="position:absolute;top:0;right:0;width:30px;height:30px;
                    border-top:2px solid {C_AMBER};border-right:2px solid {C_AMBER};"></div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;
                    color:{C_MUTED};letter-spacing:4px;margin-bottom:4px;">MODULE ANALYTIQUE</div>
        <div style="font-family:'Exo 2',sans-serif;font-weight:900;font-size:2rem;
                    color:{C_AMBER};letter-spacing:3px;text-shadow:0 0 20px rgba(255,179,0,0.3);">
            SECURITY ANALYTICS
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;
                    color:{C_MUTED};margin-top:4px;letter-spacing:2px;">
            ANALYSE APPROFONDIE · {n_total:,} ÉVÉNEMENTS · FIREWALL IPTABLES CLOUD
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPIs compacts
    n_deny   = len(df[df["action"]=="DENY"])   if "action" in df.columns else 0
    n_permit = len(df[df["action"]=="PERMIT"]) if "action" in df.columns else 0
    n_rules  = df["rule"].nunique()            if "rule"   in df.columns else 0
    n_src    = df["src_ip"].nunique()          if "src_ip" in df.columns else 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("ÉVÉNEMENTS", f"{n_total:,}")
    c2.metric("DENY",  f"{n_deny:,}",   f"{n_deny/n_total*100:.1f}%" if n_total else "0%", delta_color="inverse")
    c3.metric("PERMIT",f"{n_permit:,}", f"{n_permit/n_total*100:.1f}%" if n_total else "0%")
    c4.metric("RÈGLES / IP SRC", f"{n_rules} / {n_src:,}")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════
    # TABS
    # ══════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋  RÈGLES FIREWALL",
        "🔌  PROTOCOLES",
        "🗺   RAPPROCHEMENT TCP",
        "🕵   THREAT DETECTION",
        "📅  ANALYSE TEMPORELLE",
    ])

    # ════════════════════════════════════════
    # TAB 1 — RÈGLES
    # ════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-label">CLASSEMENT DES RÈGLES FIREWALL · PAR VOLUME</div>', unsafe_allow_html=True)

        if "rule" in df.columns:
            rule_vc = df["rule"].value_counts()
            rc = rule_vc.head(top_n).reset_index()
            rc.columns = ["Rule ID","Occurrences"]
            rc["Rule ID"] = rc["Rule ID"].astype(str)
            rc["% Total"] = (rc["Occurrences"] / n_total * 100).round(2)

            col_a, col_b = st.columns([3, 2])

            with col_a:
                # Bar horizontal avec gradient
                fig = go.Figure(go.Bar(
                    x=rc["Occurrences"], y=rc["Rule ID"],
                    orientation="h",
                    marker=dict(
                        color=rc["Occurrences"],
                        colorscale=[[0,"rgba(0,229,255,0.2)"],[0.5,C_CYAN],[1,"#00ffff"]],
                        line=dict(color="rgba(0,229,255,0.3)", width=0.5)
                    ),
                    text=[f"{v:,}  ({p:.1f}%)" for v, p in zip(rc["Occurrences"], rc["% Total"])],
                    textposition="outside",
                    textfont=dict(size=10, color=C_MUTED, family="Share Tech Mono"),
                    hovertemplate="<b>Règle %{y}</b><br>Occurrences: %{x:,}<extra></extra>",
                ))
                layout = plotly_layout(height=420, title=f"TOP {top_n} RÈGLES PLUS UTILISÉES")
                layout["yaxis"]["autorange"] = "reversed"
                layout["xaxis"]["showgrid"]  = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # DENY vs PERMIT empilé par règle
                if "action" in df.columns:
                    ra = df.groupby(["rule","action"]).size().unstack(fill_value=0).reset_index()
                    ra["rule"] = ra["rule"].astype(str)
                    top10_ids = rc["Rule ID"].head(10).tolist()
                    ra_f = ra[ra["rule"].isin(top10_ids)]

                    fig = go.Figure()
                    if "DENY" in ra_f.columns:
                        fig.add_trace(go.Bar(
                            name="DENY", x=ra_f["rule"], y=ra_f["DENY"],
                            marker_color=C_RED, opacity=0.85,
                        ))
                    if "PERMIT" in ra_f.columns:
                        fig.add_trace(go.Bar(
                            name="PERMIT", x=ra_f["rule"], y=ra_f["PERMIT"],
                            marker_color=C_GREEN, opacity=0.85,
                        ))
                    layout = plotly_layout(height=420, title="DENY vs PERMIT · PAR RÈGLE")
                    layout["barmode"] = "stack"
                    fig.update_layout(**layout)
                    st.plotly_chart(fig, use_container_width=True)

            # Tableau enrichi
            st.markdown('<div class="section-label">TABLEAU COMPLET · TOUTES RÈGLES</div>', unsafe_allow_html=True)
            rf = df["rule"].value_counts().reset_index()
            rf.columns = ["Rule ID","Occurrences"]
            rf["% Total"] = (rf["Occurrences"]/n_total*100).round(2).astype(str) + "%"
            if "action" in df.columns:
                d_cnt = df[df["action"]=="DENY"]["rule"].value_counts()
                p_cnt = df[df["action"]=="PERMIT"]["rule"].value_counts()
                rf["DENY"]    = rf["Rule ID"].map(d_cnt).fillna(0).astype(int)
                rf["PERMIT"]  = rf["Rule ID"].map(p_cnt).fillna(0).astype(int)
                rf["% DENY"]  = (rf["DENY"]/rf["Occurrences"]*100).round(1).astype(str) + "%"
                rf["STATUT"]  = rf["Rule ID"].apply(lambda x: "🧹 CLEANUP" if str(x)=="999" else "⚙ ACTIVE")
            st.dataframe(rf.style.background_gradient(
                subset=["Occurrences"], cmap="Blues"
            ), use_container_width=True, height=320)

    # ════════════════════════════════════════
    # TAB 2 — PROTOCOLES
    # ════════════════════════════════════════
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-label">HISTOGRAMME · DISTRIBUTION DES PROTOCOLES</div>', unsafe_allow_html=True)
            pc = df["PROTO"].value_counts()
            proto_colors = {"TCP": C_CYAN, "UDP": C_AMBER, "ICMP": C_GREEN}
            
            fig = go.Figure()
            for proto, count in pc.items():
                color = proto_colors.get(proto, C_MUTED)
                fig.add_trace(go.Bar(
                    name=proto, x=[proto], y=[count],
                    marker=dict(
                        color=color, opacity=0.85,
                        line=dict(color=C_BG, width=2)
                    ),
                    text=f"{count:,}<br>({count/n_total*100:.1f}%)",
                    textposition="outside",
                    textfont=dict(family="Share Tech Mono", size=11, color=color),
                ))
            layout = plotly_layout(height=360, title="PROTOCOLES DÉTECTÉS")
            layout["showlegend"] = False
            layout["xaxis"]["showgrid"] = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-label">TOP 10 RÈGLES · PROTOCOLE UDP</div>', unsafe_allow_html=True)
            df_udp = df[df["PROTO"]=="UDP"]
            if "rule" in df_udp.columns and len(df_udp) > 0:
                t10u = df_udp["rule"].value_counts().head(10).reset_index()
                t10u.columns = ["Rule ID","Occ UDP"]
                t10u["Rule ID"] = t10u["Rule ID"].astype(str)
                t10u["% UDP"]   = (t10u["Occ UDP"]/len(df_udp)*100).round(1)

                fig = go.Figure(go.Bar(
                    x=t10u["Occ UDP"], y=t10u["Rule ID"],
                    orientation="h",
                    marker=dict(
                        color=t10u["Occ UDP"],
                        colorscale=[[0,"rgba(255,179,0,0.2)"],[1,C_AMBER]],
                    ),
                    text=[f"{v:,}  ({p:.1f}%)" for v,p in zip(t10u["Occ UDP"],t10u["% UDP"])],
                    textposition="outside",
                    textfont=dict(size=9, color=C_MUTED, family="Share Tech Mono"),
                ))
                layout = plotly_layout(height=360, title="TOP 10 UDP")
                layout["yaxis"]["autorange"] = "reversed"
                layout["xaxis"]["showgrid"]  = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(t10u, use_container_width=True, height=200)

        # TOP 5 TCP
        st.markdown('<div class="section-label">TOP 5 RÈGLES · PROTOCOLE TCP</div>', unsafe_allow_html=True)
        df_tcp = df[df["PROTO"]=="TCP"]
        if "rule" in df_tcp.columns and len(df_tcp) > 0:
            t5t = df_tcp["rule"].value_counts().head(5).reset_index()
            t5t.columns = ["Rule ID","Occ TCP"]
            t5t["Rule ID"] = t5t["Rule ID"].astype(str)

            cx, cy, cz = st.columns([1.2, 1, 1])

            with cx:
                fig = go.Figure(go.Bar(
                    x=t5t["Rule ID"], y=t5t["Occ TCP"],
                    marker=dict(
                        color=t5t["Occ TCP"],
                        colorscale=[[0,"rgba(0,229,255,0.3)"],[1,C_CYAN]],
                        line=dict(color=C_BG, width=2)
                    ),
                    text=t5t["Occ TCP"].apply(lambda x: f"{x:,}"),
                    textposition="outside",
                    textfont=dict(size=11, color=C_TEXT, family="Share Tech Mono"),
                ))
                layout = plotly_layout(height=300, title="TOP 5 TCP")
                layout["xaxis"]["showgrid"] = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            with cy:
                fig = go.Figure(go.Pie(
                    values=t5t["Occ TCP"], labels=t5t["Rule ID"],
                    hole=0.6,
                    marker=dict(
                        colors=[C_CYAN,"#0099bb","#006688","#003344","#001a22"],
                        line=dict(color=C_BG, width=2)
                    ),
                    textinfo="label+percent",
                    textfont=dict(family="Share Tech Mono", size=10),
                ))
                layout = plotly_layout(height=300, title="RÉPARTITION TCP")
                layout["showlegend"] = False
                layout["margin"] = dict(l=10,r=10,t=40,b=10)
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            with cz:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="terminal-box">', unsafe_allow_html=True)
                for _, row in t5t.iterrows():
                    pct = row["Occ TCP"]/len(df_tcp)*100
                    st.markdown(f"""
                    <div class="terminal-box" style="margin-bottom:4px;padding:8px 12px;">
                        <span style="color:{C_MUTED}">RULE</span>
                        <span style="color:{C_CYAN}"> {row['Rule ID']}</span>
                        <span style="color:{C_MUTED}"> → </span>
                        <span style="color:{C_TEXT}">{row['Occ TCP']:,}</span>
                        <span style="color:{C_MUTED}"> ({pct:.1f}%)</span>
                    </div>
                    """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    # TAB 3 — RAPPROCHEMENT TCP
    # ════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-label">RAPPROCHEMENT RÈGLES × PORTS DE DESTINATION × ACTIONS · TCP UNIQUEMENT</div>', unsafe_allow_html=True)

        df_tcp = df[df["PROTO"]=="TCP"].copy()

        if "dport" in df_tcp.columns and "rule" in df_tcp.columns and len(df_tcp) > 0:
            ca, cb = st.columns(2)
            with ca: n_ports_h = st.slider("Ports dans la heatmap", 5, 20, 10, key="ph")
            with cb: n_rules_h = st.slider("Règles dans la heatmap", 5, 20, 10, key="rh")

            top_p_ids = df_tcp["dport"].value_counts().head(n_ports_h).index.tolist()
            top_r_ids = df_tcp["rule"].value_counts().head(n_rules_h).index.tolist()

            dh = df_tcp[df_tcp["dport"].isin(top_p_ids) & df_tcp["rule"].isin(top_r_ids)]
            hd = (dh.groupby(["rule","dport"]).size()
                    .reset_index(name="count")
                    .pivot(index="rule", columns="dport", values="count")
                    .fillna(0))
            hd.index   = hd.index.astype(str)
            hd.columns = [f"{int(c)}\n{PORT_NAMES.get(int(c),'?')}" for c in hd.columns]

            fig = go.Figure(go.Heatmap(
                z=hd.values,
                x=list(hd.columns),
                y=list(hd.index),
                colorscale=[[0,C_BG],[0.3,"#0d2040"],[0.7,C_CYAN],[1,"#ffffff"]],
                text=[[f"{int(v):,}" if v > 0 else "" for v in row] for row in hd.values],
                texttemplate="%{text}",
                textfont=dict(size=9, family="Share Tech Mono"),
                hovertemplate="Règle %{y}<br>Port %{x}<br>Occurrences: %{z:,}<extra></extra>",
                colorbar=dict(
                    tickfont=dict(color=C_MUTED, size=9, family="Share Tech Mono"),
                    thickness=12, len=0.9,
                    title=dict(text="OCC.", font=dict(color=C_MUTED, size=9))
                )
            ))
            layout = plotly_layout(height=420, title="HEATMAP · RÈGLES × PORTS DESTINATION (TCP)")
            layout["margin"] = dict(l=16,r=100,t=50,b=60)
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

            # Tableau croisé enrichi
            if "action" in df_tcp.columns:
                st.markdown('<div class="section-label">TABLEAU CROISÉ · RULE × PORT × ACTION</div>', unsafe_allow_html=True)
                pivot = (df_tcp.groupby(["rule","dport","action"])
                               .size().unstack(fill_value=0).reset_index())
                pivot["TOTAL"]   = pivot.select_dtypes(include="number").sum(axis=1)
                pivot            = pivot.sort_values("TOTAL", ascending=False)
                pivot["SERVICE"] = pivot["dport"].map(PORT_NAMES).fillna("UNKNOWN")
                if "DENY" in pivot.columns and "PERMIT" in pivot.columns:
                    pivot["% DENY"] = (pivot["DENY"]/pivot["TOTAL"]*100).round(1).astype(str) + "%"
                st.dataframe(pivot.head(40), use_container_width=True, height=360)

    # ════════════════════════════════════════
    # TAB 4 — THREAT DETECTION
    # ════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-label">DÉTECTION DE MENACES · COMPORTEMENTS SUSPECTS</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-label">SCAN DE PORTS · DÉTECTION NMAP</div>', unsafe_allow_html=True)
            st.caption("IP contactant le plus de ports distincts = scanner potentiel")
            if "src_ip" in df.columns and "dport" in df.columns:
                pd_div = (df.groupby("src_ip")["dport"]
                            .nunique().sort_values(ascending=False)
                            .head(top_n).reset_index())
                pd_div.columns = ["IP Source","Ports Distincts"]
                pd_div["NIVEAU"] = pd_div["Ports Distincts"].apply(
                    lambda x: "🔴 CRITIQUE" if x>1000 else "🟠 ÉLEVÉ" if x>200 else "🟡 MODÉRÉ" if x>50 else "🟢 NORMAL"
                )
                pd_div["DENY_COUNT"] = pd_div["IP Source"].map(
                    df[df["action"]=="DENY"]["src_ip"].value_counts()
                ).fillna(0).astype(int) if "action" in df.columns else 0

                fig = go.Figure(go.Bar(
                    x=pd_div["Ports Distincts"], y=pd_div["IP Source"],
                    orientation="h",
                    marker=dict(
                        color=pd_div["Ports Distincts"],
                        colorscale=[[0,"rgba(255,23,68,0.2)"],[0.5,"rgba(255,100,68,0.6)"],[1,C_RED]],
                        line=dict(color="rgba(255,23,68,0.3)", width=0.5)
                    ),
                    text=pd_div["Ports Distincts"].apply(lambda x: f"{x:,} ports"),
                    textposition="outside",
                    textfont=dict(size=9, color=C_MUTED, family="Share Tech Mono"),
                ))
                layout = plotly_layout(height=360, title="IP · DIVERSITÉ DE PORTS CONTACTÉS")
                layout["yaxis"]["autorange"] = "reversed"
                layout["xaxis"]["showgrid"]  = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(pd_div, use_container_width=True, height=200)

        with col_b:
            st.markdown('<div class="section-label">RATIO DENY% · RÈGLES LES PLUS BLOQUANTES</div>', unsafe_allow_html=True)
            if "rule" in df.columns and "action" in df.columns:
                ra = df.groupby(["rule","action"]).size().unstack(fill_value=0)
                if "DENY"   not in ra.columns: ra["DENY"]   = 0
                if "PERMIT" not in ra.columns: ra["PERMIT"] = 0
                ra["TOTAL"]      = ra["DENY"] + ra["PERMIT"]
                ra["RATIO_DENY"] = (ra["DENY"]/ra["TOTAL"]*100).round(1)
                ra = ra.sort_values("RATIO_DENY", ascending=False).head(top_n).reset_index()
                ra["rule"] = ra["rule"].astype(str)

                # Couleur progressive selon le ratio
                bar_colors = [
                    C_RED if r > 80 else C_AMBER if r > 50 else C_GREEN
                    for r in ra["RATIO_DENY"]
                ]

                fig = go.Figure(go.Bar(
                    x=ra["rule"], y=ra["RATIO_DENY"],
                    marker=dict(color=bar_colors, line=dict(color=C_BG, width=2)),
                    text=[f"{v:.0f}%" for v in ra["RATIO_DENY"]],
                    textposition="outside",
                    textfont=dict(size=11, color=C_TEXT, family="Share Tech Mono"),
                ))
                # Ligne de seuil 80%
                fig.add_hline(y=80, line=dict(color=C_RED, dash="dot", width=1),
                              annotation_text="SEUIL CRITIQUE 80%",
                              annotation_font=dict(color=C_RED, size=9, family="Share Tech Mono"))
                fig.add_hline(y=50, line=dict(color=C_AMBER, dash="dot", width=1),
                              annotation_text="SEUIL ALERTE 50%",
                              annotation_font=dict(color=C_AMBER, size=9, family="Share Tech Mono"))
                layout = plotly_layout(height=360, title="RATIO DENY% PAR RÈGLE")
                layout["yaxis"]["title"] = dict(text="% DENY", font=dict(size=9, color=C_MUTED))
                layout["yaxis"]["range"] = [0, 115]
                layout["xaxis"]["showgrid"] = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        # Treemap + Radar
        col_x, col_y = st.columns(2)

        with col_x:
            st.markdown('<div class="section-label">TREEMAP · IP LES PLUS BLOQUÉES</div>', unsafe_allow_html=True)
            if "src_ip" in df.columns and "action" in df.columns:
                dip = (df[df["action"]=="DENY"]["src_ip"]
                         .value_counts().head(20).reset_index())
                dip.columns = ["IP","Blocages"]
                fig = px.treemap(
                    dip, path=["IP"], values="Blocages",
                    color="Blocages",
                    color_continuous_scale=[[0,"rgba(255,23,68,0.1)"],[1,C_RED]],
                )
                fig.update_traces(
                    textfont=dict(family="Share Tech Mono", size=11),
                    texttemplate="<b>%{label}</b><br>%{value:,}",
                    marker=dict(line=dict(color=C_BG, width=2))
                )
                layout = plotly_layout(height=320)
                layout["margin"] = dict(l=0,r=0,t=0,b=0)
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        with col_y:
            st.markdown('<div class="section-label">PROFIL D\'ATTAQUE · PAR PROTOCOLE × ACTION</div>', unsafe_allow_html=True)
            if "PROTO" in df.columns and "action" in df.columns:
                pa = df.groupby(["PROTO","action"]).size().unstack(fill_value=0)
                protos = list(pa.index)
                
                fig = go.Figure()
                if "DENY" in pa.columns:
                    fig.add_trace(go.Scatterpolar(
                        r=pa["DENY"].values, theta=protos,
                        fill="toself", name="DENY",
                        line=dict(color=C_RED, width=2),
                        fillcolor="rgba(255,23,68,0.15)"
                    ))
                if "PERMIT" in pa.columns:
                    fig.add_trace(go.Scatterpolar(
                        r=pa["PERMIT"].values, theta=protos,
                        fill="toself", name="PERMIT",
                        line=dict(color=C_GREEN, width=2),
                        fillcolor="rgba(0,230,118,0.1)"
                    ))
                layout = plotly_layout(height=320, title="RADAR · PROTOCOLE × ACTION")
                layout["polar"] = dict(
                    bgcolor="rgba(4,13,24,0.8)",
                    radialaxis=dict(
                        gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=8),
                        linecolor=C_BORDER
                    ),
                    angularaxis=dict(
                        gridcolor=C_BORDER,
                        tickfont=dict(color=C_TEXT, size=11, family="Share Tech Mono")
                    )
                )
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════
    # TAB 5 — TEMPOREL
    # ════════════════════════════════════════
    with tab5:
        # Volume mensuel
        if "mois" in df.columns and "action" in df.columns:
            st.markdown('<div class="section-label">ÉVOLUTION MENSUELLE · DENY vs PERMIT</div>', unsafe_allow_html=True)
            monthly = df.groupby(["mois","action"]).size().reset_index(name="count")

            fig = go.Figure()
            for action, color in [("DENY", C_RED), ("PERMIT", C_GREEN)]:
                m_data = monthly[monthly["action"]==action]
                fig.add_trace(go.Bar(
                    name=action, x=m_data["mois"], y=m_data["count"],
                    marker=dict(color=color, opacity=0.85, line=dict(color=C_BG, width=1)),
                    text=m_data["count"].apply(lambda x: f"{x:,}"),
                    textposition="outside",
                    textfont=dict(size=9, color=C_MUTED, family="Share Tech Mono"),
                ))
            layout = plotly_layout(height=300, title="VOLUME MENSUEL · FLUX FIREWALL")
            layout["barmode"] = "group"
            layout["xaxis"]["showgrid"] = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            if "heure" in df.columns and "action" in df.columns:
                st.markdown('<div class="section-label">PROFIL HORAIRE MOYEN · BASELINE</div>', unsafe_allow_html=True)
                hourly = df.groupby(["heure","action"]).size().reset_index(name="count")

                fig = go.Figure()
                for action, color in [("DENY", C_RED), ("PERMIT", C_GREEN)]:
                    h_data = hourly[hourly["action"]==action].sort_values("heure")
                    fig.add_trace(go.Scatter(
                        x=h_data["heure"], y=h_data["count"],
                        mode="lines+markers", name=action,
                        line=dict(color=color, width=2, shape="spline"),
                        marker=dict(size=5, color=color),
                        fill="tozeroy",
                        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.06)"
                    ))
                layout = plotly_layout(height=300, title="CONNEXIONS PAR HEURE")
                layout["xaxis"]["tickmode"] = "linear"
                layout["xaxis"]["dtick"]    = 2
                layout["xaxis"]["ticksuffix"] = "h"
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "mois" in df.columns:
                st.markdown('<div class="section-label">ÉVOLUTION PAR PROTOCOLE · MENSUELLE</div>', unsafe_allow_html=True)
                mp = df.groupby(["mois","PROTO"]).size().reset_index(name="count")
                proto_colors = {"TCP": C_CYAN, "UDP": C_AMBER, "ICMP": C_GREEN}

                fig = go.Figure()
                for proto in df["PROTO"].unique():
                    p_data = mp[mp["PROTO"]==proto]
                    color  = proto_colors.get(proto, C_MUTED)
                    fig.add_trace(go.Scatter(
                        x=p_data["mois"], y=p_data["count"],
                        mode="lines+markers", name=proto,
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color,
                                    line=dict(color=C_BG, width=2)),
                    ))
                layout = plotly_layout(height=300, title="TRAFIC MENSUEL PAR PROTOCOLE")
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

        # Double heatmap DENY vs PERMIT
        st.markdown('<div class="section-label">HEATMAP COMPARÉE · DENY vs PERMIT · HEURE × JOUR</div>', unsafe_allow_html=True)
        if "jour_semaine" in df.columns and "heure" in df.columns and "action" in df.columns:
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=["DENY", "PERMIT"],
                horizontal_spacing=0.08
            )
            for idx, (action, colorscale) in enumerate([
                ("DENY",   [[0,C_BG],[0.3,"#300010"],[0.7,"#880020"],[1,C_RED]]),
                ("PERMIT", [[0,C_BG],[0.3,"#001a10"],[0.7,"#006040"],[1,C_GREEN]])
            ]):
                df_a = df[df["action"]==action]
                ht = df_a.groupby(["jour_semaine","heure"]).size().reset_index(name="count")
                htp = ht.pivot(index="jour_semaine", columns="heure", values="count").fillna(0)
                htp = htp.reindex([j for j in JOUR_ORDER if j in htp.index])
                htp.index = [JOUR_FR.get(j,j) for j in htp.index]

                fig.add_trace(go.Heatmap(
                    z=htp.values,
                    x=[f"{h:02d}h" for h in htp.columns],
                    y=list(htp.index),
                    colorscale=colorscale,
                    showscale=False,
                    hovertemplate=f"{action}<br>Jour: %{{y}}<br>Heure: %{{x}}<br>Count: %{{z:,}}<extra></extra>"
                ), row=1, col=idx+1)

            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(4,13,24,0.6)",
                font=dict(family="Share Tech Mono", color=C_TEXT, size=10),
                height=280,
                margin=dict(l=16,r=16,t=50,b=16),
            )
            for ann in fig.layout.annotations:
                ann.font = dict(color=C_MUTED, size=11, family="Share Tech Mono")
            fig.update_xaxes(gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=8))
            fig.update_yaxes(gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=9))
            st.plotly_chart(fig, use_container_width=True)

    # FOOTER
    st.markdown(f"""
    <div style="margin-top:40px;padding:14px 4px;border-top:1px solid {C_BORDER};
                display:flex;justify-content:space-between;align-items:center;">
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:{C_MUTED};">
            PROJET SISE-OPSIE 2026 · R.RAKOTOMALALA & D.PIERROT · UNIVERSITÉ LYON 2
        </div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:{C_MUTED};">
            {'DEMO MODE' if is_demo else 'IPTABLES CLOUD'} · {n_total:,} EVENTS PROCESSED
        </div>
    </div>
    """, unsafe_allow_html=True)   


show()    