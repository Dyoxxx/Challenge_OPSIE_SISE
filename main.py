import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Firewall Logs Dashboard", layout="wide")

st.title("Challenge OPSIE SISE - Analyse de données de pare-feu")

# Upload CSV
uploaded_file = st.file_uploader("Upload firewall CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    st.subheader("Raw Data")
    st.dataframe(df.head(200))

    col1, col2 = st.columns(2)

    # Actions distribution
    with col1:
        st.subheader("Actions Distribution")
        action_counts = df["action"].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(action_counts, labels=action_counts.index, autopct="%1.1f%%")
        ax1.set_ylabel("")
        st.pyplot(fig1)

    # Top destination ports
    with col2:
        st.subheader("Top Destination Ports")
        port_counts = df["dst_port"].value_counts().head(10)
        fig2, ax2 = plt.subplots()
        ax2.bar(port_counts.index.astype(str), port_counts.values)
        ax2.set_xlabel("Port")
        ax2.set_ylabel("Count")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    # Events over time
    with col3:
        st.subheader("Events per Day")
        daily_counts = df.groupby("date").size()
        fig3, ax3 = plt.subplots()
        daily_counts.plot(ax=ax3)
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Events")
        st.pyplot(fig3)

    # Top source IPs
    with col4:
        st.subheader("Top Source IPs")
        src_counts = df["src_ip"].value_counts().head(10)
        fig4, ax4 = plt.subplots()
        ax4.barh(src_counts.index, src_counts.values)
        ax4.set_xlabel("Count")
        st.pyplot(fig4)

    # Protocol distribution
    st.subheader("Protocol Distribution")
    proto_counts = df["protocole"].value_counts()
    fig5, ax5 = plt.subplots()
    ax5.bar(proto_counts.index, proto_counts.values)
    ax5.set_xlabel("Protocol")
    ax5.set_ylabel("Count")
    st.pyplot(fig5)

else:
    st.info("Upload a CSV file to start analysis.")
