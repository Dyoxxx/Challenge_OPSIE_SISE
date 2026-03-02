
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)
n = 50_000

# IPs sources réalistes (attaquants externes + réseau interne)
attackers = [f"77.90.{np.random.randint(0,255)}.{np.random.randint(1,255)}" for _ in range(8)]
attackers += [f"94.102.{np.random.randint(0,60)}.{np.random.randint(1,255)}" for _ in range(5)]
attackers += ["89.89.56.2","28.12.15.20","172.5.2.8","47.128.20.252","79.124.60.150"]
internal  = [f"192.168.{np.random.randint(0,5)}.{np.random.randint(1,100)}" for _ in range(10)]
internal  += [f"10.0.{np.random.randint(0,3)}.{np.random.randint(1,50)}" for _ in range(5)]

all_ips = attackers + internal

ports   = [80, 443, 22, 53, 3306, 8080, 21, 23, 3389, 123, 161, 445, 25, 110, 514]
port_p  = [.18,.16,.12,.10,.08,  .07,  .06,.05,.04,  .04, .03, .03, .02,.01, .01]
rules   = [431, 999, 153, 283, 512, 77, 202]
rule_p  = [.24, .30, .15, .12, .08, .06, .05]

dports  = np.random.choice(ports, n, p=port_p)
TCP_SET = {20,21,22,23,25,80,110,143,443,445,1433,3306,3389,5432,8080,8443}
protos  = ["TCP" if p in TCP_SET else "UDP" for p in dports]

# Simulation : plus d'attaques la nuit et en décembre
dates = pd.date_range("2025-11-01", "2026-02-28", periods=n)
hours = pd.Series(dates).dt.hour.to_numpy()   
# Biais : 60% d'attaques entre 0h-6h et 22h-23h
night_mask = (hours < 6) | (hours >= 22)
actions = np.where(
    night_mask,
    np.random.choice(["DENY","PERMIT"], n, p=[0.80, 0.20]),
    np.random.choice(["DENY","PERMIT"], n, p=[0.50, 0.50])
)

df = pd.DataFrame({
    "timestamp":     dates,
    "src_ip":        np.random.choice(all_ips, n),
    "dst_ip":        "159.84.146.99",
    "protocole":     protos,
    "dport":         dports,
    "action":        actions,
    "rule":          np.random.choice(rules, n, p=rule_p),
    "interface_in":  "eth0",
    "interface_out": "eth1",
})

df.to_parquet("data/logs_export.parquet", index=False)
print(f"✅ {len(df):,} lignes générées → data/logs_export.parquet")
print(f"\nAperçu :")
print(df.head(10).to_string())
print(f"\nDistribution actions : {dict(df['action'].value_counts())}")
print(f"Règles : {dict(df['rule'].value_counts())}")