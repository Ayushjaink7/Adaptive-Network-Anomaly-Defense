from nad.pipeline import load_artifacts, score_pcap

print("=== RUN STARTED ===")

artifacts = load_artifacts("model.pkl")

df = score_pcap("scripts/capture.pcap", artifacts, enforce=True)

print("\n========== FULL DECISION LOG ==========")
print(df[["src_ip", "dst_ip", "risk_score", "ml_anomaly", "action"]].head(30))

print("\n========== ACTION COUNTS ==========")
print(df["action"].value_counts())

print("\n========== SAMPLE HIGH RISK ==========")
print(df.sort_values("risk_score", ascending=False).head(10))
