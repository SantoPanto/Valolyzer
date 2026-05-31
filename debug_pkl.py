import joblib

# Pkl dosyasını yükle
try:
    map_stats = joblib.load('map_agent_stats.pkl')
    
    print("--- HARİTA İSİMLERİ (KEYS) ---")
    print(list(map_stats.keys()))
    
    print("\n--- İLK HARİTANIN İÇERİĞİ (ÖRNEK VERİ) ---")
    ilk_harita = list(map_stats.keys())[0]
    print(f"Harita: {ilk_harita}")
    print(map_stats[ilk_harita])
    
except Exception as e:
    print("Dosya okunamadı:", e)