import joblib
import pandas as pd
import numpy as np
from app import prepare_input_data # app.py içindeki güncel fonksiyonu çağırıyoruz

def run_diagnostics():
    print("🔍 Valolyzer Tanı ve Simetri Testi Başlatılıyor...\n")
    
    # 1. Modeli ve TÜM Yeni Meta Verileri Yükle
    try:
        model = joblib.load('valorant_lr_model.pkl')
        columns = joblib.load('model_columns.pkl')
        
        # Copilot'un yeni ürettiği pkl dosyalarını da yüklüyoruz
        agent_roles = joblib.load('agent_roles.pkl')
        agent_synergies = joblib.load('agent_synergies.pkl')
        agent_map_winrates = joblib.load('agent_map_winrates.pkl')
        
        print(f"✅ Model ve {len(columns)} sütun başarıyla yüklendi.")
        print("✅ Rol, Sinerji ve Harita Ağırlık sözlükleri başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Dosyalar yüklenemedi: {e}")
        print("Lütfen tüm .pkl dosyalarının ana dizinde olduğundan emin olun.")
        return

    # 2. Test Senaryosu: Birebir Aynı İki Takım
    test_map = "ascent" 
    team1 = ["Jett", "Omen", "Sova", "Killjoy", "KAY/O"]
    team2 = ["Jett", "Omen", "Sova", "Killjoy", "KAY/O"]

    # 3. app.py'nin Modele Gönderdiği Veriyi Üret (YENİ PARAMETRELER EKLENDİ)
    print("\n⚙️ Birebir Aynı Ajanlar İçin Input Data Üretiliyor...")
    input_df = prepare_input_data(
        test_map, 
        team1, 
        team2, 
        columns,
        agent_roles,         # Yeni eklendi
        agent_synergies,     # Yeni eklendi
        agent_map_winrates   # Yeni eklendi
    )
    
    # 4. SIFIR OLMAYAN SÜTUNLARI TESPİT ET (En Kritik Aşama)
    non_zero_cols = input_df.loc[:, (input_df != 0).any(axis=0)]
    
    print("\n⚠️ SIFIR OLMAYAN SÜTUNLAR (İdeal durumda HARİTA haricinde her şey 0 olmalı):")
    if non_zero_cols.empty:
        print("  -> Hiç sıfır olmayan sütun yok! Tamamen 0.")
    else:
        for col in non_zero_cols.columns:
            print(f"  -> {col}: {input_df[col].values[0]}")

    # 5. Tahmin Simetrisini Kontrol Et
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    
    print("\n🎯 TAHMİN SONUCU:")
    print(f"  Team 1 Kazanma Olasılığı: %{probabilities[1]*100:.4f}")
    print(f"  Team 2 Kazanma Olasılığı: %{probabilities[0]*100:.4f}")
    
    if np.isclose(probabilities[0], 0.5, atol=0.001):
         print("\n✅ BAŞARILI: Model aynı takımlar için tam %50 verdi!")
    else:
         print("\n❌ BAŞARISIZ: Model aynı takımlar için %50 VERMEDİ! Yanlışlık var.")
         
    # 6. Harita Ağırlıklarının Kontrolü (Map Bias Check)
    map_col_name = [c for c in columns if test_map in c.lower()]
    if map_col_name:
        map_idx = columns.index(map_col_name[0])
        map_weight = model.coef_[0][map_idx]
        print(f"\n🗺️ Harita Etkisi Kontrolü ({map_col_name[0]}): Ağırlık Katsayısı = {map_weight:.6f}")
        if abs(map_weight) > 0.001:
            print("   ❗ DİKKAT: Harita sütunu 0 ağırlığa sahip değil. %50 eşitliğini bozan şey harita seçimi olabilir!")

if __name__ == "__main__":
    run_diagnostics()