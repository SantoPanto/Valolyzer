import streamlit as st
import joblib
import pandas as pd

# --- Veri Sabitleri ---
VALORANT_AGENTS = sorted([
    "Jett", "Raze", "Breach", "Omen", "Brimstone", "Viper", "Killjoy", 
    "Cypher", "Sova", "Sage", "Phoenix", "Reyna", "Neon", "Fade", 
    "Astra", "KAY/O", "Chamber", "Skye", "Yoru", "Harbor", "Gekko", 
    "Deadlock", "Iso", "Clove", "Vyse"
])

# --- Modüler Fonksiyonlar ---

@st.cache_resource
def load_model_and_columns():
    """Modeli ve sütun yapılarını diske kaydedilmiş dosyalardan yükler."""
    loaded_model = joblib.load('valorant_rf_model.pkl')
    loaded_columns = joblib.load('model_columns.pkl')
    return loaded_model, loaded_columns

def setup_page_config():
    """Web arayüzünün temel yapılandırmasını ve başlıklarını ayarlar."""
    st.set_page_config(page_title="Valorant Maç Tahmincisi", page_icon="🎮", layout="wide")
    st.title("🎮 VCT 2025: Yapay Zeka Maç Tahmincisi")
    st.markdown("Harita sırasını ve takımların ajan kompozisyonlarını seçin, modelimiz maçın galibini hesaplasın!")

def get_user_inputs():
    """Kullanıcıdan harita ve ajan kompozisyonu seçimlerini alır."""
    st.subheader("📊 Maç Parametreleri")
    
    # Harita sırası
    map_order = st.radio("Bu harita, Best of 3 (Bo3) serisinin kaçıncı maçı?", [1, 2, 3], horizontal=True)
    st.markdown("---")
    
    # Ajan seçimleri
    col1, col2 = st.columns(2)
    with col1:
        st.header("🛡️ Team 1")
        team1_agents = st.multiselect("Team 1 Ajanlarını Seçin (Tam 5 Adet)", VALORANT_AGENTS, max_selections=5, key="t1")
    
    with col2:
        st.header("⚔️ Team 2")
        team2_agents = st.multiselect("Team 2 Ajanlarını Seçin (Tam 5 Adet)", VALORANT_AGENTS, max_selections=5, key="t2")
        
    return map_order, team1_agents, team2_agents

def predict_match(team1_agents, team2_agents, map_order, model, model_columns):
    """Gelen verilere göre tahminde bulunur ve kazanma yüzdelerini döndürür."""
    # 1. İçi 0 dolu bir şablon oluştur
    input_data = {col: 0 for col in model_columns}
    
    # 2. Seçimleri 1 olarak güncelle
    if 'map_order' in input_data:
        input_data['map_order'] = map_order
        
    for agent in team1_agents:
        if f"{agent}_t1" in input_data:
            input_data[f"{agent}_t1"] = 1
            
    for agent in team2_agents:
        if f"{agent}_t2" in input_data:
            input_data[f"{agent}_t2"] = 1
            
    # 3. Pandas tablosuna çevir
    input_df = pd.DataFrame([input_data])
    
    # 4. Tahmin yap
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    return prediction, probability

def display_prediction_result(prediction, probability):
    """Tahmin sonucunu ekrana yazdırır."""
    if prediction == 1:
        st.success(f"🏆 **Tahmin: Team 1 Kazanır!** (Kazanma İhtimali: %{probability[1]*100:.1f})")
        st.balloons()
    else:
        st.error(f"💀 **Tahmin: Team 2 Kazanır!** (Team 1'in Kaybetme İhtimali: %{probability[0]*100:.1f})")

def display_feature_importance(model, model_columns):
    """Modelin karar alırken kullandığı özellik ağırlıklarını grafik olarak çizer."""
    st.markdown("---")
    st.subheader("🧠 Model Bu Kararı Nasıl Verdi?")
    
    with st.expander("Arka Plandaki Matematiksel Ağırlıkları İncele"):
        st.info("Aşağıdaki grafik, galibiyete en çok etki eden 15 faktörü gösterir.")
        
        # Etki oranları
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Özellik': model_columns,
            'Etki Puanı': importances
        })
        
        # En iyi 15 özellik
        top_features = feature_imp_df.sort_values(by='Etki Puanı', ascending=False).head(15)
        top_features = top_features.set_index('Özellik')
        
        # Grafik üretimi
        st.bar_chart(top_features)

# --- Ana Akış (Program İskeleti) ---

def main():
    """Programın ana akışını kontrol eden temel (iskenlet) fonksiyondur."""
    # 1. UI Hazırlığı
    setup_page_config()
    
    # 2. Modeli Yükleme
    model, model_columns = load_model_and_columns()
    
    # 3. Girdi İşlemleri
    map_order, team1_agents, team2_agents = get_user_inputs()
    
    # 4. İşlem Hattı (Pipeline) Çalıştırma ve Çıktı Üretme
    st.markdown("---")
    if st.button("🔮 Maç Sonucunu Tahmin Et", use_container_width=True):
        
        # Validasyon
        if len(team1_agents) != 5 or len(team2_agents) != 5:
            st.warning("Lütfen tahmin yapmadan önce her iki takım için de tam 5 ajan seçtiğinizden emin olun!")
        else:
            # Hesaplama İşlemi (Girdi -> Tahmin Çıktısı)
            prediction, probability = predict_match(team1_agents, team2_agents, map_order, model, model_columns)
            
            # Sonuç Gösterimi
            display_prediction_result(prediction, probability)
            
            # Ara Çıktı Üretimi (Grafik/Tablo Gösterimi)
            display_feature_importance(model, model_columns)

# Sadece bu dosya doğrudan çalıştırıldığında main() tetiklensin
if __name__ == "__main__":
    main()