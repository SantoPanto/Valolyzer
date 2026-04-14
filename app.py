import streamlit as st
import joblib
import pandas as pd

# --- Sayfa Ayarları ---
# Sayfayı daha geniş yapmak için layout="wide" kullanıyoruz
st.set_page_config(page_title="Valorant Maç Tahmincisi", page_icon="🎮", layout="wide")

st.title("🎮 VCT 2025: Yapay Zeka Maç Tahmincisi")
st.markdown("Harita sırasını ve takımların ajan kompozisyonlarını seçin, modelimiz maçın galibini hesaplasın!")

# --- Model Yükleme ---
@st.cache_resource
def load_model_and_columns():
    loaded_model = joblib.load('valorant_rf_model.pkl')
    loaded_columns = joblib.load('model_columns.pkl')
    return loaded_model, loaded_columns

model, model_columns = load_model_and_columns()

# Güncel Valorant Ajan Listesi
VALORANT_AGENTS = sorted([
    "Jett", "Raze", "Breach", "Omen", "Brimstone", "Viper", "Killjoy", 
    "Cypher", "Sova", "Sage", "Phoenix", "Reyna", "Neon", "Fade", 
    "Astra", "KAY/O", "Chamber", "Skye", "Yoru", "Harbor", "Gekko", 
    "Deadlock", "Iso", "Clove", "Vyse"
])

# --- Kullanıcı Arayüzü (UI) ---
st.subheader("📊 Maç Parametreleri")

# Modelin en çok önem verdiği özellik olan Harita Sırası
map_order = st.radio("Bu harita, Best of 3 (Bo3) serisinin kaçıncı maçı?", [1, 2, 3], horizontal=True)

st.markdown("---")

# Ekranı iki sütuna bölüyoruz
col1, col2 = st.columns(2)

with col1:
    st.header("🛡️ Team 1")
    team1_agents = st.multiselect("Team 1 Ajanlarını Seçin (Tam 5 Adet)", VALORANT_AGENTS, max_selections=5, key="t1")

with col2:
    st.header("⚔️ Team 2")
    team2_agents = st.multiselect("Team 2 Ajanlarını Seçin (Tam 5 Adet)", VALORANT_AGENTS, max_selections=5, key="t2")

# --- Tahmin Mantığı ---
st.markdown("---")
if st.button("🔮 Maç Sonucunu Tahmin Et", use_container_width=True):
    
    # Kullanıcının eksik seçim yapmasını engelliyoruz
    if len(team1_agents) != 5 or len(team2_agents) != 5:
        st.warning("Lütfen tahmin yapmadan önce her iki takım için de tam 5 ajan seçtiğinizden emin olun!")
    else:
        # 1. Modelin eğitimde gördüğü tüm sütunları içeren, içi 0 dolu bir şablon oluştur
        input_data = {col: 0 for col in model_columns}
        
        # 2. Kullanıcının girdiği verileri şablonda 1 olarak güncelle
        if 'map_order' in input_data:
            input_data['map_order'] = map_order
            
        for agent in team1_agents:
            if f"{agent}_t1" in input_data:
                input_data[f"{agent}_t1"] = 1
                
        for agent in team2_agents:
            if f"{agent}_t2" in input_data:
                input_data[f"{agent}_t2"] = 1
                
        # 3. Şablonu Pandas tablosuna çevir
        input_df = pd.DataFrame([input_data])
        
        # 4. Yapay Zekayı Çalıştır
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0] # Kazanma yüzdesini verir
        
        # 5. Sonuçları Ekrana Yazdır
        if prediction == 1:
            st.success(f"🏆 **Tahmin: Team 1 Kazanır!** (Kazanma İhtimali: %{probability[1]*100:.1f})")
            st.balloons() # Ekranda balonlar uçurur :)
        else:
            st.error(f"💀 **Tahmin: Team 2 Kazanır!** (Team 1'in Kaybetme İhtimali: %{probability[0]*100:.1f})")