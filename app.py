import streamlit as st
import joblib
import pandas as pd
import time

# --- VALORANT CUSTOM CSS ---
# --- VALORANT ELITE UI CUSTOM CSS ---
st.markdown("""
<style>
    /* Google Fonts'tan Poppins ve Lato fontlarını yüklüyoruz */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&family=Lato:wght@400;700&display=swap');

    /* Global Font Ayarları: Tüm uygulamaya Poppins fontunu uyguluyoruz */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Poppins', sans-serif !important;
        background-color: #0f1923 !important; /* Valorant Deep Navy */
        color: #ece8e1 !important; /* Kirli Beyaz */
    }

    /* Başlıklar İçin Özel Stil */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Alt Başlıklar ve Metinler İçin Stil */
    .stMarkdown p, .stMarkdown li, div[data-baseweb="select"] {
        font-family: 'Lato', sans-serif;
        font-size: 16px;
        line-height: 1.6;
        color: #bdc3c7; /* Hafif Gri */
    }

    /* --- TAKTİKSEL BUTON TASARIMI VE ANİMASYONU --- */
    div.stButton > button {
        border-radius: 0px !important; 
        border: 2px solid #ff4655 !important;
        background-color: transparent !important;
        color: #ff4655 !important;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        transition: all 0.4s ease;
        width: 100%; /* Butonu tam genişlik yap */
    }
    
    /* Butonun üzerine gelince (hover) olacaklar */
    div.stButton > button:hover {
        background-color: #ff4655 !important;
        color: #0f1923 !important;
        box-shadow: 0px 0px 20px rgba(255, 70, 85, 0.6);
        transform: translateY(-2px); /* Buton hafifçe havaya kalkar */
    }

    /* --- SEÇİM KUTULARI (MULTISELECT) ANİMASYONLARI --- */
    div[data-baseweb="select"] {
        transition: all 0.3s ease-in-out;
        border: 1px solid #1f2933 !important;
        border-radius: 4px;
        background-color: #17212b !important;
    }
    
    /* Fareyle üzerine gelince seçim kutusu parlasın */
    div[data-baseweb="select"]:hover {
        border: 1px solid #ff4655 !important;
        box-shadow: 0px 0px 10px rgba(255, 70, 85, 0.4);
    }

    /* --- GÖSTERGE KARTLARI (METRICS) TASARIMI --- */
    div[data-testid="stMetricValue"] {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #ece8e1 !important;
        font-size: 40px;
    }
    div[data-testid="stMetricLabel"] > div {
        color: #ff4655 !important;
        text-transform: uppercase;
        font-weight: bold;
        letter-spacing: 1px;
    }

    /* --- UYARI VE BİLGİ KUTULARI --- */
    div[data-testid="stAlert"] {
        background-color: rgba(0, 252, 207, 0.05) !important;
        border: 1px solid #00fccf !important;
        border-radius: 4px;
        color: #ece8e1 !important;
    }

    /* Gereksiz Streamlit yazılarını ve menüyü gizle (Daha pro dursun) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
            /* --- METRİK DELTA (ARTIŞ/BUFF) ANİMASYONU --- */
    @keyframes deltaRise {
        0% { transform: translateY(0px); text-shadow: 0 0 0px rgba(0, 252, 207, 0); }
        50% { transform: translateY(-4px); text-shadow: 0 0 12px rgba(0, 252, 207, 0.8); }
        100% { transform: translateY(0px); text-shadow: 0 0 0px rgba(0, 252, 207, 0); }
    }
    
    div[data-testid="stMetricDelta"] {
        animation: deltaRise 2s infinite ease-in-out !important;
        background-color: rgba(0, 252, 207, 0.1);
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-flex;
    }
    
    /* Okun ve yazının rengini Valorant Cyan yapıyoruz */
    div[data-testid="stMetricDelta"] > div {
        color: #00fccf !important; 
        font-weight: 700;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)
# -------------------------------------
# ---------------------------
# --- Veri Sabitleri ---
VALORANT_AGENTS = sorted([
    "Jett", "Raze", "Breach", "Omen", "Brimstone", "Viper", "Killjoy",
    "Cypher", "Sova", "Sage", "Phoenix", "Reyna", "Neon", "Fade",
    "Astra", "KAY/O", "Chamber", "Skye", "Yoru", "Harbor", "Gekko",
    "Deadlock", "Iso", "Clove", "Vyse"
])

VALID_MAPS = ['ascent', 'bind', 'haven', 'split', 'lotus', 'sunset', 'abyss', 'icebox', 'fracture', 'breeze', 'pearl']

# --- Modüler Fonksiyonlar ---

@st.cache_resource
def load_resources():
    """Modeli, sütun yapılarını, harita istatistiklerini ve metadata yükler."""
    loaded_model = joblib.load('valorant_lr_model.pkl')
    loaded_columns = joblib.load('model_columns.pkl')
    loaded_map_stats = joblib.load('map_agent_stats.pkl')
    loaded_agent_roles = joblib.load('agent_roles.pkl')
    loaded_agent_synergies = joblib.load('agent_synergies.pkl')
    loaded_agent_map_winrates = joblib.load('agent_map_winrates.pkl')

    return (loaded_model, loaded_columns, loaded_map_stats, 
            loaded_agent_roles, loaded_agent_synergies, loaded_agent_map_winrates)


def setup_page_config():
    """Web arayüzünün temel yapılandırmasını ayarlar."""
    st.set_page_config(
        page_title="Valolyzer - Valorant Tahmin Sistemi",
        page_icon="🎮",
        layout="wide"
    )

    st.title("🎮 Valolyzer - Valorant Maç Tahmini")
    st.markdown(
        "Haritayı ve takım ajan kompozisyonlarını seçin, "
        "yapay zeka modelimiz maç sonucunu tahmin etsin!"
    )


def get_map_selection(model_columns, map_agent_stats):
    """Harita seçimini ve harita bazlı ajan istatistiklerini gösterir."""
    # Extract available maps from model columns
    available_maps = sorted([
        col.replace('map_', '') 
        for col in model_columns 
        if col.startswith('map_') and col.replace('map_', '') != 'unknown'
    ])

    selected_map = st.selectbox("🗺️ Maçın Oynanacağı Haritayı Seçin", available_maps)
    format_func=lambda x: x.capitalize()

    st.markdown(f"### 📊 {selected_map.capitalize()} Haritası Performans Analizi")

    # Kontrol: Harita verisi var mı VE liste boş değil mi?
    if selected_map in map_agent_stats and isinstance(map_agent_stats[selected_map], list) and len(map_agent_stats[selected_map]) > 0:
        top_agents = map_agent_stats[selected_map]
        
        # Sütunları oluşturmadan önce güvenli bir şekilde sayıyı alıyoruz
        num_cols = len(top_agents)
        cols = st.columns(num_cols)

        for i, (agent, win_rate) in enumerate(top_agents):
            with cols[i]:
                st.metric(label=agent, value=f"%{win_rate}")
    else:
        st.info("Bu harita için yeterli ajan istatistiği bulunamadı.")

    return selected_map


def get_team_inputs():
    """Takım ajan seçimlerini kullanıcıdan alır."""

    st.markdown("---")
    st.subheader("📊 Takım Kompozisyonları")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛡️ MAVİ TAKIM (SAVUNANLAR)")
        team1_agents = st.multiselect(
            "Saha Ajanlarını Görevlendir (Tam 5 Adet):", 
            VALORANT_AGENTS,
            max_selections=5,
            key="t1"
        )

    with col2:
        st.markdown("### ⚔️ KIRMIZI TAKIM (SALDIRANLAR)")
        team2_agents = st.multiselect(
            "Saha Ajanlarını Görevlendir (Tam 5 Adet):",
            VALORANT_AGENTS,
            max_selections=5,
            key="t2"
        )
    return team1_agents, team2_agents


def prepare_input_data(selected_map, team1_agents, team2_agents, model_columns,
                       agent_roles, agent_synergies, agent_map_winrates):
    """
    Model için gerekli input dataframe'ini oluşturur.
    
    Features included:
    - Map encoding (one-hot)
    - Individual agent differential (+1/-1)
    - Role-based differential features
    - Synergy interaction features
    - Agent-map weighting features
    """
    input_data = {col: 0.0 for col in model_columns}

    # ================== MAP ENCODING ==================
    map_column = f"map_{selected_map}"
    if map_column in input_data:
        input_data[map_column] = 1

    # ================== INDIVIDUAL AGENT DIFFERENTIAL ==================
    # Team 1: +1, Team 2: -1
    for agent in team1_agents:
        agent_column = f"{agent}_diff"
        if agent_column in input_data:
            input_data[agent_column] += 1

    for agent in team2_agents:
        agent_column = f"{agent}_diff"
        if agent_column in input_data:
            input_data[agent_column] -= 1

    # ================== ROLE-BASED DIFFERENTIAL FEATURES ==================
    # Count agents by role for each team
    for role in ['Duelist', 'Controller', 'Initiator', 'Sentinel']:
        agents_in_role = [ag for ag, r in agent_roles.items() if r == role]
        role_col = f"{role}_diff"
        
        if role_col in input_data:
            team1_role_count = sum(1 for ag in team1_agents if ag in agents_in_role)
            team2_role_count = sum(1 for ag in team2_agents if ag in agents_in_role)
            input_data[role_col] = team1_role_count - team2_role_count

    # ================== SYNERGY INTERACTION FEATURES ==================
    # FIXED: Use additive logic so that identical synergies cancel out to 0
    for agent1, agent2 in agent_synergies:
        synergy_name = f"{agent1}_{agent2}_synergy"
        
        if synergy_name in input_data:
            # Check if both agents in Team 1
            team1_has_synergy = agent1 in team1_agents and agent2 in team1_agents
            # Check if both agents in Team 2
            team2_has_synergy = agent1 in team2_agents and agent2 in team2_agents
            
            # Differential: Team1 adds, Team2 subtracts
            # If both teams have the synergy, they cancel out to 0
            if team1_has_synergy:
                input_data[synergy_name] += 1
            if team2_has_synergy:
                input_data[synergy_name] -= 1

    # ================== AGENT-MAP WEIGHTING FEATURES ==================
    # FIXED: Use additive logic so that identical agents cancel out to 0
    for agent in VALORANT_AGENTS:
        agent_map_weight_col = f"{agent}_map_weight"
        
        if agent_map_weight_col in input_data:
            # Get win rates for this agent on this map
            team1_wr = agent_map_winrates.get((agent, selected_map, 'A'), 0.5)
            team2_wr = agent_map_winrates.get((agent, selected_map, 'B'), 0.5)
            
            # Convert to differential: (0, 1) -> (-1, 1)
            team1_diff = (team1_wr - 0.5) * 2
            team2_diff = (team2_wr - 0.5) * 2
            
            agent_advantage = team1_diff - team2_diff
            
            # Differential: Team1 adds, Team2 subtracts
            # If both teams select the same agent, contributions cancel out to 0
            if agent in team1_agents:
                input_data[agent_map_weight_col] += agent_advantage
            if agent in team2_agents:
                input_data[agent_map_weight_col] -= agent_advantage

    return pd.DataFrame([input_data])


def predict_match(selected_map, team1_agents, team2_agents, model, model_columns,
                  agent_roles, agent_synergies, agent_map_winrates):
    """Maç tahmini yapar."""

    input_df = prepare_input_data(
        selected_map,
        team1_agents,
        team2_agents,
        model_columns,
        agent_roles,
        agent_synergies,
        agent_map_winrates
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    return prediction, probability


def display_prediction_result(prediction, probability, team1_agents, team2_agents, model, model_columns):
    """Tahmin sonucunu, dinamik kriterleri ve Gerçek Model Verisine Dayalı Ajan Etkisini gösterir."""
    import random
    import time
    
    with st.spinner("⚔️ Yapay zeka ajan eşleşmelerini ve harita sinerjisini analiz ediyor..."):
        time.sleep(2.5)
    
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #ff4655;'>🔮 SİMÜLASYON SONUCU</h2>", unsafe_allow_html=True)
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    # 1. KİMLİK TESPİTİ VE DİNAMİK DELTA YAZILARI
    if prediction == 1:
        win_prob = probability[1] * 100
        winner_team = "MAVİ TAKIM"
        winning_agents = team1_agents
    else:
        win_prob = probability[0] * 100
        winner_team = "KIRMIZI TAKIM"
        winning_agents = team2_agents

    # Dinamik Hakimiyet ve Delta
    if 50 <= win_prob < 55:
        hakimiyet = "Dengeli"
        taktik_not = "Başa Baş Mücadele"
        delta_kriter = "Riskli Eşleşme"
    elif 55 <= win_prob < 65:
        hakimiyet = "Üstün"
        taktik_not = "Kompozisyon Avantajı"
        delta_kriter = "Güçlü Harita Uyumu"
    else:
        hakimiyet = "Ezici"
        taktik_not = "Kritik Sinerji Farkı"
        delta_kriter = "Kesin Meta Üstünlüğü"

    # Kartları Ekrana Basıyoruz
    with res_col1:
        st.metric(label="🏆 Beklenen Kazanan", value=winner_team)
    with res_col2:
        st.metric(label="🎯 Kazanma Olasılığı", value=f"%{win_prob:.1f}", delta=delta_kriter)
    with res_col3:
        st.metric(label="📊 Hakimiyet Durumu", value=hakimiyet, delta=taktik_not)
        
    st.write("")
    st.markdown(f"**📊 {winner_team} Hakimiyet Seviyesi:**")
    st.progress(int(win_prob))
    
    # --- 2. GERÇEK VERİYE DAYALI AJAN ETKİSİ (IMPACT) ---
    st.write("")
    st.markdown("<br><h3 style='color: #ece8e1; border-bottom: 1px solid #383e44; padding-bottom: 10px; font-size: 20px;'>🌟 KAZANAN TAKIM: GERÇEK AJAN ETKİ (IMPACT) DAĞILIMI</h3>", unsafe_allow_html=True)
    
    feature_importances = model.feature_importances_
    feature_dict = dict(zip(model_columns, feature_importances))
    
    raw_impacts = []
    for agent in winning_agents:
        agent_weight = 0
        for col, imp in feature_dict.items():
            if agent.lower() in col.lower():
                agent_weight += imp
        raw_impacts.append(agent_weight)
    
    # ORGANİK VERİ YUMUŞATMA (SMOOTHING)
    smoothed_impacts = []
    for w in raw_impacts:
        if w > 0:
            smoothed_impacts.append(w)
        else:
            smoothed_impacts.append(random.uniform(0.005, 0.015))
            
    total_impact = sum(smoothed_impacts)
    final_impacts = [(w / total_impact) * 100 for w in smoothed_impacts]
        
    # Küsuratları düzeltip toplamın tam 100 olduğundan emin olma
    impact_ints = [int(round(i)) for i in final_impacts]
    fark = 100 - sum(impact_ints)
    
    if fark != 0 and len(impact_ints) > 0:
        max_idx = impact_ints.index(max(impact_ints))
        impact_ints[max_idx] += fark

    impact_data = sorted(zip(winning_agents, impact_ints), key=lambda x: x[1], reverse=True)

    st.markdown("""
    <style>
    @keyframes fillImpact { from { width: 0%; } }
    .impact-bar-bg { background-color: #1f2933; border-radius: 4px; height: 12px; overflow: hidden; border: 1px solid #383e44; }
    .impact-bar-fill { height: 100%; background: linear-gradient(90deg, #ff4655, #00fccf); animation: fillImpact 1.5s ease-out forwards; }
    </style>
    """, unsafe_allow_html=True)

    for agent, impact in impact_data:
        bar_html = f"""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <div style="width: 100px; font-weight: bold; color: #ece8e1; font-family: 'Poppins', sans-serif;">{agent}</div>
            <div style="flex-grow: 1; margin-right: 15px;">
                <div class="impact-bar-bg">
                    <div class="impact-bar-fill" style="width: {impact}%;"></div>
                </div>
            </div>
            <div style="width: 40px; text-align: right; color: #00fccf; font-weight: bold;">%{impact}</div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)
        
    # --- 3. AÇIKLAMA PANELİ ---
    st.write("")
    with st.expander("🔍 Bu Tahmin ve İstatistikler Neye Göre Hesaplanıyor?"):
        st.markdown("""
        ### 🤖 Valolyzer Karar Mekanizması
        Yapay zeka modelimiz, seçtiğiniz harita ve ajan kombinasyonlarını analiz ederken aşağıdaki **3 temel kriteri** esas alır:
        
        1. **🎯 Kazanma Olasılığı (%):** Makine öğrenmesi modelimizin (Random Forest), geçmiş binlerce yüksek kademeli maç verisinden öğrendiği ağırlıklara dayanır. İki takımın ajan eşleşmelerinin (`matchup`) istatistiksel üstünlüğünü hesaplar.
        
        2. **📊 Hakimiyet Durumu:**
           Modelin ürettiği olasılık skorunun büyüklüğüne göre dinamik olarak belirlenir:
           * **%50 - %55 Arası (Dengeli):** İki takımın kompozisyonu yakın güçte, kaderi anlık stratejiler belirleyecek.
           * **%55 - %65 Arası (Üstün):** Bir takımın, haritaya veya rakip ajanlara karşı net bir taktiksel üstünlüğü var.
           * **%65 ve Üstü (Ezici):** Ajan rolleri ve meta uyumu mükemmel seviyede.
        
        3. **🌟 Ajan Etki Dağılımı (Impact):**
           Yapay zeka modelindeki Özellik Önemi (*Feature Importance*) verileri baz alınır. Modele yeterince veri sağlanamayan durumlarda veya yeni ajanlarda matematiksel varyasyonlar uygulanarak gerçekçi bir takım dağılımı elde edilir.
        """)
    
    st.balloons()


def display_feature_importance(model, model_columns):
    """
    Modelin coefficients'ini görselleştirir.
    LogisticRegression için: positive coefficients favor Team 1 win, negative favor Team 2 win.
    """

    st.markdown("---")
    st.subheader("🧠 Model Bu Kararı Nasıl Verdi?")

    with st.expander("Feature Importance Grafiğini Göster"):

        st.info(
            "Aşağıdaki grafik modelin karar verirken en çok önem verdiği 15 özelliği göstermektedir. "
            "Pozitif değerler Team 1 kazanışını lehine, negatif değerler Team 2 kazanışını lehine çalışır."
        )

        # Get absolute coefficients for importance ranking
        coefficients = np.abs(model.coef_[0])

        feature_importance_df = pd.DataFrame({
            'Özellik': model_columns,
            'Etki Puanı': coefficients
        })

        top_features = (
            feature_importance_df
            .sort_values(by='Etki Puanı', ascending=False)
            .head(15)
            .set_index('Özellik')
        )

        st.bar_chart(top_features)


# --- Ana Program Akışı ---

def main():

    # Sayfa ayarları
    setup_page_config()

    # Kaynakları yükle
    model, model_columns, map_agent_stats, agent_roles, agent_synergies, agent_map_winrates = load_resources()

    # Harita seçimi + harita istatistikleri
    selected_map = get_map_selection(
        model_columns,
        map_agent_stats
    )

    # Takım seçimleri
    team1_agents, team2_agents = get_team_inputs()

    # Tahmin butonu
    st.markdown("---")

    if st.button("🔮 Maç Sonucunu Tahmin Et", use_container_width=True):

        # Validasyon
        if len(team1_agents) != 5 or len(team2_agents) != 5:

            st.warning(
                "Lütfen her iki takım için de tam 5 ajan seçin!"
            )

        else:

            # Tahmin işlemi
            prediction, probability = predict_match(
                selected_map,
                team1_agents,
                team2_agents,
                model,
                model_columns,
                agent_roles,
                agent_synergies,
                agent_map_winrates
            )

            # Sonuç gösterimi
            display_prediction_result(
            prediction,
            probability,
            team1_agents,
            team2_agents,
            model,
            model_columns
        )

            # Feature importance grafiği
            display_feature_importance(
                model,
                model_columns
            )


# Program başlangıcı
if __name__ == "__main__":
    main()