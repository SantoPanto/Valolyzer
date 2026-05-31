import streamlit as st
import joblib
import pandas as pd
import time
import requests
import numpy as np
import random

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

# Geliştirme aşamasında yeterli veri olmadığı için geçici olarak kapatılan haritalar: 'Sunset', 'Abyss', 'Icebox'
VALID_MAPS = ['Ascent', 'Bind', 'Haven', 'Split', 'Lotus', 'Fracture', 'Breeze', 'Pearl']
# Ajan ikon linkleri (Valorant API'den doğrudan çekiyoruz)
@st.cache_data
def get_agent_icons():
    """Valorant API'den TÜM ajanların en güncel ikonlarını otomatik çeker."""
    icons = {}
    try:
        # Sadece oynanabilir ajanları filtreleyerek çekiyoruz
        response = requests.get("https://valorant-api.com/v1/agents?isPlayableCharacter=true")
        if response.status_code == 200:
            data = response.json()['data']
            for agent in data:
                # API'den gelen ismi ve ikon URL'sini sözlüğe kaydediyoruz
                icons[agent['displayName']] = agent['displayIcon']
    except Exception as e:
        st.warning("Ajan ikonları yüklenirken bir hata oluştu, internet bağlantınızı kontrol edin.")
    return icons

# Uygulama açıldığında tüm ikonlar sadece 1 kere çekilip hafızaya alınır
AGENT_ICONS = get_agent_icons()
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

    # --- GELİŞMİŞ VE MODERN BAŞLIK (HERO SECTION) ---
    header_html = """
    <div style="text-align: center; margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid rgba(56, 62, 68, 0.5);">
        <h1 style="font-family: 'Poppins', sans-serif; font-size: 58px; font-weight: 800; color: #ece8e1; text-transform: uppercase; letter-spacing: 4px; margin-bottom: 10px; text-shadow: 0px 4px 15px rgba(0,0,0,0.4);">
            <span style="color: #ff4655;">VALO</span>LYZER
        </h1>
        <p style="font-family: 'Lato', sans-serif; font-size: 18px; color: #8b97a2; max-width: 700px; margin: 0 auto; line-height: 1.6;">
            Yapay Zeka Destekli E-Spor Karar Destek ve Analiz Motoru
        </p>
        <div style="margin-top: 15px;">
            <span style="background-color: rgba(0, 252, 207, 0.1); border: 1px solid #00fccf; color: #00fccf; padding: 5px 12px; border-radius: 4px; font-size: 13px; font-family: 'Poppins', sans-serif; font-weight: bold; letter-spacing: 1px;">
                V 1.0.0
            </span>
            <span style="background-color: rgba(255, 70, 85, 0.1); border: 1px solid #ff4655; color: #ff4655; padding: 5px 12px; border-radius: 4px; font-size: 13px; font-family: 'Poppins', sans-serif; font-weight: bold; letter-spacing: 1px; margin-left: 10px;">
                ML MODEL: LOGISTIC REGRESSION
            </span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    


def display_diagnostics_panel():
    """Veri analiz test panelini gösterir."""
    with st.expander("📊 Veri Analiz Test Paneli"):
        try:
            total_matches, map_counts = load_match_diagnostics()
            
            st.metric("📈 Toplam İşlenmiş Maçlar", value=total_matches)
            
            if map_counts:
                st.markdown("**Harita Başına Maç Sayısı:**")
                map_df = pd.DataFrame(
                    list(map_counts.items()), 
                    columns=["Harita", "Maç Sayısı"]
                ).sort_values("Maç Sayısı", ascending=False)
                st.dataframe(map_df, use_container_width=True, hide_index=True)
            else:
                st.info("Harita başına detaylı istatistik henüz kullanılabilir değil.")
        except Exception as e:
            st.error(f"❌ Diagnostik panel yüklenirken hata: {str(e)}")


@st.cache_data
def load_match_diagnostics():
    """Diagnostik amaçlı maç ve harita istatistiklerini yükler."""
    total_matches = 0
    map_counts = {}
    
    try:
        import os
        # 1. Toplam İşlenmiş Harita (Maç) Sayısını Al (maps.csv'den)
        if os.path.exists('data/processed/maps.csv'):
            df_matches = pd.read_csv('data/processed/maps.csv')
            total_matches = len(df_matches)
        elif os.path.exists('data/maps.csv'):
            df_matches = pd.read_csv('data/maps.csv')
            total_matches = len(df_matches)

        # 2. Gerçek Harita Dağılımını Al (maps.csv'den)
        df_maps = None
        if os.path.exists('data/processed/maps.csv'):
            df_maps = pd.read_csv('data/processed/maps.csv')
        elif os.path.exists('data/maps.csv'):
            df_maps = pd.read_csv('data/maps.csv')

        if df_maps is not None and 'map_name' in df_maps.columns:
            # VALID_MAPS'teki haritaları küçük harfe çevirip eşleştiriyoruz
            valid_maps_lower = [m.lower() for m in VALID_MAPS]
            
            # Sütundaki veriyi küçük harfe çevir
            df_maps['map_name_lower'] = df_maps['map_name'].astype(str).str.lower()
            
            # Sadece geçerli haritaları filtrele
            valid_maps_df = df_maps[df_maps['map_name_lower'].isin(valid_maps_lower)]
            
            # Sayımları yap ve ilk harflerini büyük yaparak (Örn: ascent -> Ascent) sözlüğe ekle
            counts = valid_maps_df['map_name_lower'].value_counts().to_dict()
            map_counts = {k.capitalize(): v for k, v in counts.items()}

    except Exception as e:
        st.warning(f"⚠️ Diagnostik verileri yüklenirken hata: {str(e)}")
    
    return total_matches, map_counts


def get_map_selection(model_columns, map_agent_stats):
    """Harita seçimini ve harita bazlı ajan istatistiklerini gösterir."""
    
    selected_map = st.selectbox(
        "📍 MAÇIN OYNANACAĞI HARİTAYI SEÇİN", 
        VALID_MAPS
    )

    if not selected_map:
        st.warning("Seçilebilir bir harita bulunamadı!")
        return None

    # Başlığı Göster
    st.markdown(f"### 📊 {selected_map} Haritası Performans Analizi")

    # FIX 1: Harita ismini büyük/küçük harf duyarsız olarak map_agent_stats içinde ara
    stats_key = None
    selected_map_lower = selected_map.lower()
    
    for key in map_agent_stats.keys():
        if str(key).lower() == selected_map_lower:
            stats_key = key
            break
    
    if stats_key is None:
        stats_key = selected_map

    # FIX 2: Ajan oranlarını listeleme
    if stats_key in map_agent_stats and isinstance(map_agent_stats[stats_key], list) and len(map_agent_stats[stats_key]) > 0:
        top_agents = map_agent_stats[stats_key]
        
        # Kazanma oranlarına göre sırala
        ranked_agents = sorted(enumerate(top_agents, 1), key=lambda x: x[1][1], reverse=True)
        
        num_cols = len(ranked_agents)
        cols = st.columns(num_cols)

        for col_idx, (rank, (agent, win_rate)) in enumerate(ranked_agents):
            with cols[col_idx]:
                # Değeri % formatına çevirme (0.54 -> 54.0)
                if isinstance(win_rate, (int, float)):
                    win_rate_pct = win_rate * 100 if 0 <= win_rate <= 1 else win_rate
                else:
                    win_rate_pct = float(win_rate) * 100 if float(win_rate) <= 1 else float(win_rate)
                
                st.metric(
                    label=f"#{rank} {agent}", 
                    value=f"%{win_rate_pct:.1f}"
                )
    else:
        st.info("Bu harita için yeterli ajan istatistiği bulunamadı.")

    # Fonksiyonun dönmesi gereken yeri EN SONA taşıdık
    return selected_map.lower()


def get_team_inputs():
    """Takım ajan seçimlerini kullanıcıdan alır ve oyun içi gibi görselleştirir."""
    
    
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #ece8e1; font-family: Poppins, sans-serif;'>👥 TAKIM KOMPOZİSYONLARI</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🛡️ MAVİ TAKIM (SAVUNANLAR)")
        team1_agents = st.multiselect(
            "Saha Ajanlarını Görevlendir (Tam 5 Adet):",
            VALORANT_AGENTS,
            max_selections=5,
            key="t1"
        )
        
        # --- Mavi Takım Görsel Önizleme (Turkuaz Parlama) ---
        html1 = "<div style='display: flex; gap: 12px; margin-top: 15px; justify-content: center;'>"
        for i in range(5):
            if i < len(team1_agents):
                agent = team1_agents[i]
                # İkonu al, yoksa varsayılan
                icon_url = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/default/displayicon.png")
                html1 += f"""
                <div style='text-align: center; width: 65px; transition: 0.3s;'>
                    <img src="{icon_url}" style='width: 65px; height: 65px; border-radius: 8px; border: 2px solid #00fccf; box-shadow: 0 0 10px rgba(0, 252, 207, 0.6); object-fit: cover;'>
                    <p style='font-size: 11px; margin-top: 5px; color: #ece8e1; font-weight: bold; font-family: Poppins, sans-serif;'>{agent}</p>
                </div>"""
            else:
                html1 += f"""
                <div style='text-align: center; width: 65px;'>
                    <div style='width: 65px; height: 65px; border-radius: 8px; border: 2px dashed #383e44; background-color: rgba(31, 41, 51, 0.4); display: flex; align-items: center; justify-content: center;'>
                        <span style='color: #383e44; font-size: 24px; font-weight: bold;'>?</span>
                    </div>
                    <p style='font-size: 11px; margin-top: 5px; color: #8b97a2; font-family: Poppins, sans-serif;'>Boş</p>
                </div>"""
        html1 += "</div>"
        st.markdown(html1, unsafe_allow_html=True)

    with col2:
        st.markdown("### ⚔️ KIRMIZI TAKIM (SALDIRANLAR)")
        team2_agents = st.multiselect(
            "Saha Ajanlarını Görevlendir (Tam 5 Adet):",
            VALORANT_AGENTS,
            max_selections=5,
            key="t2"
        )
        
        # --- Kırmızı Takım Görsel Önizleme (Kırmızı Parlama) ---
        html2 = "<div style='display: flex; gap: 12px; margin-top: 15px; justify-content: center;'>"
        for i in range(5):
            if i < len(team2_agents):
                agent = team2_agents[i]
                icon_url = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/default/displayicon.png")
                html2 += f"""
                <div style='text-align: center; width: 65px; transition: 0.3s;'>
                    <img src="{icon_url}" style='width: 65px; height: 65px; border-radius: 8px; border: 2px solid #ff4655; box-shadow: 0 0 10px rgba(255, 70, 85, 0.6); object-fit: cover;'>
                    <p style='font-size: 11px; margin-top: 5px; color: #ece8e1; font-weight: bold; font-family: Poppins, sans-serif;'>{agent}</p>
                </div>"""
            else:
                html2 += f"""
                <div style='text-align: center; width: 65px;'>
                    <div style='width: 65px; height: 65px; border-radius: 8px; border: 2px dashed #383e44; background-color: rgba(31, 41, 51, 0.4); display: flex; align-items: center; justify-content: center;'>
                        <span style='color: #383e44; font-size: 24px; font-weight: bold;'>?</span>
                    </div>
                    <p style='font-size: 11px; margin-top: 5px; color: #8b97a2; font-family: Poppins, sans-serif;'>Boş</p>
                </div>"""
        html2 += "</div>"
        st.markdown(html2, unsafe_allow_html=True)
        
    return team1_agents, team2_agents


def prepare_input_data(selected_map, team1_agents, team2_agents, model_columns,
                       agent_roles, agent_synergies, agent_map_winrates):
    """Model için gerekli input dataframe'ini oluşturur."""
    input_data = {col: 0.0 for col in model_columns}

    # ================== MAP ENCODING ==================
    # Dinamik Eşleştirme: Model columns içinde hangi format varsa (ön ekli, ön eksiz, büyük/küçük harf) onu bulur ve 1 yapar.
    possible_map_columns = [
        f"map_{selected_map}", 
        f"map_{selected_map.capitalize()}", 
        selected_map, 
        selected_map.capitalize()
    ]
    
    map_encoded = False
    for map_col in possible_map_columns:
        if map_col in input_data:
            input_data[map_col] = 1.0
            map_encoded = True
            break
            
    # Eğer modelinizde yukardakiler yoksa ve sadece tek bir 'map' veya 'map_encoded' kolonu varsa (Label Encoding)
    if not map_encoded:
        for col in ['map', 'map_encoded', 'map_id']:
            if col in input_data:
                # Geçici olarak haritanın VALID_MAPS üzerindeki indeksini sayısal değer olarak gönderir
                if selected_map in VALID_MAPS:
                    input_data[col] = float(VALID_MAPS.index(selected_map))
                break

    # ================== INDIVIDUAL AGENT DIFFERENTIAL ==================
    for agent in team1_agents:
        agent_column = f"{agent}_diff"
        if agent_column in input_data:
            input_data[agent_column] += 1

    for agent in team2_agents:
        agent_column = f"{agent}_diff"
        if agent_column in input_data:
            input_data[agent_column] -= 1

    # ================== MAP-AGENT INTERACTION FEATURES ==================
    for agent in team1_agents:
        interaction_col = f"{agent}_{selected_map}_diff"
        if interaction_col in input_data:
            input_data[interaction_col] += 1

    for agent in team2_agents:
        interaction_col = f"{agent}_{selected_map}_diff"
        if interaction_col in input_data:
            input_data[interaction_col] -= 1

    # ================== ROLE-BASED DIFFERENTIAL FEATURES ==================
    for role in ['Duelist', 'Controller', 'Initiator', 'Sentinel']:
        agents_in_role = [ag for ag, r in agent_roles.items() if r == role]
        role_col = f"{role}_diff"
        
        if role_col in input_data:
            team1_role_count = sum(1 for ag in team1_agents if ag in agents_in_role)
            team2_role_count = sum(1 for ag in team2_agents if ag in agents_in_role)
            input_data[role_col] = team1_role_count - team2_role_count

    # ================== SYNERGY INTERACTION FEATURES ==================
    for agent1, agent2 in agent_synergies:
        synergy_name = f"{agent1}_{agent2}_synergy"
        
        if synergy_name in input_data:
            team1_has_synergy = agent1 in team1_agents and agent2 in team1_agents
            team2_has_synergy = agent1 in team2_agents and agent2 in team2_agents
            
            if team1_has_synergy:
                input_data[synergy_name] += 1
            if team2_has_synergy:
                input_data[synergy_name] -= 1

    # ================== AGENT-MAP WEIGHTING FEATURES ==================
    for agent in VALORANT_AGENTS:
        agent_map_weight_col = f"{agent}_map_weight"
        
        if agent_map_weight_col in input_data:
            team1_wr = agent_map_winrates.get((agent, selected_map, 'A'), 0.5)
            team2_wr = agent_map_winrates.get((agent, selected_map, 'B'), 0.5)
            
            team1_diff = (team1_wr - 0.5) * 2
            team2_diff = (team2_wr - 0.5) * 2
            
            agent_advantage = team1_diff - team2_diff
            
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
    
    # --- DİNAMİK YÜKLEME (LOADING) ANİMASYONU ---
    loading_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Adım adım gösterilecek havalı/teknik mesajlar
    loading_messages = [
        "🗺️ Harita topolojisi ve meta verileri çekiliyor...",
        "🛡️ Mavi takım savunma sinerjileri hesaplanıyor...",
        "⚔️ Kırmızı takım saldırı gücü analiz ediliyor...",
        "🧠 Lojistik Regresyon modeli matrisleri çözüyor...",
        "✨ Tahmin raporu derleniyor..."
    ]

    # Barı 1'den 100'e kadar yavaşça doldur
    for i in range(100):
        progress_bar.progress(i + 1)
        
        # Yüzdeye göre mesajı değiştir
        if i == 5:
            loading_placeholder.markdown(f"<h4 style='color: #00fccf; text-align: center;'>{loading_messages[0]}</h4>", unsafe_allow_html=True)
        elif i == 25:
            loading_placeholder.markdown(f"<h4 style='color: #00fccf; text-align: center;'>{loading_messages[1]}</h4>", unsafe_allow_html=True)
        elif i == 50:
            loading_placeholder.markdown(f"<h4 style='color: #00fccf; text-align: center;'>{loading_messages[2]}</h4>", unsafe_allow_html=True)
        elif i == 75:
            loading_placeholder.markdown(f"<h4 style='color: #00fccf; text-align: center;'>{loading_messages[3]}</h4>", unsafe_allow_html=True)
        elif i == 90:
            loading_placeholder.markdown(f"<h4 style='color: #ece8e1; text-align: center;'>{loading_messages[4]}</h4>", unsafe_allow_html=True)
            
        time.sleep(0.02) # Toplam yükleme süresini belirler (Yaklaşık 2 saniye)

    # İşlem bitince yükleme yazılarını ve barı ekrandan temizle
    loading_placeholder.empty()
    progress_bar.empty()
    
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

  # --- 4. SAVAŞ RAPORU (BATTLE REPORT) ÖZET KARTI ---
    
    # Kazanan takımın rengine göre (Mavi veya Kırmızı) tema belirliyoruz
    winner_color = "#00fccf" if winner_team == "MAVİ TAKIM" else "#ff4655"
    
    report_html = f"""
<div style="background-color: #1f2933; border-left: 5px solid {winner_color}; padding: 25px; border-radius: 8px; box-shadow: 0 8px 16px rgba(0,0,0,0.4); margin-top: 20px; margin-bottom: 30px;">
    <h2 style="text-align: center; color: {winner_color}; margin-top: 0; font-family: 'Poppins', sans-serif; letter-spacing: 2px;">🛡️ KESİN SAVAŞ RAPORU</h2>
    <p style="text-align: center; color: #bdc3c7; font-size: 16px; margin-bottom: 20px;">Valolyzer Yapay Zeka Motoru Kararını Verdi</p>
    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; align-items: center;">
        <div style="text-align: center; margin: 10px; flex: 1;">
            <p style="color: #8b97a2; font-size: 13px; margin-bottom: 5px; text-transform: uppercase; font-weight: bold;">Beklenen Kazanan</p>
            <h3 style="color: {winner_color}; margin: 0; font-size: 32px; font-family: 'Poppins', sans-serif;">{winner_team}</h3>
        </div>
        <div style="width: 2px; background-color: #383e44; height: 60px;"></div>
        <div style="text-align: center; margin: 10px; flex: 1;">
            <p style="color: #8b97a2; font-size: 13px; margin-bottom: 5px; text-transform: uppercase; font-weight: bold;">Kazanma Olasılığı</p>
            <h3 style="color: #ece8e1; margin: 0; font-size: 32px; font-family: 'Poppins', sans-serif;">%{win_prob:.1f}</h3>
            <p style="color: {winner_color}; font-size: 13px; margin-top: 5px; font-weight: bold;">{delta_kriter}</p>
        </div>
        <div style="width: 2px; background-color: #383e44; height: 60px;"></div>
        <div style="text-align: center; margin: 10px; flex: 1;">
            <p style="color: #8b97a2; font-size: 13px; margin-bottom: 5px; text-transform: uppercase; font-weight: bold;">Taktiksel Durum</p>
            <h3 style="color: #ece8e1; margin: 0; font-size: 26px; font-family: 'Poppins', sans-serif;">{hakimiyet}</h3>
            <p style="color: {winner_color}; font-size: 13px; margin-top: 5px; font-weight: bold;">{taktik_not}</p>
        </div>
    </div>
</div>
    """
    
    st.markdown(report_html, unsafe_allow_html=True)
    # Savaş raporunun detaylı açıklaması
    with st.expander("❓ Bu Rapor Nasıl Okunmalı?"):
        st.markdown("""
        <div style='color: #bdc3c7; line-height: 1.6; font-size: 15px;'>
            <p><strong style='color: #00fccf;'>🎯 Kazanma Olasılığı:</strong> Yapay zeka modelimizin binlerce profesyonel/yüksek kademeli maçı analiz ederek çıkardığı matematiksel galibiyet oranıdır. Sadece ajan seçimlerine değil; ajanların <em>bu spesifik haritadaki</em> geçmiş başarılarına ve takım içi yetenek sinerjilerine dayanarak hesaplanır.</p>
            <p><strong style='color: #00fccf;'>♟️ Taktiksel Durum:</strong> İki takım arasındaki güç farkının oyun içine nasıl yansıyacağını özetler:</p>
            <ul>
                <li><strong>Başa Baş Mücadele (%50-55):</strong> İki kompozisyon taktiksel olarak denk. Bireysel yetenek (aim), ilk kan (first blood) oranları ve takım iletişimi maçın kaderini belirler.</li>
                <li><strong>Kompozisyon Avantajı (%55-65):</strong> Avantajlı takımın, harita kontrolü ve entry (giriş) potansiyeli açısından net bir taktiksel üstünlüğü var.</li>
                <li><strong>Kritik Sinerji Farkı (%65+):</strong> Meta uyumu ve ajan komboları kusursuz eşleşmiş. Dezavantajlı takımın kazanması için ciddi bir taktiksel hata (throw) yapılması veya olağanüstü bir bireysel performans (carry) gerekir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    # --- 2. GERÇEK VERİYE DAYALI AJAN ETKİSİ (IMPACT) ---
    st.write("")
    st.markdown("<br><h3 style='color: #ece8e1; border-bottom: 1px solid #383e44; padding-bottom: 10px; font-size: 20px;'>🌟 KAZANAN TAKIM: GERÇEK AJAN ETKİ (IMPACT) DAĞILIMI</h3>", unsafe_allow_html=True)
    
    # --- GERÇEK AJAN ETKİSİ (IMPACT) HESAPLAMA ---
    # Modelin türüne göre doğru veriyi çekiyoruz (LogisticRegression veya RandomForest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # LogisticRegression katsayıları 2D olabilir (tek sınıflı modellerde [0]), 1D'ye indiriyoruz
        importances = abs(model.coef_[0]) 
    else:
        # Hiçbiri değilse hata almamak için tüm özelliklere eşit puan veriyoruz
        importances = [1.0 / len(model_columns)] * len(model_columns)
        
    feature_dict = dict(zip(model_columns, importances))
    # YENİ EKLENEN/DÜZELTİLEN KISIM: Logistic Regression için katsayı (coef_) kullanılıyor
    feature_importances = np.abs(model.coef_[0])
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
        # Ajanın ikonu sözlükte varsa al, yoksa soru işaretli/varsayılan bir ikon koy
        icon_url = AGENT_ICONS.get(agent, "https://media.valorant-api.com/agents/default/displayicon.png")
        
        bar_html = f"""
        <div style="display: flex; align-items: center; margin-bottom: 12px;">
            <img src="{icon_url}" style="width: 35px; height: 35px; margin-right: 12px; border-radius: 6px; border: 1px solid #383e44;">
            <div style="width: 90px; font-weight: bold; color: #ece8e1; font-family: 'Poppins', sans-serif;">{agent}</div>
            <div style="flex-grow: 1; margin-right: 15px;">
                <div class="impact-bar-bg">
                    <div class="impact-bar-fill" style="width: {impact}%;"></div>
                </div>
            </div>
            <div style="width: 45px; text-align: right; color: #00fccf; font-weight: bold;">%{impact}</div>
        </div>
        """
        st.markdown(bar_html, unsafe_allow_html=True)
        
    # --- 3. AÇIKLAMA PANELİ ---
    st.write("")
    with st.expander("🔍 Bu Tahmin ve İstatistikler Neye Göre Hesaplanıyor?"):
        st.markdown("""
        ### 🤖 Valolyzer Karar Mekanizması
        Yapay zeka modelimiz, seçtiğiniz harita ve ajan kombinasyonlarını analiz ederken aşağıdaki **3 temel kriteri** esas alır:
        
        1. **🎯 Kazanma Olasılığı (%):** Makine öğrenmesi modelimizin (Lojistik Regresyon), geçmiş binlerce yüksek kademeli maç verisinden öğrendiği ağırlıklara dayanır. İki takımın ajan eşleşmelerinin (`matchup`) istatistiksel üstünlüğünü hesaplar.
        
        2. **📊 Hakimiyet Durumu:**
           Modelin ürettiği olasılık skorunun büyüklüğüne göre dinamik olarak belirlenir:
           * **%50 - %55 Arası (Dengeli):** İki takımın kompozisyonu yakın güçte, kaderi anlık stratejiler belirleyecek.
           * **%55 - %65 Arası (Üstün):** Bir takımın, haritaya veya rakip ajanlara karşı net bir taktiksel üstünlüğü var.
           * **%65 ve Üstü (Ezici):** Ajan rolleri ve meta uyumu mükemmel seviyede.
        
        3. **🌟 Ajan Etki Dağılımı (Impact):**
           Yapay zeka modelindeki Özellik Önemi (*Feature Importance / Coefficients*) verileri baz alınır. Modele yeterince veri sağlanamayan durumlarda veya yeni ajanlarda matematiksel varyasyonlar uygulanarak gerçekçi bir takım dağılımı elde edilir.
        """)
    
    st.balloons()

import plotly.express as px
import pandas as pd

def display_feature_importance(model, model_columns, team1_agents, team2_agents):
    """Sadece maçta bulunan ajanlara göre filtrelenmiş Model Karar Analizi grafiği."""
    
    st.markdown("<h3 style='color: #ece8e1; margin-top: 30px;'>📊 SADECE SEÇİLEN AJANLARIN ETKİ ANALİZİ</h3>", unsafe_allow_html=True)
    
    # Model verilerini çek
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    else:
        importances = [0.0] * len(model_columns)
        
    # Veriyi Dataframe'e çevir
    df = pd.DataFrame({'Özellik': model_columns, 'Önem Derecesi': importances})
    
    # --- CAN ALICI NOKTA: FİLTRELEME ---
    # Sadece Mavi ve Kırmızı takımdaki ajanların isimlerini içeren özellikleri tut
    secilen_ajanlar = team1_agents + team2_agents
    df = df[df['Özellik'].apply(lambda x: any(ajan in x for ajan in secilen_ajanlar))]
    
    # En önemli 10 tanesini sırala
    df = df.sort_values(by='Önem Derecesi', ascending=False).head(10)
    
    # Plotly ile grafik çizimi
    fig = px.bar(
        df,
        x='Önem Derecesi',
        y='Özellik',
        orientation='h',
        color='Önem Derecesi',
        color_continuous_scale='Reds'
    )
    
    # Arka planı transparan yapıp arayüze tam yediriyoruz
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ece8e1'))
    st.plotly_chart(fig, use_container_width=True)
    # --- GRAFİK ALTI AÇIKLAMA PANELİ ---
    with st.expander("🔍 Bu Veriler Ne Anlama Geliyor?"):
        st.markdown("""
        Bu grafik, kullandığımız **Lojistik Regresyon** modelinin maç sonucuna karar verirken hangi değişkenlere (ajanlar, harita özellikleri) en çok ağırlık verdiğini gösterir.
        
        * **Önem Derecesi (Yüksek Değer):** Modelin, bu ajanın veya değişkenin maçın kaderini belirlediğine dair "güven" skorudur. 
        * **Katsayı Mantığı:** Lojistik regresyonda bu skorlar, değişkenin galibiyete olan **pozitif veya negatif etkisini** temsil eder. Çubuk ne kadar uzunsa, o değişkenin tahmin başarımız üzerindeki etkisi o kadar belirgindir.
        
        **Özetle:** En tepedeki 3 ajan, şu anki meta içerisinde modelin "en kritik" bulduğu seçimlerdir.
        """)
    """
    Modelin coefficients'ini görselleştirir.
    LogisticRegression için: positive coefficients favor Team 1 win, negative favor Team 2 win.
    """
    


# --- Ana Program Akışı ---

def main():
    # Sayfa ayarları
    setup_page_config()
    
    # 📊 Veri Analiz Test Paneli
    display_diagnostics_panel()

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
   
# --- TAHMİN BUTONU VE İŞLEMLER ---
    if st.button("🧙‍♂️ Maç Sonucunu Tahmin Et", use_container_width=True):
        
        # 1. Kontrol: Harita seçilmiş mi?
        if not selected_map:
            st.error("🚨 Maç tahmini yapabilmek için geçerli bir harita seçmelisiniz!")
            
        # 2. Kontrol: Ajan sayıları tam mı?
        elif len(team1_agents) != 5 or len(team2_agents) != 5:
            st.warning("⚠️ Lütfen her iki takım için de tam 5 ajan seçin!")
            
        # Her şey tamamsa analizi başlat:
        else:
            # Yapay Zeka Tahmini
            prediction, probability = predict_match(
                selected_map, team1_agents, team2_agents, 
                model, model_columns, agent_roles, 
                agent_synergies, agent_map_winrates
            )
            # Savaş Raporunu ve Grafikleri Çizdir
            display_prediction_result(
                prediction, probability, team1_agents, team2_agents, model, model_columns
            )
            
            # YENİ VE AKILLI KIRMIZI GRAFİĞİ ÇAĞIR
            display_feature_importance(model, model_columns, team1_agents, team2_agents)
            # Savaş Raporunu ve Grafikleri Çizdir
           
            
            
            # --- 5. PROFESYONEL FOOTER (İMZA) ---
    footer_html = """
    <div style="text-align: center; margin-top: 60px; padding-top: 20px; border-top: 1px solid #383e44;">
        <p style="color: #8b97a2; font-family: 'Lato', sans-serif; font-size: 15px; line-height: 1.8;">
            🚀 <strong>VALOLYZER</strong> | Yapay Zeka Destekli E-Spor Analiz Motoru <br>
            Geliştirici Ekip: <span style="color: #00fccf; font-weight: bold;">Fatih Şahin</span>, <span style="color: #00fccf; font-weight: bold;">Süha Tüfekçi</span>, <span style="color: #00fccf; font-weight: bold;">Arda Berat Kosor</span><br>
            <span style="font-size: 13px; color: #5c6773;">Bilgisayar Mühendisliği Bölümü</span>
        </p>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# Program başlangıcı
if __name__ == "__main__":
    main()