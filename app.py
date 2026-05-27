import streamlit as st
import joblib
import pandas as pd
import numpy as np

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

    st.markdown(f"### 🔥 {selected_map} Haritasında En Başarılı Ajanlar")

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
        st.header("🛡️ Team 1")

        team1_agents = st.multiselect(
            "Team 1 Ajanlarını Seçin (Tam 5 Adet)",
            VALORANT_AGENTS,
            max_selections=5,
            key="t1"
        )

    with col2:
        st.header("⚔️ Team 2")

        team2_agents = st.multiselect(
            "Team 2 Ajanlarını Seçin (Tam 5 Adet)",
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


def display_prediction_result(prediction, probability):
    """Tahmin sonucunu kullanıcıya gösterir."""

    st.markdown("---")

    if prediction == 1:

        st.success(
            f"🏆 Tahmin: Team 1 Kazanır! "
            f"(Kazanma İhtimali: %{probability[1] * 100:.1f})"
        )

        st.balloons()

    else:

        st.error(
            f"💀 Tahmin: Team 2 Kazanır! "
            f"(Team 1 Kaybetme İhtimali: %{probability[0] * 100:.1f})"
        )


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
                probability
            )

            # Feature importance grafiği
            display_feature_importance(
                model,
                model_columns
            )


# Program başlangıcı
if __name__ == "__main__":
    main()