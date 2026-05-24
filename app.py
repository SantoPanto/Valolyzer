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
def load_resources():
    """Modeli, sütun yapılarını ve harita istatistiklerini yükler."""
    loaded_model = joblib.load('valorant_rf_model.pkl')
    loaded_columns = joblib.load('model_columns.pkl')
    loaded_map_stats = joblib.load('map_agent_stats.pkl')

    return loaded_model, loaded_columns, loaded_map_stats


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

    # Model sütunlarından map isimlerini çek
    available_maps = sorted([
        col.replace('map_', '')
        for col in model_columns
        if col.startswith('map_')
    ])

    selected_map = st.selectbox(
        "🗺️ Maçın Oynanacağı Haritayı Seçin",
        available_maps
    )

    # Harita bazlı en başarılı ajanlar
    st.markdown(f"### 🔥 {selected_map} Haritasında En Başarılı Ajanlar")

    if selected_map in map_agent_stats:

        top_agents = map_agent_stats[selected_map]

        cols = st.columns(len(top_agents))

        for i, (agent, win_rate) in enumerate(top_agents):
            with cols[i]:
                st.metric(
                    label=agent,
                    value=f"%{win_rate}"
                )

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


def prepare_input_data(selected_map, team1_agents, team2_agents, model_columns):
    """Model için gerekli input dataframe'ini oluşturur."""

    # Tüm feature'ları 0 yap
    input_data = {col: 0 for col in model_columns}

    # Seçilen haritayı aktif et
    map_column = f"map_{selected_map}"

    if map_column in input_data:
        input_data[map_column] = 1

    # Team 1 ajanları
    for agent in team1_agents:

        agent_column = f"{agent}_t1"

        if agent_column in input_data:
            input_data[agent_column] = 1

    # Team 2 ajanları
    for agent in team2_agents:

        agent_column = f"{agent}_t2"

        if agent_column in input_data:
            input_data[agent_column] = 1

    return pd.DataFrame([input_data])


def predict_match(selected_map, team1_agents, team2_agents, model, model_columns):
    """Maç tahmini yapar."""

    input_df = prepare_input_data(
        selected_map,
        team1_agents,
        team2_agents,
        model_columns
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
    """Modelin en önemli feature'larını grafik olarak gösterir."""

    st.markdown("---")
    st.subheader("🧠 Model Bu Kararı Nasıl Verdi?")

    with st.expander("Feature Importance Grafiğini Göster"):

        st.info(
            "Aşağıdaki grafik modelin karar verirken "
            "en çok önem verdiği 15 özelliği göstermektedir."
        )

        importances = model.feature_importances_

        feature_imp_df = pd.DataFrame({
            'Özellik': model_columns,
            'Etki Puanı': importances
        })

        top_features = (
            feature_imp_df
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
    model, model_columns, map_agent_stats = load_resources()

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
                model_columns
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