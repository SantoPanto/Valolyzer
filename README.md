# Valolyzer: AI-Powered Valorant Esports Prediction Engine

> Predict Valorant esports match outcomes with cutting-edge machine learning, fuzzy team matching, and advanced feature engineering.

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## What is Valolyzer?

Valolyzer is a **machine learning-powered Valorant esports prediction engine** that combines intelligent web scraping, advanced feature engineering, and symmetric ML training to deliver unbiased match outcome predictions. Built by Computer Engineering students, this project demonstrates professional data science, web scraping, and full-stack ML engineering practices.

---

## Key Features

### **Custom VLR.gg Web Scraper**
- Automatically extracts live match data from VLR.gg (Valorant's premier esports platform)
- Captures comprehensive match information, maps played, player stats, and team compositions
- Handles pagination and data consistency validation

### **Fuzzy String Matching (AI-Powered E-Sports Team Matcher)**
- Dynamically matches abbreviated or mistyped team names (e.g., "FNC" → "FNATIC", "T1" → "T1")
- Uses `difflib.SequenceMatcher` intelligent string comparison
- **Prevents critical data loss** during dataframe merges and ensures data integrity
- Handles real-world esports naming inconsistencies gracefully

### **Smart Data Filtering**
- Automatically filters out bugged 0-0 VLR scores and incomplete records
- Intelligently calculates true map winners and match outcomes
- Validates data quality before model training

### **Advanced Feature Engineering**
- **Agent Differentials**: Quantifies team agent pool advantages
- **Role Differentials**: Analyzes agent role composition (Duelist, Sentinel, Controller, Initiator, Flex)
- **Agent Synergies**: Captures synergistic agent combinations (e.g., Jett + Sova, Astra + Sage)
- **Historical Team Statistics**: Win rates, recent form, head-to-head records
- Multi-dimensional feature space for powerful predictions

### **Symmetric ML Training**
- Uses **Logistic Regression** with a **mirrored dataset** approach
- Ensures **side-independent, unbiased predictions** (eliminates Attacker/Defender bias)
- Model trained on both original and flipped team perspectives
- Provides confidence scores alongside predictions

### **Modern Streamlit Dashboard**
- Stunning UI with custom CSS and animations
- **Agent Impact Analysis**: Visual comparison of agent selections
- **Dynamic Map Statistics**: Real-time map winrates and team performance
- Interactive prediction interface
- Professional data visualization

---

## Project Architecture

```
Valolyzer/
├── scrapers/                  # Web scraping modules
│   ├── base.py               # Base scraper class
│   ├── vlr/
│   │   └── vlr_scraper.py    # VLR.gg scraper implementation
│   ├── rib/                  # RIB.GG scraper
│   └── tracker/              # Tracker.gg scraper
│
├── pipelines/                # Data processing pipelines
│   └── main_pipeline.py      # Main ETL pipeline
│
├── data/                     # Data storage
│   ├── raw/                  # Raw scraped data
│   ├── processed/            # Cleaned and processed data
│   │   ├── matches.csv
│   │   ├── player_stats.csv
│   │   ├── maps.csv
│   │   ├── compositions.csv
│   │   └── rounds.csv
│   ├── parquet/              # Parquet format files
│   └── debug/                # Debug data snapshots
│
├── utils/                    # Utility functions
│   ├── csv_handler.py        # CSV operations
│   ├── models.py             # Data models
│   ├── normalizers.py        # Data normalization
│   ├── validators.py         # Data validation
│   ├── logging.py            # Logging utilities
│   ├── http.py               # HTTP utilities
│   └── debug.py              # Debug utilities
│
├── database/                 # Database operations
│
├── app.py                    # Streamlit dashboard application
├── main.py                   # Main entry point
├── retrain_valolyzer.py      # Model retraining script
├── config.py                 # Configuration management
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Installation & Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (for data fetching)

### Step 1: Install Dependencies
Install all required Python packages:
```bash
pip install -r requirements.txt
```

### Step 2: Fetch Fresh Data
Scrape the latest match data from VLR.gg:
```bash
python3 -m pipelines.main_pipeline
```
This will automatically:
- Connect to VLR.gg
- Extract match results and team compositions
- Store raw data in `data/raw/`

### Step 3: Clean, Match, & Train Model
Process data with fuzzy team matching and train the ML model:
```bash
python3 retrain_valolyzer.py
```
This performs:
- Data cleaning and validation
- Fuzzy team name matching
- Feature engineering (agent differentials, synergies, etc.)
- Symmetric ML model training
- Model serialization for predictions

### Step 4: Launch the Dashboard
Start the interactive Streamlit application:
```bash
streamlit run app.py
```
The dashboard will open in your default browser at `http://localhost:8501`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Web Scraping** | BeautifulSoup4, Requests |
| **Frontend/Dashboard** | Streamlit |
| **Data Format** | CSV, Parquet, JSON |

---

## How It Works

1. **Data Collection**: Custom web scraper harvests match data from VLR.gg
2. **Data Cleaning**: Smart filters remove corrupted records and normalize team names using fuzzy matching
3. **Feature Engineering**: Advanced calculations generate predictive features (agent synergies, differentials, etc.)
4. **Model Training**: Logistic Regression trained on mirrored dataset ensures unbiased predictions
5. **Predictions**: Dashboard accepts team matchups and delivers ML-powered outcome predictions with confidence scores
6. **Continuous Learning**: Automated retraining keeps model updated with latest esports meta

---

## Performance & Metrics

- **Data Coverage**: Multiple esports platforms (VLR.gg, RIB.GG, Tracker.gg)
- **Feature Dimensions**: 20+ engineered features per match
- **Model Type**: Symmetric Logistic Regression with side-balancing
- **Prediction Confidence**: Probability-based scores (0-1)
- **Data Freshness**: Automated pipeline for continuous updates

---

## Project Credits

**Development Team:**
- **Fatih Şahin** - Lead Developer
- **Süha Tüfekçi** - Data Engineer
- **Arda Berat Kosor** - ML Engineer

*Computer Engineering Program*

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Related Resources

- [Valorant Official Site](https://www.valorant.com/)
- [VLR.gg - Esports Platform](https://www.vlr.gg/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-Learn ML Documentation](https://scikit-learn.org/)

---

## Support & Contributions

For questions, bug reports, or contributions:
1. Check existing documentation in the `docs/` folder
2. Review the project architecture and code structure
3. Submit issues or pull requests via GitHub

---

## Model Dosyaları Hakkında Bilgilendirme
Projemizde kullandığımız eğitilmiş makine öğrenmesi model dosyalarının (`.pkl`) boyutları oldukça küçük olduğu için harici bir indirme bağlantısına gerek duyulmamıştır. Tüm model dosyaları doğrudan teslim edilen proje klasörünün (.zip) içerisine dahil edilmiştir ve anında çalıştırılmaya hazırdır.

---

## Örnek Giriş ve Çıktı (Sample Input & Output)

Sistemimizin tahminleme modeline gönderilen örnek veri formatı ve modelin ürettiği çıktı şu şekildedir:

**Örnek Giriş (Input):**
Kullanıcı Streamlit arayüzünden iki takım seçtiğinde arka planda modele şu özellikler (features) gönderilir:
- Takım 1: Mavi Takım(Savunanlar)
- Takım 2: Kırmızı Takım(Saldıranlar)
- Seçilen Harita: Ascent

**Örnek Çıktı (Output):**
Model, bu verileri işledikten sonra takımların kazanma olasılıklarını ve güven skorunu döndürür:
- Kazanan Tahmini: Mavi Takım(Savunanlar)
- Kazanma İhtimali (Confidence Score): %68.5
- Taktiksel Durum: Üstün / Kritik Sinerji Farkı 

--- 

<div align="center">

**Built with for the Valorant Esports Community**

If this project helps you, consider giving it a star!

</div>
