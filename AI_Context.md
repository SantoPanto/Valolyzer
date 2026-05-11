# Valolyzer: E-Spor Karşılaşma Tahmin Modeli - Proje Bağlamı (AI Context)

Bu doküman, başka bir Yapay Zeka asistanına bu projenin (Valolyzer) amacını, teknik altyapısını, dosya mimarisini ve çalışma mantığını tek seferde kavratmak için hazırlanmıştır. Bu metni yeni bir sohbete kopyalayıp yapıştırarak yapay zekanın projeye tam hakim olmasını sağlayabilirsiniz.

---

## 📌 Projenin Amacı ve Özeti
**Proje Adı:** E-Spor Analitiği: Taktiksel Oyunlar İçin Karşılaşma Tahmin Modeli (Valolyzer)
**Ders:** Python Programlamaya Giriş
**Temel Hedef:** Profesyonel Valorant karşılaşmalarında, takımların seçtikleri ajan kompozisyonlarına (karakterlere) ve haritanın serideki sırasına bakarak, maçın galibini ve kazanma olasılıklarını istatistiksel olarak tahmin eden bir makine öğrenmesi uygulaması geliştirmek.

## 🛠️ Teknoloji Yığını (Tech Stack)
- **Dil:** Python
- **Veri İşleme & Analizi:** Pandas, NumPy
- **Makine Öğrenmesi (Model Eğitimi):** Scikit-Learn (Random Forest Classifier)
- **Arayüz (Frontend):** Streamlit
- **Model Kayıt/Yükleme:** Joblib

## 📂 Dosya ve Dizin Mimarisi
Proje, temel olarak veri eğitimi ve arayüz sunumu olmak üzere iki ana bacaktan oluşur:
1. `01_veri_incelemesi.ipynb`: Ham verinin işlendiği, keşifçi veri analizi (EDA) yapıldığı, ajan/harita gibi kategorik verilerin **One-Hot Encoding** yöntemi ile sayısallaştırıldığı ve `RandomForestClassifier` algoritmasının eğitildiği çalışma dosyası.
2. `valorant_rf_model.pkl`: Jupyter notebook'ta eğitilip dışa aktarılmış, kararları veren ana model dosyası.
3. `model_columns.pkl`: Modelin eğitim sırasında gördüğü tüm veri sütunlarının listesi. (Tahmin aşamasında kullanıcı verilerini modele uyumlu hale getirmek için kullanılıyor).
4. `app.py`: Streamlit ile yazılmış, modüler fonksiyonlardan oluşan web uygulaması (Arayüz).
5. `Proje_Raporu.md` / `Ödev.md`: Projenin akademik isterlerini ve yazılı rapor formatını tutan belgeler.

## ⚙️ Uygulamanın Çalışma Mantığı ve Veri Akışı (app.py)
Uygulama tam modüler bir yapıda (`main()` fonksiyonu etrafında) kurulmuştur. Süreç şu şekilde işler:
1. **Girdi (Input):** Kullanıcı, Streamlit arayüzünden `map_order` (Harita Sırası: 1, 2 veya 3) seçer. Ardından Team 1 ve Team 2 için tam olarak **beşer adet ajan** seçer.
2. **Veri Dönüşümü:** `predict_match()` fonksiyonu içerisinde, `model_columns.pkl` listesinde bulunan tüm sütun adları için içi "0" ile dolu bir Python sözlüğü (dict) yaratılır. 
3. **One-Hot Encoding Simülasyonu:** Kullanıcının girdiği `map_order` doğrudan işlenir. Seçilen ajanlar ise şablon üzerinde `AjanAdı_t1 = 1` ve `AjanAdı_t2 = 1` olacak şekilde güncellenir.
4. **Hesaplama:** Oluşturulan 0 ve 1'lerden ibaret veri şablonu bir Pandas DataFrame'ine çevrilir ve `valorant_rf_model.pkl` dosyasına beslenir.
5. **Çıktı (Output):** Modelin `predict()` fonksiyonu ile kazanan takım (1 veya 0), `predict_proba()` fonksiyonu ile de kazanma yüzdeleri alınarak arayüzde gösterilir. Ayrıca `model.feature_importances_` değeri okunarak, galibiyette en çok rol oynayan ilk 15 özelliğin Bar Chart grafiği (Streamlit üzerinden) çizdirilir.

## 🤖 Yapay Zekaya (Sana) Talimatlar
Eğer bu projeyle ilgili benden (kullanıcıdan) bir geliştirme, hata çözümü veya özellik ekleme talebi alırsan şunlara dikkat etmelisin:
- Kodlar "Python Programlamaya Giriş" seviyesi ödevi olsa da profesyonel modülerlikte (`if __name__ == '__main__':` bloğu ve ayrıştırılmış fonksiyonlar ile) yazılmıştır; bu mimariyi bozma.
- Yeni bir UI elemanı eklenecekse, Streamlit'in standartlarını (örn. `st.columns`, `st.sidebar`) kullanarak `app.py` içerisindeki uygun fonksiyona entegre et.
- Makine öğrenmesi modeli veya veriseti değişmediği sürece `predict_match()` içerisindeki One-Hot Encoding sözlük (dict) oluşturma mantığına dokunma.
- Önceliğin, kodların her zaman hatasız çalışması ve kullanıcı arayüzünde temiz görünmesidir.

*(Bu bağlam notu projeye tam adaptasyon sağlaman içindir, şimdi nasıl yardımcı olmamı istersin?)*
