










## Python Programlamaya Giriş Dersi




## Grup Üyeleri

## GRUP-12

## 1. Süha Tüfekçi / 032490079
## 2. Fatih Şahin / 032490059
## 3. Arda Berat Kosor / 032390048














## Proje Başlığı:
E-Spor Analitiği: Taktiksel Oyunlar İçin Karşılaşma Tahmin Modeli
## Proje Konusu:
Profesyonel e-spor (özellikle Valorant) karşılaşmalarında; maç öncesi ve maç içi dinamik
verilerin (harita seçimleri, ajan seçimleri, ekonomi yönetimi ve ilk kan oranları gibi)
işlenerek, maç sonuçlarının ve anlık kazanma olasılıklarının makine öğrenmesi
algoritmalarıyla istatistiksel olarak tahmin edilmesi.
## Gerçek Hayat Problemi:
E-spor, devasa bir endüstri haline gelmesine rağmen, geleneksel sporlardaki (futbol,
basketbol) gelişmiş veri analitiği araçlarına kıyasla hala geridedir. Profesyonel takımlar,
stratejilerini ve "meta" okumalarını çoğunlukla oyuncu sezgilerine veya kısıtlı istatistiklere
dayandırmaktadır.
Bu proje hangi ihtiyaca cevap veriyor?
Bu proje, "Hangi haritada hangi kompozisyonun istatistiksel olarak kazanma ihtimali daha
yüksek?" sorusuna nesnel, veriye dayalı ve algoritmik bir cevap vererek takımların analitik
strateji kurma ihtiyacını karşılar.
Nerelerde kullanılabilir?
- E-Spor Takımları: Koçlar ve analistler tarafından rakip takımın zayıflıklarını bulmak
ve maç öncesi strateji belirlemek için.
- Canlı Yayınlar (Broadcasting): Turnuva yayıncıları tarafından izleyicilere ekranda
"Anlık Kazanma Olasılığı (Win Probability)" grafikleri sunmak için.
- Analiz Platformları: E-spor veri tabanları, fantezi ligleri veya tahmin yürüten
platformlar için altyapı motoru olarak.
## Projenin Amacı:
Yüksek hacimli ve karmaşık e-spor verilerini anlamlandırarak, maç sonuçlarını yüksek
doğruluk payıyla tahmin edebilen bir makine öğrenmesi modeli geliştirmek. Böylece espor
ekosistemindeki veri eksikliğini gidermek ve oyundaki taktiksel kararların matematiksel bir
temele oturtulmasını sağlamak.



Projede Kullanılacak Yöntemler:   Web Scraping (BeautifulSoup/Selenium) veya API
entegrasyonu ile profesyonel maç verilerinin toplanması.
- Veri Ön İşleme (Data Preprocessing): Toplanan maç verileri, (Jupyter
Notebook) üzerinde pandas kütüphanesi kullanılarak temizlenmiş ve analiz edilmiştir.
Ajan ve harita isimleri gibi kategorik değişkenler makine öğrenmesine uygun hale
getirilmesi için sayısal formatlara (One-Hot Encoding) dönüştürülmüştür.
- Modelleme ve Kayıt: scikit-learn kütüphanesi kullanılarak Random Forest
algoritması eğitilmiştir. Eğitilen model ve modelin sütun yapısı, web uygulamasında
kullanılmak üzere joblib kütüphanesi ile dışa aktarılarak kaydedilmiştir.
- Web Arayüzü Geliştirme: Kullanıcıların takımları ve haritayı seçip anlık tahmin
alabilmesi için Streamlit kütüphanesi kullanılarak interaktif bir web arayüzü
oluşturulmuştur.
## Program Mimarisi:
- Veri Katmanı (Data Layer):  “01_veri_incelemesi.ipynb “dosyasının bulunduğu
kısımdır. Ham verilerin “Kaggle” platformu üzerinden temin edilip veri seti yapısı,
“Riot Games API” standartlarına uygun şekilde işlendiği, keşifçi veri analizinin (EDA)
yapıldığı ve Random Forest modelinin eğitildiği katmandır.
- İş Mantığı ve İşlem Katmanı (Processing & Logic Layer): Eğitilmiş yapay
zeka modelinin tutulduğu katmandır. Eğitim katmanından çıkan
“valorant_rf_model.pkl” (modelin kendisi) ve “model_columns.pkl” (veri sütun
şablonları) dosyaları bu katmanı oluşturur. Sunum katmanından gelen verileri işleyip
matematiksel kazanma olasılıklarını hesaplar.
- Sunum Katmanı (Presentation Layer): app.py dosyası ile temsil edilen
katmandır. Streamlit altyapısı ile çalışır. Kullanıcıdan harita ve ajan kompozisyonu
seçimlerini alır, arka planda (Logic Layer) modeli çalıştırır ve "Hangi Takım Kazanır?"
sonucunu, modelin karar verme ağırlıklarıyla birlikte ekrana görsel olarak yansıtır.
Gerekli kütüphane bağımlılıkları “requirements.txt” dosyasında tanımlanmıştır.
Projenin Beklenen Çıktısı:   Geçmiş profesyonel maç verileriyle eğitilmiş; iki takım karşı
karşıya geldiğinde harita, ajan seçimleri ve takım istatistiklerini girdi (input) olarak alıp,
maçın hangi takım tarafından kazanılacağını yüzdelik bir olasılıkla çıktı (output) olarak
verebilen, doğruluğu test edilmiş bir yapay zeka algoritması yazılımı.
Github Reposu İçin Link:   https://github.com/SantoPanto/Valolyzer