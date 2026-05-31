import pandas as pd

VALID_MAPS = ['ascent', 'bind', 'haven', 'split', 'lotus', 'sunset', 'abyss', 'icebox', 'fracture', 'breeze', 'pearl']

# İşlenmiş harita verisini oku
df = pd.read_csv('data/processed/maps.csv')

# İçinde geçen gerçek harita adını bulup ayıklayan fonksiyon
def clean_map_name(name):
    name_lower = str(name).lower()
    for valid in VALID_MAPS:
        if valid in name_lower:
            return valid.capitalize() # İlk harfi büyük olarak temiz halini döndür (Örn: Ascent)
    return name

# Temizliği uygula ve üstüne yaz
df['map_name'] = df['map_name'].apply(clean_map_name)
df.to_csv('data/processed/maps.csv', index=False)

print("Harita isimleri başarıyla temizlendi! (Örn: 'BreezePICK57:08' -> 'Breeze')")