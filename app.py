import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Chargement du modèle, scaler et des colonnes attendues
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("expected_columns.pkl")

# Dictionnaire pour original_language
lang_dict = {
    'en': 0, 'de': 1, 'fr': 2, 'fi': 3, 'he': 4, 'es': 5, 'zh': 6, 'ja': 7, 'da': 8, 'ko': 9,
    'pl': 10, 'sv': 11, 'it': 12, 'bs': 13, 'hi': 14, 'ru': 15, 'no': 16, 'pt': 17, 'nl': 18, 'el': 19,
    'cs': 20, 'bn': 21, 'cn': 22, 'tn': 23, 'sr': 24, 'mn': 25, 'et': 26, 'uk': 27, 'is': 28, 'ca': 29,
    'ro': 30, 'hu': 31, 'se': 32, 'ps': 33, 'th': 34, 'xx': 35, 'tr': 36, 'vi': 37, 'sh': 38, 'fa': 39,
    'ht': 40, 'bg': 41, 'zu': 42, 'ar': 43, 'mr': 44, 'ku': 45, 'bo': 46, 'ta': 47, 'tl': 48, 'kk': 49,
    'hr': 50, 'id': 51, 'am': 52, 'lt': 53, 'iu': 54, 'pa': 55, 'te': 56, 'sl': 57
}

# Listes
genres_list = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
               'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
               'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western']

languages_list = ['', 'Afrikaans', 'Bahasa indonesia', 'Bahasa melayu', 'Bamanankan',
       'Bosanski', 'Català', 'Cymraeg', 'Dansk', 'Deutsch', 'Eesti', 'English',
       'Español', 'Esperanto', 'Français', 'Gaeilge', 'Galego', 'Hrvatski',
       'Italiano', 'Kinyarwanda', 'Kiswahili', 'Latin', 'Lietuvių', 'Magyar',
       'Malti', 'Nederlands', 'No Language', 'Norsk', 'Polski', 'Português',
       'Pусский', 'Română', 'Slovenčina', 'Slovenščina', 'Somali', 'Srpski',
       'Tiếng Việt', 'Türkçe', 'euskera', 'isiZulu', 'shqip', 'suomi',
       'svenska', 'Èʋegbe', 'Íslenska', 'Český', 'ελληνικά', 'Український',
       'беларуская мова', 'български език', 'қазақ', 'עִבְרִית', 'اردو',
       'العربية', 'فارسی', 'پښتو', 'हिन्दी', 'বাংলা', 'ਪੰਜਾਬੀ', 'தமிழ்',
       'తెలుగు', 'සිංහල', 'ภาษาไทย', 'ქართული', '广州话 / 廣州話', '日本語', '普通话',
       '한국어/조선말']

country_list = ['AE', 'AF', 'AN', 'AR', 'AT', 'AU', 'AZ', 'BA', 'BE', 'BG', 'BO', 'BR',
       'BT', 'BW', 'BY', 'CA', 'CH', 'CL', 'CN', 'CO', 'CU', 'CY', 'CZ', 'DE',
       'DK', 'DZ', 'EE', 'ES', 'ET', 'FI', 'FR', 'GB', 'GR', 'HK', 'HR', 'HT',
       'HU', 'ID', 'IE', 'IL', 'IN', 'IR', 'IS', 'IT', 'JM', 'JP', 'KR', 'KZ',
       'LB', 'LI', 'LR', 'LT', 'LU', 'LV', 'ME', 'MN', 'MT', 'MX', 'NI', 'NL',
       'NO', 'NZ', 'PE', 'PH', 'PK', 'PL', 'PS', 'PT', 'PY', 'RO', 'RS', 'RU',
       'SA', 'SE', 'SG', 'SI', 'SK', 'SU', 'SZ', 'TH', 'TJ', 'TN', 'TR', 'TW',
       'UA', 'US', 'UY', 'UZ', 'VE', 'VI', 'XC', 'XG', 'YU', 'ZA']

# Formulaire
st.title("🎬 Prédiction de la Popularité d’un Film")

# Champs utilisateur
budget = st.number_input("💰 Budget (USD)", min_value=0, value=1_000_000, step=10000)
revenue = st.number_input("💰 Revenue (USD)", min_value=0, value=1_000_000, step=10000)
runtime = st.number_input("⏱️ Durée du film (en minutes)", min_value=0, value=90)
vote_count = st.number_input("🗳️ Nombre de votes", min_value=0, value=100)
vote_average = st.slider("⭐ Note moyenne", 0.0, 10.0, 5.5)
release_year = st.number_input("📆 Année de sortie", min_value=1900, max_value=2100, value=2020)
adult = st.checkbox("🔞 Film adulte ?", value=False)
video = st.checkbox("🎞️ Trailler disponible ?", value=False)

original_language = st.selectbox("🌍 Langue originale", options=list(lang_dict.keys()))
selected_genres = st.multiselect("🎭 Genres", genres_list)
selected_languages = st.multiselect("🗣️ Langues parlées", languages_list)
selected_countries = st.multiselect("🌐 Pays d’origine", country_list)

if st.button("📊 Prédire la popularité"):
    data = {
        'adult': int(adult),
        'video': int(video),
        'budget': budget,
        'revenue': revenue,
        'runtime': runtime,
        'vote_count': vote_count,
        'release_year': release_year,
        'vote_average': vote_average,
        'original_language': lang_dict.get(original_language, -1),
    }

    # Genres
    for genre in genres_list:
        data[genre] = int(genre in selected_genres)

    # Langues parlées
    for lang in languages_list:
        data[lang] = int(lang in selected_languages)

    # Pays
    for c in country_list:
        data[c] = int(c in selected_countries)

    df_input = pd.DataFrame([data])

    # Ajout des colonnes manquantes avec 0
    for col in expected_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    # Réorganisation des colonnes dans le bon ordre
    df_input = df_input[expected_columns]

    df_scaled = scaler.transform(df_input)
    prediction = model.predict(df_scaled)

    st.success(f"📈 Popularité prédite : **{prediction[0]:.2f}**")
