import streamlit as st
import pandas as pd
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import mysql.connector
from werkzeug.security import check_password_hash


def get_db_connection():
    db_settings = st.secrets["mysql"]  # Mengambil data dari Streamlit Secrets
    
    return mysql.connector.connect(
        host=db_settings["host"],
        user=db_settings["user"],
        password=db_settings["password"],
        database=db_settings["database"]
    )

# Fungsi autentikasi admin
def authenticate(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT password FROM admin WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    return user and check_password_hash(user['password'], password)

# Fungsi untuk mengambil data obat
def get_drugs():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM datadrug")
    drugs = cursor.fetchall()
    conn.close()
    return drugs

# Fungsi untuk menambah data obat
def add_drug(new_drug_id, new_drug_name, new_condition, new_side_effects, new_benefits_review, new_review, new_comments):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO datadrug (id, urlDrugName, `condition`, sideEffects, benefitsReview, sideEffectsReview, commentsReview) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (new_drug_id, new_drug_name, new_condition, new_side_effects,
         new_benefits_review, new_review, new_comments)
    )
    conn.commit()
    conn.close()


# Load the trained model
svm_model = joblib.load('svmig_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load StandardScaler
vocab = joblib.load('tfidf_vocab.pkl')  # Load vocabulary
# Load selected features based on IG
selected_features = joblib.load('selected_features.pkl')

# Function to clean text
def clean_text(text):
    if pd.isna(text):  # Handle NaN
        return ''
    words = text.split()
    words = [word for word in words if not word.startswith(('http', '@', '#'))]
    text = ' '.join(words)
    text = ''.join(char if char.isalpha() or char.isspace()
                   else ' ' for char in text)
    text = ' '.join(text.split())
    return text.lower()

# Tokenization function
def tokenization(text):
    return text.split()


# Stopword removal
stop_words = {
    'and', 'or', 'but', 'so', 'because', 'if', 'while', 'with', 'at', 'by',
    'for', 'to', 'of', 'in', 'on', 'a', 'an', 'the', 'is', 'it', 'this',
    'that', 'these', 'those', 'i', 'we', 'you', 'he', 'she', 'they', 'me',
    'him', 'her', 'them', 'my', 'our', 'your', 'his', 'their', 'its', 'be',
    'am', 'are', 'was', 'were', 'been', 'can', 'will', 'would', 'could',
    'should', 'do', 'did', 'does', 'have', 'has', 'had'
}


def remove_stopwords(token_list):
    return [word for word in token_list if word.lower() not in stop_words]

# Lemmatization
def lemmatize_text(tokens):
    return [Word(word).lemmatize() for word in tokens]

# Function to preprocess review text
def preprocess_review(review):
    cleaned = clean_text(review)
    tokens = tokenization(cleaned)
    tokens = remove_stopwords(tokens)
    lemmatized = lemmatize_text(tokens)
    return lemmatized

# Function to predict the review sentiment
def svm_predict(text):
    preprocessed_text = preprocess_review(text)

    # Gunakan vocabulary yang sama dari training
    text_tfidf = np.zeros((1, len(vocab)))
    for term in preprocessed_text:
        if term in vocab:
            idx = vocab.index(term)
            text_tfidf[0, idx] = 1  # Menggunakan frekuensi biner

    # Hanya pilih fitur yang dipilih saat training (berdasarkan Information Gain)
    reduced_text_tfidf = text_tfidf[:, [vocab.index(
        f) for f in selected_features if f in vocab]]

    # Pastikan jumlah fitur setelah seleksi sesuai dengan training
    if reduced_text_tfidf.shape[1] != len(selected_features):
        raise ValueError(
            f"Jumlah fitur prediksi ({reduced_text_tfidf.shape[1]}) tidak sesuai dengan jumlah fitur training ({len(selected_features)})")

    # Standardisasi menggunakan scaler yang sama
    text_tfidf_scaled = scaler.transform(reduced_text_tfidf)

    # Prediksi menggunakan model SVM
    prediction = svm_model.predict(text_tfidf_scaled)[0]
    return prediction


def get_all_drugs():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT urlDrugName FROM datadrug")
    drugs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return drugs


def get_drug_info(drug_name):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM datadrug WHERE urlDrugName = %s", (drug_name,))
    drug_info = cursor.fetchall()
    conn.close()
    return drug_info


# Set page config with title and layout
st.set_page_config(page_title="Ask a Medicine", page_icon="💊", layout="wide")
# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f5f7fa;
        }
        .stSidebar {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        .stSidebar .markdown-text-container {
            text-align: center !important;
            width: 100%;
        }
        .sidebar-logo {
            display: block;
            margin: auto;
        }
        .sidebar-title {
            font-size: 20px;
            font-weight: bold;
        }
        .sidebar-subtitle {
            font-size: 16px;
            font-weight: normal;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            border: 2px solid white;
            padding: 8px 15px;
            width: 100%;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
            border: 1px solid #1f77b4;
            padding: 10px;
        }
        .stDataFrameContainer .dataframe {
            text-align: left;
            white-space: normal !important;
            word-wrap: break-word !important;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            word-wrap: break-word;
            white-space: normal;
        }
        th {
            background-color: #1f77b4;
            color: white;
            text-align: center !important;
        }
        td {
            text-align: left;
        }
        .feature-box {
            border: 2px solid #1f77b4;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            background-color: #ffffff;
        }
        .predict-button {
            background-color: #ff6f61 !important;
            color: white;
            border-radius: 8px;
            padding: 5px 10px !important;
            font-size: 14px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Layout
with st.sidebar:
    st.image("logo-unud.png", width=80,
             use_container_width=True, output_format="PNG")
    st.markdown(
        """
        <h3 style="text-align: center; font-size: 16px; font-weight: bold;">
            Kameliya Putri
        </h3>
        <p style="text-align: center; font-size: 16px; font-weight: bold;">
            2108561019
        </p>
        <h3 style="text-align: center; font-size: 18px;">
            PENERAPAN ALGORITMA MULTICLASS SUPPORT VECTOR MACHINE DALAM KLASIFIKASI EFEK SAMPING OBAT BERDASARKAN REVIEW PENGGUNA
        </h3>
        """,
        unsafe_allow_html=True,
    )


# Title of the application
st.title("💊 Ask a Medicine")
st.subheader(
    "Your Personal Assistant for Drug Reviews & Side Effect Predictions")

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "show_login" not in st.session_state:
    st.session_state.show_login = False
if "menu" not in st.session_state:
    st.session_state["menu"] = "home"

# Header dengan tombol login/logout
col1, col2 = st.columns([0.88, 0.12])
with col2:
    if st.session_state.logged_in:
        if st.button("🚪 Logout", help="Click to logout"):
            st.session_state.logged_in = False
            st.session_state.show_login = False  # Reset form login
            st.rerun()
    elif not st.session_state.show_login:  # Pastikan tombol Login hanya muncul jika form login belum terbuka
        if st.button("🔑 Login", help="Click to login admin to add drug data"):
            st.session_state.show_login = True

# Form Login
if st.session_state.show_login:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    # Kolom pertama untuk form dan kolom kedua untuk tombol
    col1, col2 = st.columns([0.85, 0.15])
    with col2:
        # Tombol kecil di sebelah kanan
        if st.button("Login", key="login_button", use_container_width=False):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                # Sembunyikan form login setelah berhasil login
                st.session_state.show_login = False
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau password salah.")


if "show_form" not in st.session_state:
    st.session_state.show_form = False

# Admin Panel - Hanya tampil jika sudah login
if st.session_state.logged_in:
    st.sidebar.header(f"Admin Panel - {st.session_state.username}")

# Menampilkan data obat jika sudah login
if st.session_state.logged_in and not st.session_state.show_form:
    st.subheader("📋 Data Obat")
    # Kolom pertama untuk judul dan kolom kedua untuk tombol
    col1, col2 = st.columns([0.75, 0.25])
    with col2:
        # Tombol kecil di sebelah kanan
        if st.button("➕ Tambah Data Obat", key="add_drug_btn", use_container_width=False):
            st.session_state.show_form = True
            st.rerun()
    drugs = get_drugs()
    if drugs:
        st.dataframe(drugs, use_container_width=True)
    else:
        st.write("❌ Tidak ada data obat.")

# Form input data obat untuk admin
if st.session_state.logged_in and st.session_state.show_form:
    st.subheader("📝 Tambah Data Obat")
    new_drug_id = st.text_input("ID Obat")
    new_drug_name = st.text_input("Nama Obat")
    new_condition = st.text_input("Kondisi")
    new_side_effects = st.text_area("Efek Samping")
    new_benefits_review = st.text_area("Manfaat")
    new_review = st.text_area("Review Efek Samping")
    new_comments = st.text_area("Komentar")

    if st.button("💾 Simpan Data"):
        add_drug(new_drug_id, new_drug_name, new_condition,
                 new_side_effects, new_benefits_review, new_review, new_comments)
        st.success("✅ Data obat berhasil ditambahkan!")
        st.session_state.show_form = False
        st.rerun()

# Menu navigation
if not st.session_state.logged_in and not st.session_state.show_login and not st.session_state.show_form:
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Home"):
            st.session_state["menu"] = "home"
    with col2:
        if st.button("Review Side Effect"):
            st.session_state["menu"] = "side_effect"
    with col3:
        if st.button("Review Classification"):
            st.session_state["menu"] = "classification"

# Display content based on selected button
if not st.session_state.logged_in and not st.session_state.show_login and not st.session_state.show_form:
    if st.session_state["menu"] == "home":
        st.header("📋 List of Available Drugs")
        drug_list = get_all_drugs()
        if drug_list:
            grouped_drugs = {}
            for drug in drug_list:
                initial = drug[0].upper()
                if initial not in grouped_drugs:
                    grouped_drugs[initial] = []
                grouped_drugs[initial].append(drug)

            for initial, drugs in sorted(grouped_drugs.items()):
                st.subheader(f"🔤 {initial}")

                # Buat 5 kolom
                cols = st.columns(5)

                # Tampilkan obat secara horizontal dalam 5 kolom
                for i, drug in enumerate(drugs):
                    cols[i % 5].markdown(f"- *{drug}*")
        else:
            st.warning("⚠️ No medicines found.")

    # Display content based on selected button
    elif st.session_state["menu"] == "side_effect":

        st.header("Review Side Effect")
        drug_name = st.text_input(
            "🔎 Search for a Drug", placeholder="Enter drug name...")

        if st.button("🔍 Search", key="search_button", help="Click to search information about drug"):
            if drug_name:
                # Ambil data berdasarkan drug_name
                drug_info = get_drug_info(drug_name)

                if drug_info:  # Pastikan data tidak kosong
                    df_drug_info = pd.DataFrame(
                        drug_info)  # Konversi ke DataFrame

                    st.subheader(f"📝 Drug Review for: {drug_name}")

                    # Pilih hanya kolom yang diperlukan
                    required_columns = ["condition", "sideEffects",
                                        "sideEffectsReview", "commentsReview"]

                    # Pastikan kolom ada dalam DataFrame sebelum digunakan
                    available_columns = [
                        col for col in required_columns if col in df_drug_info.columns]

                    if set(available_columns) == set(required_columns):
                        review_data = df_drug_info[available_columns].copy()
                        review_data.reset_index(drop=True, inplace=True)
                        review_data.index += 1  # Set index mulai dari 1
                        review_data.insert(0, "No", review_data.index)

                        # Konversi DataFrame ke HTML dengan CSS agar rapi
                        html_table = review_data.to_html(
                            escape=False, index=False, classes="table table-striped")

                        # Tambahkan CSS agar header kolom rata tengah
                        html_table = html_table.replace(
                            '<th>', '<th style="text-align:center;">')

                        st.markdown(html_table, unsafe_allow_html=True)
                    else:
                        st.warning(
                            "⚠️ Some expected columns are missing in the database.")
                else:
                    st.warning("⚠️ No drug found with that name.")

        st.markdown("</div>", unsafe_allow_html=True)

    elif st.session_state["menu"] == "classification":

        st.header("Review Classification")
        review_text = st.text_area(
            "📝 Enter a drug review", placeholder="Type your review here...")

        if st.button("🔍 Predict", key="predict_button", help="Click to predict side effect severity"):
            if review_text:
                prediction = svm_predict(review_text)
                st.success(
                    f"✅ *Predicted Side Effect Severity:* {prediction} Side Effect")
            else:
                st.error("⚠️ Please enter a review first.")

        st.markdown("</div>", unsafe_allow_html=True)
