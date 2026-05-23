import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# PAGE CONFIG

st.set_page_config(
    page_title="Job Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Job Classification Dashboard")
st.markdown(
    "Dashboard Analisis dan Prediksi Kategori Lowongan Kerja"
)

# LOAD DATA

df = pd.read_csv("B4.csv")

if "final_category" in df.columns:
    category_col = "final_category"

elif "label" in df.columns:
    category_col = "label"

else:
    st.error(
        f"""
        Kolom kategori tidak ditemukan.

        Kolom yang tersedia:
        {df.columns.tolist()}
        """
    )
    st.stop()

#KPI 
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Lowongan",
        len(df)
    )

with col2:
    st.metric(
        "Jumlah Kategori",
        df[category_col].nunique()
    )

with col3:

    if "job_type" in df.columns:

        st.metric(
            "Jenis Pekerjaan",
            df["job_type"].nunique()
        )

    else:

        st.metric(
            "Jenis Pekerjaan",
            "-"
        )

st.divider()

st.subheader("Filter Data")

kategori = st.multiselect(
    "Pilih Kategori",
    sorted(
        df[category_col]
        .dropna()
        .unique()
    )
)

if kategori:

    filtered_df = df[
        df[category_col]
        .isin(kategori)
    ]

else:

    filtered_df = df.copy()

# DATA TABLE
st.subheader("Data Lowongan Kerja")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# DISTRIBUSI KATEGORI

st.subheader(
    "Distribusi Kategori Pekerjaan"
)

kategori_count = (
    filtered_df[category_col]
    .value_counts()
    .reset_index()
)

kategori_count.columns = [
    "Kategori",
    "Jumlah"
]

fig_bar = px.bar(
    kategori_count,
    x="Kategori",
    y="Jumlah",
    text_auto=True,
    title="Jumlah Lowongan per Kategori"
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# PIE CHART

fig_pie = px.pie(
    kategori_count,
    names="Kategori",
    values="Jumlah",
    title="Persentase Kategori Pekerjaan"
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# TOP SKILL

st.subheader("Top Skill")

if "job_skill" in filtered_df.columns:

    skills_text = ", ".join(
        filtered_df["job_skill"]
        .astype(str)
        .tolist()
    )

    skill_list = []

    for item in skills_text.split(","):

        item = item.strip()

        if item:
            skill_list.append(item)

    if len(skill_list) > 0:

        skill_df = (
            pd.Series(skill_list)
            .value_counts()
            .head(10)
            .reset_index()
        )

        skill_df.columns = [
            "Skill",
            "Jumlah"
        ]

        fig_skill = px.bar(
            skill_df,
            x="Skill",
            y="Jumlah",
            text_auto=True,
            title="Top 10 Skill"
        )

        st.plotly_chart(
            fig_skill,
            use_container_width=True
        )

#LOAD MODEL

model = pickle.load(
    open(
        "B4_model.pkl",
        "rb"
    )
)

vectorizer = pickle.load(
    open(
        "B4_vectorizer.pkl",
        "rb"
    )
)

#PREDIKSI

st.divider()

st.subheader(
    "Prediksi Kategori Pekerjaan"
)

job_title = st.text_input(
    "Job Title",
    placeholder="Contoh: Data Analyst"
)

job_skill = st.text_area(
    "Skill",
    placeholder="Contoh: SQL, Python, Tableau"
)

if st.button("Prediksi"):

    if (
        job_title.strip() == ""
        and
        job_skill.strip() == ""
    ):

        st.warning(
            "Masukkan Job Title atau Skill terlebih dahulu."
        )

    else:

        text_input = (
            str(job_title)
            + " "
            + str(job_skill)
        )

        vector = vectorizer.transform(
            [text_input]
        )

        prediction = model.predict(
            vector
        )[0]

        valid_categories = (
            df[category_col]
            .dropna()
            .unique()
            .tolist()
        )

        if prediction not in valid_categories:
            prediction = "general_role"

        st.success(
            f"Kategori Prediksi: {prediction}"
        )

# FOOTER

st.markdown("---")

st.caption(
    "Dashboard Machine Learning Klasifikasi Lowongan Kerja menggunakan TF-IDF dan LinearSVC"
)