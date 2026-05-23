# Job Classification Dashboard

Dashboard analisis dan klasifikasi lowongan kerja berbasis Machine Learning menggunakan TF-IDF dan LinearSVC yang dibangun dengan Streamlit.

## Project Overview

Proyek ini bertujuan untuk:

- Menganalisis data lowongan kerja
- Mengelompokkan pekerjaan ke dalam kategori tertentu
- Menampilkan visualisasi distribusi kategori pekerjaan
- Menampilkan skill yang paling banyak dicari
- Melakukan prediksi kategori pekerjaan berdasarkan Job Title dan Skill

## Machine Learning Workflow

Analisis Data
↓
Data Cleaning & Preprocessing
↓
Feature Extraction (TF-IDF)
↓
Model Training (LinearSVC)
↓
Model Evaluation
↓
Dashboard Deployment (Streamlit)

## Categories

Model mengklasifikasikan lowongan kerja ke beberapa kategori seperti:

- administration
- backend_dev
- cloud_devops
- customer_service
- data_analyst_bi
- data_science_ai
- design_uiux
- digital_marketing
- finance_accounting
- hr_recruitment
- internship
- management
- mobile_dev
- network_security
- procurement
- sales
- software_dev
- technical_support

## Dataset Features

| Feature | Description |
|----------|------------|
| job_title | Nama posisi pekerjaan |
| job_type | Jenis pekerjaan |
| salary | Informasi gaji |
| job_skill | Skill yang dibutuhkan |
| gaji_perbulan | Estimasi gaji per bulan |
| label | Kategori pekerjaan |

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Plotly
- Pickle

## Project Structure

```text
data-science-job/
│
├── B4.csv
├── B4_model.pkl
├── B4_vectorizer.pkl
├── Penyaringan.py
├── app.py
├── notebook.ipynb
├── README.md
└── requirements.txt
```

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run app.py
```

## Deployment

