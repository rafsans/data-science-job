import pandas as pd
import re
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("\n[INFO] START PIPELINE")
print("Working dir:", os.getcwd())

# =========================================
# LOAD DATA
# =========================================

df_main = pd.read_csv("analisis.csv")

print("\n[INFO] Dataset Loaded")
print("Shape:", df_main.shape)

# =========================================
# RENAME COLUMN
# =========================================

df_main = df_main.rename(columns={
    "Job_Title": "job_title",
    "Job_Type": "job_type",
    "Salary": "salary",
    "Job_Skill": "job_skill",
    "Gaji_Perbulan": "gaji_perbulan"
})

# =========================================
# PILIH KOLOM
# =========================================

df_main = df_main[
    [
        "job_title",
        "job_type",
        "salary",
        "job_skill",
        "gaji_perbulan"
    ]
]

# =========================================
# CLEAN DATA
# =========================================

df_main = df_main.fillna("")

df_main = df_main.drop_duplicates()

df_main = df_main.drop_duplicates(
    subset=[
        "job_title",
        "job_skill"
    ]
)

# =========================================
# CLEAN TEXT
# =========================================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = text.encode(
        "ascii",
        "ignore"
    ).decode()

    words = re.findall(
        r"[a-zA-Z][a-zA-Z0-9+#/. -]*",
        text
    )

    text = " ".join(words)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =========================================
# APPLY CLEANING
# =========================================

df_main["job_title"] = (
    df_main["job_title"]
    .astype(str)
    .apply(clean_text)
)

df_main["job_skill"] = (
    df_main["job_skill"]
    .astype(str)
    .apply(clean_text)
)

# =========================================
# COMBINE TEXT
# =========================================

df_main["clean_text"] = (
    df_main["job_title"]
    + " "
    + df_main["job_skill"]
)

# =========================================
# CATEGORY KEYWORDS
# =========================================
CATEGORY_KEYWORDS = {

    "frontend_dev": [
        "frontend", 
        "front end", 
        "react", 
        "vue",
        "angular", 
        "html", 
        "css", 
        "javascript",
        "typescript", 
        "ui developer"
    ],

    "backend_dev": [
        "backend", 
        "back end", 
        "java", 
        "spring",
        "node", 
        "express", 
        "laravel", 
        "django",
        "flask", 
        "php", 
        "golang", 
        "api", 
        ".net"
    ],

    "mobile_dev": [
        "android", 
        "ios", 
        "flutter",
        "react native", 
        "kotlin", 
        "swift"
    ],

    "software_dev": [
        "software engineer",
        "software developer",
        "programmer",
        "web developer",
        "fullstack",
        "full stack",
        "application developer",
        "qa analyst",
        "quality assurance",
        "software tester",
        "test engineer",
        "qa engineer",
        "automation tester",
        "manual tester",
        "quality control",
        "software quality assurance",
        "qa qc",
        "testing engineer",
        "quality engineer"
    ],

    "data_science_ai": [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai engineer",
        "data scientist",
        "nlp",
        "computer vision",
        "predictive modeling",
        "statistical modeling",
        "data mining"
    ],

    "data_analyst_bi": [
        "data analyst",
        "business analyst",
        "research analyst",
        "reporting analyst",
        "market analyst",
        "bi analyst",
        "power bi",
        "tableau",
        "data visualization",
        "report analyst",
        "reporting",
        "mis analyst",
        "business intelligence",
        "analytics",
        "kpi",
        "dashboard"
    ],

    "cloud_devops": [
        "aws",
        "azure",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "devops",
        "jenkins",
        "cloud engineer",
        "site reliability engineer",
        "data engineer",
        "platform engineer",
        "infrastructure engineer",
        "sre",
        "linux",
        "unix",
        "openshift",
        "ansible",
        "ci/cd",
        "gitlab",
        "github actions",
        "helm",
        "prometheus",
        "grafana",
        "system administrator",
        "linux administrator",
        "server administrator"
    ],

    "network_security": [
        "network engineer",
        "security engineer",
        "cybersecurity",
        "network security",
        "firewall",
        "siem",
        "penetration testing",
        "soc analyst",
        "information security",
        "iso 27001",
        "vulnerability assessment"
    ],

    "technical_support": [
        "technical engineer",
        "service engineer",
        "field engineer",
        "system engineer",
        "technical support",
        "technical specialist"
    ],

    "digital_marketing": [
        "digital marketing",
        "seo",
        "sem",
        "social media",
        "content creator",
        "google ads",
        "meta ads"
    ],

    "finance_accounting": [
        "finance",
        "accounting",
        "tax",
        "taxation",
        "accountant",
        "auditor",
        "bookkeeping",
        "financial analyst",
        "finance analyst",
        "payroll",
        "accounts payable",
        "accounts receivable",
        "treasury",
        "controller",
        "accounts executive",
        "accounts assistant",
        "account assistant",
        "audit assistant",
        "audit associate",
        "finance assistant",
        "finance executive",
        "account clerk",
        "accounts clerk"
    ],

    "hr_recruitment": [
        "human resource",
        "hr",
        "recruitment",
        "talent acquisition",
        "recruiter",
        "hr generalist",
        "training",
        "human capital",
        "people operations",
        "people development",
        "hrbp",
        "organizational development",
        "talent management",
    ],

    "sales": [
        "sales",
        "sales executive",
        "account executive",
        "business development",
        "business executive",
        "sales consultant",
        "sales representative",
        "relationship manager",
        "account manager",
        "sales advisor",
        "sales promoter"
    ],

    "customer_service": [
        "customer service",
        "customer support",
        "call center",
        "helpdesk",
        "service desk",
        "it support"
    ],

    "administration": [
        "administrator",
        "administration",
        "office admin",
        "administrative assistant",
        "admin staff",
        "office staff",
        "executive assistant",
        "clerical",
        "office assistant",
        "administrative officer",
        "general clerk",
        "administration executive",
        "office coordinator",
        "receptionist",
        "secretary"
    ],

"management": [
    "project manager",
    "product manager",
    "operations manager",
    "general manager",
    "branch manager",
    "department head",
    "head of",
    "team lead",
    "project coordinator"
    ],

    "internship": [
        "intern",
        "internship",
        "trainee"
    ],

    "procurement": [
        "procurement",
        "purchasing",
        "buyer",
        "sourcing",
        "supply chain",
        "logistics",
        "warehouse",
        "vendor management",
        "inventory",
        "inventory control",
        "inventory management",
        "supply planning",
        "demand planning",
        "warehouse officer",
        "logistic officer",
        "logistics coordinator"
    ],

    "design_uiux": [
        "ui ux",
        "ux designer",
        "ui designer",
        "graphic designer",
        "product designer",
        "visual designer"
    ]
}

# =========================================
# CATEGORY MAPPING
# =========================================
def map_category(title, skill):

    title = str(title).lower()
    skill = str(skill).lower()

    text = f"{title} {skill}"

    # =====================================
    # PRIORITAS TITLE (PALING PENTING)
    # =====================================

    if "data analyst" in title:
        return "data_analyst_bi"

    if "business analyst" in title:
        return "data_analyst_bi"

    if "reporting analyst" in title:
        return "data_analyst_bi"

    if "financial analyst" in title:
        return "finance_accounting"

    if "finance analyst" in title:
        return "finance_accounting"

    if "human capital" in title:
        return "hr_recruitment"

    if "hr manager" in title:
        return "hr_recruitment"

    if "data engineer" in title:
        return "cloud_devops"

    if "network engineer" in title:
        return "network_security"

    if "security engineer" in title:
        return "network_security"

    if "infrastructure engineer" in title:
        return "cloud_devops"
        
    if "cloud engineer" in title:
        return "cloud_devops"

    if "software engineer" in title:
        return "software_dev"

    if "qa analyst" in title:
        return "software_dev"

    if "quality assurance" in title:
        return "software_dev"

    if "software tester" in title:
        return "software_dev"

    if "accounting analyst" in title:
        return "finance_accounting"

    if "sales analyst" in title:
        return "sales"

    # =====================================
    # KEYWORD SCORING
    # =====================================

    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():

        score = 0

        for kw in keywords:

            if kw in text:

                if kw in title:
                    score += 3
                else:
                    score += 1

        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)

    # =====================================
    # FALLBACK
    # =====================================
    
    if "infrastructure engineer" in title:
        return "cloud_devops"

    if "developer" in text:
        return "software_dev"

    if "programmer" in text:
        return "software_dev"

    if "data scientist" in text:
        return "data_science_ai"

    if "machine learning" in text:
        return "data_science_ai"

    if "analyst" in text:
        return "data_analyst_bi"

    if "business intelligence" in text:
        return "data_analyst_bi"

    if "cloud" in text:
        return "cloud_devops"

    if "devops" in text:
        return "cloud_devops"

    if "security" in text:
        return "network_security"

    if "network" in text:
        return "network_security"

    if "service engineer" in text:
        return "technical_support"

    if "field engineer" in text:
        return "technical_support"

    if "technical engineer" in text:
        return "technical_support"

    if "support" in text:
        return "customer_service"

    if "service" in text:
        return "customer_service"

    if "marketing" in text:
        return "digital_marketing"

    if "finance" in text:
        return "finance_accounting"

    if "accounting" in text:
        return "finance_accounting"

    if "tax" in text:
        return "finance_accounting"

    if "hr" in text:
        return "hr_recruitment"

    if "recruit" in text:
        return "hr_recruitment"

    if "sales" in text:
        return "sales"

    if "manager" in text:
        return "management"

    if "intern" in text:
        return "internship"

    if "designer" in text:
        return "design_uiux"

    if "procurement" in text:
        return "procurement"

    if "logistics" in text:
        return "procurement"

    if "warehouse" in text:
        return "procurement"

    if "engineer" in text:
        return "technical_support"

    if "manager" in text:
        return "management"

    if "admin" in text:
        return "administration"

    return "administration"

# =========================================
# LABELING
# =========================================

df_main["label"] = df_main.apply(
    lambda x: map_category(
        x["job_title"],
        x["job_skill"]
    ),
    axis=1
)

# =========================================
# ONLY VALID LABELS
# =========================================

allowed_labels = [
    "frontend_dev",
    "backend_dev",
    "mobile_dev",
    "software_dev",
    "data_science_ai",
    "data_analyst_bi",
    "cloud_devops",
    "network_security",
    "technical_support",
    "digital_marketing",
    "finance_accounting",
    "hr_recruitment",
    "sales",
    "customer_service",
    "administration",
    "management",
    "internship",
    "procurement",
    "design_uiux"
]

df_main = df_main[
    df_main["label"].isin(
        allowed_labels
    )
]

print("\n[INFO] LABEL DISTRIBUTION\n")
print(df_main["label"].value_counts())

# =========================================
# TRAINING DATA
# =========================================

X = df_main["clean_text"]
y = df_main["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================
# TF-IDF
# =========================================

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(
    X_train
)

X_test_vec = vectorizer.transform(
    X_test
)

# =========================================
# MODEL
# =========================================

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_vec,
    y_train
)

# =========================================
# EVALUATION
# =========================================

y_pred = model.predict(
    X_test_vec
)

print("\nAccuracy:")
print(
    accuracy_score(
        y_test,
        y_pred
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# =========================================
# SAVE OUTPUT
# =========================================

os.makedirs(
    "output",
    exist_ok=True
)

df_main.to_csv(
    "output/B4.csv",
    index=False,
    encoding="utf-8-sig"
)

pickle.dump(
    model,
    open(
        "output/B4_model.pkl",
        "wb"
    )
)

pickle.dump(
    vectorizer,
    open(
        "output/B4_vectorizer.pkl",
        "wb"
    )
)

print("\n[SUCCESS]")
print("output/B4.csv")
print("output/B4_model.pkl")
print("output/B4_vectorizer.pkl")
print("\nPIPELINE SELESAI")