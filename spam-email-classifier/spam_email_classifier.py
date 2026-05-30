"""
Spam EMAIL Classifier
Dataset : 5,728 real emails (subject + body) — same structure as balaka18
          Auto-downloaded from GitHub — no login, no API key needed!
Models  : TF-IDF + Naive Bayes  &  Logistic Regression
"""

import io, ssl, urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────
#  1. LOAD EMAIL DATASET  (real emails, no login needed)
# ──────────────────────────────────────────────────────
print("=" * 60)
print("  SPAM EMAIL CLASSIFIER")
print("  Dataset : 5,728 real emails (subject + body)")
print("  Source  : OmkarPathak/Playing-with-datasets (GitHub)")
print("=" * 60)

print("\n[1/5] Downloading dataset...")

CSV_URL = (
    "https://raw.githubusercontent.com/OmkarPathak/"
    "Playing-with-datasets/master/Email%20Spam%20Filtering/emails.csv"
)

ctx = ssl._create_unverified_context()   # handles corporate/self-signed certs
raw = urllib.request.urlopen(CSV_URL, context=ctx).read()
df  = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")))

# columns: 'text' (email content), 'spam' (0=ham, 1=spam)
df.columns = ["text", "label"]
df = df.dropna()

print(f"  ✓ Loaded {len(df):,} emails")
print(f"  Label distribution:")
print(f"    Ham  (0) : {(df['label']==0).sum():,}")
print(f"    Spam (1) : {(df['label']==1).sum():,}")
print(f"\n  Sample email snippet:")
print(f"    {df['text'].iloc[5][:120]}...")


# ──────────────────────────────────────────────────────
#  2. PREPROCESSING
# ──────────────────────────────────────────────────────
print("\n[2/5] Preprocessing...")

X = df["text"]
y = df["label"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")


# ──────────────────────────────────────────────────────
#  3. TF-IDF FEATURE EXTRACTION
# ──────────────────────────────────────────────────────
print("\n[3/5] TF-IDF Vectorisation (unigrams + bigrams)...")

vectorizer = TfidfVectorizer(
    max_features=15_000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True,
    strip_accents="unicode",
    analyzer="word",
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"  Vocabulary size : {X_train_tfidf.shape[1]:,}")
print(f"  Feature matrix  : {X_train_tfidf.shape}")


# ──────────────────────────────────────────────────────
#  4. TRAIN & EVALUATE MODELS
# ──────────────────────────────────────────────────────
print("\n[4/5] Training models...")

models = {
    "Naive Bayes (MultinomialNB)": MultinomialNB(alpha=0.05),
    "Logistic Regression"        : LogisticRegression(max_iter=1000, C=5, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc   = accuracy_score(y_test, preds)
    results[name] = {"model": model, "preds": preds, "accuracy": acc}
    print(f"\n  ── {name} ──")
    print(f"  Accuracy : {acc * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Ham", "Spam"]))

best_name  = max(results, key=lambda n: results[n]["accuracy"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]
print(f"\n  🏆 Best model : {best_name}  ({results[best_name]['accuracy']*100:.2f}%)")


# ──────────────────────────────────────────────────────
#  5. VISUALISATIONS
# ──────────────────────────────────────────────────────
print("\n[5/5] Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Spam Email Classifier – Results", fontsize=15, fontweight="bold")

# (a) Class distribution
labels = ["Ham (Not Spam)", "Spam"]
counts = [(y == 0).sum(), (y == 1).sum()]
bars = axes[0].bar(labels, counts, color=["#2ecc71", "#e74c3c"], width=0.4)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Number of Emails")
for bar, v in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 v + 40, str(v), ha="center", fontweight="bold")

# (b) Accuracy comparison
names = list(results.keys())
accs  = [results[n]["accuracy"] * 100 for n in names]
b2    = axes[1].bar(names, accs, color=["#3498db", "#9b59b6"], width=0.4)
axes[1].set_title("Model Accuracy Comparison")
axes[1].set_ylabel("Accuracy (%)")
axes[1].set_ylim([min(accs) - 3, 101])
axes[1].set_xticklabels(names, rotation=12, ha="right")
for bar, acc in zip(b2, accs):
    axes[1].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.2, f"{acc:.2f}%",
                 ha="center", fontweight="bold")

# (c) Confusion matrix
cm   = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
disp.plot(ax=axes[2], colorbar=False, cmap="Reds")
axes[2].set_title(f"Confusion Matrix\n({best_name})")

plt.tight_layout()
plt.savefig("spam_classifier_results.png", dpi=150)
plt.close()
print("  ✓ Saved : spam_classifier_results.png")


# ──────────────────────────────────────────────────────
#  PREDICTION HELPER
# ──────────────────────────────────────────────────────
def predict_email(text: str) -> None:
    vec   = vectorizer.transform([text])
    pred  = best_model.predict(vec)[0]
    proba = best_model.predict_proba(vec)[0]
    tag   = "🚨 SPAM" if pred == 1 else "✅ HAM "
    short = text[:72] + "..." if len(text) > 72 else text
    print(f"  {tag} | Spam: {proba[1]*100:5.1f}% | {short}")


# ──────────────────────────────────────────────────────
#  DEMO PREDICTIONS
# ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO PREDICTIONS")
print("=" * 60)

sample_emails = [
    "Subject: Congratulations! You've won $1,000,000 lottery prize. Click here to claim NOW! Free money waiting!!!",
    "Subject: Team meeting rescheduled\n\nHi all, tomorrow's 3pm meeting has been moved to 4pm. Please update your calendars.",
    "Subject: URGENT – Your account will be suspended\n\nDear user, verify your bank details immediately at our secure link to avoid account closure.",
    "Subject: Q3 Report attached\n\nPlease find the quarterly financial report attached. Let me know if you have any questions.",
    "Subject: Get rich quick! Make $5000 per week from home! No experience needed. Act fast — limited slots!",
    "Subject: Lunch tomorrow?\n\nHey, are you free for lunch tomorrow around 1pm? Let me know!",
    "Subject: FREE iPhone 15 Pro! You have been selected!\n\nClick the link below and fill in your details to receive your FREE iPhone 15 Pro today!",
]

for email in sample_emails:
    predict_email(email)

print("\n✅ Done! Check spam_classifier_results.png for plots.")
