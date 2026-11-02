# 🎬 Movie Recommendation System (NLP)

A content-based movie recommender built with **TF-IDF** and **cosine similarity**. Give it a movie title and it returns the most similar movies based on their genres, overview, and tagline.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange)
![pandas](https://img.shields.io/badge/pandas-data-lightgrey)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

This project recommends movies using Natural Language Processing. Each movie's text (genres + overview + tagline) is cleaned into a single `tags` field and converted into a TF-IDF vector. Similar movies are found by comparing these vectors with cosine similarity.

- **Type:** Content-based filtering
- **Catalog size:** 45,447 movies
- **Feature space:** 50,000 TF-IDF features (unigrams, bigrams, and trigrams)

---

## 🧠 How It Works

```
Raw movie data
      │
      ▼
Combine genres + overview + tagline  ──►  clean text (`tags`)
      │
      ▼
TF-IDF Vectorizer (1–3 grams, 50k features, English stop words)
      │
      ▼
Sparse TF-IDF matrix  (45,447 × 50,000)
      │
      ▼
Cosine similarity against the selected movie
      │
      ▼
Top-N most similar movies
```

### TF-IDF configuration

| Parameter      | Value       |
| -------------- | ----------- |
| `max_features` | 50,000      |
| `ngram_range`  | (1, 3)      |
| `stop_words`   | `english`   |
| `lowercase`    | `True`      |
| `norm`         | `l2`        |
| `use_idf`      | `True`      |

---

## 📁 Project Files

| File               | Description                                                                 |
| ------------------ | --------------------------------------------------------------------------- |
| `tfidf.pkl`        | Fitted `TfidfVectorizer`                                                    |
| `tfidf_matrix.pkl` | Sparse CSR matrix of TF-IDF vectors for all movies (45,447 × 50,000)        |
| `df.pkl`           | Movie metadata DataFrame                                                    |
| `indices.pkl`      | Pandas Series mapping movie `title` → row index                             |

### `df.pkl` columns

| Column         | Description                           |
| -------------- | ------------------------------------- |
| `title`        | Movie title                           |
| `genres`       | Space-separated genres                |
| `overview`     | Plot summary                          |
| `tagline`      | Movie tagline                         |
| `vote_average` | Average user rating                   |
| `popularity`   | Popularity score                      |
| `tags`         | Cleaned text used to build the vectors |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install pandas numpy scipy scikit-learn==1.6.1
```

> ⚠️ The pickle files were created with **scikit-learn 1.6.1**. Using the same version avoids compatibility warnings.

### 3. Run recommendations

```python
import pickle
from sklearn.metrics.pairwise import linear_kernel

# Load saved artifacts
tfidf        = pickle.load(open("tfidf.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
df           = pickle.load(open("df.pkl", "rb"))
indices      = pickle.load(open("indices.pkl", "rb"))


def recommend(title, n=10):
    if title not in indices:
        return f"'{title}' not found in the dataset."

    idx = indices[title]

    # TF-IDF vectors are L2-normalized, so linear_kernel == cosine similarity
    scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Top-N most similar movies (excluding the movie itself)
    top = scores.argsort()[::-1][1:n + 1]

    return df.iloc[top][["title", "genres", "vote_average"]]


print(recommend("Toy Story"))
```

---

## 📊 Example

**Input:** `Toy Story`

**Output:** a ranked list of the most similar movies (title, genres, rating), such as other animated family films with similar themes.

---

## 🛠️ Tech Stack

- **Python**
- **pandas**: data handling
- **scikit-learn**: TF-IDF vectorization and similarity
- **SciPy**: sparse matrix operations
- **pickle**: model persistence

---

## 🔮 Future Improvements

- Hybrid recommender (content + collaborative filtering)
- Semantic embeddings (Sentence-BERT) instead of TF-IDF
- Web app interface (Streamlit / FastAPI / Flutter client)
- Filter by genre, rating, or release year
- Fuzzy title matching for typos

---

## 👤 Author

**Ibrahim**
BS Software Engineering, National Textile University (NTU), Faisalabad

🔗 [LinkedIn](https://linkedin.com/in/muhammad-ibrahim-python)

---

## 📄 License

This project is licensed under the MIT License.
