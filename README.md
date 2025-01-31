# Country Clustering Using Socio-Economic and Health Factors

## 🌍 Project Overview
This project categorizes countries into distinct groups using **socio-economic and health factors** to identify developmental patterns. The clusters can help policymakers, NGOs, and international organizations design targeted interventions and track progress toward global goals like the SDGs.

**Key Objectives**:
- Cluster countries using **K-Means**, **Hierarchical Clustering**, and **DBSCAN**.
- Analyze cluster characteristics (e.g., GDP, child mortality).
- Compare algorithm performance and interpret results.

**Dataset**: [Kaggle Country Dataset](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)  
**Tools**: Python, Scikit-Learn, Pandas, Matplotlib, Scipy.

---

## ✨ Features
- **Clustering Algorithms**:
  - K-Means
  - Hierarchical Clustering (Ward’s Method)
  - DBSCAN (Density-Based Clustering)
- **Visualization**:
  - PCA and t-SNE for dimensionality reduction
- **Evaluation Metrics**:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score

---

## 💻 Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/abe16s/Categorize-Countries.git
   cd Categorize-Countries
   ```
2. **Install Dependencies**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib scipy jupyter
   ```
3. **Download the Dataset**:
   - Place `country-data.csv` in the `data/` directory.

---

## 📊 Data Description
The dataset contains **10 features** for 167 countries:
- `child_mort`: Child mortality rate (per 1,000 live births)
- `exports`: Exports as a % of GDP
- `health`: Healthcare spending as a % of GDP
- `imports`: Imports as a % of GDP
- `income`: Net income per capita (USD)
- `inflation`: Annual inflation rate (%)
- `life_expec`: Life expectancy at birth (years)
- `total_fer`: Fertility rate (children per woman)
- `gdpp`: GDP per capita (USD)
- `country`: Country name

**Preprocessing**:
- Standardized using `StandardScaler`.

---

## 🔄 Workflow
1. **Data Preprocessing**:
   - Handle missing values.
   - Standardize features.
2. **Clustering**:
   - K-Means (Elbow Method for optimal `k`).
   - Hierarchical Clustering (Dendrogram for `k` selection).
   - DBSCAN (k-Distance Graph for `eps` tuning).
3. **Evaluation**:
   - Internal validation metrics.
4. **Visualization**:
   - PCA/t-SNE for 2D cluster plots.

---

## 📚 Algorithms Used
### 1. **K-Means Clustering**
- **Strengths**: Fast, works well for globular clusters.
- **Weaknesses**: Assumes spherical clusters, sensitive to outliers.
- **Use Case**: Baseline clustering for tiered categorization.

### 2. **Hierarchical Clustering**
- **Strengths**: No need to pre-specify `k`, dendrogram visualization.
- **Weaknesses**: Computationally expensive (O(n³)).
- **Use Case**: Hierarchical analysis of development stages.

### 3. **DBSCAN**
- **Strengths**: Handles noise, detects arbitrary-shaped clusters.
- **Weaknesses**: Struggles with varying densities.
- **Use Case**: Outlier detection (e.g., Qatar, Niger).

### All algorithms are custom written in the `src` directory and also standard libraries are used for comparison. 
---


## 📈 **Key Insights**
- **K-Means**:
  - 3 clear tiers: **Developed**, **Developing**, **Underdeveloped**.
  - High-income countries (e.g., USA, Germany) vs. low-GDP nations (e.g., Mali, Haiti).
- **DBSCAN**:
  - Identified outliers (e.g., Qatar: high GDP, low population).
- **Hierarchical**:
  - Similar results to K-Means but with dendrogram insights.

---

## 🚀 Usage
1. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/comparison.ipynb
   ```
2. **Steps**:
   - Execute cells sequentially.
   - Adjust hyperparameters (e.g., `eps` for DBSCAN).
   - Analyze visualizations and metrics.

---

**Happy Clustering!** 🎯