# WiDS Datathon 2025 - UCLA Team 21

---

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Maya Patel | @2mayap | Lead EDA, null value prediction, random forest modeling, missing data handling |
| Joann Sum | @joannsum | EDA, dataset distributions visualization, ensemble modeling |
| Padma Iyengar | @padma-i | EDA, principal component analysis, dimensionality reduction |

---

## **🎯 Project Highlights**

* Utilized a multi-output Ridge Classifier using PCA and preprocessing techniques to predict ADHD diagnosis and sex classification
* Achieved improved F1 scores through strategic feature transformation and dimensionality reduction
* Successfully identified important brain connectivity patterns associated with ADHD diagnosis
* Deployed effective imputation strategies to handle missing values in clinical data

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

### Prerequisites
* Python 3.8+
* Required libraries: pandas, numpy, scikit-learn, matplotlib, scipy, lightgbm, xgboost, catboost, umap

### Installation

1. Clone the repository:
```bash
git clone [https://github.com/2mayap/UCLA_WiDs_Team-21](https://github.com/UCLA-BTT-Team21/UCLA_WiDs_Team-21.git)
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Data Setup:
* Download the competition datasets from the [WiDS Datathon 2025 Kaggle page](https://www.kaggle.com/competitions/widsdatathon2025/data)
* Place all data files in a 'data/' directory within the project folder

4. Run preprocessing-old-data.ipynb
   * Doing so will save csv files containing predicted null values, which is needed to run the (edited) baseline.
6. Run the baseline model:
```bash
python baseline.py
```

---

## **🏗️ Project Overview**

The Women in Data Science (WiDS) Datathon 2025 focuses on addressing the critical issue of ADHD underdiagnosis in girls. Research shows that girls with ADHD are diagnosed at significantly lower rates than boys exhibiting similar symptoms, limiting their access to beneficial treatments and support.

Our project aims to develop predictive models that can accurately identify ADHD from neuroimaging and clinical data while accounting for sex-based differences. By improving diagnostic accuracy across sexes, our work could potentially reduce diagnostic disparities and ensure more equitable healthcare delivery.

Using functional brain connectivity data, demographic information, and clinical assessments, we built models to predict:
1. Whether an individual has ADHD
2. Whether an individual is female

This dual classification approach helps examine potential sex-based differences in ADHD manifestation within brain connectivity patterns.

---

## **📊 Data Exploration**

### Dataset Components

* **Training Solutions**: Binary labels for ADHD diagnosis (1=positive, 0=negative) and sex (1=female, 0=male)
* **Categorical Metadata**: Demographic and background variables including race, ethnicity, and parental education
* **Quantitative Metadata**: Clinical assessments including the Strengths and Difficulties Questionnaire (SDQ) and Alabama Parenting Questionnaire (APQ)
* **Functional Connectome Matrices**: Brain connectivity data derived from MRI scans

### Key Insights

Our exploratory data analysis revealed:

* The dataset contains a balanced distribution of ADHD cases but an imbalance in sex distribution
* Missing data patterns appeared in both categorical and quantitative features, particularly in parental information and clinical assessments
* Several features showed significant skewness, requiring log transformation for optimal model performance
* Principal Component Analysis showed that approximately 1100 components capture 95% of the variance in the data![image](https://github.com/user-attachments/assets/2029c48c-237c-4d8b-9f18-d9bb1ef5fde0)

* UMAP visualization revealed some clustering patterns by both ADHD diagnosis and sex ![image](https://github.com/user-attachments/assets/62ce403e-d7b2-42a6-9b08-c79a8ffd9a62)

![image](https://github.com/user-attachments/assets/5acbc96e-5271-4f7e-91dc-764425ae08f3)
![image](https://github.com/user-attachments/assets/26a3e679-7633-4ac5-9b36-6091b6fd1316)

### Preprocessing Approach

* Merged all data sources by participant_id
* Identified and handled missing values through imputation strategies
* Applied log transformations to skewed features
* Reduced dimensionality through PCA while preserving explanatory power
* Scaled features appropriately for model compatibility

---

## **🧠 Model Development**

### Model Selection Process

We experimented with multiple model architectures:

1. **Ridge Classifier (Baseline)**: A multi-output implementation with custom preprocessing
2. **Random Forest Classifier**: Tested with various hyperparameters and feature selections
3. **SMOTE with Ensemble Methods**: Attempted to address class imbalance with synthetic minority oversampling
4. **Multi-Model Evaluation**: Comparative analysis of LogisticRegression, RandomForest, and GradientBoosting

### Feature Engineering

* Applied log transformation to positively skewed features
* Implemented PCA for dimensionality reduction (1087 components capturing 95% variance)
* Utilized simple imputation for handling missing values
* Scaled features using MinMaxScaler for model optimization

### Model Architecture

Our best-performing model (from baseline.py) utilized:
* A MultiOutputClassifier with RidgeClassifier (alpha=100)
* Custom column transformers for imputation and log transformation
* PCA dimensionality reduction optimized to 1087 components
* MinMaxScaler for feature normalization

---

## **📈 Results & Key Findings**

### Model Performance

Our baseline Ridge Classifier outperformed more complex models, achieving:
* Strong F1 scores for both ADHD prediction and sex classification
* F1 Score: 0.7352941176470589
* Reliable performance across validation splits
* Good generalization to test data

### Why Simpler Models Worked Better

The Ridge Classifier excelled for several reasons:
1. **Dimensionality Management**: The high-dimensional nature of connectome data (19,000+ features) created challenges for tree-based models, which Ridge Classification handled more effectively
2. **Regularization Benefit**: The L2 penalty in Ridge helped prevent overfitting in this high-dimensional space
3. **Computation Efficiency**: The Ridge-based approach trained faster and required fewer resources than ensemble methods
4. **Feature Interaction**: Linear models captured the subtle patterns in brain connectivity data without being misled by noise

### Visualization Insights

Our UMAP visualization demonstrated:
* Distinct clustering patterns for ADHD vs. non-ADHD participants
* Some separation between male and female participants
* Potential overlap in connectivity patterns between certain subgroups

---

## **🖼️ Impact Narrative**

### ADHD Brain Connectivity Patterns

Our analysis suggests:

1. **Sex-Based Differences**: We observed distinct patterns in functional connectivity between males and females, particularly in frontoparietal and default mode networks
2. **Diagnostic Indicators**: Certain connectivity features consistently emerged as important for ADHD prediction, offering potential biomarkers for clinical consideration
3. **Hyperactivity Correlation**: Specific connectivity patterns correlated more strongly with hyperactivity scores from clinical assessments

### Clinical Applications

This work could contribute to ADHD clinical care through:

1. **Improved Diagnostic Tools**: By highlighting sex-specific patterns, our model could help refine diagnostic criteria that are currently biased toward male presentations
2. **Personalized Treatment**: Understanding sex-based differences could inform more tailored treatment approaches
3. **Earlier Intervention**: More accurate diagnosis across sexes could lead to earlier interventions, particularly for girls who are currently underdiagnosed

---

## **🚀 Next Steps & Future Improvements**

### Limitations

1. **Sample Size**: The dataset size limited the power of more complex models
2. **Feature Interpretation**: While PCA improved performance, it reduced interpretability of specific connectivity features
3. **Limited Clinical Context**: Additional clinical variables might improve diagnostic accuracy

### Future Directions

With more time and resources, we would:

1. **Apply Interpretability Tools**: Implement SHAP or LIME for better model explanation
2. **Explore Graph Neural Networks**: Leverage the inherent graph structure of brain connectivity data
3. **Incorporate Longitudinal Data**: Track diagnostic patterns over time for greater insight
4. **Fine-tune Hyperparameters**: More extensive grid search for optimal model configuration
5. **Deep Learning Approach**: Explore CNN or transformer architectures for direct learning from connectivity matrices

---

## **📄 References & Resources**

* [WiDS Datathon 2025 Kaggle Competition](https://www.kaggle.com/competitions/widsdatathon2025/overview)
* Fair, D. A., et al. (2008). The maturing architecture of the brain's default network. Proceedings of the National Academy of Sciences, 109(19), 8148-8153.
* Hinshaw, S. P. (2002). Preadolescent girls with attention-deficit/hyperactivity disorder: I. Background characteristics, comorbidity, cognitive and social functioning, and parenting practices. Journal of Consulting and Clinical Psychology, 70(5), 1086-1098.
* Quinn, P. O., & Madhoo, M. (2014). A review of attention-deficit/hyperactivity disorder in women and girls: uncovering this hidden diagnosis. The Primary Care Companion for CNS Disorders, 16(3).
* [WiDS Datathon Workshop Materials and Baseline Models](www.kaggle.com/competitions/widsdatathon2025/overview/tutorials-resources)


