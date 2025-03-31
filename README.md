# IMDB Movie Reviews Sentiment Analysis

## Table of Contents
- [Overview](#overview)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Folder Structure](#folder-structure)
- [Tools and Technologies](#tools-and-technologies)
- [Installation Steps](#installation-steps)
- [Conclusion](#conclusion)
- [Future enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
This project focuses on analyzing sentiment in IMDB movie reviews using machine learning techniques. The dataset contains **50,000 movie reviews**, and the primary goal is to classify reviews as either positive or negative. The project involves data preprocessing, exploratory data analysis (EDA), dimensionality reduction, and the implementation of various machine learning models to achieve accurate sentiment classification.

## Objectives
- Perform **sentiment analysis** on IMDB movie reviews.
- Preprocess text data to extract meaningful features for analysis.
- Explore the dataset to identify trends and insights.
- Implement and optimize machine learning models for sentiment classification.
- Reduce dimensionality to improve clustering and feature selection.

## Methodology

### 1. Data Preprocessing
- Utilized **Natural Language Toolkit (NLTK)** for text preprocessing:
  - Removed stop words.
  - Tokenized and lemmatized text data.
- Extracted key features to enhance model performance.

### 2. Exploratory Data Analysis (EDA)
- Analyzed the dataset to uncover:
  - Trends in positive and negative reviews.
  - Frequently occurring words in each sentiment category.
  - A "Plots" folder is automatically generated storing all the plot images in an organized, structured format.

### 3. Dimensionality Reduction
- Applied **Principal Component Analysis (PCA)**:
  - Reduced dimensionality while retaining **65% variance**.
  - Improved clustering performance with a PCA ratio of **2.5:1**.

### 4. Machine Learning Models
Implemented multiple models for sentiment classification:
- **KMeans Clustering**:
  - Used PCA-transformed data for clustering.
  - Achieved an inertia score of approximately **37k**.
- **Logistic Regression**:
  - Optimized using **OpenGridCV** with L2 regularization.
  - Achieved an accuracy of **89%**.
- **Naive Bayes Classifier**:
  - Tuned with an alpha value of 2.0 using OpenGridCV.
  - Achieved an accuracy of **85%**.
- **Decision Tree Classifier**:
  - Optimized using **RandomizedSearchCV** with a max depth of 24.
  - Achieved an accuracy of **75%**.

## Folder Structure
```
IMDB-Sentiment-Analysis/
├── IMDB_reviews_sentiment_analysis.ipynb
├── Plots/
│   └── …generated plot images…
├── requirements.txt
└── README.md
```

## Tools and Technologies
The following tools and libraries were used in this project:
- Programming Language: **Python**
- Libraries:  
  - **Natural Language Toolkit (NLTK)**  
  - **Scikit-Learn**  
  - **Matplotlib**
- Techniques:  
  - Logistic Regression  
  - Naive Bayes Classification  
  - Decision Trees  
  - KMeans Clustering  
  - Principal Component Analysis (PCA)  
- Optimization Frameworks:  
  - OpenGridCV  
  - RandomizedSearchCV  

## Installation Steps
1. **Clone the Repository**  
   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/jagrit-sharma/IMDB-Sentiment-Analysis.git
   cd IMDB-Sentiment-Analysis
   ```
2.	**Create a Virtual Environment (Optional but recommended)**
    
    It’s recommended to create a virtual environment to manage dependencies for your project:

	•	For Windows:

    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

	•	For macOS/Linux:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.	**Install Dependencies**
    
    Install the required dependencies by running the following command:

    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook IMDB_reviews_sentiment_analysis.ipynb
   ```
5.	**Output Plots**

    The code generates several plots and stores them in folders under Plots. These include:
    - EDA Plots
    - KMeans Plots
    - Logit Plots
    - DT Plots
    - NB Plots

    You can view these plots after running the code.

## Conclusion
This project demonstrates how effective preprocessing, EDA, dimensionality reduction, and machine learning techniques can be utilized to perform sentiment analysis on textual data. The results highlight the importance of model optimization in achieving high accuracy for classification tasks.

## Future Enhancements
- Explore advanced deep learning models for improved accuracy.
- Implement Natural Language Processing models for more detailed analysis.
- Develop an interactive sentiment analysis system to accept user input.
- Integrate real-time sentiment analysis on streaming data.
- Enhance visualization of EDA findings.

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/jagrit-sharma/IMDB-Sentiment-Analysis/blob/main/LICENSE) file for details.

## Acknowledgments

- Kaggle for providing the 50,000 IMDB Reviews Dataset.