# Spam Classification using Machine Learning

## 📌 Project Overview
This project is a **Spam Classification System** that uses **Machine Learning** to classify messages as **spam** or **ham (not spam)**. The model is trained on a dataset of SMS messages and deployed using **Streamlit** to provide a user-friendly interface for spam detection.

## 📊 Dataset
The dataset consists of labeled SMS messages with two categories:
- **Spam:** Unwanted or fraudulent messages.
- **Ham:** Legitimate messages.

The dataset includes text messages and their respective labels, used for training and evaluating the machine learning model.

## 🚀 Features
- **Preprocessing:** Handles missing values, removes duplicates, and cleans text data.
- **Feature Engineering:** Converts text into numerical features using TF-IDF Vectorization.
- **Machine Learning Models:** Utilizes models like Naïve Bayes, Logistic Regression, and Random Forest.
- **Evaluation Metrics:** Uses accuracy, precision, recall, and F1-score for performance assessment.
- **User Interface:** Built with **Streamlit** to classify messages in real-time.
- **Deployment:** Hosted locally for testing and demonstration.

## 📌 Project Phases
### **Phase 1: Project Execution**
#### 1️⃣ Understanding the Project Requirements
- Defined the project goal of spam classification.
- Analyzed dataset features and structure.
- Outlined the steps: Data Collection → Preprocessing → Model Training → Deployment.

#### 2️⃣ Development Environment Setup
- Installed required libraries:
  ```bash
  pip install pandas numpy scikit-learn streamlit imblearn
  ```
- Used **Jupyter Notebook** for EDA & Model Training.
- Built the web app using **Streamlit**.

#### 3️⃣ Data Preprocessing
- **Removed special characters & stopwords**.
- **Tokenized & vectorized** the text using TF-IDF.
- **Balanced the dataset** using SMOTE (Synthetic Minority Oversampling Technique).

#### 4️⃣ Model Training & Evaluation
- Trained models using **Naïve Bayes, Logistic Regression, Random Forest**.
- Evaluated models using accuracy, F1-score, precision, and recall.
- Selected the best model based on performance metrics.

#### 5️⃣ Model Deployment
- Built a **Streamlit UI** for message classification.
- Users can enter a message to check if it is spam or not.
- Ran the application using:
  ```bash
  streamlit run app.py
  ```

### **Phase 2: Research Paper Writing**
#### 1️⃣ Research Topic Selection
- Focused on improving spam classification using **NLP & ML techniques**.
- Reviewed literature on spam detection models.

#### 2️⃣ Research Methodology
- **Dataset Used:** Publicly available SMS Spam dataset.
- **Techniques Applied:**
  - Text Preprocessing: Tokenization, Lemmatization, Stopword Removal.
  - Model Comparison: Logistic Regression vs. Naïve Bayes vs. Random Forest.
  - Performance Evaluation: Accuracy, Precision, Recall, F1-score.

#### 3️⃣ Experimentation & Results
- Compared model performance across different feature extraction techniques.
- **Final Model:** Naïve Bayes achieved the highest F1-score of **0.99**.
- Conducted **statistical significance testing** to validate results.

#### 4️⃣ Research Paper Writing & Formatting
- Structured the paper in **IEEE format**.
- Included **Abstract, Introduction, Methodology, Results, Conclusion, and References**.

#### 5️⃣ Peer Review & Submission
- Reviewed the research paper with peers.
- Submitted to **conference/journal** for publication.

### **Phase 3: Final Presentation & Submission**
- Created a **PowerPoint presentation** summarizing the project.
- **Key Sections:** Introduction, Methodology, Results, Live Demo.
- Demonstrated live predictions using **Streamlit UI**.
- Submitted the **project files, code, documentation, and research paper**.

## 📁 Project Structure
```
├── dataset/
│   ├── spam_data.csv
│
├── src/
│   ├── spam_classifier.py  # Machine Learning Model
│   ├── app.py  # Streamlit Application
│
├── README.md
├── requirements.txt
```

## ⚡ How to Run the Project
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/spam-classification.git
   cd spam-classification
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 📌 Results
- Achieved **99% accuracy** on the test dataset.
- Model successfully detects spam messages with high precision.
- Deployed a **fully functional Streamlit app** for real-time spam classification.

## 🤝 Contributing
Contributions are welcome! If you find issues or want to enhance the project, feel free to fork the repo and submit a pull request.

## 📜 License
This project is open-source and available under the **MIT License**.

---
📧 **Contact:** For any queries, reach out via prabhunandan016@gmail.com

