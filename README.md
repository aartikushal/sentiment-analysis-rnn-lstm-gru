# 🧠 Sentiment Analysis using Recurrent Neural Networks (RNN, LSTM, GRU)

## 📌 Project Overview
This project focuses on **sentiment analysis of Twitter data** using **Recurrent Neural Networks (RNNs)** and their advanced variants such as **LSTM** and **GRU**.  
The goal is to classify tweets into multiple sentiment categories by effectively learning **sequential and contextual patterns in text data**.

The project covers the **complete NLP lifecycle** — from foundational understanding and data exploration to model training, evaluation, and optimization.

---

## 1️⃣ Foundational Knowledge

### 🔹 Understanding Recurrent Neural Networks (RNNs)
Recurrent Neural Networks are a class of neural networks designed specifically for **sequential data**.  
Unlike traditional feedforward networks, RNNs contain **feedback loops** that allow information from previous time steps to influence current predictions.

This internal memory, known as the **hidden state**, enables RNNs to learn **temporal dependencies** and contextual relationships across sequences.

---

### 🔹 Why RNNs are Suitable for Sequential Data
Many real-world datasets are sequential by nature:
- Text
- Speech
- Time series
- Sensor data
- Video frames

RNNs process data **one element at a time**, maintaining context from earlier inputs.  
This makes them well-suited for tasks where **order and history matter**, such as:
- Sentiment analysis
- Language modeling
- Speech recognition
- Machine translation

---

### 🔹 RNN Architectures

#### Vanilla RNN
- Basic recurrent architecture
- Maintains a hidden state
- Suffers from **vanishing and exploding gradient problems**
- Limited ability to learn long-term dependencies

#### Long Short-Term Memory (LSTM)
- Introduces **memory cells** and gating mechanisms
- Uses input, forget, and output gates
- Effectively captures **long-range dependencies**
- Solves gradient-related issues in vanilla RNNs

#### Gated Recurrent Unit (GRU)
- Simplified alternative to LSTM
- Uses update and reset gates
- Fewer parameters and faster training
- Comparable performance to LSTM in many NLP tasks

---

### 🔹 Advantages of RNNs
- Capture **temporal dependencies** through hidden states
- Preserve **context over time**
- Handle **variable-length sequences**
- Perform well on language and time-dependent problems
- LSTM and GRU models handle **long-term memory** efficiently

---

## 2️⃣ Data Exploration

### 🔍 Dataset Description
- Source: Twitter sentiment dataset
- Columns:
  - ID
  - Entity
  - Sentiment Label
  - Tweet Text
- Multi-class sentiment classification:
  - Positive
  - Negative
  - Neutral
  - Irrelevant

---

### 📊 Exploratory Analysis Insights
- Dataset contains **over 70,000 tweets**
- Missing values were identified and removed
- Duplicate records were eliminated
- **Negative sentiment** appears more frequently than other classes
- Tweet lengths vary significantly, influencing padding strategy

---

### 📈 Visualization Observations
- Sentiment label distribution reveals **class imbalance**
- Text length distribution helps determine optimal sequence length
- Word clouds highlight:
  - Positive sentiment keywords (supportive and appreciative terms)
  - Negative sentiment keywords (complaints and dissatisfaction)

---

## 3️⃣ Text Preprocessing & Feature Engineering

### 🎯 Goal
Convert raw tweet text into a **numerical format** that RNN-based models can process efficiently.

---

### 🔹 Key Preprocessing Steps

#### Text Cleaning
- Convert text to lowercase
- Remove URLs, mentions, hashtags, emojis, and special characters
- Normalize whitespace

#### Tokenization
- Break text into individual tokens (words)
- Build a vocabulary mapping words to integers

#### Integer Encoding
- Convert tokens into numerical sequences
- Handle out-of-vocabulary (OOV) words safely

#### Padding & Truncation
- Ensure all sequences have equal length
- Shorter sequences are padded
- Longer sequences are truncated

#### Embedding Preparation
- Convert integer sequences into dense vector representations
- Enables semantic understanding of words

---

## 4️⃣ Label Encoding & Data Splitting

### 🔹 Sentiment Encoding
- Textual sentiment labels are converted into numerical values
- Enables compatibility with machine learning models
- Supports both binary and multi-class classification

---

### 🔹 Train-Test Split
- Dataset split into training and testing sets
- **Stratified sampling** ensures balanced class distribution
- Prevents bias toward dominant sentiment classes

---

## 5️⃣ RNN Model Construction

### 🏗️ Architecture Selection
A **Bidirectional LSTM model** was chosen to:
- Capture context from both past and future words
- Improve understanding of sentence semantics

---

### 🔹 Design Considerations
- Sentiment analysis requires understanding **word order and context**
- Dataset size supports moderately complex architectures
- Dropout layers help prevent overfitting
- Softmax activation enables multi-class prediction

---

## 6️⃣ Model Training

### 🚀 Training Strategy
- Model trained using categorical loss functions
- Performance monitored using:
  - Training accuracy
  - Validation accuracy
  - Loss curves

---

### 🔹 Overfitting Prevention Techniques
- Dropout regularization
- Early stopping based on validation loss
- Best model weights restored automatically

---

### 📈 Training Insights
- Model accuracy improved steadily across epochs
- Validation performance stabilized, indicating good generalization
- Early stopping prevented unnecessary training

---

## 7️⃣ Model Evaluation

### 📊 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

### 🔍 Performance Analysis
- Strong overall classification accuracy (~85%)
- Balanced precision and recall across sentiment classes
- Slight confusion between Neutral and Irrelevant sentiments
- Negative sentiment showed highest recall

---

### 🧠 Interpretation
- Model effectively learns sentiment patterns
- Performs consistently across all classes
- Minimal bias toward any single sentiment

---

## 8️⃣ Fine-Tuning & Optimization

### 🔧 Hyperparameter Tuning
- Learning rate adjustments improved convergence
- Batch size optimized for stability and performance
- Dropout rate controlled overfitting

---

### ⚙️ Optimization Techniques
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling to improve convergence
- Early stopping combined with learning rate reduction

---

## 🏁 Final Conclusion
- RNN-based architectures are highly effective for sentiment analysis
- LSTM and GRU significantly outperform vanilla RNNs
- Proper preprocessing is critical for NLP success
- Bidirectional models capture richer contextual information
- Fine-tuning and regularization greatly enhance model performance

---

## 🛠️ Technologies Used
- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- TensorFlow / Keras
- Natural Language Processing (NLP)
- Deep Learning (RNN, LSTM, GRU)

---

## 👩‍💻 Author
**Aarti Potdar**  
Senior Consultant | AI & Machine Learning Enthusiast  
Expertise in NLP, Deep Learning, and Enterprise AI Solutions

---

⭐ If you found this project useful, please **star the repository**  
📋 Feel free to **fork and experiment further**
