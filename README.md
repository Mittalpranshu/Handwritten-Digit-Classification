# 🧠 Handwritten Digit Classification (MNIST) — ANN vs CNN

This project demonstrates the application of **Artificial Neural Networks (ANNs)** and **Convolutional Neural Networks (CNNs)** on the classic **MNIST dataset** of handwritten digits.  
The aim is to **train, evaluate, and compare** the performance of both models using accuracy metrics and confusion matrices.

---

## 🚀 Project Overview
- **Dataset**: [MNIST handwritten digits](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)  
- **Models Implemented**: 
  - Fully Connected **ANN**
  - Convolutional **CNN**
- **Frameworks**: TensorFlow, Keras, Scikit-learn, Matplotlib, Seaborn  
- **Evaluation**: Validation accuracy, Confusion matrix, and accuracy comparison plots  

---

## 📂 Workflow
1. **Load and preprocess data**  
   - Normalization, one-hot encoding  
   - Separate ANN input (flattened) vs CNN input (28x28x1)  

2. **ANN Model**  
   - Input layer (784 units) → Hidden layers (128, 64, ReLU) → Output layer (10 softmax)  

3. **CNN Model**  
   - Conv2D + MaxPooling → Conv2D + MaxPooling → Flatten → Dense (64 ReLU) → Output (softmax)  

4. **Training**  
   - Both models trained for 5 epochs with batch size = 128  
   - Optimizer = Adam, Loss = categorical crossentropy  

5. **Evaluation**  
   - Accuracy comparison between ANN and CNN  
   - Confusion matrices for detailed classification insights  
   - Validation accuracy curve visualization  

---

## 📊 Results

- **ANN Test Accuracy**: ~97%  
- **CNN Test Accuracy**: ~99%  
- CNN significantly outperformed ANN due to its ability to capture **spatial features** in images.

### Validation Accuracy Comparison
![Validation Accuracy](./assets/accuracy_plot.png)

### Confusion Matrices
| ANN | CNN |
|-----|-----|
| ![ANN Confusion Matrix](./assets/cm_ann.png) | ![CNN Confusion Matrix](./assets/cm_cnn.png) |

---

## 🛠️ Tech Stack
- **Language**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Scikit-learn  

---

## 📌 Key Learnings
- Importance of **data preprocessing and normalization**.  
- ANN performs well on simple datasets but CNN leverages **spatial hierarchies** for higher accuracy.  
- Use of **confusion matrices** for detailed model evaluation.  
- Visualization helps in interpreting model improvements across epochs.  

---



