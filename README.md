# 🤖 AI Text Detection System

A sophisticated Streamlit web application that detects whether text is AI-generated or human-written using advanced machine learning techniques.

## 🎯 Features

- **Dual Input Methods**: Paste text directly or upload PDF files
- **Sentence-Level Analysis**: Analyzes each sentence individually with AI probability scores
- **Feature Visualization**: Interactive radar charts showing linguistic feature analysis
- **Real-Time Processing**: Live progress tracking through the prediction pipeline
- **Multiple Export Options**: PDF reports, CSV data, and JSON exports
- **Interactive UI**: Modern, responsive design with animations and micro-interactions

## 🧠 Model Pipeline

The application uses a hybrid approach combining:

1. **TF-IDF Features** (5,000 features, n-grams 1-2)
2. **Custom Linguistic Features** (10 handcrafted features):
   - chat_word_ratio
   - punctuation_ratio  
   - TTR (Type-Token Ratio)
   - function_word_ratio
   - discourse_marker_ratio
   - sentence_length_std
   - sentence_length_cv
   - contraction_ratio
   - transition_ratio
   - formal_start_ratio

3. **Ensemble Model**: Logistic Regression + LinearSVC voting classifier

## 📁 Project Structure

```
Text_AI_detection/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .dockerignore         # Docker ignore file
├── README.md             # Project documentation
├── best_model.pkl        # Trained ensemble model
├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
├── feature_scaler.pkl    # Custom feature scaler
├── feature_columns.pkl   # Feature column names
├── chat_words.txt        # Chat abbreviations list
├── function_words.txt    # Function words list
└── discourse_markers.txt # Discourse markers list
```

## 🚀 Local Development

### Prerequisites

- Python 3.11+
- Git

### Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Text_AI_detection
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Download NLTK data**:
   ```bash
   python -c "import nltk; nltk.download('punkt')"
   ```

6. **Run the application**:
   ```bash
   streamlit run app.py
   ```

The app will be available at `http://localhost:8501`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and run**:
   ```bash
   docker-compose up --build
   ```

2. **Access the application**: `http://localhost:8501`

3. **Stop the application**:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. **Build the image**:
   ```bash
   docker build -t ai-detection-app .
   ```

2. **Run the container**:
   ```bash
   docker run -p 8501:8501 ai-detection-app
   ```

## ☁️ Cloud Deployment

### Streamlit Community Cloud

1. **Push to GitHub** (see GitHub setup below)
2. **Connect to Streamlit Community Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select the `app.py` file
   - Deploy

### AWS/Azure/GCP

1. **Build and push Docker image**:
   ```bash
   docker build -t ai-detection-app .
   docker tag ai-detection-app your-registry/ai-detection-app
   docker push your-registry/ai-detection-app
   ```

2. **Deploy to cloud service** using the provided Docker image

## 📊 GitHub Upload Instructions

### Required Files for GitHub

**Essential files (must be uploaded)**:
- `app.py` - Main application
- `requirements.txt` - Dependencies
- `Dockerfile` - Container configuration
- `README.md` - Documentation
- `best_model.pkl` - Trained model
- `tfidf_vectorizer.pkl` - Text vectorizer
- `feature_scaler.pkl` - Feature scaler
- `feature_columns.pkl` - Feature names
- `chat_words.txt` - Chat words list
- `function_words.txt` - Function words list
- `discourse_markers.txt` - Discourse markers

**Optional files**:
- `docker-compose.yml` - For local Docker development
- `.dockerignore` - Docker optimization

### GitHub Setup Steps

1. **Initialize Git repository**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: AI Text Detection System"
   ```

2. **Create GitHub repository**:
   - Go to [github.com](https://github.com)
   - Click "New repository"
   - Name: `ai-text-detection`
   - Description: "AI vs Human text detection system with Streamlit"
   - Make it Public (for free deployment)
   - Don't initialize with README (we already have one)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/yourusername/ai-text-detection.git
   git branch -M main
   git push -u origin main
   ```

4. **Verify deployment**:
   - Check that all files are uploaded
   - Ensure model files are included (they should be <100MB total)

## 🔧 Configuration

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Server address (default: 0.0.0.0)

### Model Files

The application expects these model files to be present:
- `best_model.pkl` (~2MB)
- `tfidf_vectorizer.pkl` (~200KB)
- `feature_scaler.pkl` (~1KB)
- `feature_columns.pkl` (~200B)

## 📈 Performance Metrics

Based on the training evaluation:
- **Overall Accuracy**: 92.67%
- **Cross-Validation F1-Score**: 91.39% ± 0.11%
- **Domain-Specific Performance**:
  - Academic: 87-88%
  - Web Articles: 85-99%
  - Conversational: 92-95%

## 🐛 Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **Model files missing**:
   - Ensure all `.pkl` files are in the root directory
   - Check file sizes and permissions

3. **Docker build fails**:
   - Check `requirements.txt` versions
   - Ensure all model files are included

4. **Memory issues**:
   - Reduce `max_features` in TF-IDF vectorizer
   - Use smaller PDF files

### Performance Optimization

- Use caching for model loading
- Limit PDF file sizes
- Implement request rate limiting for production

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with Streamlit
- Machine learning models powered by scikit-learn
- Text processing with spaCy and NLTK
- Visualization with Plotly
