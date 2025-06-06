# MLOps Injury Duration Prediction Pipeline

A complete machine learning operations (MLOps) pipeline that predicts sports injury recovery duration using XGBoost regression, with automated hyperparameter tuning, containerized deployment, and CI/CD integration.

## 🏗️ Architecture Overview

```
Data → Model Training (Optuna) → Artifact Generation → Docker Container → AWS Lambda → API Gateway → Production API
```

## 🎯 What This Repository Demonstrates

- **ML Model Training**: XGBoost regression with Optuna hyperparameter optimization
- **Artifact Management**: Model serialization with metadata and feature statistics
- **Containerization**: Docker-based AWS Lambda deployment
- **Infrastructure as Code**: SAM (Serverless Application Model) templates
- **CI/CD Pipeline**: Automated training and deployment with AWS CodeBuild
- **Production API**: RESTful endpoint for real-time injury duration predictions

## 📊 Model Performance

- **Algorithm**: XGBoost Regressor
- **Optimization**: 200 trials with Optuna (RMSE minimization)
- **Performance**: ~3.6 days RMSE on 4-40 day injury duration range
- **Features**: 18 injury characteristics (age, contact sport, swelling, etc.)

## 🚀 Quick Start

### Local Development
```bash
# Train the model
cd injuries_lambda
python train_injuries.py

# Deploy locally
cd PredictInjuryDurationLambda
sam build
sam deploy --guided
```

### API Usage
```bash
curl -X POST https://your-api-endpoint/predict-injury-duration \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age": 25,
      "is_contact": 1,
      "swelling": 2,
      ...
    }
  }'
```

## 📁 Project Structure

```
├── injuries_lambda/
│   ├── train_injuries.py          # Model training with Optuna
│   ├── requirements.txt           # Training dependencies
│   └── data/                      # Training datasets
├── PredictInjuryDurationLambda/
│   ├── lambda_handler.py          # Inference endpoint
│   ├── Dockerfile                 # Container definition
│   ├── template.yaml              # SAM infrastructure
│   ├── requirements.txt           # Lambda dependencies
│   └── samconfig.toml             # SAM configuration
└── buildspec.yaml                 # CodeBuild CI/CD pipeline
```

## 🛠️ Technology Stack

- **ML Framework**: XGBoost, scikit-learn, Optuna
- **Containerization**: Docker, AWS Lambda Container Images
- **Infrastructure**: AWS SAM, CloudFormation
- **CI/CD**: AWS CodeBuild
- **API**: AWS API Gateway (HTTP API)
- **Languages**: Python 3.12

## 🎓 Learning Outcomes

Perfect for understanding:
- MLOps best practices and patterns
- Containerized ML model deployment
- AWS serverless architecture for ML
- Automated ML pipeline design
- Model artifact management and versioning

## 📄 License

MIT License - feel free to use this as a template for your own MLOps projects!