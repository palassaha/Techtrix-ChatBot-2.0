# 🤖 Techtrix Chatbot 2.0

Welcome to **Techtrix Chatbot 2.0**! 🚀 This chatbot is designed to assist users with queries related to **Techtrix 2025**, the annual tech fest of our college. It provides event details, schedules, and other essential information in real time.

## 📂 Project Structure
```
Techtrix-chatbot-2.0/
├── app/
│   ├── __init__.py
│   ├── api.py          # API endpoints for chatbot
│   ├── langchain_utils.py # Language model utilities
│   ├── config.py       # Configuration settings
│   ├── chat_history.py # Chat history management
├── data/
│   ├── TechFest2025-EventDetails.pdf  # Event details in PDF
├── tests/
│   ├── test_api.py     # API testing scripts
│   ├── test_langchain.py # LangChain utility tests
├── venv/  # Virtual environment (optional)
├── requirements.txt    # Dependencies
├── .gitignore          # Git ignored files
├── README.md           # Project documentation
├── .env                # Environment variables
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
├── .github/
│   ├── workflows/
│   │   ├── ci-cd.yml  # CI/CD Pipeline
```

## 🚀 Features
- Answers queries related to **Techtrix 2025** 📅
- Uses **LangChain** for natural language understanding 🧠
- Provides event details from PDF 📄
- API-based chatbot with history tracking 💬

## 🛠️ Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/palassaha/Techtrix-chatbot-2.0.git
cd Techtrix-chatbot-2.0
```

### 2️⃣ Create & Activate Virtual Environment (Optional)
```sh
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 4️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and configure necessary environment variables.

### 5️⃣ Run the Chatbot Locally
```sh
uvicorn app.api:app --reload
```

### 6️⃣ Use Custom Data
To use your own event details, replace the existing PDF file under the `data/` folder with your own.
Update the corresponding file reference in the langchain_utils.py file under the `app\` folder.


---
Made with ❤️ for Techtrix 2025!

