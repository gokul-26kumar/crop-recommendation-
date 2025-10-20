🌱 Crop Recommendation Web App using Random Forest
📌 Overview
This is a Flask-based web application that recommends the most suitable crop to cultivate based on soil and climate conditions. It uses a trained Random Forest model to make predictions and provides a user-friendly interface for farmers, agronomists, and researchers.
🧠 Core Features
- 🔍 Predicts optimal crop based on user input
- 🧪 Uses Random Forest Classifier for robust predictions
- 🖥️ Web interface with login/signup functionality
- 📊 Scaled inputs using pre-trained scaler
- 🗃️ User authentication via SQLite database
🗂️ Project Structure
newproject/
├── static/                          # Static assets (CSS, JS, images)
├── templates/                      # HTML templates
│   ├── about.html
│   ├── home.html
│   ├── index.html
│   ├── login.html
│   ├── predict.html
│   └── signup.html
├── app.py                          # Main Flask application
├── model.py                        # ML model loading and prediction logic
├── Crop_recommendation.csv         # Dataset used for training
├── crop_recommendation_model.pkl   # Trained Random Forest model
├── scaler.pkl                      # Scaler for input normalization
├── users.db                        # SQLite database for user accounts
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies


🛠️ Installation
- Clone the repository:
git clone https://github.com/yourusername/newproject.git
cd newproject
- Install dependencies:
pip install -r requirements.txt
- Run the Flask app:
python app.py


🌐 Web Pages
- index.html: Landing page
- login.html / signup.html: User authentication
- home.html: Dashboard after login
- predict.html: Crop prediction form
- about.html: Info about the system
📈 Model Details
- Algorithm: Random Forest Classifier
- Inputs:
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- pH
- Rainfall (mm)
- Output: Recommended crop name
🧪 Sample Prediction Flow
# model.py
import pickle

model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def predict_crop(data):
    scaled = scaler.transform([data])
    prediction = model.predict(scaled)
    return prediction[0]


🔐 User Management
- SQLite database (users.db) stores user credentials
- Flask sessions used for login/logout flow
📦 Dependencies
- Flask
- scikit-learn
- pandas
- numpy
- sqlite3
- Werkzeug (for password hashing)

🙌 Credits
Developed by Gokul
Student & Aspiring Data Scientist
Dayananda Sagar Academy of Technology and Management

