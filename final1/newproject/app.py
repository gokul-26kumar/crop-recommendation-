from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import numpy as np
import joblib
import sqlite3
import hashlib

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained model and scaler
model = joblib.load('crop_recommendation_model.pkl')
scaler = joblib.load('scaler.pkl')

# Crop dictionary
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
    10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean",
    18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

# Initialize the SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Hashing the password for security
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Home route (redirects to login page if not logged in)
@app.route('/')
def home():
    if 'username' in session:
        return render_template("home.html")
    return redirect(url_for('login'))

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return "Invalid credentials"

    return render_template('login.html')

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = hash_password(request.form['password'])

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()

        # Check if username already exists
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            return "Username already exists!"

        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()

        return redirect(url_for('login'))

    return render_template('signup.html')

# Route to logout and clear session
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Predict route (handles form submission for crop prediction)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect form data
            N = int(request.form["Nitrogen"])
            P = int(request.form["Phosporus"])
            K = int(request.form["Potassium"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])

            # Create an array of the input features
            feature_list = [N, P, K, temperature, humidity, ph, rainfall]
            single_pred = np.array(feature_list).reshape(1, -1)

            # Scale features and make prediction
            scaled_features = scaler.transform(single_pred)
            prediction = model.predict(scaled_features)[0]

            # Get the predicted crop
            crop = crop_dict.get(prediction, "Unknown")
            result = f"{crop} is the best crop for cultivation." if crop != "Unknown" else "Sorry, we can't recommend a crop here."

            return render_template('predict.html', result=result)

        except Exception as e:
            return render_template('predict.html', result=f"Error: {str(e)}")
    else:
        return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    init_db()  # Initialize the database on app startup
    app.run(debug=True)
