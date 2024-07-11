from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load trained model and scaler
model = joblib.load(r'C:\sudhanshu_projects\project-task-training-course\Customer Churn Prevention Model\random_forest_model.pkl')
scaler = joblib.load(r'C:\sudhanshu_projects\project-task-training-course\Customer Churn Prevention Model\scaler.pkl')

# DataFrame to store user credentials
users_df = pd.DataFrame(columns=['username', 'password'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    if username in users_df['username'].values:
        flash('Username already exists. Please log in.')
        return redirect(url_for('index'))
    else:
        users_df.loc[len(users_df)] = [username, password]
        flash('Signup successful. Please log in.')
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if ((users_df['username'] == username) & (users_df['password'] == password)).any():
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            flash('Login failed. Please try again.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))
    
    columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    if request.method == 'POST':
        # Collect input values and convert to appropriate types
        input_data = {col: request.form[col] for col in columns}
        input_data['Geography'] = int(LabelEncoder().fit(['delhi', 'bangalore', 'mumbai']).transform([input_data['Geography']])[0])
        input_data['Gender'] = int(LabelEncoder().fit(['Male', 'Female']).transform([input_data['Gender']])[0])
        
        # Perform scaling
        input_df = pd.DataFrame([input_data])
        input_df = scaler.transform(input_df)
        
        # Perform prediction
        prediction = int(model.predict(input_df)[0])  # Convert prediction to standard int
        session['input_data'] = {col: value for col, value in input_data.items()}
        session['email'] = request.form['email']
        session['prediction'] = prediction
        return redirect(url_for('result'))
    return render_template('predict.html', columns=columns)

@app.route('/result')
def result():
    if 'username' not in session:
        return redirect(url_for('login'))
    input_data = session['input_data']
    email = session['email']
    prediction = session['prediction']
    result = 'exited the company' if prediction == 1 else 'did not exit the company'
    
    return render_template('result.html', username=session['username'], input_data=input_data, result=result, email=email)

if __name__ == '__main__':
    app.run(debug=True)
