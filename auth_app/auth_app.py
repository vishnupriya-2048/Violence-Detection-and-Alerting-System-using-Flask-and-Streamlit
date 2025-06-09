import os
from dotenv import load_dotenv # Import load_dotenv for .env file support
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables from .env file (for local development)
# This line should be at the very top of your script.
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# Get SECRET_KEY from environment variable
# In production, ensure FLASK_SECRET_KEY is always set in your deployment environment.
# 'fallback_dev_key' is only for local testing if the env var isn't set.
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_fallback_dev_key_if_env_not_set')

# Database configuration. For production, consider using a managed database service.
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' # Route name for login page, used for redirection

# User Model for database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    phone_number = db.Column(db.String(20), nullable=False)

    def set_password(self, password):
        """Hashes the given password and stores it."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Checks if the given password matches the stored hash."""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        """String representation of the User object."""
        return f'<User {self.email}>'

# User loader function required by Flask-Login
@login_manager.user_loader
def load_user(user_id):
    """Loads a user from the database given their ID."""
    return User.query.get(int(user_id))

# Login route
@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    # If the user is already authenticated, redirect them directly to the Streamlit app
    if current_user.is_authenticated:
        # Get Streamlit app URL from environment variable
        return redirect(os.environ.get('STREAMLIT_APP_URL', 'http://localhost:8501'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        # Check if user exists and password is correct
        if user and user.check_password(password):
            login_user(user, remember=True) # Log in the user; 'remember=True' for persistent session
            flash('Logged in successfully!', 'success') # Flash a success message
            # Redirect to the Streamlit app after successful login
            return redirect(os.environ.get('STREAMLIT_APP_URL', 'http://localhost:8501'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger') # Flash an error message
    
    return render_template('login.html') # Render the login form for GET requests or failed POSTs

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # If the user is already authenticated, redirect them directly to the Streamlit app
    if current_user.is_authenticated:
        return redirect(os.environ.get('STREAMLIT_APP_URL', 'http://localhost:8501'))

    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        phone_number = request.form.get('phone_number')

        # Check if email already exists
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email already registered. Please login or use a different email.', 'warning')
            return redirect(url_for('signup'))
        
        # Create a new user instance
        new_user = User(name=name, email=email, phone_number=phone_number)
        new_user.set_password(password) # Hash and set the password
        
        try:
            db.session.add(new_user) # Add new user to the database session
            db.session.commit() # Commit the transaction
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login')) # Redirect to login page after successful signup
        except Exception as e:
            db.session.rollback() # Rollback in case of an error
            flash(f'An error occurred during registration: {e}', 'danger')
            return redirect(url_for('signup'))

    return render_template('signup.html') # Render the signup form

# Logout route
@app.route('/logout')
@login_required # Requires user to be logged in to access this route
def logout():
    logout_user() # Log out the current user
    flash('You have been logged out.', 'info') # Flash a logout message
    return redirect(url_for('login')) # Redirect to the login page

if __name__ == '__main__':
    # Ensure database tables are created within the application context
    with app.app_context():
        db.create_all() 
    # Run the Flask development server
    # debug=True should be False in production
    app.run(debug=True, port=5000)