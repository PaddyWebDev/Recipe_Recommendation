#!/bin/bash

# Name of your virtual environment
VENV_DIR=".venv"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "🛠️ Creating virtual environment..."
  py -m venv "$VENV_DIR"
fi

# 2. Activate the virtual environment
echo "🚀 Activating virtual environment..."
source "$VENV_DIR/Scripts/activate"

# 3. Upgrade pip and install required packages
echo "📦 Installing packages..."
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt
else
  # Add the basic ML/Flask dependencies you need
  pip install flask scikit-learn numpy pandas
fi

# 4. Set Flask environment variables and run the app
echo "🌐 Starting Flask app..."
export FLASK_APP=app.py
export FLASK_ENV=development  # optional: enables debug mode
flask run
