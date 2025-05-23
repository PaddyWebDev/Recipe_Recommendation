#!/bin/bash

# Name of your virtual environment
VENV_DIR=".venv"

# 1. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "🛠️ Creating virtual environment..."
  py -m venv "$VENV_DIR"
else
    source .venv/Scripts/activate
fi

# 4. Set Flask environment variables and run the app
clear
echo "🌐 Starting Flask app..."
export FLASK_APP=app.py
export FLASK_ENV=development  # optional: enables debug mode
flask run
