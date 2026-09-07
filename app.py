import os
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

from train_model import train_and_save_model, MODEL_FILE

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
DATABASE = "users.db"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

FEATURES = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]

def init_db():
    con = sqlite3.connect(DATABASE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

def get_model():
    if not os.path.exists(MODEL_FILE):
        train_and_save_model()
    return joblib.load(MODEL_FILE)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_from_form(form):
    row = {
        "Age": float(form["Age"]),
        "Gender": form["Gender"],
        "TB": float(form["TB"]),
        "DB": float(form["DB"]),
        "Alkphos": float(form["Alkphos"]),
        "Sgpt": float(form["Sgpt"]),
        "Sgot": float(form["Sgot"]),
        "TP": float(form["TP"]),
        "ALB": float(form["ALB"]),
        "A/G Ratio": float(form["AG"]),
    }
    df = pd.DataFrame([row], columns=FEATURES)
    model = get_model()
    prediction = int(model.predict(df)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1]) * 100

    if prediction == 1:
        result = "Higher predicted risk in this model"
        message = (
            "The model classified this input as the liver-disease class in the training dataset. "
            "This is a screening result, not a medical diagnosis."
        )
    else:
        result = "Lower predicted risk in this model"
        message = (
            "The model classified this input as the non-liver-disease class in the training dataset. "
            "This is a screening result, not a medical diagnosis."
        )

    return result, message, probability

@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        if not username or not password:
            flash("Enter both username and password.")
            return redirect(url_for("register"))

        con = sqlite3.connect(DATABASE)
        try:
            con.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            con.commit()
            flash("Registration successful. Please log in.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("That username already exists.")
            return redirect(url_for("register"))
        finally:
            con.close()

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        con = sqlite3.connect(DATABASE)
        row = con.execute(
            "SELECT username FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        con.close()

        if row:
            session["username"] = username
            return redirect(url_for("dashboard"))

        flash("Invalid username or password.")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            result, message, probability = predict_from_form(request.form)
            return render_template(
                "result.html",
                result=result,
                message=message,
                probability=probability
            )
        except Exception as e:
            flash("Please enter valid numeric values in all laboratory fields.")
            return redirect(url_for("predict"))

    return render_template("predict.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("report")
        if not file or file.filename == "":
            flash("Please select an LFT report image.")
            return redirect(url_for("upload"))

        if not allowed_file(file.filename):
            flash("Use a PNG, JPG, or JPEG image.")
            return redirect(url_for("upload"))

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        return render_template(
            "upload_result.html",
            filename=filename,
            message="The report image was uploaded. Because medical report layouts vary, verify/read the values manually and enter them in the prediction form before running the model."
        )

    return render_template("upload.html")

if __name__ == "__main__":
    init_db()
    print("Starting Liver Disease Prediction App...")
    print("The ML model will be trained automatically the first time it is needed.")
    app.run(debug=True)
