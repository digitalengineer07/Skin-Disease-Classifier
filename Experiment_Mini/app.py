import os
import numpy as np
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, render_template, jsonify, redirect, url_for, send_file, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

UPLOAD_FOLDER = 'static/uploads'
DATABASE = 'skin_analysis.db'
GOOGLE_CLIENT_ID = "532097536200-dsvfqemgmgnigioha6o651gm3v2t4fkp.apps.googleusercontent.com"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'derm_secure_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT id, username FROM users WHERE id = ?", (int(user_id),))
    row = c.fetchone()
    conn.close()
    if row:
        return User(id=row[0], username=row[1])
    return None


def init_db():                                 ##db initialization
    conn =sqlite3.connect(DATABASE)    
    c=conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            disease_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL,
            notes TEXT,
            user_id INTEGER,
            top3_json TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    try:
        c.execute("ALTER TABLE analysis_history ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass
        
    try:
        c.execute("ALTER TABLE analysis_history ADD COLUMN top3_json TEXT")
    except sqlite3.OperationalError:
        pass

    c.execute('''
        CREATE TABLE IF NOT EXISTS disease_info (
            name TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            symptoms TEXT NOT NULL,
            treatments TEXT NOT NULL,
            prevention TEXT NOT NULL
        )
    ''')

    c.execute("SELECT COUNT(*) FROM disease_info")
    count = c.fetchone()[0]

    if count == 0:
        disease_info = [
            ("Actinic keratosis",
             "A rough, scaly patch on the skin caused by years of sun exposure.",
             "Rough, dry, scaly patches; May be red, tan, pink, or flesh-colored; Usually less than 1 inch in diameter.",
             "Cryotherapy, Topical medications, Photodynamic therapy, Curettage and electrosurgery, Chemical peeling.",
             "Use sunscreen daily; Wear protective clothing; Avoid peak sun hours; Regular skin checks."),
            ("Atopic Dermatitis",
             "A chronic skin condition characterized by itchy, inflamed skin.",
             "Red to brownish-gray patches; Itching, which may be severe; Small, raised bumps; Dry, cracked, scaly skin.",
             "Moisturize regularly; Topical corticosteroids; Immunomodulators; Antihistamines; Light therapy.",
             "Moisturize daily; Identify and avoid triggers; Use mild soaps; Manage stress."),
            ("Benign keratosis",
             "A non-cancerous growth on the skin that develops from skin cells.",
             "Waxy, stuck-on appearance; Light brown to black color; Round or oval shape; Flat or slightly raised.",
             "Often no treatment needed; Cryotherapy; Curettage; Laser therapy.",
             "No specific prevention; Regular skin examinations."),
            ("Dermatofibroma",
             "A common benign skin tumor that presents as a firm nodule.",
             "Firm, round bump; Pink, red, or brown color; May be tender to touch; Usually less than 1 cm in diameter.",
             "Often no treatment needed; Surgical excision if bothersome; Cryotherapy; Steroid injections.",
             "No specific prevention; Protect skin from trauma."),
            ("Melanocytic nevus",
             "A common mole that forms when melanocytes grow in clusters.",
             "Brown or black color; Round shape with well-defined borders; Usually less than 6 mm in diameter; Uniform appearance.",
             "Usually no treatment needed; Surgical removal if suspicious.",
             "Monitor for changes; Protect from sun exposure; Regular skin self-exams."),
            ("Melanoma",
             "The most serious type of skin cancer that develops from pigment-producing cells.",
             "Asymmetrical shape; Irregular border; Varied color; Diameter larger than 6 mm; Evolving size, shape, or color.",
             "Surgical excision; Sentinel lymph node biopsy; Immunotherapy; Targeted therapy; Radiation therapy.",
             "Avoid excessive sun exposure; Use sunscreen; Avoid tanning beds; Regular skin self-exams; Professional skin checks."),
            ("Squamous cell carcinoma",
             "A common form of skin cancer that develops from squamous cells.",
             "Firm, red nodule; Flat sore with crusted surface; New sore or raised area on old scar; Rough, scaly patch on lip.",
             "Surgical excision; Mohs surgery; Radiation therapy; Curettage and electrodesiccation; Topical medications.",
             "Use sunscreen daily; Wear protective clothing; Avoid tanning beds; Check skin regularly."),
            ("Tinea Ringworm Candidiasis",
             "Fungal infections affecting the skin, causing ring-shaped rashes.",
             "Ring-shaped rash; Red, scaly, or cracked skin; Itching; Abnormal nail appearance for nail infections.",
             "Antifungal creams; Oral antifungal medications; Keep affected areas clean and dry.",
             "Practice good hygiene; Don't share personal items; Keep skin dry; Wear clean clothes."),
            ("Vascular lesion",
             "Abnormalities of blood vessels that are visible on the skin.",
             "Red or purple discoloration; May be flat or raised; Can appear anywhere on the body; Sometimes painful.",
             "Laser therapy; Sclerotherapy; Surgical removal; Compression therapy.",
             "Protect skin from sun damage; Avoid trauma to skin; Maintain healthy weight and blood pressure.")
        ]

        c.executemany(
            "INSERT INTO disease_info (name, description, symptoms, treatments, prevention) VALUES (?, ?, ?, ?, ?)",
            disease_info
        )

    conn.commit()
    conn.close()

class_names = ["Actinic keratosis", "Atopic Dermatitis", "Benign keratosis", "Dermatofibroma",
               "Melanocytic nevus", "Melanoma", "Squamous cell carcinoma",
               "Tinea Ringworm Candidiasis", "Vascular lesion"]

# Load the model architecture manually to avoid Keras 3 serialization bugs
base_model = DenseNet121(weights=None, include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])
model.load_weights("skin_disease_model.h5")

# Function to preprocess images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to save analysis history
def save_to_history(image_path, disease_class, confidence, user_id, top3_json="[]"):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    analysis_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute(
        "INSERT INTO analysis_history (id, image_path, disease_class, confidence, timestamp, notes, user_id, top3_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (analysis_id, image_path, disease_class, confidence, timestamp, "", user_id, top3_json)
    )
    conn.commit()
    conn.close()
    return analysis_id

# Function to get analysis history
def get_history(user_id, limit=10):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM analysis_history WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    history = [dict(row) for row in c.fetchall()]
    conn.close()
    return history

# Function to get disease information
def get_disease_info(disease_name):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM disease_info WHERE name = ?",
        (disease_name,)
    )
    info = c.fetchone()
    conn.close()
    if info:
        return dict(info)
    return None

# Auth Routes
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if not username or not password:
            return render_template("register.html", error="Username and password required")
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, generate_password_hash(password)))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("register.html", error="Username already exists")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT id, username, password_hash FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        
        if row and check_password_hash(row[2], password):
            user = User(id=row[0], username=row[1])
            login_user(user)
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/google_auth", methods=["POST"])
def google_auth():
    token = request.form.get('credential')
    if not token:
        flash("Google login failed. No credential received.")
        return redirect(url_for('login'))
        
    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        
        email = idinfo['email']
        
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT id, username FROM users WHERE username = ?", (email,))
        row = c.fetchone()
        
        if row:
            user = User(id=row[0], username=row[1])
            login_user(user)
        else:
            c.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (email, "OAUTH_LOGIN"))
            conn.commit()
            c.execute("SELECT id, username FROM users WHERE username = ?", (email,))
            row = c.fetchone()
            user = User(id=row[0], username=row[1])
            login_user(user)
            
        conn.close()
        return redirect(url_for('index'))
        
    except ValueError:
        flash("Invalid Google token. Check your CLIENT ID configuration.")
        return redirect(url_for('login'))

# Main route for the application
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        file.save(image_path)

        img_array = preprocess_image(image_path)
        predictions = model.predict(img_array)[0]
        
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = [{"disease": class_names[i], "confidence": float(predictions[i] * 100)} for i in top_3_indices]
        
        predicted_class = top_3[0]["disease"]
        confidence = top_3[0]["confidence"]
        top3_json = json.dumps(top_3)

        analysis_id = save_to_history(image_path, predicted_class, confidence, current_user.id, top3_json)
        
        return redirect(url_for('index', analysis_id=analysis_id))

    analysis_id = request.args.get('analysis_id')
    if analysis_id:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM analysis_history WHERE id = ? AND user_id = ?", (analysis_id, current_user.id))
        record = c.fetchone()
        conn.close()
        
        if record:
            disease_info = get_disease_info(record['disease_class'])
            history = get_history(current_user.id, 5)
            
            top_3 = []
            if 'top3_json' in record.keys() and record['top3_json']:
                try:
                    top_3 = json.loads(record['top3_json'])
                except:
                    pass
            if not top_3:
                top_3 = [{"disease": record['disease_class'], "confidence": record['confidence']}]
                
            return render_template("index.html",
                                  image_path=record['image_path'],
                                  predicted_class=record['disease_class'],
                                  confidence=record['confidence'],
                                  top_3=top_3,
                                  disease_info=disease_info,
                                  history=history,
                                  analysis_id=record['id'])

    history = get_history(current_user.id, 5)
    return render_template("index.html", history=history)

# Route for history page
@app.route("/history")
@login_required
def history():
    all_history = get_history(current_user.id, limit=100)
    return render_template("history.html", history=all_history)

# API endpoint for history
@app.route("/api/history", methods=["GET"])
@login_required
def api_history():
    limit = request.args.get('limit', 10, type=int)
    history = get_history(current_user.id, limit)
    return jsonify(history)

# API endpoint for disease info
@app.route("/api/disease/<disease_name>")
def api_disease_info(disease_name):
    info = get_disease_info(disease_name)
    if info:
        return jsonify(info)
    return jsonify({"error": "Disease not found"}), 404

# API endpoint to delete analysis
@app.route("/api/history/<analysis_id>", methods=["DELETE"])
@login_required
def delete_analysis(analysis_id):
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT image_path FROM analysis_history WHERE id = ? AND user_id = ?", (analysis_id, current_user.id))
    result = c.fetchone()

    if result:
        image_path = result[0]
        c.execute("DELETE FROM analysis_history WHERE id = ?", (analysis_id,))
        conn.commit()

        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({"success": True})

    conn.close()
    return jsonify({"error": "Analysis not found"}), 404

# API endpoint to clear history
@app.route("/api/history/clear", methods=["POST"])
@login_required
def clear_history():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()

    c.execute("SELECT image_path FROM analysis_history WHERE user_id = ?", (current_user.id,))
    image_paths = [row[0] for row in c.fetchall()]

    c.execute("DELETE FROM analysis_history WHERE user_id = ?", (current_user.id,))
    conn.commit()
    conn.close()

    for path in image_paths:
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"success": True})

# API endpoint to export analysis as JSON
@app.route("/api/export/<analysis_id>")
@login_required
def export_analysis(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM analysis_history WHERE id = ? AND user_id = ?", (analysis_id, current_user.id))
    analysis = c.fetchone()

    if not analysis:
        conn.close()
        return jsonify({"error": "Analysis not found"}), 404

    analysis_dict = dict(analysis)
    disease_info = get_disease_info(analysis_dict["disease_class"])

    export_data = {
        "analysis": analysis_dict,
        "disease_info": disease_info
    }

    conn.close()
    return jsonify(export_data)

# API endpoint to export analysis as PDF
@app.route("/api/export/<analysis_id>/pdf")
@login_required
def export_analysis_pdf(analysis_id):
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM analysis_history WHERE id = ? AND user_id = ?", (analysis_id, current_user.id))
    analysis = c.fetchone()

    if not analysis:
        conn.close()
        return jsonify({"error": "Analysis not found"}), 404

    analysis_dict = dict(analysis)
    disease_info = get_disease_info(analysis_dict["disease_class"])

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Add content to the PDF
    story.append(Paragraph("Skin Disease Analysis Report", styles['h1']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Analysis ID: {analysis_id}", styles['Normal']))
    story.append(Paragraph(f"Disease Class: {analysis_dict['disease_class']}", styles['Normal']))
    story.append(Paragraph(f"Confidence: {analysis_dict['confidence']}", styles['Normal']))

    if disease_info:
        story.append(Paragraph("Disease Information", styles['h2']))
        story.append(Paragraph(f"Description: {disease_info['description']}", styles['Normal']))
        story.append(Paragraph(f"Symptoms: {disease_info['symptoms']}", styles['Normal']))
        story.append(Paragraph(f"Treatments: {disease_info['treatments']}", styles['Normal']))
        story.append(Paragraph(f"Prevention: {disease_info['prevention']}", styles['Normal']))

    doc.build(story)
    buffer.seek(0)

    conn.close()

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'skin_analysis_{analysis_id}.pdf'
    )

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
