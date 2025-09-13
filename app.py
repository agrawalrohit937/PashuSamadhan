# # # app.py (No major changes needed, your backend is solid)

# # import os
# # import json
# # from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
# # import tensorflow as tf
# # from tensorflow.keras.applications.efficientnet import preprocess_input
# # import numpy as np
# # import cv2
# # from werkzeug.utils import secure_filename
# # import logging
# # import pandas as pd

# # # Configure logging
# # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # # Initialize Flask app
# # app = Flask(__name__)
# # app.config['SECRET_KEY'] = 'a-very-strong-secret-key-for-siH-2025' # Better to use a strong key
# # app.config['UPLOAD_FOLDER'] = 'uploads/'
# # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max upload size: 16MB
# # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# # # --- Model Loading (Global Scope for efficiency) ---
# # MODEL = None
# # CLASS_NAMES = []
# # IMG_SIZE = 300 # Must match your training img_size

# # def build_model_architecture(num_classes, img_size=300, base='efficientnetb3', dropout=0.3):
# #     """
# #     Rebuilds the exact model architecture as in your notebook.
# #     This function should exactly mirror the model definition from your training script.
# #     """
# #     inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    
# #     if base.lower() == 'efficientnetb3':
# #         from tensorflow.keras.applications import EfficientNetB3
# #         backbone = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
# #     else:
# #         raise ValueError(f'Unsupported base model: {base}')
    
# #     x = backbone.outputs[0]
# #     x = tf.keras.layers.Dropout(dropout)(x)
# #     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
# #     model = tf.keras.models.Model(inputs, outputs)
# #     return model

# # def load_model_and_labels():
# #     global MODEL, CLASS_NAMES
# #     logging.info("Attempting to load model and labels...")
# #     try:
# #         # Load label map
# #         label_map_path = os.path.join(app.root_path, 'model', 'label_map.json')
# #         if not os.path.exists(label_map_path):
# #             logging.error(f"label_map.json not found at {label_map_path}")
# #             return
# #         with open(label_map_path, 'r') as f:
# #             label_map = json.load(f)
# #             CLASS_NAMES = sorted(label_map, key=label_map.get)
# #         logging.info(f"Successfully loaded {len(CLASS_NAMES)} class names: {CLASS_NAMES}")

# #         num_classes = len(CLASS_NAMES)
        
# #         # Build the model architecture
# #         MODEL = build_model_architecture(num_classes=num_classes, img_size=IMG_SIZE, base='efficientnetb3', dropout=0.3)

# #         # Load the trained weights
# #         model_weights_path = os.path.join(app.root_path, 'model', 'best_model_weights.h5')
# #         if not os.path.exists(model_weights_path):
# #             logging.error(f"Model weights not found at {model_weights_path}")
# #             return
# #         MODEL.load_weights(model_weights_path)
# #         logging.info("Model weights loaded successfully.")
# #     except Exception as e:
# #         logging.exception(f"CRITICAL ERROR: Failed to load model or labels: {e}")
# #         MODEL = None
# #         CLASS_NAMES = []

# # with app.app_context():
# #     load_model_and_labels()

# # # Your BREED_INFO dictionary (no changes)
# # BREED_INFO = {
# #     "Sahiwal": {
# #         "description": "The Sahiwal is a prominent breed of Zebu cattle...",
# #         "characteristics": ["High milk production", "Excellent heat tolerance", "Strong disease resistance", "Gentle and docile temperament", "Reddish-brown coat"],
# #         "origin": "Punjab, Pakistan",
# #         "image": "Sahiwal.png"
# #     },
# #     "Gir": {
# #         "description": "The Gir or Gyr is one of the most famous Zebu dairy cattle breeds...",
# #         "characteristics": ["High milk fat content", "Resilient and adaptable", "Disease resistant", "Distinctive prominent forehead and long ears", "Red, yellowish-white, or spotted coat"],
# #         "origin": "Gir Hills, Gujarat, India",
# #         "image": "Gir.png"
# #     },
# #     "Holstein_Friesian": {
# #         "description": "The Holstein Friesian, commonly known as Holstein, is the world's most widespread dairy cattle breed...",
# #         "characteristics": ["Highest milk yield", "Large body size", "Distinctive black and white (or red and white) markings", "Efficient feed conversion"],
# #         "origin": "Netherlands / Germany",
# #         "image": "Holstein_Friesian.png"
# #     },
# #     "Jaffrabadi": {
# #         "description": "The Jaffrabadi is a heavy breed of water buffalo...",
# #         "characteristics": ["High butterfat milk", "Large and robust body", "Distinctive curled horns", "Good for draught power", "Strong and hardy constitution"],
# #         "origin": "Gujarat, India",
# #         "image": "Jaffrabadi.png"
# #     },
# #     "Jersey": {
# #         "description": "Jersey cattle are a small breed of dairy cattle...",
# #         "characteristics": ["High butterfat and protein milk", "Smaller body size", "Fawn to dark brown coat", "Docile temperament", "Efficient grazers"],
# #         "origin": "Island of Jersey",
# #         "image": "Jersey.png"
# #     },
# #     "Murrah": {
# #         "description": "The Murrah is a renowned breed of water buffalo...",
# #         "characteristics": ["High milk yield with rich fat content", "Jet black coat", "Short, tightly curled horns", "Adaptable to various climates", "Strong and healthy calves"],
# #         "origin": "Punjab / Haryana, India",
# #         "image": "Murrah.png"
# #     }
# # }

# # def allowed_file(filename):
# #     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # @app.route('/')
# # def index():
# #     return render_template('index.html', title="Home")

# # @app.route('/about')
# # def about():
# #     return render_template('about.html', title="About Us")

# # @app.route('/breeds')
# # def list_breeds():
# #     breeds_for_display = []
# #     for breed_name in CLASS_NAMES:
# #         info = BREED_INFO.get(breed_name, {})
# #         breeds_for_display.append({
# #             'name': breed_name,
# #             'description_short': info.get('description', 'No description available.')[:100] + '...',
# #             'image': info.get('image', 'default_cattle.jpg')
# #         })
# #     return render_template('breeds.html', title="Explore Breeds", breeds=breeds_for_display)

# # @app.route('/breeds/<breed_name>')
# # def breed_detail(breed_name):
# #     info = BREED_INFO.get(breed_name)
# #     if not info:
# #         flash(f"Information for '{breed_name}' not found.", "warning")
# #         return redirect(url_for('list_breeds'))
# #     return render_template('breed_detail.html', title=f"{breed_name} Breed Info", breed_name=breed_name, info=info)

# # @app.route('/predict', methods=['GET'])
# # def predict_get():
# #     return render_template('predict.html', title="Predict Breed")

# # @app.route('/api/predict', methods=['POST'])
# # def api_predict():
# #     if MODEL is None or not CLASS_NAMES:
# #         return jsonify({'error': 'Model not loaded on server.', 'category': 'error'}), 500

# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file part in the request.', 'category': 'error'}), 400
# #     file = request.files['file']

# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file.', 'category': 'error'}), 400

# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
# #         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# #         file.save(filepath)
# #         logging.info(f"File saved to {filepath}")

# #         try:
# #             img_bgr = cv2.imread(filepath)
# #             if img_bgr is None:
# #                 os.remove(filepath)
# #                 return jsonify({'error': 'Could not read image. Please ensure it is a valid image file.', 'category': 'error'}), 400
                
# #             img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# #             img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
# #             x = img_resized.astype('float32')
# #             x = preprocess_input(x) # Using EfficientNet's preprocess_input
# #             x = np.expand_dims(x, 0)

# #             preds = MODEL.predict(x)[0]
# #             idxs = preds.argsort()[-3:][::-1] # Top 3 predictions

# #             results = []
# #             for i in idxs:
# #                 results.append({'breed': CLASS_NAMES[i], 'score': float(f"{preds[i]*100:.2f}")})
            
# #             return jsonify({
# #                 'success': True,
# #                 'results': results,
# #                 'image_url': url_for('uploaded_file', filename=filename)
# #             })

# #         except Exception as e:
# #             logging.exception("An error occurred during prediction API call.")
# #             if os.path.exists(filepath):
# #                 os.remove(filepath) 
# #             return jsonify({'error': f"An internal server error occurred: {str(e)}", 'category': 'error'}), 500
# #     else:
# #         return jsonify({'error': 'Invalid file type. Allowed types are png, jpg, jpeg, webp.', 'category': 'error'}), 400

# # @app.route('/uploads/<filename>')
# # def uploaded_file(filename):
# #     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# # @app.route('/static/img/breeds/<filename>')
# # def serve_breed_image(filename):
# #     return send_from_directory(os.path.join(app.root_path, 'static', 'img', 'breeds'), filename)

# # if __name__ == '__main__':
# #     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# #     os.makedirs(os.path.join(app.root_path, 'static', 'img', 'breeds'), exist_ok=True)
# #     app.run(debug=True)

# # # app.py (Add these new sections)

# # # ... (existing imports)
# # import joblib

# # # ... (after existing Flask App Initialization)

# # # --- NEW: Disease Diagnosis Model Loading ---
# # DISEASE_MODEL = None
# # DISEASE_ENCODER = None
# # ANIMAL_ENCODER = None
# # ALL_SYMPTOMS = [] # To store all possible symptoms for feature engineering
# # FEATURE_COLUMNS = [] # To store all columns model was trained on

# # def load_disease_model():
# #     global DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER, ALL_SYMPTOMS, FEATURE_COLUMNS
# #     logging.info("Attempting to load disease diagnosis model...")
# #     try:
# #         model_path = os.path.join(app.root_path, 'disease_model', 'best_disease_model.pkl')
# #         DISEASE_MODEL = joblib.load(model_path)
        
# #         disease_encoder_path = os.path.join(app.root_path, 'disease_model', 'disease_encoder.pkl')
# #         DISEASE_ENCODER = joblib.load(disease_encoder_path)
        
# #         animal_encoder_path = os.path.join(app.root_path, 'disease_model', 'animal_encoder.pkl')
# #         ANIMAL_ENCODER = joblib.load(animal_encoder_path)
        
# #         symptoms_path = os.path.join(app.root_path, 'disease_model', 'symptom_list.json')
# #         with open(symptoms_path, 'r') as f:
# #             ALL_SYMPTOMS = json.load(f)
            
# #         # Create the list of feature columns in the exact order the model expects
# #         # It should be ['Animal'] followed by all the symptom columns
# #         FEATURE_COLUMNS = ['Animal'] + ALL_SYMPTOMS
        
# #         logging.info("Disease diagnosis model and encoders loaded successfully.")
# #     except Exception as e:
# #         logging.exception(f"CRITICAL ERROR: Failed to load disease model: {e}")

# # # Call the new load function within the app context
# # with app.app_context():
# #     load_model_and_labels() # Your existing function
# #     load_disease_model()    # The new function


# # # ... (before existing API routes for breed prediction)

# # # --- NEW: Disease Diagnosis Page and API Routes ---
# # @app.route('/disease')
# # def disease_page():
# #     return render_template('disease.html', title="Disease Diagnosis")

# # @app.route('/api/symptoms')
# # def get_symptoms():
# #     """Provides the full list of symptoms to the frontend for autocomplete."""
# #     if not ALL_SYMPTOMS:
# #         return jsonify({"error": "Symptom list not available"}), 500
# #     return jsonify(ALL_SYMPTOMS)

# # @app.route('/api/diagnose', methods=['POST'])
# # def api_diagnose():
# #     if not all([DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER]):
# #         return jsonify({"error": "Diagnosis model is not ready."}), 500
        
# #     data = request.get_json()
# #     if not data or 'animal' not in data or 'symptoms' not in data:
# #         return jsonify({"error": "Invalid input. Animal and symptoms are required."}), 400

# #     animal = data['animal']
# #     selected_symptoms = data['symptoms']

# #     try:
# #         # --- Create input vector for the model ---
# #         # 1. Create a DataFrame with a single row and all the necessary columns, initialized to 0
# #         input_df = pd.DataFrame(columns=FEATURE_COLUMNS, index=[0])
# #         input_df = input_df.fillna(0)

# #         # 2. Encode the animal type
# #         encoded_animal = ANIMAL_ENCODER.transform([animal])[0]
# #         input_df['Animal'] = encoded_animal

# #         # 3. Multi-hot encode the selected symptoms
# #         for symptom in selected_symptoms:
# #             if symptom in input_df.columns:
# #                 input_df[symptom] = 1
        
# #         # Ensure column order is exactly as the model expects
# #         input_df = input_df[FEATURE_COLUMNS]

# #         # --- Prediction ---
# #         prediction_proba = DISEASE_MODEL.predict_proba(input_df)[0]
# #         top_prediction_index = prediction_proba.argmax()
        
# #         predicted_disease = DISEASE_ENCODER.inverse_transform([top_prediction_index])[0]
# #         confidence_score = round(prediction_proba[top_prediction_index] * 100, 2)

# #         return jsonify({
# #             "success": True,
# #             "disease": predicted_disease,
# #             "confidence": confidence_score,
# #             "key_symptoms": selected_symptoms # For the "Explainable AI" part
# #         })

# #     except Exception as e:
# #         logging.exception("An error occurred during diagnosis.")
# #         return jsonify({"error": str(e)}), 500

# import os
# import json
# from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
# import tensorflow as tf
# from tensorflow.keras.applications.efficientnet import preprocess_input
# import numpy as np
# import cv2
# from werkzeug.utils import secure_filename
# import logging
# import joblib  # For loading .pkl model
# import pandas as pd  # For creating the model's input DataFrame

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize Flask app
# app = Flask(__name__)
# app.config['SECRET_KEY'] = 'a-very-strong-secret-key-for-siH-2025'
# app.config['UPLOAD_FOLDER'] = 'uploads/'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# # --- Breed Recognition Model Loading ---
# MODEL = None
# CLASS_NAMES = []
# IMG_SIZE = 300

# def build_model_architecture(num_classes, img_size=300, base='efficientnetb3', dropout=0.3):
#     inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
#     if base.lower() == 'efficientnetb3':
#         from tensorflow.keras.applications import EfficientNetB3
#         backbone = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
#     else:
#         raise ValueError(f'Unsupported base model: {base}')
#     x = backbone.outputs[0]
#     x = tf.keras.layers.Dropout(dropout)(x)
#     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
#     model = tf.keras.models.Model(inputs, outputs)
#     return model

# def load_model_and_labels():
#     global MODEL, CLASS_NAMES
#     logging.info("Attempting to load breed recognition model and labels...")
#     try:
#         label_map_path = os.path.join(app.root_path, 'model', 'label_map.json')
#         with open(label_map_path, 'r') as f:
#             label_map = json.load(f)
#             CLASS_NAMES = sorted(label_map, key=label_map.get)
#         logging.info(f"Successfully loaded {len(CLASS_NAMES)} breed class names: {CLASS_NAMES}")
#         num_classes = len(CLASS_NAMES)
#         MODEL = build_model_architecture(num_classes=num_classes, img_size=IMG_SIZE)
#         model_weights_path = os.path.join(app.root_path, 'model', 'best_model_weights.h5')
#         MODEL.load_weights(model_weights_path)
#         logging.info("Breed recognition model weights loaded successfully.")
#     except Exception as e:
#         logging.exception(f"CRITICAL ERROR: Failed to load breed model or labels: {e}")
#         MODEL = None
#         CLASS_NAMES = []

# # --- Disease Diagnosis Model Loading ---
# DISEASE_MODEL = None
# DISEASE_ENCODER = None
# ANIMAL_ENCODER = None
# ALL_SYMPTOMS = []
# FEATURE_COLUMNS = []

# # app.py

# # ... (imports)

# # --- Disease Diagnosis Model Loading ---
# # ... (existing variables)

# def load_disease_model():
#     global DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER, ALL_SYMPTOMS, FEATURE_COLUMNS
#     logging.info("Attempting to load disease diagnosis model...")
#     try:
#         model_path = os.path.join(app.root_path, 'disease_model', 'best_disease_model.pkl')
#         DISEASE_MODEL = joblib.load(model_path)
        
#         disease_encoder_path = os.path.join(app.root_path, 'disease_model', 'disease_encoder.pkl')
#         DISEASE_ENCODER = joblib.load(disease_encoder_path)
        
#         animal_encoder_path = os.path.join(app.root_path, 'disease_model', 'animal_encoder.pkl')
#         ANIMAL_ENCODER = joblib.load(animal_encoder_path)
        
#         # --- CHANGE HERE ---
#         # Ab hum column order ko manually nahi banayenge, balki file se load karenge
#         columns_path = os.path.join(app.root_path, 'disease_model', 'feature_columns.json')
#         with open(columns_path, 'r') as f:
#             FEATURE_COLUMNS = json.load(f)
            
#         # Symptom list abhi bhi autocomplete ke liye zaroori hai
#         symptoms_path = os.path.join(app.root_path, 'disease_model', 'symptom_list.json')
#         with open(symptoms_path, 'r') as f:
#             ALL_SYMPTOMS = json.load(f)
        
#         logging.info("Disease diagnosis model, encoders, and feature columns loaded successfully.")
#     except Exception as e:
#         logging.exception(f"CRITICAL ERROR: Failed to load disease model: {e}")

# # ... (rest of your app.py file)
# # def load_disease_model():
# #     global DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER, ALL_SYMPTOMS, FEATURE_COLUMNS
# #     logging.info("Attempting to load disease diagnosis model...")
# #     try:
# #         model_path = os.path.join(app.root_path, 'disease_model', 'best_disease_model.pkl')
# #         DISEASE_MODEL = joblib.load(model_path)
        
# #         disease_encoder_path = os.path.join(app.root_path, 'disease_model', 'disease_encoder.pkl')
# #         DISEASE_ENCODER = joblib.load(disease_encoder_path)
        
# #         animal_encoder_path = os.path.join(app.root_path, 'disease_model', 'animal_encoder.pkl')
# #         ANIMAL_ENCODER = joblib.load(animal_encoder_path)
        
# #         symptoms_path = os.path.join(app.root_path, 'disease_model', 'symptom_list.json')
# #         with open(symptoms_path, 'r') as f:
# #             ALL_SYMPTOMS = json.load(f)
            
# #         # Create the list of feature columns in the exact order the model expects
# #         FEATURE_COLUMNS = ['Animal', 'Age', 'Temperature'] + ALL_SYMPTOMS
        
# #         logging.info("Disease diagnosis model and encoders loaded successfully.")
# #     except Exception as e:
# #         logging.exception(f"CRITICAL ERROR: Failed to load disease model: {e}")

# # Load all models when the app starts
# with app.app_context():
#     load_model_and_labels()
#     load_disease_model()

# # --- Breed Info Data ---
# BREED_INFO = {
#     "Sahiwal": {
#         "description": "The Sahiwal is a prominent breed of Zebu cattle, originally from the Sahiwal district of Punjab, Pakistan. Renowned for its exceptional milk production, it exhibits strong heat tolerance and natural resistance to various parasites, making it ideal for tropical climates.",
#         "characteristics": ["High milk production", "Excellent heat tolerance", "Strong disease resistance", "Gentle and docile temperament", "Reddish-brown coat"],
#         "origin": "Punjab, Pakistan",
#         "image": "sahiwal.png" # Using .jpg as generated
#     },
#     "Gir": {
#         "description": "The Gir or Gyr is one of the most famous Zebu dairy cattle breeds, originating from the Gir Hills of Gujarat, India. It's easily recognizable by its distinctive convex forehead, long pendulous ears, and a generally robust, hardy constitution. Gir cattle are known for their milk quality and adaptability.",
#         "characteristics": ["High milk fat content", "Resilient and adaptable", "Disease resistant", "Distinctive prominent forehead and long ears", "Red, yellowish-white, or spotted coat"],
#         "origin": "Gir Hills, Gujarat, India",
#         "image": "gir.png" # Using .jpg as generated
#     },
#     "Holstein_Friesian": { # Changed to match your model output key
#         "description": "The Holstein Friesian, commonly known as Holstein, is the world's most widespread dairy cattle breed, originating from the Netherlands and Germany. They are celebrated for their unmatched milk production capabilities, making them the backbone of commercial dairy farming globally.",
#         "characteristics": ["Highest milk yield", "Large body size", "Distinctive black and white (or red and white) markings", "Efficient feed conversion"],
#         "origin": "Netherlands / Germany",
#         "image": "Holstein_Friesian.png" # Using .jpg as generated
#     },
#     "Jaffrabadi": {
#         "description": "The Jaffrabadi is a heavy breed of water buffalo, primarily found in the Saurashtra region of Gujarat, India. These majestic animals are known for their massive build, strong horns that curve back and then loop forward, and their impressive milk production, especially for butterfat content. They are highly adaptable and well-suited for both milk and draught purposes.",
#         "characteristics": ["High butterfat milk", "Large and robust body", "Distinctive curled horns", "Good for draught power", "Strong and hardy constitution"],
#         "origin": "Gujarat, India",
#         "image": "jaffrabadi.png" # You'll need to generate this image
#     },
#     "Jersey": {
#         "description": "Jersey cattle are a small breed of dairy cattle originating from the British Channel Island of Jersey. Despite their smaller size, they are highly valued for the high butterfat and protein content of their milk, often having a rich golden color. They are known for their docile nature and efficient grazing.",
#         "characteristics": ["High butterfat and protein milk", "Smaller body size", "Fawn to dark brown coat", "Docile temperament", "Efficient grazers"],
#         "origin": "Island of Jersey",
#         "image": "Jersy.png" # Using .jpg as generated
#     },
#     "Murrah": {
#         "description": "The Murrah is a renowned breed of water buffalo primarily kept for milk production, originating from the Punjab and Haryana states of India. They are characterized by their jet-black coat, short and tightly curled horns, and impressive milk yield. Murrah buffaloes are a cornerstone of dairy farming in many parts of Asia due to their consistent production and high-quality milk.",
#         "characteristics": ["High milk yield with rich fat content", "Jet black coat", "Short, tightly curled horns", "Adaptable to various climates", "Strong and healthy calves"],
#         "origin": "Punjab / Haryana, India",
#         "image": "murrah.png" # You'll need to generate this image
#     }
# }


# # app.py

# # ... (BREED_INFO dictionary ke baad isko add karein)

# # Static data for average milk yield per breed (in Liters per day)
# MILK_YIELD_DATA = {
#     "Sahiwal": "8 - 10 L/day",
#     "Gir": "6 - 10 L/day",
#     "Holstein_Friesian": "25 - 30 L/day",
#     "Jersey": "18 - 20 L/day",
#     "Murrah": "7 - 10 L/day",
#     "Jaffrabadi": "8 - 12 L/day",
#     "Default": "Not Available" # Agar breed match na ho to
# }

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# # --- General Page Routes ---
# @app.route('/')
# def index():
#     return render_template('index.html', title="Home")

# @app.route('/about')
# def about():
#     return render_template('about.html', title="About Us")

# # --- Breed Information Routes ---
# @app.route('/breeds')
# def list_breeds():
#     breeds_for_display = []
#     for breed_name in CLASS_NAMES:
#         info = BREED_INFO.get(breed_name, {})
#         breeds_for_display.append({
#             'name': breed_name,
#             'description_short': info.get('description', 'No description available.')[:100] + '...',
#             'image': info.get('image', 'default_cattle.jpg')
#         })
#     return render_template('breeds.html', title="Explore Breeds", breeds=breeds_for_display)

# @app.route('/breeds/<breed_name>')
# def breed_detail(breed_name):
#     info = BREED_INFO.get(breed_name)
#     if not info:
#         flash(f"Information for '{breed_name}' not found.", "warning")
#         return redirect(url_for('list_breeds'))
#     return render_template('breed_detail.html', title=f"{breed_name} Breed Info", breed_name=breed_name, info=info)

# # --- Breed Prediction Route ---
# @app.route('/predict')
# def predict_get():
#     return render_template('predict.html', title="Predict Breed")

# # --- Disease Diagnosis Route ---
# @app.route('/disease')
# def disease_page():
#     return render_template('disease.html', title="Disease Diagnosis")

# # --- API Endpoints ---

# # @app.route('/api/predict', methods=['POST'])
# # def api_predict():
# #     if MODEL is None or not CLASS_NAMES:
# #         return jsonify({'error': 'Breed model not loaded on server.', 'category': 'error'}), 500
# #     # ... (Your existing breed prediction logic here, no changes needed)
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file part in the request.', 'category': 'error'}), 400
# #     file = request.files['file']
# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file.', 'category': 'error'}), 400
# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# #         file.save(filepath)
# #         try:
# #             img_bgr = cv2.imread(filepath)
# #             if img_bgr is None:
# #                 os.remove(filepath)
# #                 return jsonify({'error': 'Could not read image file.', 'category': 'error'}), 400
# #             img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# #             img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
# #             x = img_resized.astype('float32')
# #             x = preprocess_input(x)
# #             x = np.expand_dims(x, 0)
# #             preds = MODEL.predict(x)[0]
# #             idxs = preds.argsort()[-3:][::-1]
# #             results = [{'breed': CLASS_NAMES[i], 'score': float(f"{preds[i]*100:.2f}")} for i in idxs]
# #             return jsonify({'success': True, 'results': results, 'image_url': url_for('uploaded_file', filename=filename)})
# #         except Exception as e:
# #             logging.exception("An error occurred during breed prediction.")
# #             return jsonify({'error': str(e), 'category': 'error'}), 500
# #     else:
# #         return jsonify({'error': 'Invalid file type.', 'category': 'error'}), 400

# # app.py

# # @app.route('/api/predict', methods=['POST'])
# # def api_predict():
# #     if MODEL is None or not CLASS_NAMES:
# #         return jsonify({'error': 'Breed model not loaded on server.', 'category': 'error'}), 500
    
# #     if 'file' not in request.files:
# #         return jsonify({'error': 'No file part in the request.', 'category': 'error'}), 400
# #     file = request.files['file']

# #     if file.filename == '':
# #         return jsonify({'error': 'No selected file.', 'category': 'error'}), 400

# #     if file and allowed_file(file.filename):
# #         filename = secure_filename(file.filename)
# #         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
# #         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# #         file.save(filepath)
        
# #         try:
# #             img_bgr = cv2.imread(filepath)
# #             if img_bgr is None:
# #                 os.remove(filepath)
# #                 return jsonify({'error': 'Could not read image file.', 'category': 'error'}), 400
            
# #             img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# #             img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
# #             x = img_resized.astype('float32')
# #             x = preprocess_input(x)
# #             x = np.expand_dims(x, 0)
            
# #             preds = MODEL.predict(x)[0]
# #             idxs = preds.argsort()[-3:][::-1]
# #             results = [{'breed': CLASS_NAMES[i], 'score': float(f"{preds[i]*100:.2f}")} for i in idxs]
            
# #             # --- NEW: Milk Yield Logic ---
# #             # Get the top predicted breed
# #             top_breed = results[0]['breed']
# #             # Look up the milk yield from our dictionary
# #             milk_yield = MILK_YIELD_DATA.get(top_breed, MILK_YIELD_DATA["Default"])

# #             return jsonify({
# #                 'success': True,
# #                 'results': results,
# #                 'image_url': url_for('uploaded_file', filename=filename),
# #                 'milk_yield': milk_yield  # Add milk yield to the response
# #             })

# #         except Exception as e:
# #             logging.exception("An error occurred during breed prediction.")
# #             return jsonify({'error': str(e), 'category': 'error'}), 500
# #     else:
# #         return jsonify({'error': 'Invalid file type.', 'category': 'error'}), 400

# # app.py

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     if MODEL is None or not CLASS_NAMES:
#         return jsonify({'error': 'Breed model not loaded on server.', 'category': 'error'}), 500
    
#     # --- NEW: Define your confidence threshold here ---
#     # Aap is value ko apni zaroorat ke hisaab se 40.0 se 60.0 ke beech rakh sakte hain
#     CONFIDENCE_THRESHOLD = 50.0  # Represents 50%

#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request.', 'category': 'error'}), 400
#     file = request.files['file']

#     if file.filename == '':
#         return jsonify({'error': 'No selected file.', 'category': 'error'}), 400

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#         file.save(filepath)
        
#         try:
#             img_bgr = cv2.imread(filepath)
#             if img_bgr is None:
#                 os.remove(filepath)
#                 return jsonify({'error': 'Could not read image file.', 'category': 'error'}), 400
            
#             img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#             img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
#             x = img_resized.astype('float32')
#             x = preprocess_input(x)
#             x = np.expand_dims(x, 0)
            
#             preds = MODEL.predict(x)[0]
#             idxs = preds.argsort()[-3:][::-1]
#             results = [{'breed': CLASS_NAMES[i], 'score': float(f"{preds[i]*100:.2f}")} for i in idxs]
            
#             # --- NEW: Check the confidence of the top prediction ---
#             top_prediction_score = results[0]['score']
            
#             if top_prediction_score < CONFIDENCE_THRESHOLD:
#                 # If the score is too low, send a specific error message
#                 return jsonify({
#                     'success': False, 
#                     'error': 'Please upload a clear image of a cattle or buffalo. The AI was not confident in its prediction.',
#                     'category': 'warning' # Use 'warning' for a yellow alert
#                 })

#             # If the score is high enough, proceed as normal
#             top_breed = results[0]['breed']
#             milk_yield = MILK_YIELD_DATA.get(top_breed, MILK_YIELD_DATA["Default"])

#             return jsonify({
#                 'success': True,
#                 'results': results,
#                 'image_url': url_for('uploaded_file', filename=filename),
#                 'milk_yield': milk_yield
#             })

#         except Exception as e:
#             logging.exception("An error occurred during breed prediction.")
#             return jsonify({'error': str(e), 'category': 'error'}), 500
#     else:
#         return jsonify({'error': 'Invalid file type.', 'category': 'error'}), 400


# @app.route('/api/symptoms')
# def get_symptoms():
#     """Provides the full list of symptoms to the frontend for autocomplete."""
#     if not ALL_SYMPTOMS:
#         return jsonify({"error": "Symptom list not available"}), 500
#     return jsonify(ALL_SYMPTOMS)

# @app.route('/api/diagnose', methods=['POST'])
# def api_diagnose():
#     """Handles the disease diagnosis request from the chatbot."""
#     if not all([DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER]):
#         return jsonify({"error": "Diagnosis model is not ready."}), 500
        
#     data = request.get_json()
#     if not data or 'animal' not in data or 'symptoms' not in data or 'age' not in data or 'temperature' not in data:
#         return jsonify({"error": "Invalid input. Animal, age, temperature, and symptoms are required."}), 400

#     animal = data['animal']
#     selected_symptoms = data['symptoms']
    
#     try:
#         age = float(data['age'])
#         temperature = float(data['temperature'])
#     except ValueError:
#         return jsonify({"error": "Age and Temperature must be valid numbers."}), 400

#     try:
#         input_df = pd.DataFrame(columns=FEATURE_COLUMNS, index=[0])
#         input_df = input_df.fillna(0)

#         input_df['Age'] = age
#         input_df['Temperature'] = temperature
        
#         # Use .lower() to match the training data ('cow', 'buffalo')
#         encoded_animal = ANIMAL_ENCODER.transform([animal.lower()])[0]
#         input_df['Animal'] = encoded_animal

#         for symptom in selected_symptoms:
#             if symptom in input_df.columns:
#                 input_df[symptom] = 1
        
#         input_df = input_df[FEATURE_COLUMNS]

#         prediction_proba = DISEASE_MODEL.predict_proba(input_df)[0]
#         top_prediction_index = prediction_proba.argmax()
        
#         predicted_disease = DISEASE_ENCODER.inverse_transform([top_prediction_index])[0]
#         confidence_score = round(prediction_proba[top_prediction_index] * 100, 2)

#         return jsonify({
#             "success": True,
#             "disease": predicted_disease,
#             "confidence": confidence_score,
#             "key_symptoms": selected_symptoms
#         })
#     except Exception as e:
#         logging.exception("An error occurred during diagnosis.")
#         return jsonify({"error": str(e)}), 500

# # --- File Server Routes ---
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# @app.route('/static/img/breeds/<filename>')
# def serve_breed_image(filename):
#     return send_from_directory(os.path.join(app.root_path, 'static', 'img', 'breeds'), filename)

# if __name__ == '__main__':
#     os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
#     os.makedirs(os.path.join(app.root_path, 'static', 'img', 'breeds'), exist_ok=True)
#     app.run(debug=True)

import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import cv2
from werkzeug.utils import secure_filename
import logging
import joblib
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-strong-secret-key-for-siH-2025'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# --- Breed Recognition Model ---
MODEL = None
CLASS_NAMES = []
IMG_SIZE = 300

# === CORRECTED FUNCTION TO FIX SHAPE MISMATCH ===
def build_model_architecture(num_classes, img_size=300, base='efficientnetb3', dropout=0.3):
    """
    Rebuilds the exact model architecture, explicitly defining a 3-channel input.
    """
    # Explicitly define the input layer for a 3-channel (RGB) image
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    
    if base.lower() == 'efficientnetb3':
        from tensorflow.keras.applications import EfficientNetB3
        # Pass the explicit 'inputs' tensor to the backbone model.
        # This forces it to build with the correct 3-channel shape on any server.
        backbone = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=inputs, pooling='avg')
    else:
        raise ValueError(f'Unsupported base model: {base}')
    
    x = backbone.outputs[0]
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model

def load_model_and_labels():
    global MODEL, CLASS_NAMES
    logging.info("Attempting to load breed recognition model and labels...")
    try:
        label_map_path = os.path.join(app.root_path, 'model', 'label_map.json')
        with open(label_map_path, 'r') as f:
            label_map = json.load(f)
            CLASS_NAMES = sorted(label_map, key=label_map.get)
        logging.info(f"Successfully loaded {len(CLASS_NAMES)} breed class names: {CLASS_NAMES}")
        
        num_classes = len(CLASS_NAMES)
        
        # Build the model architecture using the corrected function
        MODEL = build_model_architecture(num_classes=num_classes, img_size=IMG_SIZE)
        
        # Load the weights into the correctly built architecture
        model_weights_path = os.path.join(app.root_path, 'model', 'best_model_weights.h5')
        MODEL.load_weights(model_weights_path)
        
        logging.info("Breed recognition model weights loaded successfully.")
    except Exception as e:
        logging.exception(f"CRITICAL ERROR: Failed to load breed model or labels: {e}")
        MODEL = None
        CLASS_NAMES = []

# --- Disease Diagnosis Model ---
DISEASE_MODEL = None
DISEASE_ENCODER = None
ANIMAL_ENCODER = None
ALL_SYMPTOMS = []
FEATURE_COLUMNS = []

def load_disease_model():
    global DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER, ALL_SYMPTOMS, FEATURE_COLUMNS
    logging.info("Attempting to load disease diagnosis model...")
    try:
        model_path = os.path.join(app.root_path, 'disease_model', 'best_disease_model.pkl')
        DISEASE_MODEL = joblib.load(model_path)
        
        disease_encoder_path = os.path.join(app.root_path, 'disease_model', 'disease_encoder.pkl')
        DISEASE_ENCODER = joblib.load(disease_encoder_path)
        
        animal_encoder_path = os.path.join(app.root_path, 'disease_model', 'animal_encoder.pkl')
        ANIMAL_ENCODER = joblib.load(animal_encoder_path)
        
        columns_path = os.path.join(app.root_path, 'disease_model', 'feature_columns.json')
        with open(columns_path, 'r') as f:
            FEATURE_COLUMNS = json.load(f)
            
        symptoms_path = os.path.join(app.root_path, 'disease_model', 'symptom_list.json')
        with open(symptoms_path, 'r') as f:
            ALL_SYMPTOMS = json.load(f)
        
        logging.info("Disease diagnosis model and assets loaded successfully.")
    except Exception as e:
        logging.exception(f"CRITICAL ERROR: Failed to load disease model: {e}")

# Load all models when the app starts
with app.app_context():
    load_model_and_labels()
    load_disease_model()

# --- Static Data (Breed Info & Milk Yield) ---
# --- Breed Info Data ---

BREED_INFO = {
    "Sahiwal": {
        "description": "The Sahiwal is a prominent breed of Zebu cattle, originally from the Sahiwal district of Punjab, Pakistan. Renowned for its exceptional milk production, it exhibits strong heat tolerance and natural resistance to various parasites, making it ideal for tropical climates.",
        "characteristics": ["High milk production", "Excellent heat tolerance", "Strong disease resistance", "Gentle and docile temperament", "Reddish-brown coat"],
        "origin": "Punjab, Pakistan",
        "image": "sahiwal.png" # Using .jpg as generated
    },
    "Gir": {
        "description": "The Gir or Gyr is one of the most famous Zebu dairy cattle breeds, originating from the Gir Hills of Gujarat, India. It's easily recognizable by its distinctive convex forehead, long pendulous ears, and a generally robust, hardy constitution. Gir cattle are known for their milk quality and adaptability.",
        "characteristics": ["High milk fat content", "Resilient and adaptable", "Disease resistant", "Distinctive prominent forehead and long ears", "Red, yellowish-white, or spotted coat"],
        "origin": "Gir Hills, Gujarat, India",
        "image": "gir.png" # Using .jpg as generated
    },
    "Holstein_Friesian": { # Changed to match your model output key
        "description": "The Holstein Friesian, commonly known as Holstein, is the world's most widespread dairy cattle breed, originating from the Netherlands and Germany. They are celebrated for their unmatched milk production capabilities, making them the backbone of commercial dairy farming globally.",
        "characteristics": ["Highest milk yield", "Large body size", "Distinctive black and white (or red and white) markings", "Efficient feed conversion"],
        "origin": "Netherlands / Germany",
        "image": "Holstein_Friesian.png" # Using .jpg as generated
    },
    "Jaffrabadi": {
        "description": "The Jaffrabadi is a heavy breed of water buffalo, primarily found in the Saurashtra region of Gujarat, India. These majestic animals are known for their massive build, strong horns that curve back and then loop forward, and their impressive milk production, especially for butterfat content. They are highly adaptable and well-suited for both milk and draught purposes.",
        "characteristics": ["High butterfat milk", "Large and robust body", "Distinctive curled horns", "Good for draught power", "Strong and hardy constitution"],
        "origin": "Gujarat, India",
        "image": "jaffrabadi.png" # You'll need to generate this image
    },
    "Jersey": {
        "description": "Jersey cattle are a small breed of dairy cattle originating from the British Channel Island of Jersey. Despite their smaller size, they are highly valued for the high butterfat and protein content of their milk, often having a rich golden color. They are known for their docile nature and efficient grazing.",
        "characteristics": ["High butterfat and protein milk", "Smaller body size", "Fawn to dark brown coat", "Docile temperament", "Efficient grazers"],
        "origin": "Island of Jersey",
        "image": "Jersy.png" # Using .jpg as generated
    },
    "Murrah": {
        "description": "The Murrah is a renowned breed of water buffalo primarily kept for milk production, originating from the Punjab and Haryana states of India. They are characterized by their jet-black coat, short and tightly curled horns, and impressive milk yield. Murrah buffaloes are a cornerstone of dairy farming in many parts of Asia due to their consistent production and high-quality milk.",
        "characteristics": ["High milk yield with rich fat content", "Jet black coat", "Short, tightly curled horns", "Adaptable to various climates", "Strong and healthy calves"],
        "origin": "Punjab / Haryana, India",
        "image": "murrah.png" # You'll need to generate this image
    }
}
MILK_YIELD_DATA = {
    "Sahiwal": "8 - 10 L/day",
    "Gir": "6 - 10 L/day",
    "Holstein_Friesian": "25 - 30 L/day",
    "Jersey": "18 - 20 L/day",
    "Murrah": "7 - 10 L/day",
    "Jaffrabadi": "8 - 12 L/day",
    "Default": "Not Available"
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Page Routes ---
@app.route('/')
def index():
    return render_template('index.html', title="Home")

@app.route('/about')
def about():
    return render_template('about.html', title="About Us")

@app.route('/breeds')
def list_breeds():
    breeds_for_display = []
    # This loop will now work because CLASS_NAMES will not be empty
    for breed_name in CLASS_NAMES:
        info = BREED_INFO.get(breed_name, {})
        breeds_for_display.append({
            'name': breed_name,
            'description_short': info.get('description', 'No description available.')[:100] + '...',
            'image': info.get('image', 'default_cattle.jpg')
        })
    return render_template('breeds.html', title="Explore Breeds", breeds=breeds_for_display)

@app.route('/breeds/<breed_name>')
def breed_detail(breed_name):
    info = BREED_INFO.get(breed_name)
    if not info:
        flash(f"Information for '{breed_name}' not found.", "warning")
        return redirect(url_for('list_breeds'))
    return render_template('breed_detail.html', title=f"{breed_name} Breed Info", breed_name=breed_name, info=info)

@app.route('/predict')
def predict_get():
    return render_template('predict.html', title="Predict Breed")

@app.route('/disease')
def disease_page():
    return render_template('disease.html', title="Disease Diagnosis")

# --- API Endpoints ---
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if MODEL is None:
        return jsonify({'error': 'Breed model is not loaded on server. Check logs for details.', 'category': 'error'}), 500
    
    CONFIDENCE_THRESHOLD = 50.0
    
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file part.', 'category': 'error'}), 400
    if file.filename == '': return jsonify({'error': 'No selected file.', 'category': 'error'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)
        try:
            img_bgr = cv2.imread(filepath)
            if img_bgr is None:
                os.remove(filepath)
                return jsonify({'error': 'Could not read image file.', 'category': 'error'}), 400
            
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            x = img_resized.astype('float32')
            x = preprocess_input(x)
            x = np.expand_dims(x, 0)
            
            preds = MODEL.predict(x)[0]
            idxs = preds.argsort()[-3:][::-1]
            results = [{'breed': CLASS_NAMES[i], 'score': float(f"{preds[i]*100:.2f}")} for i in idxs]
            
            top_prediction_score = results[0]['score']
            if top_prediction_score < CONFIDENCE_THRESHOLD:
                return jsonify({ 'success': False, 'error': 'Please upload a clear image of a cattle or buffalo.', 'category': 'warning'})

            top_breed = results[0]['breed']
            milk_yield = MILK_YIELD_DATA.get(top_breed, MILK_YIELD_DATA["Default"])

            return jsonify({
                'success': True,
                'results': results,
                'image_url': url_for('uploaded_file', filename=filename),
                'milk_yield': milk_yield
            })

        except Exception as e:
            logging.exception("An error occurred during breed prediction.")
            return jsonify({'error': str(e), 'category': 'error'}), 500
    else:
        return jsonify({'error': 'Invalid file type.', 'category': 'error'}), 400

@app.route('/api/symptoms')
def get_symptoms():
    if not ALL_SYMPTOMS:
        return jsonify({"error": "Symptom list not available"}), 500
    return jsonify(ALL_SYMPTOMS)

@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    if not all([DISEASE_MODEL, DISEASE_ENCODER, ANIMAL_ENCODER]):
        return jsonify({"error": "Diagnosis model is not ready."}), 500
    
    data = request.get_json()
    if not data or 'animal' not in data or 'symptoms' not in data or 'age' not in data or 'temperature' not in data:
        return jsonify({"error": "Invalid input."}), 400

    animal = data['animal']
    selected_symptoms = data['symptoms']
    
    try:
        age = float(data['age'])
        temperature = float(data['temperature'])
    except ValueError:
        return jsonify({"error": "Age and Temperature must be valid numbers."}), 400

    try:
        input_df = pd.DataFrame(columns=FEATURE_COLUMNS, index=[0])
        input_df = input_df.fillna(0)
        
        input_df['Age'] = age
        input_df['Temperature'] = temperature
        input_df['Animal'] = ANIMAL_ENCODER.transform([animal.lower()])[0]

        for symptom in selected_symptoms:
            if symptom in input_df.columns:
                input_df[symptom] = 1
        
        input_df = input_df[FEATURE_COLUMNS]

        prediction_proba = DISEASE_MODEL.predict_proba(input_df)[0]
        top_prediction_index = prediction_proba.argmax()
        
        predicted_disease = DISEASE_ENCODER.inverse_transform([top_prediction_index])[0]
        confidence_score = round(prediction_proba[top_prediction_index] * 100, 2)

        return jsonify({
            "success": True,
            "disease": predicted_disease,
            "confidence": confidence_score,
            "key_symptoms": selected_symptoms
        })

    except Exception as e:
        logging.exception("An error occurred during diagnosis.")
        return jsonify({"error": str(e)}), 500

# --- File Server Routes ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/img/breeds/<filename>')
def serve_breed_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'static', 'img', 'breeds'), filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.join(app.root_path, 'static', 'img', 'breeds'), exist_ok=True)
    app.run(debug=True)
