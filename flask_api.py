#!/usr/bin/env python3
"""
Flask API for Fall Detection — Cloud Version (Railway.app)
Uses PORT environment variable for cloud deployment
"""

from flask import Flask, request, jsonify
import pickle, os, csv, threading, numpy as np, pandas as pd
from collections import deque
from datetime import datetime
from scipy.signal import butter, filtfilt

app = Flask(__name__)

# ── Load model ────────────────────────────────────────────────────────────────
with open('simple_detector.pkl', 'rb') as f:
    detector = pickle.load(f)

best_name = max(detector.models, key=lambda x: detector.models[x]['accuracy'])
print(f"Model: {best_name}  accuracy: {detector.models[best_name]['accuracy']:.4f}")

# ── Constants ─────────────────────────────────────────────────────────────────
SAMPLING_RATE  = 50
WINDOW_SIZE    = 125
STEP_SIZE      = 50
ACC_SCALE      = 16.0 / 32768.0 * 9.81
GYRO_SCALE     = 2000.0 / 32768.0 * (3.14159265 / 180.0)
FALL_THRESHOLD = 0.60

# ── State ─────────────────────────────────────────────────────────────────────
buffer         = deque(maxlen=WINDOW_SIZE)
sample_count   = 0
new_since_pred = 0
pred_running   = False
lock           = threading.Lock()
log_lock       = threading.Lock()

latest_result = {
    "prediction": 0, "label": "NO FALL", "probability": 0.0,
    "confidence": "low", "timestamp": datetime.now().isoformat(),
    "total_samples": 0, "status": "waiting"
}
fall_history = []

ESP32_LOG = 'esp32_raw_data.csv'
if not os.path.exists(ESP32_LOG):
    with open(ESP32_LOG, 'w', newline='') as f:
        csv.writer(f).writerow(['timestamp','ax','ay','az','gx','gy','gz'])


def preprocess_window(raw_df):
    df = raw_df.copy()
    for col in ['ax','ay','az']:
        df[col] = df[col] * ACC_SCALE
    for col in ['gx','gy','gz']:
        df[col] = df[col] * GYRO_SCALE
    b, a = butter(4, 20/(0.5*SAMPLING_RATE), btype='low', analog=False)
    for col in ['ax','ay','az','gx','gy','gz']:
        df[col] = filtfilt(b, a, df[col].values)
    return df


def run_prediction():
    global latest_result, pred_running
    try:
        with lock:
            rows = list(buffer)
        if len(rows) < WINDOW_SIZE:
            pred_running = False
            return

        raw_df = pd.DataFrame(rows, columns=['ax','ay','az','gx','gy','gz'])
        processed = preprocess_window(raw_df) if raw_df[['ax','ay','az']].abs().max().max() > 1000 else raw_df

        features = detector._extract_window_features(processed)
        features_scaled = detector.scaler.transform([features])
        prob = float(detector.models[best_name]['model'].predict_proba(features_scaled)[0][1])
        prediction = 1 if prob >= FALL_THRESHOLD else 0
        confidence = 'high' if abs(prob-0.5) > 0.35 else 'medium' if abs(prob-0.5) > 0.15 else 'low'

        with lock:
            latest_result.update({
                "prediction": prediction,
                "label": "FALL DETECTED" if prediction == 1 else "NO FALL",
                "probability": round(prob, 4),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "total_samples": sample_count,
                "status": "ready"
            })

        if prediction == 1:
            now = datetime.now()
            log_it = not fall_history or (now - datetime.fromisoformat(fall_history[-1]['time'])).total_seconds() > 3
            if log_it:
                fall_history.append({"time": now.isoformat(), "probability": round(prob, 4)})
                if len(fall_history) > 50: fall_history.pop(0)

        print(f"{'FALL' if prediction else 'safe'}  prob={prob:.3f}")
    except Exception as e:
        print(f"Prediction error: {e}")
    pred_running = False


@app.route('/data', methods=['POST'])
def receive_data():
    global sample_count, new_since_pred, pred_running
    try:
        samples = request.get_json(force=True).get('samples', [])
        if not samples:
            return jsonify({"error": "no samples"}), 400

        should_predict = False
        with lock:
            for s in samples:
                if len(s) == 6:
                    buffer.append(s)
                    sample_count += 1
                    new_since_pred += 1
            latest_result['total_samples'] = sample_count
            if latest_result['status'] == 'waiting':
                latest_result['status'] = 'collecting'
            if len(buffer) >= WINDOW_SIZE and new_since_pred >= STEP_SIZE and not pred_running:
                new_since_pred = 0
                pred_running = True
                should_predict = True

        # Log to CSV
        ts = datetime.now().isoformat()
        with log_lock:
            with open(ESP32_LOG, 'a', newline='') as f:
                w = csv.writer(f)
                for s in samples:
                    if len(s) == 6:
                        w.writerow([ts] + s)

        if should_predict:
            threading.Thread(target=run_prediction, daemon=True).start()

        return jsonify({"ok": True, "buffer": len(buffer), "total": sample_count}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/result', methods=['GET'])
def get_result():
    with lock:
        return jsonify(latest_result), 200


@app.route('/history', methods=['GET'])
def get_history():
    return jsonify({"falls": fall_history, "total": len(fall_history)}), 200


@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({
        "status": "online", "model": best_name,
        "accuracy": detector.models[best_name]['accuracy'],
        "buffer": len(buffer), "window_size": WINDOW_SIZE
    }), 200


@app.route('/reset', methods=['POST'])
def reset():
    global sample_count, new_since_pred, pred_running
    with lock:
        buffer.clear()
        sample_count = new_since_pred = 0
        pred_running = False
        latest_result.update({
            "prediction": 0, "label": "NO FALL", "probability": 0.0,
            "confidence": "low", "status": "waiting", "total_samples": 0,
            "timestamp": datetime.now().isoformat()
        })
    return jsonify({"status": "reset done"}), 200


if __name__ == '__main__':
    # Railway uses PORT env variable
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
