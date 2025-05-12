from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

app = Flask(__name__)

# Konfigurasi kamera
CAMERA_INDEX = 1
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print(f"PERINGATAN: Tidak dapat membuka kamera {CAMERA_INDEX} saat startup.")
    cap = None # Set ke None agar bisa dicek nanti
else:
    # Coba mengatur resolusi kamera ke 16:9 (1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Konfigurasi model MediaPipe Gesture Recognizer
recognizer = None
try:
    # Pastikan path 'models/gesture_recognizer.task' benar relatif terhadap lokasi app.py
    base_options = mp.tasks.BaseOptions(model_asset_path='models/gesture_recognizer.task')
    options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
    print("Model Gesture Recognizer berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model Gesture Recognizer: {e}")
    # Aplikasi bisa tetap jalan, tapi get_prediction akan return error

# Variabel global
detection_log = []   # Untuk menyimpan log deteksi
MAX_LOG_SIZE = 50    # Batasi ukuran log
latest_prediction = {"gesture": "Belum ada prediksi", "confidence": 0.0}
mirror_mode = True  # Default mirror mode (True = mode cermin aktif)
confidence_threshold = 0.5  # Threshold untuk confidence

# Variabel untuk membangun kalimat
sentence_buffer = []
current_gesture = "" # Menyimpan gestur efektif terakhir (setelah threshold & konfirmasi) untuk cek consecutive
last_prediction_time = time.time()
prediction_cooldown = 1.0  # Jeda antar penambahan KATA ke kalimat
consecutive_same_predictions = 0
required_consecutive = 3  # Butuh 3 prediksi sama berturut-turut untuk konfirmasi
max_sentence_length = 10  # Maksimal 10 kata dalam kalimat

# --- Fungsi Placeholder Frame ---
def create_failed_camera_frame():
    # Buat frame hitam atau dengan pesan error
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    text = "Gagal Mengakses Kamera"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 2, 3)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2   
    # Gambar teks dengan warna mencolok (misalnya merah)
    cv2.putText(frame, text, (text_x, text_y), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    ret, buffer = cv2.imencode('.jpg', frame)
    # Jika encode gagal, kembalikan bytes kosong atau handle error
    return buffer.tobytes() if ret else b''

failed_camera_frame_bytes = create_failed_camera_frame()

# Fungsi untuk memproses frame dan memprediksi gesture
def process_frame(frame):
    if recognizer is None:
         return "Model Error", 0.0 # Kembalikan error jika model tidak ada

    # Resize frame (pastikan ukurannya sesuai ekspektasi model jika perlu)
    # frame_resized = cv2.resize(frame, (300, 300)) # Contoh resize jika diperlukan model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Langsung konversi frame asli
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    try:
        recognition_result = recognizer.recognize(mp_image)
    except Exception as e:
        print(f"Error saat recognize: {e}")
        return "Prediction Error", 0.0

    if recognition_result.gestures:
        # Asumsi gestur teratas adalah yang paling relevan
        top_gesture = recognition_result.gestures[0][0]
        confidence = top_gesture.score
        # Kembalikan hasil mentah, thresholding dilakukan di get_prediction
        return top_gesture.category_name, confidence
    # Jika tidak ada gestur terdeteksi oleh model
    return "Tidak dikenali", 0.0

# --- Generator untuk streaming video feed ---
def generate_frames():
    global cap, mirror_mode, failed_camera_frame_bytes
    while True:
        # Cek jika objek kamera valid dan terbuka
        if cap is None or not cap.isOpened():
            print("Mencoba membuka ulang kamera di generate_frames...")
            if cap: cap.release() # Release jika ada objek tapi tertutup
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                print("Kamera berhasil dibuka ulang.")
            else:
                # Jika masih gagal, kirim frame error & tunggu
                print("Masih gagal membuka kamera.")
                cap = None # Pastikan tetap None
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + failed_camera_frame_bytes + b'\r\n')
                time.sleep(2) # Tunggu sebelum mencoba lagi
                continue # Kembali ke awal loop

        # Jika kamera OK, baca frame
        success, frame = cap.read()
        if not success:
            # Jika gagal baca frame, asumsikan kamera terputus
            print("Gagal membaca frame, kamera mungkin terputus. Mencoba membuka ulang...")
            cap.release()
            cap = None # Set ke None agar loop berikutnya coba buka ulang
            # Kirim frame error sementara
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + failed_camera_frame_bytes + b'\r\n')
            time.sleep(1)
            continue # Kembali ke awal loop

        # Proses frame jika berhasil dibaca
        if mirror_mode:
            frame = cv2.flip(frame, 1) # Flip horizontal untuk efek cermin

        # Encode frame ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
             print("Gagal encode frame ke JPEG")
             continue # Lewati frame ini jika encode gagal

        frame_bytes = buffer.tobytes()
        # Kirim frame sebagai bagian dari stream multipart
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        # Tambahkan jeda kecil jika CPU usage terlalu tinggi
        # time.sleep(0.01)

# --- Route Utama ---
@app.route('/')
def index():
    # Hanya render template HTML utama
    return render_template('index.html')

# --- Route untuk Video Feed ---
@app.route('/video_feed')
def video_feed():
    # Pastikan model recognizer siap sebelum memulai stream
    if recognizer is None:
         # Kembalikan error server jika model tidak bisa dimuat
         return "Error: Gesture recognizer model not loaded.", 500
    # Mulai stream menggunakan generator generate_frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Route untuk Mendapatkan Prediksi ---
@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global cap, latest_prediction, mirror_mode, current_gesture, last_prediction_time
    global consecutive_same_predictions, sentence_buffer, confidence_threshold
    global detection_log

    current_time = time.time()

    # Validasi awal: Kamera dan Model harus siap
    if cap is None or not cap.isOpened():
        latest_prediction = {"gesture": "Gagal Membuka Kamera", "confidence": 0.0, "sentence": " ".join(sentence_buffer)}
        return jsonify(latest_prediction)
    if recognizer is None:
         latest_prediction = {"gesture": "Model Error", "confidence": 0.0, "sentence": " ".join(sentence_buffer)}
         return jsonify(latest_prediction)

    # Baca frame terbaru dari kamera
    success, frame = cap.read()
    if not success:
        # Jika gagal baca frame (meskipun sudah dicek di generate_frames)
        latest_prediction = {"gesture": "Gagal Membaca Frame", "confidence": 0.0, "sentence": " ".join(sentence_buffer)}
        return jsonify(latest_prediction)

    # Apply mirror jika aktif (untuk konsistensi dengan video feed)
    if mirror_mode:
        frame = cv2.flip(frame, 1)

    # Lakukan prediksi pada frame saat ini
    detected_gesture_this_cycle, confidence_this_cycle = process_frame(frame)

    # --- Logika Cooldown dan Konfirmasi untuk Penambahan Kalimat ---
    # Cek apakah waktu cooldown antar *penambahan kata* sudah terlewati
    if current_time - last_prediction_time >= prediction_cooldown:
        # Tentukan gestur efektif untuk konfirmasi (setelah threshold)
        if confidence_this_cycle < confidence_threshold or detected_gesture_this_cycle in ["Model Error", "Prediction Error"]:
             effective_gesture = "Tidak dikenali"
        else:
             effective_gesture = detected_gesture_this_cycle

        # Bandingkan gestur efektif saat ini dengan gestur efektif dari siklus *sebelumnya* (current_gesture)
        if effective_gesture == current_gesture and effective_gesture != "Tidak dikenali":
            consecutive_same_predictions += 1 # Tambah hitungan jika sama
        else:
            # Jika gestur efektif berubah (atau dari 'Tidak Dikenali' ke valid), reset counter
            current_gesture = effective_gesture # Update state gestur efektif terakhir
            consecutive_same_predictions = 1 # Mulai hitung dari 1 untuk gestur baru ini

        # Cek apakah jumlah deteksi berturut-turut sudah memenuhi syarat
        if consecutive_same_predictions >= required_consecutive:
            # Hanya proses jika gestur efektif saat ini valid (bukan 'Tidak Dikenali')
            if current_gesture != "Tidak dikenali":
                 # --- MODIFIKASI: Langsung tambahkan gestur terkonfirmasi ke buffer ---
                 sentence_buffer.append(current_gesture)

                 # --- Aksi setelah penambahan berhasil ---
                 # 1. Log Deteksi
                 timestamp = datetime.now().strftime('%H:%M:%S') # Format Waktu
                 log_entry = {
                     "timestamp": timestamp,
                     "gesture": current_gesture,
                     "confidence": round(confidence_this_cycle, 2) # Log confidence asli dari siklus ini
                 }
                 detection_log.append(log_entry)
                 # Jaga ukuran log
                 if len(detection_log) > MAX_LOG_SIZE:
                     detection_log.pop(0) # Hapus entri terlama

                 # 2. Reset counter consecutive HANYA setelah berhasil menambah kata
                 consecutive_same_predictions = 0

                 # 3. Batasi panjang kalimat
                 if len(sentence_buffer) > max_sentence_length:
                     sentence_buffer = sentence_buffer[-max_sentence_length:]
                 # --- Akhir Aksi Setelah Penambahan ---

        # Reset waktu cooldown (last_prediction_time) setelah siklus pengecekan ini selesai
        last_prediction_time = current_time
    # --- Akhir Logika Cooldown dan Konfirmasi ---

    # Siapkan respons JSON untuk dikirim ke frontend
    # Kembalikan hasil deteksi MENTAH dari siklus INI untuk ditampilkan di #result
    latest_prediction = {
        "gesture": detected_gesture_this_cycle,
        "confidence": round(confidence_this_cycle, 2),
        "sentence": " ".join(sentence_buffer) # Kalimat yang sudah terbangun (terkonfirmasi)
    }

    return jsonify(latest_prediction)


# --- Endpoint untuk Operasi Lain ---

@app.route('/get_log', methods=['GET'])
def get_log():
    # Mengembalikan riwayat deteksi (log)
    global detection_log
    # Balik urutan agar yang terbaru di atas saat ditampilkan
    return jsonify({"log": list(reversed(detection_log))})

@app.route('/toggle_mirror', methods=['POST'])
def toggle_mirror():
    # Mengaktifkan/menonaktifkan mode cermin
    global mirror_mode
    data = request.get_json()
    # Jika 'non_mirror' True, maka mirror_mode False, dan sebaliknya
    mirror_mode = not data.get('non_mirror', False)
    print(f"Mirror mode set to: {mirror_mode}") # Debug
    return jsonify({"mirror_mode": mirror_mode})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    # Menghapus kalimat yang sudah terbentuk
    global sentence_buffer
    sentence_buffer = []
    print("Sentence buffer cleared.") # Debug
    return jsonify({"success": True, "message": "Kalimat telah dihapus"})

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    # Memperbarui ambang batas kepercayaan
    global confidence_threshold
    data = request.get_json()
    try:
        new_threshold = float(data.get('threshold', 0.5))
        if 0.0 <= new_threshold <= 1.0:
             confidence_threshold = new_threshold
             print(f"Confidence threshold updated to: {confidence_threshold}") # Debug
             return jsonify({"success": True, "threshold": confidence_threshold})
        else:
             return jsonify({"success": False, "message": "Threshold harus antara 0.0 dan 1.0"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Nilai threshold tidak valid"}), 400

@app.route('/update_cooldown', methods=['POST'])
def update_cooldown():
    # Memperbarui jeda waktu antar penambahan kata
    global prediction_cooldown
    data = request.get_json()
    try:
        new_cooldown = float(data.get('cooldown', 1.0))
        if new_cooldown >= 0.1: # Beri batas minimal
             prediction_cooldown = new_cooldown
             print(f"Prediction cooldown updated to: {prediction_cooldown}") # Debug
             return jsonify({"success": True, "cooldown": prediction_cooldown})
        else:
            return jsonify({"success": False, "message": "Cooldown minimal 0.1 detik"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Nilai cooldown tidak valid"}), 400

@app.route('/update_consecutive', methods=['POST'])
def update_consecutive():
    # Memperbarui jumlah prediksi beruntun yang diperlukan
    global required_consecutive
    data = request.get_json()
    try:
        new_consecutive = int(data.get('consecutive', 3))
        if new_consecutive >= 1: # Minimal 1
             required_consecutive = new_consecutive
             print(f"Required consecutive predictions updated to: {required_consecutive}") # Debug
             return jsonify({"success": True, "consecutive": required_consecutive})
        else:
             return jsonify({"success": False, "message": "Prediksi beruntun minimal 1"}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "message": "Nilai prediksi beruntun tidak valid"}), 400

# --- Menjalankan Aplikasi Flask ---
if __name__ == '__main__':
    try:
        detection_log = [] # Kosongkan log saat start
        # Pastikan model sudah siap sebelum menjalankan server
        if recognizer is None:
             print("############################################################")
             print("ERROR: Model Gesture Recognizer tidak dapat dimuat.")
             print("Pastikan file 'models/gesture_recognizer.task' ada dan valid.")
             print("Aplikasi tidak dapat dimulai tanpa model.")
             print("############################################################")
             # Exit jika model gagal load? Atau biarkan jalan tapi fitur deteksi error?
             # exit(1) # Uncomment jika ingin menghentikan aplikasi
        # Jalankan server Flask
        # host='0.0.0.0' agar bisa diakses dari luar localhost
        # use_reloader=False penting jika ada state global & hardware access
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    finally:
        # Pastikan kamera dilepas saat aplikasi berhenti
        print("Releasing camera...")
        if cap and cap.isOpened():
            cap.release()
        print("Application stopped.")