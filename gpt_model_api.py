import sys
import subprocess
import os
import json
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from waitress import serve
from datetime import datetime, timedelta
import uuid
import warnings

# Disable the Sliding Window Attention warning
warnings.filterwarnings("ignore", message="Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.")

app = Flask(__name__)
user_ids = {}
request_log = []
users_file = 'users.json'
invalid_attempts = {}
blocked_ips = {}

def install_requirements():
    try:
        print("Bağımlılıklar kontrol ediliyor...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "scikit-build", "setuptools", "distro", "typing-extensions", "packaging", "wheel", "flask", "waitress"], check=True)
        required_packages = ["transformers", "torch", "requests"]
        for package in required_packages:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package, "--quiet"], check=True)
        print("Gerekli paketler başarıyla yüklendi!")
    except subprocess.CalledProcessError as e:
        print(json.dumps({"error": f"Paket yükleme sırasında hata oluştu: {e}"}))
        print(traceback.format_exc())
        sys.exit(1)

def load_model_and_tokenizer(model_dir):
    try:
        print(f"Model ve tokenizer yükleniyor... Model dizini: {model_dir}")
        model_subdir = os.path.join(model_dir, 'meta-llama/Llama-3.2-1B')
        model_file = os.path.join(model_subdir, 'model.safetensors')
        print(f"Checking for model file: {model_file}")
        if not os.path.exists(model_file):
            print(f"Model dosyası bulunamadı: {model_file}. Model indiriliyor...")

        config_file = os.path.join(model_subdir, 'config.json')
        tokenizer_file = os.path.join(model_subdir, 'tokenizer.json')
        print(f"Checking for tokenizer files: {config_file, tokenizer_file}")
        if not os.path.exists(config_file) or not os.path.exists(tokenizer_file):
            print(f"Tokenizer dosyaları bulunamadı: {config_file, tokenizer_file}. Tokenizer indiriliyor...")
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir=model_subdir)
            tokenizer.save_pretrained(model_subdir)
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir=model_subdir)
            model.save_pretrained(model_subdir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_subdir)
            model = AutoModelForCausalLM.from_pretrained(model_subdir)

        # Verify the model file integrity
        try:
            model = AutoModelForCausalLM.from_pretrained(model_subdir)
        except Exception as e:
            print(f"Model file is corrupted or incomplete: {e}. Re-downloading the model...")
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir=model_subdir, force_download=True)
            tokenizer.save_pretrained(model_subdir)
            model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B', cache_dir=model_subdir, force_download=True)
            model.save_pretrained(model_subdir)

        tokenizer.pad_token = tokenizer.eos_token
        model.to('cpu')
        print("Model ve tokenizer başarıyla yüklendi!")
        return tokenizer, model
    except Exception as e:
        print(json.dumps({"error": f"Model veya tokenizer yüklenirken hata oluştu: {str(e)}"}))
        print(traceback.format_exc())
        sys.exit(1)

# Load the model and tokenizer once and reuse them
model_dir = os.path.dirname(__file__)
tokenizer, model = load_model_and_tokenizer(model_dir)

def query_model(question, tokenizer, model):
    try:
        print(f"Querying model with question: {question}")
        inputs = tokenizer.encode(question, return_tensors='pt', padding=True).to('cpu')
        attention_mask = (inputs != tokenizer.pad_token_id).long().to('cpu')
        outputs = model.generate(inputs, attention_mask=attention_mask, max_length=512, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, repetition_penalty=1.2)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not response_text:
            response_text = f"Cevap alınamadı. Model çıktısı boş. Inputs: {inputs}, Outputs: {outputs}"
        print(f"Model response: {response_text}")
        return response_text
    except Exception as e:
        error_message = f"Model çalıştırılırken hata oluştu: {str(e)}"
        print(traceback.format_exc())
        return error_message

@app.route('/', methods=['GET'])
def list_endpoints():
    endpoints = {
        "GET": ["/requests", "/requestid/<request_id>"],
        "POST": ["/query", "/user_id", "/requestid"]
    }
    return jsonify(endpoints)

@app.route('/query', methods=['POST'])
def query():
    client_ip = request.remote_addr
    if client_ip in blocked_ips and datetime.now() < blocked_ips[client_ip]:
        return jsonify({"error": "IP adresiniz geçici olarak engellendi."}), 403

    data = request.json
    print("Received JSON:", data)  # Hata ayıklama için
    if not data or "message" not in data:
        return jsonify({"error": "Eksik veri"}), 400
    user_id = data.get('user_id')
    message = data.get('message')
    request_id = data.get('request_id')
    if not user_id or not message or not request_id:
        return jsonify({"error": "user_id, message ve request_id gereklidir."}), 400
    if user_id not in user_ids:
        invalid_attempts[client_ip] = invalid_attempts.get(client_ip, 0) + 1
        if invalid_attempts[client_ip] > 5:
            blocked_ips[client_ip] = datetime.now() + timedelta(minutes=10)
            return jsonify({"error": "IP adresiniz geçici olarak engellendi."}), 403
        return jsonify({"error": "Geçersiz user_id."}), 400
    question = message + " bu model sadece seo sem smm ve e pazarlama alanında cevap verir"
    request_log.append({"request_id": request_id, "user_id": user_id, "message": message, "response": None})
    if len(request_log) > 10:
        request_log.pop(0)
    response = query_model(question, tokenizer, model)
    for req in request_log:
        if req['request_id'] == request_id:
            req['response'] = response
            break
    return jsonify({"request_id": request_id, "response": response})

@app.route('/requests', methods=['GET'])
def get_requests():
    # Exclude user_id from the response
    sanitized_requests = [
        {key: value for key, value in req.items() if key != 'user_id'}
        for req in request_log
    ]
    return jsonify(sanitized_requests)

@app.route('/requestid/<request_id>', methods=['GET'])
def get_request_by_id(request_id):
    for req in request_log:
        if req['request_id'] == request_id:
            return jsonify(req)
    return jsonify({"error": "request_id bulunamadı."}), 404

@app.route('/requestid', methods=['POST'])
def add_request_id():
    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return jsonify({"error": "request_id gereklidir."}), 400
    for req in request_log:
        if req['request_id'] == request_id:
            return jsonify(req)
    return jsonify({"error": "request_id bulunamadı."}), 404

@app.route('/user_id', methods=['POST'])
def add_user_id():
    data = request.json
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id gereklidir."}), 400
    user_ids[user_id] = True
    save_users()
    return jsonify({"message": f"user_id '{user_id}' kaydedildi."})

def load_users():
    global user_ids
    if os.path.exists(users_file):
        with open(users_file, 'r') as f:
            user_ids = json.load(f)
    # Automatically add the fixed user ID if it does not already exist
    fixed_user_id = "test"
    if fixed_user_id not in user_ids:
        user_ids[fixed_user_id] = True

def save_users():
    with open(users_file, 'w') as f:
        json.dump(user_ids, f)

def start_server():
    global server_thread
    server_thread = executor.submit(lambda: serve(app, host='0.0.0.0', port=1881))
    print("Sunucu başlatıldı.")

def stop_server():
    global server_thread
    if server_thread:
        server_thread.cancel()
        print("Sunucu durduruldu.")

if __name__ == "__main__":
    install_requirements()
    load_users()

    executor = ThreadPoolExecutor(max_workers=1)

    # Start the server automatically
    start_server()
