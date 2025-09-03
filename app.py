import os
import sqlite3
import cv2
import numpy as np
import pandas as pd
import pytesseract
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify, flash
from werkzeug.utils import secure_filename
from PIL import Image
import io
import tempfile
import re

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS parking_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  plate_number TEXT NOT NULL,
                  entry_date DATE NOT NULL,
                  exit_date DATE,
                  is_active INTEGER DEFAULT 1,
                  image_path TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def extract_plate_number(image_path):
    try:
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # グレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ノイズ除去
        gray = cv2.medianBlur(gray, 3)
        
        # コントラスト調整
        gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        
        # エッジ検出
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # ナンバープレートらしい矩形を探す
        plate_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 最小面積
                rect = cv2.boundingRect(contour)
                x, y, w, h = rect
                aspect_ratio = w / h
                if 2 < aspect_ratio < 6:  # ナンバープレートのアスペクト比
                    plate_candidates.append((rect, area))
        
        # 面積が最大の候補を選択
        if plate_candidates:
            best_rect = max(plate_candidates, key=lambda x: x[1])[0]
            x, y, w, h = best_rect
            plate_region = gray[y:y+h, x:x+w]
            
            # OCRでテキスト抽出
            text = pytesseract.image_to_string(plate_region, config='--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            
            # 日本のナンバープレート形式に合うかチェック
            text = text.strip().replace(' ', '').replace('\n', '')
            
            # 簡単なパターンマッチング（例：品川301あ1234）
            if len(text) >= 4:
                return text
        
        # 候補が見つからない場合は全体にOCRを適用
        text = pytesseract.image_to_string(gray, config='--psm 6')
        text = text.strip().replace(' ', '').replace('\n', '')
        
        if len(text) >= 4:
            return text
            
        return "認識できませんでした"
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return "エラーが発生しました"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    file = request.files['file']
    action = request.form.get('action', 'entry')
    
    if file.filename == '':
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # アップロードディレクトリを作成
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # ナンバープレート認識
        plate_number = extract_plate_number(filepath)
        
        if plate_number and plate_number not in ["認識できませんでした", "エラーが発生しました"]:
            conn = sqlite3.connect('parking.db')
            c = conn.cursor()
            
            if action == 'entry':
                # 入場記録
                c.execute("INSERT INTO parking_records (plate_number, entry_date, image_path) VALUES (?, ?, ?)",
                         (plate_number, date.today(), filepath))
                flash(f'入場記録を追加しました: {plate_number}')
            else:
                # 退場記録
                c.execute("UPDATE parking_records SET exit_date = ?, is_active = 0 WHERE plate_number = ? AND is_active = 1",
                         (date.today(), plate_number))
                if c.rowcount > 0:
                    flash(f'退場記録を更新しました: {plate_number}')
                else:
                    flash(f'該当する入場記録が見つかりません: {plate_number}')
            
            conn.commit()
            conn.close()
        else:
            flash(f'ナンバープレートを認識できませんでした: {plate_number}')
        
        return redirect(url_for('records'))
    
    flash('無効なファイル形式です')
    return redirect(request.url)

@app.route('/records')
def records():
    conn = sqlite3.connect('parking.db')
    
    # 現在駐車中の車両
    active_cars = pd.read_sql_query("""
        SELECT plate_number, entry_date, 
               (julianday('now') - julianday(entry_date)) as days_parked
        FROM parking_records 
        WHERE is_active = 1 
        ORDER BY entry_date DESC
    """, conn)
    
    # 全記録
    all_records = pd.read_sql_query("""
        SELECT plate_number, entry_date, exit_date, 
               CASE WHEN exit_date IS NOT NULL 
                    THEN (julianday(exit_date) - julianday(entry_date))
                    ELSE (julianday('now') - julianday(entry_date))
               END as days_parked
        FROM parking_records 
        ORDER BY created_at DESC
        LIMIT 100
    """, conn)
    
    conn.close()
    
    return render_template('records.html', 
                         active_cars=active_cars.to_dict('records'),
                         all_records=all_records.to_dict('records'))

@app.route('/analytics')
def analytics():
    conn = sqlite3.connect('parking.db')
    
    # 統計データを取得
    stats = {}
    
    # 現在の駐車台数
    current_count = pd.read_sql_query("SELECT COUNT(*) as count FROM parking_records WHERE is_active = 1", conn).iloc[0]['count']
    stats['current_count'] = current_count
    
    # 今日の入場台数
    today_entries = pd.read_sql_query("SELECT COUNT(*) as count FROM parking_records WHERE entry_date = date('now')", conn).iloc[0]['count']
    stats['today_entries'] = today_entries
    
    # 平均駐車日数
    avg_days = pd.read_sql_query("""
        SELECT AVG(julianday(COALESCE(exit_date, 'now')) - julianday(entry_date)) as avg_days 
        FROM parking_records
    """, conn).iloc[0]['avg_days']
    stats['avg_days'] = round(avg_days, 1) if avg_days else 0
    
    # 長期駐車車両（7日以上）
    long_term = pd.read_sql_query("""
        SELECT plate_number, entry_date,
               (julianday('now') - julianday(entry_date)) as days_parked
        FROM parking_records 
        WHERE is_active = 1 AND (julianday('now') - julianday(entry_date)) >= 7
        ORDER BY days_parked DESC
    """, conn)
    
    # 月別統計
    monthly_stats = pd.read_sql_query("""
        SELECT strftime('%Y-%m', entry_date) as month,
               COUNT(*) as entries
        FROM parking_records 
        GROUP BY strftime('%Y-%m', entry_date)
        ORDER BY month DESC
        LIMIT 12
    """, conn)
    
    conn.close()
    
    return render_template('analytics.html',
                         stats=stats,
                         long_term=long_term.to_dict('records'),
                         monthly_stats=monthly_stats.to_dict('records'))

@app.route('/export')
def export_csv():
    conn = sqlite3.connect('parking.db')
    
    df = pd.read_sql_query("""
        SELECT plate_number as 'ナンバープレート',
               entry_date as '入場日',
               exit_date as '退場日',
               CASE WHEN is_active = 1 THEN '駐車中' ELSE '退場済み' END as '状態',
               CASE WHEN exit_date IS NOT NULL 
                    THEN (julianday(exit_date) - julianday(entry_date))
                    ELSE (julianday('now') - julianday(entry_date))
               END as '駐車日数',
               created_at as '記録作成日時'
        FROM parking_records 
        ORDER BY created_at DESC
    """, conn)
    
    conn.close()
    
    # CSVファイルを一時的に作成
    output = io.StringIO()
    df.to_csv(output, index=False, encoding='utf-8-sig')
    output.seek(0)
    
    # バイナリストリームに変換
    output_bytes = io.BytesIO()
    output_bytes.write(output.getvalue().encode('utf-8-sig'))
    output_bytes.seek(0)
    
    return send_file(
        output_bytes,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'parking_records_{datetime.now().strftime("%Y%m%d")}.csv'
    )

@app.route('/manual_entry', methods=['GET', 'POST'])
def manual_entry():
    if request.method == 'POST':
        plate_number = request.form.get('plate_number')
        action = request.form.get('action')
        
        if plate_number:
            conn = sqlite3.connect('parking.db')
            c = conn.cursor()
            
            if action == 'entry':
                c.execute("INSERT INTO parking_records (plate_number, entry_date) VALUES (?, ?)",
                         (plate_number, date.today()))
                flash(f'手動で入場記録を追加しました: {plate_number}')
            else:
                c.execute("UPDATE parking_records SET exit_date = ?, is_active = 0 WHERE plate_number = ? AND is_active = 1",
                         (date.today(), plate_number))
                if c.rowcount > 0:
                    flash(f'手動で退場記録を更新しました: {plate_number}')
                else:
                    flash(f'該当する入場記録が見つかりません: {plate_number}')
            
            conn.commit()
            conn.close()
            
            return redirect(url_for('records'))
    
    return render_template('manual_entry.html')

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)