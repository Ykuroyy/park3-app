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
import traceback
import logging

# EasyOCRをオプショナルインポート
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    # EasyOCRリーダーを初期化（日本語・英語対応）
    easyocr_reader = easyocr.Reader(['ja', 'en'])
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr_reader = None

# ログ設定
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['DEBUG'] = True

# Tesseractの設定
if os.path.exists('/usr/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
elif os.path.exists('/opt/homebrew/bin/tesseract'):
    pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_db():
    try:
        logger.info("データベース初期化開始")
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
        logger.info("データベース初期化完了")
    except Exception as e:
        logger.error(f"データベース初期化エラー: {e}")
        raise

def extract_plate_number(image_path):
    try:
        logger.info(f"画像処理開始: {image_path}")
        
        # ファイルの存在確認
        if not os.path.exists(image_path):
            logger.error(f"画像ファイルが存在しません: {image_path}")
            return "ファイルが見つかりません"
        
        # Tesseractがインストールされているか確認
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract利用可能")
        except Exception as e:
            logger.error(f"Tesseractエラー: {e}")
            # Tesseractが利用できない場合は手動入力を促す
            return "OCRエンジンが利用できません。手動入力をご利用ください。"
        
        # 画像を読み込み
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"画像の読み込みに失敗: {image_path}")
            # PILで試してみる
            try:
                pil_image = Image.open(image_path)
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                logger.info("PILで画像読み込み成功")
            except Exception as e:
                logger.error(f"PILでも画像読み込み失敗: {e}")
                return "画像の読み込みに失敗しました"
        
        logger.info(f"画像サイズ: {image.shape}")
        
        # グレースケールに変換
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 簡単な前処理のみ実行
        gray = cv2.medianBlur(gray, 3)
        
        # EasyOCRを優先して使用
        if EASYOCR_AVAILABLE and easyocr_reader:
            logger.info("EasyOCR開始")
            try:
                results = easyocr_reader.readtext(gray)
                if results:
                    # 最も信頼度の高いテキストを選択
                    best_result = max(results, key=lambda x: x[2])
                    text = best_result[1].strip().replace(' ', '').replace('\n', '').replace('\t', '')
                    logger.info(f"EasyOCR結果: {text} (信頼度: {best_result[2]:.2f})")
                    
                    if len(text) >= 3 and best_result[2] > 0.5:  # 信頼度50%以上
                        return text
            except Exception as e:
                logger.error(f"EasyOCRエラー: {e}")
        
        # TesseractでOCRを試す
        logger.info("Tesseract OCR開始")
        try:
            text = pytesseract.image_to_string(gray, config='--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZあいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん')
            text = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
            logger.info(f"Tesseract結果: {text}")
            
            if len(text) >= 3:
                return text if text else "認識できませんでした"
        except Exception as e:
            logger.error(f"Tesseract OCRエラー: {e}")
        
        # より高度な処理を試す
        try:
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
                if area > 500:  # 最小面積を小さくする
                    rect = cv2.boundingRect(contour)
                    x, y, w, h = rect
                    if h > 0:  # ゼロ除算を防ぐ
                        aspect_ratio = w / h
                        if 1.5 < aspect_ratio < 8:  # アスペクト比の範囲を広げる
                            plate_candidates.append((rect, area))
            
            # 面積が最大の候補を選択
            if plate_candidates:
                best_rect = max(plate_candidates, key=lambda x: x[1])[0]
                x, y, w, h = best_rect
                
                # 範囲チェック
                if x >= 0 and y >= 0 and x + w <= gray.shape[1] and y + h <= gray.shape[0]:
                    plate_region = gray[y:y+h, x:x+w]
                    
                    # OCRでテキスト抽出
                    text = pytesseract.image_to_string(plate_region, config='--psm 8')
                    text = text.strip().replace(' ', '').replace('\n', '').replace('\t', '')
                    
                    if len(text) >= 3:
                        return text
            
        except Exception as e:
            logger.error(f"高度な処理でエラー: {e}")
        
        return "認識できませんでした"
        
    except Exception as e:
        logger.error(f"OCR全般エラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")
        return f"エラーが発生しました: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logger.info("ファイルアップロード開始")
        
        if 'file' not in request.files:
            logger.warning("ファイルが選択されていません")
            flash('ファイルが選択されていません')
            return redirect(url_for('index'))
        
        file = request.files['file']
        action = request.form.get('action', 'entry')
        
        logger.info(f"ファイル名: {file.filename}, アクション: {action}")
        
        if file.filename == '':
            logger.warning("空のファイル名")
            flash('ファイルが選択されていません')
            return redirect(url_for('index'))
        
        if file and allowed_file(file.filename):
            try:
                # アップロードディレクトリを作成
                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                
                filename = secure_filename(file.filename)
                # ユニークなファイル名を生成
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                logger.info(f"ファイル保存先: {filepath}")
                
                file.save(filepath)
                
                # ファイルが正しく保存されたか確認
                if not os.path.exists(filepath):
                    logger.error("ファイル保存に失敗")
                    flash('ファイルの保存に失敗しました')
                    return redirect(url_for('index'))
                
                # ナンバープレート認識
                logger.info("ナンバープレート認識開始")
                plate_number = extract_plate_number(filepath)
                logger.info(f"認識結果: {plate_number}")
                
                if plate_number and not plate_number.startswith("認識できませんでした") and not plate_number.startswith("エラーが発生しました"):
                    try:
                        conn = sqlite3.connect('parking.db')
                        c = conn.cursor()
                        
                        if action == 'entry':
                            # 入場記録
                            c.execute("INSERT INTO parking_records (plate_number, entry_date, image_path) VALUES (?, ?, ?)",
                                     (plate_number, date.today(), filepath))
                            flash(f'入場記録を追加しました: {plate_number}')
                            logger.info(f"入場記録追加: {plate_number}")
                        else:
                            # 退場記録
                            c.execute("UPDATE parking_records SET exit_date = ?, is_active = 0 WHERE plate_number = ? AND is_active = 1",
                                     (date.today(), plate_number))
                            if c.rowcount > 0:
                                flash(f'退場記録を更新しました: {plate_number}')
                                logger.info(f"退場記録更新: {plate_number}")
                            else:
                                flash(f'該当する入場記録が見つかりません: {plate_number}')
                                logger.warning(f"入場記録なし: {plate_number}")
                        
                        conn.commit()
                        conn.close()
                    except Exception as e:
                        logger.error(f"データベースエラー: {e}")
                        flash('データベースエラーが発生しました')
                else:
                    flash(f'ナンバープレートの認識結果: {plate_number}')
                
                return redirect(url_for('records'))
                
            except Exception as e:
                logger.error(f"ファイル処理エラー: {e}")
                logger.error(f"トレースバック: {traceback.format_exc()}")
                flash(f'ファイル処理でエラーが発生しました: {str(e)}')
                return redirect(url_for('index'))
        
        flash('無効なファイル形式です')
        return redirect(url_for('index'))
    
    except Exception as e:
        logger.error(f"アップロード全般エラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")
        flash(f'予期しないエラーが発生しました: {str(e)}')
        return redirect(url_for('index'))

@app.route('/records')
def records():
    try:
        logger.info("記録一覧画面の表示開始")
        
        conn = sqlite3.connect('parking.db')
        
        # データベースの初期化を確認
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parking_records'")
        if not c.fetchone():
            logger.info("テーブルが存在しないため初期化")
            init_db()
        
        # 現在駐車中の車両
        try:
            active_cars = pd.read_sql_query("""
                SELECT plate_number, entry_date, 
                       (julianday('now') - julianday(entry_date)) as days_parked
                FROM parking_records 
                WHERE is_active = 1 
                ORDER BY entry_date DESC
            """, conn)
            logger.info(f"現在駐車中の車両数: {len(active_cars)}")
        except Exception as e:
            logger.error(f"現在駐車中車両の取得エラー: {e}")
            active_cars = pd.DataFrame()
        
        # 全記録
        try:
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
            logger.info(f"全記録数: {len(all_records)}")
        except Exception as e:
            logger.error(f"全記録の取得エラー: {e}")
            all_records = pd.DataFrame()
        
        conn.close()
        
        # DataFrameをdictに変換（エラー処理付き）
        try:
            active_cars_dict = active_cars.to_dict('records') if not active_cars.empty else []
            all_records_dict = all_records.to_dict('records') if not all_records.empty else []
        except Exception as e:
            logger.error(f"データ変換エラー: {e}")
            active_cars_dict = []
            all_records_dict = []
        
        logger.info("記録一覧画面の表示完了")
        
        return render_template('records.html', 
                             active_cars=active_cars_dict,
                             all_records=all_records_dict)
    
    except Exception as e:
        logger.error(f"記録一覧画面のエラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")
        flash(f'記録の取得中にエラーが発生しました: {str(e)}')
        return render_template('records.html', 
                             active_cars=[],
                             all_records=[])

@app.route('/analytics')
def analytics():
    try:
        logger.info("分析画面の表示開始")
        
        conn = sqlite3.connect('parking.db')
        
        # データベースの初期化を確認
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parking_records'")
        if not c.fetchone():
            logger.info("テーブルが存在しないため初期化")
            init_db()
            conn.close()
            conn = sqlite3.connect('parking.db')
        
        # 統計データを取得
        stats = {}
        
        # 現在の駐車台数
        try:
            result = pd.read_sql_query("SELECT COUNT(*) as count FROM parking_records WHERE is_active = 1", conn)
            stats['current_count'] = int(result.iloc[0]['count']) if not result.empty else 0
        except Exception as e:
            logger.error(f"現在の駐車台数取得エラー: {e}")
            stats['current_count'] = 0
        
        # 今日の入場台数
        try:
            result = pd.read_sql_query("SELECT COUNT(*) as count FROM parking_records WHERE entry_date = date('now')", conn)
            stats['today_entries'] = int(result.iloc[0]['count']) if not result.empty else 0
        except Exception as e:
            logger.error(f"今日の入場台数取得エラー: {e}")
            stats['today_entries'] = 0
        
        # 平均駐車日数
        try:
            result = pd.read_sql_query("""
                SELECT AVG(julianday(COALESCE(exit_date, 'now')) - julianday(entry_date)) as avg_days 
                FROM parking_records
            """, conn)
            avg_days = result.iloc[0]['avg_days'] if not result.empty and result.iloc[0]['avg_days'] is not None else 0
            stats['avg_days'] = round(float(avg_days), 1) if avg_days else 0
        except Exception as e:
            logger.error(f"平均駐車日数取得エラー: {e}")
            stats['avg_days'] = 0
        
        # 長期駐車車両（7日以上）
        try:
            long_term = pd.read_sql_query("""
                SELECT plate_number, entry_date,
                       (julianday('now') - julianday(entry_date)) as days_parked
                FROM parking_records 
                WHERE is_active = 1 AND (julianday('now') - julianday(entry_date)) >= 7
                ORDER BY days_parked DESC
            """, conn)
            long_term_dict = long_term.to_dict('records') if not long_term.empty else []
        except Exception as e:
            logger.error(f"長期駐車車両取得エラー: {e}")
            long_term_dict = []
        
        # 月別統計
        try:
            monthly_stats = pd.read_sql_query("""
                SELECT strftime('%Y-%m', entry_date) as month,
                       COUNT(*) as entries
                FROM parking_records 
                GROUP BY strftime('%Y-%m', entry_date)
                ORDER BY month DESC
                LIMIT 12
            """, conn)
            monthly_stats_dict = monthly_stats.to_dict('records') if not monthly_stats.empty else []
        except Exception as e:
            logger.error(f"月別統計取得エラー: {e}")
            monthly_stats_dict = []
        
        conn.close()
        
        logger.info("分析画面の表示完了")
        
        return render_template('analytics.html',
                             stats=stats,
                             long_term=long_term_dict,
                             monthly_stats=monthly_stats_dict)
    
    except Exception as e:
        logger.error(f"分析画面のエラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")
        flash(f'分析データの取得中にエラーが発生しました: {str(e)}')
        return render_template('analytics.html',
                             stats={'current_count': 0, 'today_entries': 0, 'avg_days': 0},
                             long_term=[],
                             monthly_stats=[])

@app.route('/export')
def export_csv():
    try:
        logger.info("CSV出力開始")
        
        conn = sqlite3.connect('parking.db')
        
        # データベースの初期化を確認
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='parking_records'")
        if not c.fetchone():
            logger.info("テーブルが存在しないため初期化")
            init_db()
            conn.close()
            conn = sqlite3.connect('parking.db')
        
        try:
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
            logger.info(f"CSV出力データ件数: {len(df)}")
        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            # 空のDataFrameを作成
            df = pd.DataFrame(columns=['ナンバープレート', '入場日', '退場日', '状態', '駐車日数', '記録作成日時'])
        
        conn.close()
        
        # CSVファイルを一時的に作成
        output = io.StringIO()
        df.to_csv(output, index=False, encoding='utf-8-sig')
        output.seek(0)
        
        # バイナリストリームに変換
        output_bytes = io.BytesIO()
        output_bytes.write(output.getvalue().encode('utf-8-sig'))
        output_bytes.seek(0)
        
        logger.info("CSV出力完了")
        
        return send_file(
            output_bytes,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'parking_records_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    
    except Exception as e:
        logger.error(f"CSV出力エラー: {e}")
        logger.error(f"トレースバック: {traceback.format_exc()}")
        flash(f'CSV出力中にエラーが発生しました: {str(e)}')
        return redirect(url_for('records'))

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

@app.route('/test_ocr')
def test_ocr():
    """OCR機能のテスト"""
    test_results = {
        'tesseract_available': False,
        'easyocr_available': EASYOCR_AVAILABLE,
        'opencv_version': cv2.__version__,
        'errors': []
    }
    
    # Tesseractテスト
    try:
        version = pytesseract.get_tesseract_version()
        test_results['tesseract_available'] = True
        test_results['tesseract_version'] = str(version)
    except Exception as e:
        test_results['errors'].append(f"Tesseract: {str(e)}")
    
    # EasyOCRテスト
    if EASYOCR_AVAILABLE:
        try:
            test_results['easyocr_languages'] = easyocr_reader.lang_list
        except Exception as e:
            test_results['errors'].append(f"EasyOCR: {str(e)}")
    
    return jsonify(test_results)

if __name__ == '__main__':
    init_db()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)