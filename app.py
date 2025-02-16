import shutil
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
import os
import time
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import get_device

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

def emit_progress(job_id: str, pageIndex: int, is_done: bool, content: dict, img_dir: str) -> None:
    """
    透過 SocketIO 發送進度更新訊息。

    參數:
        job_id: 工作識別碼。
        pageIndex: 當前處理頁面。
        is_done: 是否為最後一頁。
        content: 頁面處理後的中間結果。
    """
    print(f"img_dir: {img_dir}")
    progress_data = {
        'jobId': job_id,
        'pageIndex': pageIndex,
        'imgDir': img_dir,
        'isDone': is_done,
        'content': content
    }
    socketio.emit('page_processed', progress_data)
    socketio.sleep(0)

def process_single_page(ds: PymuDocDataset, page_id: int, image_writer: FileBasedDataWriter, is_ocr: bool):
    """
    處理單一頁面，依據是否為 OCR 模式分別呼叫不同的處理方法。

    回傳該頁面的處理結果。
    """
    if is_ocr:
        # OCR 模式處理：進行 layout 分析後進行文字提取
        infer_result = ds.apply(doc_analyze, ocr=True, start_page_id=page_id, end_page_id=page_id)
        page_result = infer_result.pipe_ocr_mode(image_writer, start_page_id=page_id, end_page_id=page_id)
    else:
        # 非 OCR 模式處理：直接進行文字提取
        infer_result = ds.apply(doc_analyze, ocr=False, start_page_id=page_id, end_page_id=page_id)
        page_result = infer_result.pipe_txt_mode(image_writer, start_page_id=page_id, end_page_id=page_id)
    return page_result

def process_pdf(pdf_file_path: str, job_id: str):
    """
    處理上傳的 PDF 檔案，依序處理每一頁並透過 SocketIO 回傳進度訊息。

    該函式會在背景執行緒中運作，不會阻塞主線程。
    """
    start_time = time.time()
    pdf_file_name = os.path.basename(pdf_file_path)

    # 建立輸出目錄
    output_image_dir = os.path.join("output", job_id, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    # 取得輸出目錄的絕對路徑
    output_abs_dir = os.path.abspath(output_image_dir)
    # 初始化資料寫入器 (md_writer 目前未使用，如有需要可啟用)
    image_writer = FileBasedDataWriter(output_image_dir)

    # 讀取 PDF 檔案內容
    pdf_reader = FileBasedDataReader("")
    pdf_bytes = pdf_reader.read(pdf_file_path)

    # 初始化資料集
    ds = PymuDocDataset(pdf_bytes)
    page_count = ds.apply(lambda d: len(d))
    print(f"總頁數: {page_count}")

    # 根據 classify 結果判斷是否使用 OCR 模式
    is_ocr = ds.classify() == SupportedPdfParseMethod.OCR
    pipe_results = []

    # 逐頁處理 PDF
    for page_id in range(page_count):
        print(f"開始處理第 {page_id + 1} 頁")
        page_start_time = time.time()

        # 處理單一頁面
        page_result = process_single_page(ds, page_id, image_writer, is_ocr)
        pipe_results.append(page_result)

        # 發送進度更新訊息
        emit_progress(
            job_id=job_id,
            pageIndex=page_id + 1,
            is_done=(page_id == page_count - 1),
            content=page_result.get_middle_json(),
            img_dir=output_abs_dir
        )

        print(f"第 {page_id + 1} 頁處理完畢，耗時 {time.time() - page_start_time:.2f} 秒")

    # 清除 GPU 或其他資源記憶體
    clean_memory(get_device())
    total_time = time.time() - start_time
    print(f"所有頁面處理完畢，總耗時 {total_time:.2f} 秒")

    # 回傳最終處理結果
    return {
        "status": "success",
        "processing_time": total_time,
        "pdf_name": pdf_file_name
    }

@app.route('/process-pdf', methods=['POST'])
def handle_pdf():
    """
    處理上傳 PDF 的 HTTP 請求，將 PDF 儲存後交由背景執行緒處理，並回傳初始狀態。
    """
    try:
        print('開始處理 PDF 請求')
        # 從 JSON 請求體中獲取資料
        data = request.get_json()
        
        # 檢查是否有提供檔案路徑
        fileAbsolutePath = data.get('fileAbsolutePath')
        if not fileAbsolutePath:
            return jsonify({"error": "未提供檔案路徑"}), 400
        print(f"fileAbsolutePath: {fileAbsolutePath}")
        job_id = data.get('jobId')
        print(f"jobId: {job_id}")

        # 在背景執行緒中處理 PDF，避免阻塞 HTTP 主線程
        try:
            result = process_pdf(fileAbsolutePath, job_id)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # 檢查資料夾是否存在後再刪除
        image_folder = os.path.join("output", job_id)
        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

@socketio.on('connect')
def handle_connect():
    print('用戶端已連線')

@socketio.on('disconnect')
def handle_disconnect():
    print('用戶端已斷線')
    
if __name__ == '__main__':
    print('啟動 Flask 伺服器')
    socketio.run(app, host='0.0.0.0', port=5050, allow_unsafe_werkzeug=True)
