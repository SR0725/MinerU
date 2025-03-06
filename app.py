import shutil
from flask import Flask, request, jsonify
import os
import time
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.libs.clean_memory import clean_memory
from magic_pdf.libs.config_reader import get_device

app = Flask(__name__)

# 儲存中間處理結果的字典
processing_results = {}

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

def process_pdf_start(pdf_file_path: str, job_id: str):
    """
    準備開始處理PDF，進行預處理並回傳總頁數。
    
    參數:
        pdf_file_path: PDF檔案的絕對路徑
        job_id: 工作識別碼
        
    回傳:
        包含總頁數的字典
    """
    start_time = time.time()
    pdf_file_name = os.path.basename(pdf_file_path)

    # 建立輸出目錄
    output_image_dir = os.path.join("output", job_id, "images")
    os.makedirs(output_image_dir, exist_ok=True)

    # 取得輸出目錄的絕對路徑
    output_abs_dir = os.path.abspath(output_image_dir)
    
    # 讀取 PDF 檔案內容
    pdf_reader = FileBasedDataReader("")
    pdf_bytes = pdf_reader.read(pdf_file_path)

    # 初始化資料集
    ds = PymuDocDataset(pdf_bytes)
    page_count = ds.apply(lambda d: len(d))
    print(f"總頁數: {page_count}")
    
    # 根據 classify 結果判斷是否使用 OCR 模式
    is_ocr = ds.classify() == SupportedPdfParseMethod.OCR
    
    # 儲存處理狀態到全域變數
    processing_results[job_id] = {
        "ds": ds,
        "is_ocr": is_ocr,
        "output_image_dir": output_image_dir,
        "output_abs_dir": output_abs_dir,
        "pdf_file_name": pdf_file_name,
        "start_time": start_time,
        "pages": [],
        "page_count": page_count
    }
    
    return {
        "status": "success",
        "pageCount": page_count,
        "pdfName": pdf_file_name
    }

def process_pdf_with(pdf_file_path: str, job_id: str, page: int):
    """
    處理PDF的特定頁面
    
    參數:
        pdf_file_path: PDF檔案的絕對路徑
        job_id: 工作識別碼
        page: 要處理的頁面索引
        
    回傳:
        該頁面的處理結果
    """
    if job_id not in processing_results:
        return {"error": "未找到此工作，請先呼叫 process_pdf_start"}, 400
    
    job_data = processing_results[job_id]
    
    if page < 0 or page >= job_data["page_count"]:
        return {"error": f"頁面索引超出範圍(0-{job_data['page_count']-1})"}, 400
    
    # 檢查該頁是否已被處理
    for processed_page in job_data["pages"]:
        if processed_page["pageIndex"] == page:
            return processed_page
    
    print(f"開始處理第 {page} 頁")
    page_start_time = time.time()
    
    # 初始化資料寫入器
    image_writer = FileBasedDataWriter(job_data["output_image_dir"])
    
    # 處理單一頁面
    page_result = process_single_page(job_data["ds"], page, image_writer, job_data["is_ocr"])
    page_middle_json = page_result.get_middle_json()
    
    # 建立頁面結果
    page_data = {
        "pageIndex": page,
        "content": page_middle_json,
        "imgDir": job_data["output_abs_dir"],
        "processingTime": time.time() - page_start_time
    }
    
    # 將頁面結果加入到工作結果中
    job_data["pages"].append(page_data)
    
    print(f"第 {page} 頁處理完畢，耗時 {time.time() - page_start_time:.2f} 秒")
    
    return page_data

def process_pdf_final_result(pdf_file_path: str, job_id: str):
    """
    完成PDF處理，回傳最終結果並清理資源
    
    參數:
        pdf_file_path: PDF檔案的絕對路徑
        job_id: 工作識別碼
        
    回傳:
        處理的最終結果
    """
    if job_id not in processing_results:
        return {"error": "未找到此工作，請先呼叫 process_pdf_start"}, 400
    
    job_data = processing_results[job_id]
    
    # 清除 GPU 或其他資源記憶體
    clean_memory(get_device())
    
    total_time = time.time() - job_data["start_time"]
    print(f"所有頁面處理完畢，總耗時 {total_time:.2f} 秒")
    
    # 整理最終結果
    result = {
        "status": "success",
        "processing_time": total_time,
        "pdf_name": job_data["pdf_file_name"],
        "pages": job_data["pages"]
    }
    
    # 刪除處理資料
    del processing_results[job_id]
    
    # 刪除臨時資料夾
    image_folder = os.path.join("output", job_id)
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)
    
    return result

@app.route('/process-pdf-start', methods=['POST'])
def handle_pdf_start():
    """
    處理PDF開始預處理的HTTP請求，回傳PDF的頁數。
    """
    try:
        print('開始PDF預處理請求')
        # 從 JSON 請求體中獲取資料
        data = request.get_json()
        
        # 檢查是否有提供檔案路徑
        fileAbsolutePath = data.get('fileAbsolutePath')
        if not fileAbsolutePath:
            return jsonify({"error": "未提供檔案路徑"}), 400
        print(f"fileAbsolutePath: {fileAbsolutePath}")
        job_id = data.get('jobId')
        print(f"jobId: {job_id}")

        try:
            result = process_pdf_start(fileAbsolutePath, job_id)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-pdf-page', methods=['POST'])
def handle_pdf_page():
    """
    處理PDF特定頁面的HTTP請求。
    """
    try:
        # 從 JSON 請求體中獲取資料
        data = request.get_json()
        
        # 檢查是否有提供必要資料
        fileAbsolutePath = data.get('fileAbsolutePath')
        if not fileAbsolutePath:
            return jsonify({"error": "未提供檔案路徑"}), 400
        job_id = data.get('jobId')
        if not job_id:
            return jsonify({"error": "未提供jobId"}), 400
        page = data.get('page')
        if page is None:
            return jsonify({"error": "未提供頁碼"}), 400
        
        print(f"處理PDF '{fileAbsolutePath}' 的第 {page} 頁")

        try:
            result = process_pdf_with(fileAbsolutePath, job_id, page)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process-pdf-complete', methods=['POST'])
def handle_pdf_complete():
    """
    完成PDF處理的HTTP請求，清理資源並回傳最終結果。
    """
    try:
        # 從 JSON 請求體中獲取資料
        data = request.get_json()
        
        # 檢查是否有提供必要資料
        fileAbsolutePath = data.get('fileAbsolutePath')
        if not fileAbsolutePath:
            return jsonify({"error": "未提供檔案路徑"}), 400
        job_id = data.get('jobId')
        if not job_id:
            return jsonify({"error": "未提供jobId"}), 400
        
        print(f"完成PDF '{fileAbsolutePath}' 的處理")

        try:
            result = process_pdf_final_result(fileAbsolutePath, job_id)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    print('啟動 Flask 伺服器')
    app.run(host='0.0.0.0', port=5050)
