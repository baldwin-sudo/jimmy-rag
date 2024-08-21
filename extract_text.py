import os
import pytesseract
from pdf2image import convert_from_path
import pdfplumber
from PIL import Image
import openpyxl

# Paths
files_dir = "files"
texts_dir = "texts"

# Create texts_dir if it doesn't exist
os.makedirs(texts_dir, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        # Try using pdfplumber first for text extraction
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        if not text.strip():
            raise ValueError("PDF is likely an image-based PDF")
    except Exception as e:
        # Use OCR if pdfplumber fails (likely due to an image-based PDF)
        images = convert_from_path(file_path)
        for image in images:
            text += pytesseract.image_to_string(image)
    return text

def extract_text_from_xlsx(file_path):
    workbook = openpyxl.load_workbook(file_path)
    text = ""
    for sheet in workbook:
        for row in sheet.iter_rows(values_only=True):
            text += " ".join([str(cell) for cell in row if cell is not None]) + "\n"
    return text

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.xlsx':
        return extract_text_from_xlsx(file_path)
    else:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

# Process each file in the files_dir
for filename in os.listdir(files_dir):
    file_path = os.path.join(files_dir, filename)
    if os.path.isfile(file_path):
        print(f"Processing {filename}...")
        try:
            text = extract_text_from_file(file_path)
            output_filename = os.path.splitext(filename)[0] + ".txt"
            output_path = os.path.join(texts_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            print(f"Text extracted and saved to {output_filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
