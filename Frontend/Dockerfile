# ใช้ Python เป็น Base Image
FROM python:3.9-slim

# ตั้งค่า Working Directory
WORKDIR /app

# คัดลอกไฟล์ที่จำเป็นลง Container
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดของ Frontend
COPY . .

# รัน Frontend บน port 8090
CMD ["python", "Interface.py"]
