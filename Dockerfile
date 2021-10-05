FROM python:3.9

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y \
    coinor-cbc

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r  requirements.txt 


ENV PORT=


COPY app.py . 

CMD streamlit run app.py --server.port=${PORT} --browser.serverAddress="0.0.0.0"