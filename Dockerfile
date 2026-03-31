


FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH="/app"
RUN git clone https://github.com/clarkmaio/dkppa.git .

RUN pip install -r requirements.txt
RUN pip list
CMD ["streamlit", "run", "src/app/app.py", "--server.port", "7860"]