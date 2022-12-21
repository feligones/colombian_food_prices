FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10-slim

RUN pip --no-cache-dir install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --deploy --system && \
    rm -rf /root/.cache

EXPOSE 80

COPY . .

CMD ["python", "app/bridge.py"]