FROM python:3.10

COPY ./requirements.txt /src/inference/requirements.txt

COPY src/training/artifacts/champion_model.pkl /src/inference/champion_model.pkl

COPY src/inference/* /src/inference

WORKDIR /src/inference

RUN pip install -r requirements.txt

EXPOSE 8000

ENTRYPOINT [ "uvicorn" ]

CMD [ "--host", "0.0.0.0", "api_server:app" ]
