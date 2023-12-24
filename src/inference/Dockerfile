FROM python:3.10

COPY ./requirements.txt /src/inference/requirements.txt

WORKDIR /src/inference

RUN pip install -r requirements.txt

COPY src/inference/* /src/inference

EXPOSE 8000

ENTRYPOINT [ "uvicorn" ]

CMD [ "--host", "0.0.0.0", "main:app" ]