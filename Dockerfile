ARG IMAGE_VARIANT=slim-buster
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.6.13

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM openjdk:${OPENJDK_VERSION}-${IMAGE_VARIANT}

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.1.2

WORKDIR .
COPY . .

RUN pip install --upgrade pip
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION}
RUN pip install -r requirements.txt

CMD ["main.py"]
ENTRYPOINT ["python"]






