FROM python:3.10.0

RUN mkdir /src

COPY ./requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY . /src/
WORKDIR /src

CMD python server.py