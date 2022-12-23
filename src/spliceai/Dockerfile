FROM python:3.10.0

RUN mkdir /dnalm_service

COPY ../../. /dnalm_service/

RUN pip install -r /dnalm_service/requirements.txt
RUN python setup.py install

WORKDIR /dnalm_service

CMD gunicorn server:app