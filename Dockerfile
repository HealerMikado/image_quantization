FROM public.ecr.aws/lambda/python:3.9

RUN yum install gcc openssl-devel bzip2-devel libffi-devel wget tar gzip zip make -y

COPY lambda/requirements.txt requirements.txt

RUN python3.9 -m pip install -r requirements.txt

COPY lambda/app.py app.py


CMD ['app.lambda_handler']