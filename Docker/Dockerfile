FROM python:3.6

RUN pip3 install 'tensorflow==1.15.0'
RUN pip3 install 'matplotlib==3.1.2'
RUN pip3 install 'opencv-python==4.1.2.30'

RUN git clone https://github.com/StefanCosminR/crack-detector.git

EXPOSE 8080

CMD cd ./crack-detector/CracksDetectionApp && \
    python3 predict.py --meta_file="./alg2_output/model_complete.meta" --CP_dir="./alg2_output" --start_as_server=True --port=8080


