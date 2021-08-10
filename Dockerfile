FROM pytorch/pytorch:latest

WORKDIR /interactive_segmentation/

COPY isegm /interactive_segmentation/isegm
COPY misc  /interactive_segmentation/misc
COPY models /interactive_segmentation/models
COPY ritm_api /interactive_segmentation/ritm_api
COPY train.py /interactive_segmentation/
COPY config.yml /interactive_segmentation/
COPY inferenceAPI_tutorial.ipynb /interactive_segmentation/
COPY requirements.txt /interactive_segmentation/

RUN pip install --upgrade pip
RUN conda install jupyterlab
RUN pip install -r requirements.txt
RUN echo $(ls)
RUN echo $PWD
WORKDIR /