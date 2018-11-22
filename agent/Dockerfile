FROM python:3.6

ADD ./agent /agent

# @TODO to be replaced with `pip install pommerman`
RUN pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
ADD ./agent/playground /pommerman
RUN cd /pommerman && pip install .
# end @TODO

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
