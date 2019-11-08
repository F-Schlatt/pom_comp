FROM python:3.7

ADD ./agent

RUN pip install torch
RUN pip install git+https://www.github.com/MultiAgentLearning/playground
RUN pip install git+https://www.github.com/fschlatt/arm

EXPOSE 10080

ENV NAME Agent

# Run app.py when the container launches
WORKDIR /agent
ENTRYPOINT ["python"]
CMD ["run.py"]
