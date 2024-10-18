FROM ghcr.io/pyvista/pyvista:latest-slim

RUN apt update && apt install -y build-essential
RUN pip install noise trame trame-vtk trame-vuetify scipy pyserial

COPY main.py .
COPY assets ./assets


CMD [ "python3", "main.py" ]