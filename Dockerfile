FROM ghcr.io/luxonis/robothub-app-v2:2023.108.0914-regular

RUN pip3 install --target /lib --upgrade depthai==2.21.2.0
RUN pip3 install -U av twilio humanize pandas Pillow pytz