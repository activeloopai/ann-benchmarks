FROM ann-benchmarks
ENV BUGGER_OFF=true
ENV ACTIVELOOP_TOKEN=eyJhbGciOiJIUzUxMiIsImlhdCI6MTY4MTI4Mzc3NywiZXhwIjoxNzEyOTA2MDk5fQ.eyJpZCI6Im5vdGlmeSJ9.C3zTjQODfq0TUhkdRye639aKJ0FOanZuCwL2ks2NiKXJ6YecYVMBrdFu3AabGsk7iuS-ELtQYp7WxITv76hcSg
ENV ACTIVELOOP_ORG=notify

RUN apt-get install -y python-setuptools python-pip
RUN pip3 install deeplake
RUN python3 -c 'import deeplake'
