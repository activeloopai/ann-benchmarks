#WORKDIR /home/app
FROM ann-benchmarks
ENV BUGGER_OFF=true
#COPY requirements.txt run_algorithm.py ./

RUN python3.10 -m pip install deeplake
RUN python3.10 -c 'import deeplake'
RUN python3.10 -c 'pip install data/libdeeplake-0.0.84-cp310-cp310-manylinux2010_x86_64.whl'