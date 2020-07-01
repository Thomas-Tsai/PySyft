FROM billylin26/syft-proto as base

FROM base as builder

WORKDIR install

RUN apt-get update
# RUN apt-get install -y python3-pip python3-dev

COPY . /install
RUN pip install --user -r ./pip-dep/requirements.txt
COPY . /install
RUN pip install --user .

FROM billylin26/syft-proto as syft

COPY --from=builder root/.local root/.local
COPY --from=builder /install /install

ENV PATH=/root/.local/bin:$PATH
