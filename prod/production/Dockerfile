ARG buildImage
ARG runtimeImage

FROM ${buildImage} AS build

WORKDIR ${APP_ROOT}

COPY --chown=${USER} ./requirements.txt ./model-requirements.txt
RUN python3 -m pip install --no-cache-dir --no-warn-script-location -r requirements.txt -r model-requirements.txt --compile --user

FROM ${runtimeImage} AS runtime

WORKDIR ${APP_ROOT}

COPY --chown=${USER} --from=build ${APP_ROOT}/index.py .
COPY --chown=${USER} --from=build ${APP_ROOT}/.local ./.local
COPY --chown=${USER} --from=build ${APP_ROOT}/core ./core
COPY --chown=${USER} ./model ./core/model
COPY --chown=${USER} ./tests ./core/tests

USER ${USER}

ENV PATH ${APP_ROOT}/.local/bin:${PATH}
WORKDIR ${APP_ROOT}/core
#RUN coverage run --source=model -m unittest discover
#RUN coverage report
WORKDIR ${APP_ROOT}

#ENTRYPOINT ["python3 index.py"]
RUN echo "python3 index.py" > entrypoint.sh
RUN chmod +x entrypoint.sh
CMD ./entrypoint.sh
ENV fprocess="python3 index.py"
ENV cgi_headers="true"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:5000"
