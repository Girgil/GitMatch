FROM mongo:latest

COPY /mongo/importscript.sh /docker-entrypoint-initdb.d/
COPY /mongo/repertoire_features.json /data/repertoire_features.json
COPY /mongo/users.json /data/users.json

RUN chmod +x /docker-entrypoint-initdb.d/importscript.sh