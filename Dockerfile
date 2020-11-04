FROM ubuntu
MAINTAINER Michal Mika
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y python3 python3-pip libsuitesparse-dev python3-numpy vim
ADD . /home/tensiga/
RUN pip3 install -e /home/tensiga/
ENTRYPOINT [ "/bin/bash" ]

