FROM gitpod/workspace-full-vnc

RUN sudo apt-get update \
&& sudo pip3 install requirements.txt
