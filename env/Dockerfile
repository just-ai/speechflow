FROM nvcr.io/nvidia/tritonserver:21.03-py3

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Europe/Moscow apt-get -y install tzdata

# install system requirements
RUN apt-get update
RUN apt-get install -y libssl1.1
RUN apt-get install -y g++
RUN apt-get install -y -qq sox zip unzip p7zip-full ffmpeg

# install .NET SDK
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN rm packages-microsoft-prod.deb

RUN apt-get install -y apt-transport-https && apt-get update
RUN apt-get install -y dotnet-sdk-5.0
RUN apt-get install -y aspnetcore-runtime-5.0
RUN apt-get install -y dotnet-runtime-5.0
RUN apt-get install -y nuget

# install eSpeak
RUN apt-get install -y espeak-ng

# install Python 3.10
RUN wget https://repo.continuum.io/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
RUN bash Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -b -p /ext3/miniconda3
RUN rm Miniconda3-py310_24.1.2-0-Linux-x86_64.sh

RUN /usr/bin/update-alternatives --install /usr/bin/python python /ext3/miniconda3/bin/python 1
RUN /usr/bin/update-alternatives --install /usr/bin/python3 python3 /ext3/miniconda3/bin/python3 1
RUN /usr/bin/update-alternatives --install /usr/bin/pip pip /ext3/miniconda3/bin/pip 1
RUN /usr/bin/update-alternatives --install /usr/bin/pip3 pip3 /ext3/miniconda3/bin/pip3 1

# install python requirements
RUN pip install git+https://github.com/just-ai/speechflow.git
RUN pip install git+https://github.com/just-ai/multilingual-text-parser.git
RUN pip install fairseq==0.12.2 --no-deps
RUN pip install streamlit==1.44.1
RUN pip install st-annotated-text==4.0.2
RUN pip install grpcio-channelz==1.40
RUN pip install protobuf==3.20
