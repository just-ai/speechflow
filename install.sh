git submodule update --init --recursive -f
sh env/build.sh

IMG_NAME="torch_$(date '+%d%m%Y').img"
singularity run -w --nv --no-home -B .:/src --pwd /src $IMG_NAME python -m pytest libs/multilingual_text_parser/tests tests
