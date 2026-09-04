module purge
ml python/3.12.13 uv/0.11.2

rm -rf $SCRATCH/venvs/sae-llm-venv

uv venv $SCRATCH/venvs/sae-llm-venv --python 3.12.13
source $SCRATCH/venvs/sae-llm-venv/bin/activate

cd $SCRATCH/repos/sae-llm
uv pip install -e .