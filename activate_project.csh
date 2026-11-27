setenv PROJECT_ROOT $cwd

source $PROJECT_ROOT/.venv/bin/activate.csh

setenv HF_HOME $PROJECT_ROOT/.cache/huggingface
setenv TRANSFORMERS_CACHE $PROJECT_ROOT/.cache/huggingface/transformers
setenv HF_DATASETS_CACHE $PROJECT_ROOT/.cache/huggingface/datasets
setenv TORCH_HOME $PROJECT_ROOT/.cache/torch
setenv PIP_CACHE_DIR $PROJECT_ROOT/.cache/pip
setenv TMPDIR $PROJECT_ROOT/.tmp

mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE $TORCH_HOME $PIP_CACHE_DIR $TMPDIR

cd $PROJECT_ROOT
