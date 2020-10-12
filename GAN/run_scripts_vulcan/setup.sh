# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/vulcan/scratch/yogesh22/Software/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/vulcan/scratch/yogesh22/Softwares/anaconda/etc/profile.d/conda.sh" ]; then
        . "/vulcan/scratch/yogesh22/Softwares/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/vulcan/scratch/yogesh22/Softwares/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup

conda deactivate
conda activate main_env
module add cuda/10.0.130
module add cudnn/v7.5.0
