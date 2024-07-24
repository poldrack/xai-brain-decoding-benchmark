# find run with best validation performance

import os
import pandas as pd
from pathlib import Path
import numpy as np
import shutil


def get_mean_loss(data):
    # get mean loss from last epoch in each fold
    run_losses = {}
    run_accs = {}
    for run in data['run'].unique():
        run_data = data[(data['run'] == run)]
        run_losses[run] = run_data['loss'].iloc[-1]
        run_accs[run] = run_data['accuracy'].iloc[-1]
        print(f'Run {run} has mean loss {run_losses[run]} and accuracy {run_accs[run]}')

    print('Best run:', min(run_losses, key=run_losses.get))
    return min(run_losses, key=run_losses.get)


if __name__ == "__main__":

    basedir = Path('results/models').absolute()
    datafiles = list(basedir.glob('task*/validation_history.csv'))
    for datafile in datafiles:
        data = pd.read_csv(datafile)
        best_run = get_mean_loss(data)

        # symlink best run dir to best_model_fit
        taskdir = datafile.parent
        best_run_dir = taskdir / 'best_model_dir'

        if best_run_dir.exists():
            print('Removing existing best_model_dir')
            shutil.rmtree(best_run_dir)

        if not best_run_dir.exists():
            os.makedirs(best_run_dir)
        
        print('Symlinking', taskdir / f'run-{best_run}', best_run_dir / f'run-{best_run}')
        os.symlink(taskdir / f'run-{best_run}', best_run_dir / f'run-{best_run}')
        os.symlink(taskdir / f'run-{best_run}' / 'config.json', best_run_dir / 'config.json')
        