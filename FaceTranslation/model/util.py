from datetime import datetime
import os

import torch

from FaceTranslation.util import project_root


def save_model(model, timestamp):
    tstamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    model_dir = os.path.join(project_root, 'models', tstamp)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, os.path.join(model_dir, 'model.pt'))
    torch.save(model.state_dict(), os.path.join(model_dir, 'state.pt'))


def load_model(model_dir=None):
    if model_dir is None:
        models = os.listdir(os.path.join(project_root, 'models'))
        model_dir = sorted(models)[-1]

    full_model_dir = os.path.join(project_root, 'models', model_dir)
    model = torch.load(os.path.join(full_model_dir, 'model.pt'))
    state = torch.load(os.path.join(full_model_dir, 'state.pt'))
    model.load_state_dict(state)
    return model
