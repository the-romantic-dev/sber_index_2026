"""Чтение гиперпараметров из JSON; пути отсчитываются от файла конфигурации."""
import json
from pathlib import Path


def load_config(path):
    """Вернуть параметры и абсолютные data_dir/output_dir без зависимости от cwd."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    for key in ['data_dir', 'output_dir']:
        config[key] = (path.parent / config[key]).resolve()
    return config
