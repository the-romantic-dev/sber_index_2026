"""Командный запуск: python -m solution --config config.json."""
import argparse
import os

from .config import load_config


def main():
    """Прочитать конфигурацию и выполнить полный расчёт."""
    parser = argparse.ArgumentParser(description='Динамические группы муниципалитетов СберИндекса')
    parser.add_argument('--config', default='config.json', help='Путь к JSON с параметрами проекта')
    args = parser.parse_args()
    config = load_config(args.config)
    os.environ.setdefault('OMP_NUM_THREADS', str(config['runtime']['threads']))
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from .pipeline import run
    run(config)


if __name__ == '__main__':
    main()
