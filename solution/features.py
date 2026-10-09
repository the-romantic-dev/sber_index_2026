"""Региональная нормировка, масштабирование и выбор EWMA только по 2023."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def regionalize(x, regions):
    """Отклонения конечных потребительских координат от сглаженного регионального уровня."""
    codes, inverse, counts = np.unique(regions, return_inverse=True, return_counts=True)
    counts = counts[:, None]
    grand_mean = x.mean(axis=0)
    means = np.stack([x[inverse == i].mean(axis=0) for i in range(len(codes))])
    within_variance = np.sum([
        ((x[inverse == i] - mean) ** 2).sum(axis=0)
        for i, mean in enumerate(means)
    ], axis=0) / max(len(x) - len(codes), 1)
    between_variance = (counts * (means - grand_mean) ** 2).sum(axis=0) / len(x)
    noise_variance = within_variance * len(codes) / len(x)
    signal_variance = np.maximum(between_variance - noise_variance, 0)
    numerator = counts * signal_variance
    denominator = numerator + within_variance
    strength = np.divide(numerator, denominator, out=np.zeros_like(means), where=denominator > 0)
    strength = np.where(counts >= 2, strength, 0)
    return x - (grand_mean + strength * (means - grand_mean))[inverse]


def ewma(x, alpha):
    """Сгладить массив по первой оси (времени), сохранив его форму."""
    smoothed = x.copy()
    for month in range(1, len(smoothed)):
        smoothed[month] = alpha * smoothed[month] + (1 - alpha) * smoothed[month - 1]
    return smoothed


def forecasts(cube, regions, config):
    """Из (24, n, 7) получить ошибки прогнозов, выбранную alpha и две панели."""
    relative = np.stack([regionalize(month, regions) for month in cube])
    scaler = RobustScaler().fit(relative[:config["features"]["scaler_months"]].reshape(-1, cube.shape[2]))
    scaled = scaler.transform(relative.reshape(-1, cube.shape[2])).reshape(relative.shape)
    scaled[:, :, :6] /= np.sqrt(6)
    scaled /= np.sqrt(2)

    candidates = {
        f'ewma_{alpha}': np.concatenate([
            np.zeros_like(scaled[:1]), ewma(scaled, alpha)[:-1],
        ])
        for alpha in config["features"]["alpha_candidates"]
    }
    candidates['train_median'] = np.zeros_like(scaled)
    background = (
        scaler.transform(np.zeros_like(relative).reshape(-1, cube.shape[2]))
        .reshape(relative.shape)
    )
    background[:, :, :6] /= np.sqrt(6)
    background /= np.sqrt(2)
    candidates['regional_background'] = background
    candidates['lag12'] = np.concatenate([
        np.zeros_like(scaled[:12]), scaled[:-12],
    ])

    rows = []
    for name, prediction in candidates.items():
        for split, months in [
            ('validation', range(config['features']['scaler_months'], 12)),
            ('test', range(12, len(scaled))),
        ]:
            if name == 'lag12' and split == 'validation':
                continue
            for month in months:
                rows.append(dict(
                    model=name,
                    split=split,
                    t=month,
                    MSE=np.mean((scaled[month] - prediction[month]) ** 2),
                ))

    forecast_errors = pd.DataFrame(rows)
    validation_errors = forecast_errors[
        (forecast_errors.split == 'validation')
        & forecast_errors.model.str.startswith('ewma')
    ]
    chosen_model = (
        validation_errors.groupby('model').MSE.mean()
        .sort_values(kind='stable').index[0]
    )
    return forecast_errors, float(chosen_model.split('_')[1]), scaled, relative
