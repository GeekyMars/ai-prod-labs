import random
from typing import Dict
import numpy as np
import pandas as pd


def seed_everything(seed: int = 76) -> None:
    """
    Фиксирует зерно генератора случайных чисел для обеспечения
    воспроизводимости экспериментов
    """
    random.seed(seed)
    np.random.seed(seed)


def load_variant_data(filepath: str, variant: int = 30) -> pd.DataFrame:
    """
    Загружает данные указанного варианта из файла.
    Использует строгую математическую привязку колонок, чтобы обойти 
    любые проблемы с кодировками кириллицы и скрытыми пробелами в заголовках.
    """
    # Читаем только матрицу чисел, пропуская 3 строки текстовых заголовков.
    # Кодировка текста больше не имеет значения, так как цифры (ASCII) идентичны.
    df_raw = pd.read_csv(
        filepath,
        sep=";",
        decimal=",",
        skiprows=3,
        header=None,
        encoding="cp1251"
    )

    # Вычисляем индексы нужных столбцов.
    # 0-й столбец — дни. Каждый вариант занимает ровно 6 столбцов.
    start_col = 1 + (variant - 1) * 6
    
    var_subset = pd.DataFrame()
    var_subset["day"] = df_raw.iloc[:, 0].astype(int)
    
    col_names = ["мыло", "порошок", "средство", "краска", "пена", "прибыль"]
    
    for i, name in enumerate(col_names):
        col_idx = start_col + i
        col_data = df_raw.iloc[:, col_idx]
        
        # Защита от строковых чисел с запятой, если pandas не распознал их сам
        if col_data.dtype == object:
            col_data = col_data.astype(str).str.replace(",", ".").astype(float)
        else:
            col_data = col_data.astype(float)
            
        var_subset[name] = col_data

    return var_subset


def compute_metrics(original: np.ndarray, filtered: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет количественные метрики качества фильтрации:
      - MSE: среднеквадратичное отклонение от зашумленного оригинала
      - MAE: среднее абсолютное отклонение
      - Smoothness_Ratio: отношение дисперсии первой разности отфильтрованного ряда
        к дисперсии первой разности оригинала: чем меньше, тем лучше подавлен шум
      - Correlation: коэффициент линейной корреляции Пирсона с исходным рядом
    """
    orig = np.asarray(original, dtype=float)
    filt = np.asarray(filtered, dtype=float)

    mse = float(np.mean((orig - filt) ** 2))
    mae = float(np.mean(np.abs(orig - filt)))

    diff_orig = np.diff(orig)
    diff_filt = np.diff(filt)
    var_orig = float(np.var(diff_orig))
    var_filt = float(np.var(diff_filt))

    smoothness_ratio = float(var_filt / (var_orig + 1e-9))
    corr = float(np.corrcoef(orig, filt)[0, 1])

    return {
        "MSE": round(mse, 4),
        "MAE": round(mae, 4),
        "Smoothness_Ratio": round(smoothness_ratio, 4),
        "Correlation": round(corr, 4),
    }


def compute_autocorrelation(series: np.ndarray, max_lags: int = 20) -> np.ndarray:
    """
    Вычисляет выборочную автокорреляционную функцию (ACF) для анализа
    скрытой периодичности и сезонности временного ряда для Задачи 3
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    s_norm = s - np.mean(s)
    autocov = np.correlate(s_norm, s_norm, mode="full")[n - 1 :]
    acf = autocov / (autocov[0] + 1e-9)
    return acf[: max_lags + 1]