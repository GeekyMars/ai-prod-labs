from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import savgol_filter


class BaseFilter(ABC):
    """
    Базовый абстрактный класс для алгоритмов фильтрации временных рядов
    """
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def apply(self, data: np.ndarray | list) -> np.ndarray:
        """
        Основной метод применения фильтра к одномерному временному ряду
        :param data: Входной ряд наблюдений
        :return: Отфильтрованный массив той же длины
        """
        pass


class ExponentialSmoothing(BaseFilter):
    """
    Экспоненциальное сглаживание (EMA)
    Формула: s_t = alpha * x_t + (1 - alpha) * s_{t-1}
    """
    def __init__(self, alpha: float = 0.2, name: str | None = None):
        super().__init__(name)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Параметр alpha должен лежать в диапазоне (0, 1].")
        self.alpha = alpha

    def apply(self, data: np.ndarray | list) -> np.ndarray:
        series = np.asarray(data, dtype=float)
        smoothed = np.empty_like(series)
        smoothed[0] = series[0]

        for t in range(1, len(series)):
            smoothed[t] = self.alpha * series[t] + (1.0 - self.alpha) * smoothed[t - 1]

        return smoothed


class KalmanFilter1D(BaseFilter):
    """
    Одномерный скалярный фильтр Калмана
    Модель случайного блуждания:
      x_t = x_{t-1} + w_t,  w_t ~ N(0, Q)  (процесс)
      z_t = x_t + v_t,      v_t ~ N(0, R)  (измерения)
    """

    def __init__(
        self,
        q: float = 0.05,
        r: float = 1.0,
        initial_p: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name)
        self.q = q  # дисперсия шума процесса
        self.r = r  # дисперсия шума измерений
        self.initial_p = initial_p

    def apply(self, data: np.ndarray | list) -> np.ndarray:
        series = np.asarray(data, dtype=float)
        n = len(series)
        filtered = np.empty(n, dtype=float)

        x_hat = series[0]
        p = self.initial_p

        for t in range(n):
            # Шаг прогноза
            p = p + self.q
            # Шаг коррекции
            k = p / (p + self.r)
            x_hat = x_hat + k * (series[t] - x_hat)
            p = (1.0 - k) * p

            filtered[t] = x_hat

        return filtered


class SavitzkyGolayFilter(BaseFilter):
    """
    Фильтр Савицкого-Голея
    Локальное сглаживание ряда полиномом методом наименьших квадратов
    """
    def __init__(
        self,
        window_size: int = 7,
        poly_order: int = 2,
        name: str | None = None,
    ):
        super().__init__(name)
        if window_size % 2 == 0:
            raise ValueError("Параметр window_size должен быть нечетным числом!")
        if window_size <= poly_order:
            raise ValueError("Размер окна должен быть строго больше степени полинома!")
        self.window_size = window_size
        self.poly_order = poly_order

    def apply(self, data: np.ndarray | list[float]) -> np.ndarray:
        series = np.asarray(data, dtype=float)
        result = savgol_filter(
            series,
            window_length=self.window_size,
            polyorder=self.poly_order,
        )

        return np.asarray(result, dtype=float)


class LMSFilter(BaseFilter):
    """
    Адаптивный фильтр Нормализованного МНК (NLMS) в режиме линейного предиктора
    Настраивает веса w для минимизации ошибки e_t = x_t - w^T * x_{t-1:t-p}
    """
    def __init__(
        self,
        order: int = 3,
        mu: float = 0.2,
        eps: float = 1e-4,
        name: str | None = None,
    ):
        super().__init__(name)
        self.order = order  # порядок фильтра - число предыдущих точек
        self.mu = mu        # темп обучения
        self.eps = eps      # регуляризатор от деления на ноль

    def apply(self, data: np.ndarray | list) -> np.ndarray:
        series = np.asarray(data, dtype=float)
        n = len(series)
        filtered = np.copy(series)

        # Центрирование для исключения влияния постоянного смещения ряда
        mean_offset = float(np.mean(series[:min(10, n)]))
        centered = series - mean_offset

        weights = np.zeros(self.order, dtype=float)

        for t in range(self.order, n):
            x_hist = centered[t - self.order:t][::-1]
            y_pred = float(np.dot(weights, x_hist))
            err = centered[t] - y_pred

            # Шаг градиентного спуска с нормализацией по энергии входного вектора
            norm_sq = float(np.dot(x_hist, x_hist))
            weights += (self.mu / (norm_sq + self.eps)) * err * x_hist

            filtered[t] = y_pred + mean_offset

        return filtered