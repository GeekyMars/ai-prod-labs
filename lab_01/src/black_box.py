from abc import ABC, abstractmethod
from typing import List, Union

# Импортируем наш тип для аннотаций
from .constructive_number import ConstructiveNumber

# Создаем псевдоним типа для удобства 
# Наш вектор может состоять как из обычных чисел, так и из интервалов
Vector = List[Union[float, 'ConstructiveNumber']]

class BaseFunction(ABC):
    """
    Абстрактный базовый класс для целевых функций
    """
    def __init__(self):
        """
        Инициализирует счетчики вызовов целевой функции, 
        вектора градиента и матрицы Гессе
        """
        self.f_calls = 0
        self.grad_calls = 0
        self.hessian_calls = 0

    def reset_counters(self):
        """
        Обнуляет все счетчики вызовов. Необходимо вызывать перед 
        каждым новым запуском алгоритма оптимизации
        """
        self.f_calls = 0
        self.grad_calls = 0
        self.hessian_calls = 0

    @abstractmethod
    def __call__(self, x: Vector) -> Union[float, 'ConstructiveNumber']:
        """
        Вычисляет значение функции f(x) в заданной точке
        Использование __call__ позволяет вызывать объект как обычную функцию: obj(x)
        """
        pass

    @abstractmethod
    def gradient(self, x: Vector) -> Vector:
        """
        Вычисляет вектор градиента
        """
        pass

    @abstractmethod
    def hessian(self, x: Vector) -> List[Vector]:
        """
        Вычисляет матрицу Гессе
        Необходимо для метода Ньютона
        Возвращает двумерный массив
        """
        pass



class IdealQuadraticFunction(BaseFunction):
    """
    Идеальная 6-арная квадратичная функция. Число обусловленности равно 1!
    """
    def __init__(self):
        super().__init__()
        self.dim = 6

    def __call__(self, x: Vector) -> Union[float, 'ConstructiveNumber']:
        self.f_calls += 1  # Увеличиваем счетчик вызова функции
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
        
        # Считаем сумму квадратов вручную, не используя numpy, чтобы
        # python исопользовал наши методы __mul__ и __add__
        result = x[0] * x[0]
        for i in range(1, self.dim):
            result = result + (x[i] * x[i])
            
        return result / 2.0

    def gradient(self, x: Vector) -> Vector:
        self.grad_calls += 1 # Увеличиваем счетчик вызова градиента
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
        
        # Аналитическая первая производная от 0.5 * x^2 - это само x
        # Просто возвращаем копию входящего вектора.
        return [xi for xi in x]

    def hessian(self, x: Vector) -> List[Vector]:
        self.hessian_calls += 1 # Увеличиваем счетчик вызова гессиана
        # Матрица вторых производных - единичная матрица 6x6.
        hess = []
        for i in range(self.dim):
            row = []
            for j in range(self.dim):
                # На главной диагонали 1.0, остальные элементы 0.0
                row.append(1.0 if i == j else 0.0)
            hess.append(row)
        return hess


class BadConditionQuadraticFunction(BaseFunction):
    """
    Овражная 4-арная квадратичная функция
    Число обусловленности равно 100
    """
    def __init__(self):
        super().__init__()
        self.dim = 4
        # Задаем коэффициенты для диагонали матрицы Гессе
        # Максимальный / Минимальный = 100 / 1 = 100
        self.coefficients = [1.0, 33.0, 66.0, 100.0]

    def __call__(self, x: Vector) -> Union[float, 'ConstructiveNumber']:
        self.f_calls += 1  # Увеличиваем счетчик вызова функции
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
        
        # Вычисляем сумму (c_i * x_i^2) / 2
        result = self.coefficients[0] * x[0] * x[0]
        for i in range(1, self.dim):
            result = result + (self.coefficients[i] * x[i] * x[i])
            
        return result / 2.0

    def gradient(self, x: Vector) -> Vector:
        self.grad_calls += 1 # Увеличиваем счетчик вызова градиента
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
        
        # Первая производная от (c * x^2) / 2 равна c * x
        return [self.coefficients[i] * x[i] for i in range(self.dim)]

    def hessian(self, x: Vector) -> List[Vector]:
        self.hessian_calls += 1 # Увеличиваем счетчик вызова гессиана
        # Вторая производная - диагональная матрица с нашими коэффициентами
        hess = []
        for i in range(self.dim):
            row = []
            for j in range(self.dim):
                row.append(self.coefficients[i] if i == j else 0.0)
            hess.append(row)
        return hess


class RosenbrockFunction(BaseFunction):
    """
    3-арная функция Розенброка
    """
    def __init__(self):
        super().__init__()
        self.dim = 3

    def __call__(self, x: Vector) -> Union[float, 'ConstructiveNumber']:
        self.f_calls += 1  # Увеличиваем счетчик вызова функции
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
        
        # Вычисляем 100(x_1 - x_0^2)^2 + (1 - x_0)^2
        term1 = x[1] - (x[0] * x[0])
        part1 = (term1 * term1) * 100.0 + (1.0 - x[0]) * (1.0 - x[0])
        
        # Вычисляем 100(x_2 - x_1^2)^2 + (1 - x_1)^2
        term2 = x[2] - (x[1] * x[1])
        part2 = (term2 * term2) * 100.0 + (1.0 - x[1]) * (1.0 - x[1])
        
        return part1 + part2

    def gradient(self, x: Vector) -> Vector:
        self.grad_calls += 1 # Увеличиваем счетчик вызова градиента
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
        
        # Аналитические первые частные производные
        df_dx0 = (x[0] * x[0] - x[1]) * 400.0 * x[0] - (1.0 - x[0]) * 2.0
        df_dx1 = (x[1] - x[0] * x[0]) * 200.0 + (x[1] * x[1] - x[2]) * 400.0 * x[1] - (1.0 - x[1]) * 2.0
        df_dx2 = (x[2] - x[1] * x[1]) * 200.0
        
        return [df_dx0, df_dx1, df_dx2]

    def hessian(self, x: Vector) -> List[Vector]:
        self.hessian_calls += 1 # Увеличиваем счетчик вызова гессиана
        if len(x) != self.dim:
            raise ValueError(f"Вектор x должен содержать ровно {self.dim} координат!")
            
        # Аналитические вторые производные (матрица 3x3)
        h00 = 1200.0 * (x[0] * x[0]) - 400.0 * x[1] + 2.0
        h01 = -400.0 * x[0]
        h02 = 0.0
        
        h10 = -400.0 * x[0]
        h11 = 202.0 - 400.0 * x[2] + 1200.0 * (x[1] * x[1])
        h12 = -400.0 * x[1]
        
        h20 = 0.0
        h21 = -400.0 * x[1]
        h22 = 200.0
        
        return [
            [h00, h01, h02],
            [h10, h11, h12],
            [h20, h21, h22]
        ]