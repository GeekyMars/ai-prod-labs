from abc import ABC, abstractmethod
from typing import List, Tuple

# Импортируем базовую функцию и тип вектора из нашего второго модуля
from black_box import BaseFunction, Vector

from constructive_number import ConstructiveNumber


def to_real_value(value: float | ConstructiveNumber) -> float:
    """
    Возвращает числовой центр обычного или конструктивного числа
    """
    if isinstance(value, ConstructiveNumber):
        return value.get_center()
    return float(value)


class BaseOptimizer(ABC):
    """
    Абстрактный базовый класс для алгоритмов оптимизации
    Обеспечивает единый интерфейс и встроенную систему запоминания траектории
    """
    def __init__(self):
        # Хранилище истории для построения графиков
        self.history_x: List[Vector] = []
        self.history_f: List[float] = []

    def clear_history(self) -> None:
        """
        Очищает историю перед новым запуском
        """
        self.history_x.clear()
        self.history_f.clear()

    def _save_step(self, x: Vector, f_val) -> None:
        """
        Сохраняет текущую точку и значение функции
        Вектор x копируется, чтобы избежать перезаписи ссылок в памяти
        """
        # Копируем вектор. Если внутри ConstructiveNumber, они тоже сохранятся.
        self.history_x.append([xi for xi in x])
        
        # Если f_val - конструктивное число, для графиков нам нужен его центр.
        if isinstance(f_val, ConstructiveNumber):
            self.history_f.append(f_val.get_center())
        else:
            self.history_f.append(float(f_val))

    @abstractmethod
    def optimize(self, func: BaseFunction, x0: Vector, *args, **kwargs) -> Vector:
        """
        Основной метод оптимизации.
        :param func: Целевая функция, наследник BaseFunction
        :param x0: Стартовая точка - вектор
        :return: Найденная точка минимума
        """
        pass


class GradientDescentOptimizer(BaseOptimizer):
    """
    Классический градиентный спуск
    """
    def optimize(self, func: BaseFunction, x0: Vector, learning_rate: float = 0.01, max_iter: int = 1000, tol: float = 1e-6) -> Vector:
        """
        Запуск оптимизации.
        func: Целевая функция
        x0: Начальная точка
        learning_rate: Шаг обучения (alpha)
        max_iter: Максимальное количество шагов
        tol: Толерантность - критерий остановки по норме градиента
        """
        self.clear_history()
        
        # Делаем копию стартового вектора, чтобы не перезаписать оригинал в памяти
        x = [xi for xi in x0]

        for i in range(max_iter):
            # Вычисляем текущее значение функции и сохраняем шаг в историю
            f_val = func(x)
            self._save_step(x, f_val)

            # Вычисляем вектор градиента в текущей точке
            grad = func.gradient(x)

            # Критерий остановки: проверяем, не достигли ли мы минимума
            # Для проверки длины вектора градиента используем только скалярные центры чисел
            grad_real = [g.get_center() if isinstance(g, ConstructiveNumber) else float(g) for g in grad]
                   
            grad_norm = sum(g * g for g in grad_real) ** 0.5
            
            # Если градиент близок к нулю, значит мы в точке минимума
            if grad_norm < tol:
                print(f"Градиентный спуск успешно сошелся за {i} итераций (norm: {grad_norm:.6f})")
                break

            # Делаем шаг: x_{new} = x_{old} - learning_rate * grad
            for j in range(len(x)):
                x[j] = x[j] - (grad[j] * learning_rate)

        else:
            # Сработает, если цикл завершился по max_iter, а не по break
            print(f"Внимание: Градиентный спуск остановлен по лимиту итераций ({max_iter})")

        # Сохраняем финальную точку
        self._save_step(x, func(x))
        return x


class NelderMeadOptimizer(BaseOptimizer):
    """
    Метод Нелдера-Мида / деформируемого многогранника
    """
    def optimize(self, func: BaseFunction, x0: Vector, step: float = 0.5, max_iter: int = 1000, tol: float = 1e-6) -> Vector:
        self.clear_history()
        if max_iter <= 0:
            raise ValueError("max_iter должен быть положительным")

        n = len(x0)

        # Классические гиперпараметры деформации симплекса
        alpha = 1.0   # Коэффициент отражения
        gamma = 2.0   # Коэффициент растяжения
        rho = 0.5     # Коэффициент сжатия
        sigma = 0.5   # Коэффициент глобального сжатия (shrink)

        # Инициализация стартового симплекса: создаем N + 1 вершин
        simplex = []
        simplex.append([xi for xi in x0])  # Первая вершина - стартовая точка
        
        for i in range(n):
            point = [xi for xi in x0]
            # Сдвигаем одну из координат на величину step
            point[i] = point[i] + step
            simplex.append(point)

        evaluated = []
        for it in range(max_iter):
            # Оценка функции во всех вершинах
            evaluated = []
            for pt in simplex:
                f_val = func(pt)
                # Извлекаем скалярный центр для безопасной сортировки и ветвлений
                f_center = to_real_value(f_val)
                evaluated.append((f_center, f_val, pt))

            # Сортируем вершины по значению функции от лучшей к худшей
            evaluated.sort(key = lambda item: item[0])

            # Распаковываем нужные нам точки
            best_f_center, best_f, best_pt = evaluated[0]
            good_f_center, _, _ = evaluated[-2]  # Предпоследняя - хорошая
            worst_f_center, _, worst_pt = evaluated[-1] # Последняя - худшая

            # Сохраняем текущую лучшую точку в историю для графиков
            self._save_step(best_pt, best_f)

            # Критерий остановки - разница между худшей и лучшей точками
            if abs(worst_f_center - best_f_center) < tol:
                print(f"Нелдер-Мид успешно сошелся за {it} итераций (diff: {abs(worst_f_center - best_f_center):.8f})")
                break

            # Вычисление центра тяжести не учитываем худшую точку
            centroid = []
            for i in range(n):
                c_i = evaluated[0][2][i]
                for j in range(1, n):
                    c_i = c_i + evaluated[j][2][i]
                centroid.append(c_i / n)

            # ОПЕРАЦИИ ДЕФОРМАЦИИ
            
            # Отражение
            xr = [centroid[i] + (centroid[i] - worst_pt[i]) * alpha for i in range(n)]
            fr_val = func(xr)
            fr_center = to_real_value(fr_val)

            if best_f_center <= fr_center < good_f_center:
                # Отраженная точка лучше худшей, но не лучше лучшей, поэтому просто меняем худшую на отраженную
                simplex = [pt for _, _, pt in evaluated[:-1]] + [xr]
                continue

            # Растяжение, если мы нашли отличный склон
            if fr_center < best_f_center:
                xe = [centroid[i] + (xr[i] - centroid[i]) * gamma for i in range(n)]
                fe_val = func(xe)
                fe_center = to_real_value(fe_val)
                
                if fe_center < fr_center:
                    simplex = [pt for _, _, pt in evaluated[:-1]] + [xe]
                else:
                    simplex = [pt for _, _, pt in evaluated[:-1]] + [xr]
                continue

            # Сжатие, если отражение оказалось плохим
            xc = [centroid[i] + (worst_pt[i] - centroid[i]) * rho for i in range(n)]
            fc_val = func(xc)
            fc_center = to_real_value(fc_val)

            if fc_center < worst_f_center:
                # Сжатая точка лучше худшей - принимаем
                simplex = [pt for _, _, pt in evaluated[:-1]] + [xc]
                continue

            # Глобальное сжатие если любое другое действие ведет к ухудшению, стягиваем все точки к лучшей
            new_simplex = [best_pt]
            for i in range(1, len(simplex)):
                pt = evaluated[i][2]
                shrunk_pt = [best_pt[j] + (pt[j] - best_pt[j]) * sigma for j in range(n)]
                new_simplex.append(shrunk_pt)
            simplex = new_simplex

        else:
            print(f"Внимание: Нелдер-Мид остановлен по лимиту итераций ({max_iter})")

        # Сохраняем финальный результат
        final_pt = evaluated[0][2]
        final_f = evaluated[0][1]
        self._save_step(final_pt, final_f)
        
        return final_pt