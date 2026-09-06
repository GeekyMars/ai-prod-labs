import math
import random
from typing import Sequence

# Импортируем типы и базовые классы из наших модулей
from .black_box import BaseFunction, Vector
from .constructive_number import ConstructiveNumber
from .optimizers import BaseOptimizer, to_real_value


class SimulatedAnnealingOptimizer(BaseOptimizer):
    """
    Имитация отжига (Simulated Annealing - SA).
    Алгоритм локального поиска, который с некоторой вероятностью принимает 
    ухудшающие шаги, чтобы выбраться из локальных минимумов.
    """

    def optimize(
        self,
        func: BaseFunction,
        x0: Sequence[float | ConstructiveNumber],
        initial_temp: float = 100.0,
        cooling_rate: float = 0.95,
        min_temp: float = 1e-5,
        max_iter: int = 1000,
        step_size: float = 1.0,
    ) -> Vector:
        
        self.clear_history()

        # Инициализация стартовой точки
        current_x = [to_real_value(xi) for xi in x0]
        current_f = func(current_x)
        current_f_val = to_real_value(current_f)

        # Отслеживаем глобально лучший найденный результат
        best_x = current_x[:]
        best_f_val = current_f_val

        self._save_step(best_x, current_f)

        T = initial_temp

        for i in range(max_iter):
            if T < min_temp:
                print(f"Отжиг успешно сошелся: достигнута минимальная температура ({min_temp}) на {i} итерации.")
                break

            # Генерация соседней точки случайным сдвигом
            new_x = [xi + random.uniform(-1, 1) * step_size for xi in current_x]

            new_f = func(new_x)
            new_f_val = to_real_value(new_f)

            # Разница между новым и текущим значением
            delta_f = new_f_val - current_f_val

            # Условие принятия новой точки
            # Если новая точка лучше (delta_f < 0) - принимаем всегда
            # Если хуже - принимаем с вероятностью, зависящей от температуры
            if delta_f < 0 or random.random() < math.exp(-delta_f / T):
                current_x = new_x
                current_f = new_f
                current_f_val = new_f_val

                # Обновляем глобальный оптимум
                if current_f_val < best_f_val:
                    best_x = current_x[:]
                    best_f_val = current_f_val

            # Сохраняем в историю именно лучший найденный вектор для красивых графиков
            self._save_step(best_x, func(best_x))

            # Понижаем температуру
            T *= cooling_rate
        else:
            print(f"Внимание: Имитация отжига остановлена по лимиту итераций ({max_iter}).")

        return best_x


class PSOOptimizer(BaseOptimizer):
    """
    Метод роя частиц (Particle Swarm Optimization - PSO).
    Частицы перемещаются в пространстве поиска, 
    учитывая свой лучший опыт и лучший опыт всего роя.
    """

    def optimize(
        self,
        func: BaseFunction,
        x0: Sequence[float | ConstructiveNumber],
        num_particles: int = 30,
        max_iter: int = 200,
        w: float = 0.5,   # Коэффициент инерции - насколько частица сохраняет направление
        c1: float = 1.5,  # Когнитивный коэффициент - тяга к собственной лучшей точке
        c2: float = 1.5,  # Социальный коэффициент - тяга к лучшей точке всего роя
    ) -> Vector:
        
        self.clear_history()

        dim = len(x0)
        base_x = [to_real_value(xi) for xi in x0]

        # Инициализация структур данных роя
        particles_x = []
        particles_v = []
        particles_pbest_x = []
        particles_pbest_f = []

        gbest_x = base_x[:]
        gbest_f_val = float('inf')

        # Создаем частицы, разбрасывая их случайным образом вокруг стартовой точки
        for _ in range(num_particles):
            px = [bx + random.uniform(-5, 5) for bx in base_x]
            pv = [random.uniform(-1, 1) for _ in range(dim)]

            pf = func(px)
            pf_val = to_real_value(pf)

            particles_x.append(px)
            particles_v.append(pv)
            particles_pbest_x.append(px[:])
            particles_pbest_f.append(pf_val)

            # Проверяем, не стала ли эта частица глобальным лидером
            if pf_val < gbest_f_val:
                gbest_f_val = pf_val
                gbest_x = px[:]

        self._save_step(gbest_x, func(gbest_x))

        # Основной цикл полета роя
        for it in range(max_iter):
            for i in range(num_particles):
                # Обновление скорости и позиции каждой частицы
                for j in range(dim):
                    r1, r2 = random.random(), random.random()
                    # Формула изменения вектора скорости
                    particles_v[i][j] = (
                        w * particles_v[i][j] +
                        c1 * r1 * (particles_pbest_x[i][j] - particles_x[i][j]) +
                        c2 * r2 * (gbest_x[j] - particles_x[i][j])
                    )
                    # Сдвиг частицы
                    particles_x[i][j] += particles_v[i][j]

                # Оценка новой позиции черным ящиком
                current_f = func(particles_x[i])
                current_f_val = to_real_value(current_f)

                # Обновление персонального рекорда
                if current_f_val < particles_pbest_f[i]:
                    particles_pbest_f[i] = current_f_val
                    particles_pbest_x[i] = particles_x[i][:]

                    # Обновление рекорда всего роя
                    if current_f_val < gbest_f_val:
                        gbest_f_val = current_f_val
                        gbest_x = particles_x[i][:]

            # В историю записываем положение глобального лидера
            self._save_step(gbest_x, func(gbest_x))

        print(f"Рой частиц завершил работу. Выполнено {max_iter} итераций.")
        return gbest_x