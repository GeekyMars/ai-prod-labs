from typing import Union

class ConstructiveNumber:

    def __init__(self, x_or_a: float, eps_or_b: float, is_interval: bool = False):
        """
        Инициализация конструктивного числа
        :param is_interval: если True, воспринимаем вход как (a, b)
                            если False, воспринимаем как (x, epsilon)
        """
        if is_interval:
            self.a = float(x_or_a)
            self.b = float(eps_or_b)
        else:
            self.a = float(x_or_a - eps_or_b)
            self.b = float(x_or_a + eps_or_b)
            
        if self.a > self.b:
            self.a, self.b = self.b, self.a


    def get_real(self, alpha: float) -> float:
        """
        Возвращает действительное число на основе параметра alpha в диапазоне [0, 1]
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Параметр alpha должен быть в диапазоне [0, 1]")
        return self.a * (1.0 - alpha) + self.b * alpha
    

    def __add__(self, other: Union['ConstructiveNumber', float, int]) -> 'ConstructiveNumber':
        """
        Нативное сложение: [a, b] + [c, d] = [a + c, b + d] или [a, b] + k = [a + k, b + k]
        """
        if isinstance(other, (float, int)):
            return ConstructiveNumber(self.a + other, self.b + other, is_interval=True)
        
        elif isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a + other.a, self.b + other.b, is_interval=True)
        
        return NotImplemented

    def __radd__(self, other):
        """
        Обратное сложение: k + [a, b] = [a + k, b + k]
        """
        return self.__add__(other)


    def __repr__(self):
        return f"ConstructiveNumber(a={self.a:.6f}, b={self.b:.6f})"


    def __sub__(self, other: Union['ConstructiveNumber', float, int]) -> 'ConstructiveNumber':
        """
        Нативное вычитание: [a, b] - [c, d] = [a - d, b - c] или [a, b] - k = [a - k, b - k]
        """
        if isinstance(other, (float, int)):
            return ConstructiveNumber(self.a - other, self.b - other, is_interval=True)
        
        elif isinstance(other, ConstructiveNumber):
            return ConstructiveNumber(self.a - other.b, self.b - other.a, is_interval=True)
        
        return NotImplemented

    def __rsub__(self, other: Union[float, int]) -> 'ConstructiveNumber':
        """
        Обратное вычитание: k - [a, b] = [k - b, k - a]
        """
        if isinstance(other, (float, int)):
            return ConstructiveNumber(other - self.b, other - self.a, is_interval=True)
        
        return NotImplemented


    def __mul__(self, other: Union['ConstructiveNumber', float, int]) -> 'ConstructiveNumber':
        """
        Нативное умножение: вычисляет все 4 комбинации произведений для точного поиска границ
        """
        if isinstance(other, (float, int)):
            # Если умножаем на константу, достаточно умножить границы и отсортировать
            p1, p2 = self.a * other, self.b * other
            return ConstructiveNumber(min(p1, p2), max(p1, p2), is_interval=True)
            
        elif isinstance(other, ConstructiveNumber):
            # Умножение интервала на интервал
            p1 = self.a * other.a
            p2 = self.a * other.b
            p3 = self.b * other.a
            p4 = self.b * other.b
            return ConstructiveNumber(min(p1, p2, p3, p4), max(p1, p2, p3, p4), is_interval=True)
            
        return NotImplemented
    
    def __rmul__(self, other):
        """
        Обратное умножение: k * [a, b] = [a * k, b * k]
        """
        return self.__mul__(other)

    
    def __truediv__(self, other: Union['ConstructiveNumber', float, int]) -> 'ConstructiveNumber':
        """
        Нативное деление. Эквивалентно умножению на обратный интервал [1/d, 1/c]
        """
        if isinstance(other, (float, int)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль!!!")
            p1, p2 = self.a / other, self.b / other
            return ConstructiveNumber(min(p1, p2), max(p1, p2), is_interval=True)
            
        elif isinstance(other, ConstructiveNumber):
            if other.a <= 0 <= other.b:
                raise ZeroDivisionError("Интервал-делитель содержит ноль! Деление невозможно!!!")
            # Деление на [c, d] = умножение на [1/d, 1/c]
            inverse_other = ConstructiveNumber(1.0 / other.b, 1.0 / other.a, is_interval=True)
            return self * inverse_other  # Переиспользуем написанный метод __mul__
           
        return NotImplemented

    def __rtruediv__(self, other: Union[float, int]) -> 'ConstructiveNumber':
        """
        Обратное деление: k / [a, b] = k * [1/b, 1/a]
        """
        if isinstance(other, (float, int)):
            if self.a <= 0 <= self.b:
                raise ZeroDivisionError("Интервал-делитель содержит ноль! Деление невозможно!!!")
            
            inverse_self = ConstructiveNumber(1.0 / self.b, 1.0 / self.a, is_interval=True)
            return other * inverse_self
        
        return NotImplemented


    def get_center(self) -> float:
        """ Возвращает центр интервала """
        return self.get_real(0.5)


    def __lt__(self, other: Union['ConstructiveNumber', float, int]) -> bool:
        """ Оператор < (less than). Сравниваем по центрам интервалов """
        if isinstance(other, (float, int)):
            return self.get_center() < other
        
        elif isinstance(other, ConstructiveNumber):
            return self.get_center() < other.get_center()
        return NotImplemented


    def __eq__(self, other: object) -> bool:
        """ Оператор == (equal). """
        if isinstance(other, (float, int)):
            return self.a == other and self.b == other
        elif isinstance(other, ConstructiveNumber):
            return self.a == other.a and self.b == other.b
        return False




