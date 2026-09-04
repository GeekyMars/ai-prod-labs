import src.ComplexNumber as CN

class Function1:
    def func(x: CN.ComplexNumber):
        return x * CN.ComplexNumber(1, 2) + CN.ComplexNumber(2, 1)
    
    def name():
        return "x * (1 + 2 * i) + 2 + i"

class Function2:
    def func(x: CN.ComplexNumber):
        return x * CN.ComplexNumber(2, 0) + CN.ComplexNumber(1, 1)
    
    def name():
        return "x * 2 + 1 + i"  