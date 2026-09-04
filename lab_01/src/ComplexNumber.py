class ComplexNumber:
    def __init__(self, real, imaginary):
        self.real = real
        self.imaginary = imaginary
    
    def __add__(self, right):
        if isinstance(right, ComplexNumber):
            return ComplexNumber(self.real + right.real, self.imaginary + right.imaginary)
        raise Exception(f"Unexpected data type is {type(right)}")
        
    def __mul__(self, right):
        if isinstance(right, ComplexNumber):
            return ComplexNumber(self.real * right.real - self.imaginary * right.imaginary, 
                                 self.real * right.imaginary + self.imaginary * right.real)
        raise Exception(f"Unexpected data type is {type(right)}")
    
    def __eq__(self, right):
        if isinstance(right, ComplexNumber):
            return self.real == right.real and self.imaginary == right.imaginary
        raise Exception(f"Unexpected data type is {type(right)}")

    def __str__(self):
        return f"{self.real} + i * {self.imaginary}"
    
    def __repr__(self):
        return self.__str__()