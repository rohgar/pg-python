class MyClass:
    def __init__(self):
        try:
            self.var1 = self.initialize_var1()
            self.var2 = self.initialize_var2()
            self.var3 = self.initialize_var3()  # This might fail
        except Exception as e:
            print(f"Initialization error: {e}")
            # Optionally set default values for failed attributes
            self.var3 = None
            raise e  # Re-raise the exception if desired

    def initialize_var1(self):
        return "Value 1"

    def initialize_var2(self):
        return "Value 2"

    def initialize_var3(self):
        raise ValueError("Failed to initialize var3")  # Simulate an error


class CallingClass:
    def __init__(self):
        try:
            self.obj = MyClass()
        except Exception as e:
            print(f"Exception caught: {e}")
            # Access partially initialized variables
            if hasattr(self, 'obj'):
                print(f"Accessible variables: {getattr(self.obj, 'var1', None)}, {getattr(self.obj, 'var2', None)}")

# Example usage
calling_instance = CallingClass()