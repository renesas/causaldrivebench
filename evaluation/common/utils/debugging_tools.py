import inspect

def inspect_object(obj):
    obj_type = type(obj)
    separator = "=" * 50

    print(f"\n{separator}")
    print(f"  Object Inspector: {obj_type.__name__}")
    print(f"{separator}")

    # --- Class Info ---
    print(f"\n  {'Type':<20}: {obj_type}")
    print(f"  {'Module':<20}: {inspect.getmodule(obj_type)}")
    if inspect.getdoc(obj):
        print(f"  {'Docstring':<20}: {inspect.getdoc(obj)}")

    # --- Member Variables ---
    print(f"\n--- Member Variables {'-' * 30}")
    member_vars = {k: v for k, v in inspect.getmembers(obj)
                   if not callable(v) and not k.startswith("__")}

    if member_vars:
        for name, value in member_vars.items():
            print(f"  {name:<30} = {repr(value)}")
    else:
        print("  No member variables found.")

    # --- Member Functions ---
    print(f"\n--- Member Functions {'-' * 30}")
    member_funcs = {k: v for k, v in inspect.getmembers(obj)
                    if callable(v) and not k.startswith("__")}

    if member_funcs:
        for name, func in member_funcs.items():
            try:
                sig = inspect.signature(func)
                params = []
                for param_name, param in sig.parameters.items():
                    if param.default is inspect.Parameter.empty:
                        params.append(param_name)
                    else:
                        params.append(f"{param_name}={repr(param.default)}")
                print(f"  {name}({', '.join(params)})")

                # Print docstring if available
                if inspect.getdoc(func):
                    print(f"    >> {inspect.getdoc(func)}")
            except (ValueError, TypeError):
                print(f"  {name}(...) [signature unavailable]")
    else:
        print("  No member functions found.")

    # --- Dunder/Special Methods ---
    print(f"\n--- Special Methods {'-' * 31}")
    special_methods = [k for k, v in inspect.getmembers(obj)
                       if callable(v) and k.startswith("__")]
    if special_methods:
        print(f"  {', '.join(special_methods)}")
    else:
        print("  No special methods found.")

    print(f"\n{separator}\n")


# # --- Example Usage ---
# class Car:
#     """A simple Car class."""

#     def __init__(self, make, model, year):
#         self.make = make
#         self.model = model
#         self.year = year
#         self._mileage = 0

#     def drive(self, distance, speed=60):
#         """Drive the car a given distance at a given speed."""
#         self._mileage += distance

#     def get_mileage(self):
#         """Return the current mileage."""
#         return self._mileage


# my_car = Car("Toyota", "Corolla", 2022)
# inspect_object(my_car)