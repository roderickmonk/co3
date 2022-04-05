#!/usr/bin/env python

arr_a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr_b = [x * -10 for x in arr_a]


if __name__ == "__main__":

    iterator_a = iter(arr_a)
    iterator_b = iter(arr_b)

    def func_a():

        try:
            while True:
                try:
                    yield next(iterator_a)
                except StopIteration:
                    raise StopIteration
        except StopIteration:
            raise StopIteration

    def func_b():

        try:
            while True:
                try:
                    yield next(iterator_b)
                except StopIteration:
                    raise StopIteration
        except StopIteration:
            raise StopIteration

    while True:

        it_func_a = func_a()
        it_func_b = func_b()

        try:
            print(next(it_func_a))
            print(next(it_func_b))

        except RuntimeError:
            print("StopIteration")
            break
