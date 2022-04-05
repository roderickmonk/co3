# Import math for the infinity functionality
from copy import copy
import copyreg
from itertools import tee, islice
import math
from time import perf_counter
from bson.objectid import ObjectId
from devtools import debug
import random

# The Sieve of Eratosthenes method of calculating the primes less than the limit
def _nearest_prime(limit: int) -> int:
    # The list of prime numbers
    primes: list[int] = []
    # The boolean list of whether a number is prime
    numbers = [True] * limit
    # Loop all of the numbers in numbers starting from 2
    for i in range(2, limit):
        # If the number is prime
        if numbers[i]:
            # Add it onto the list of prime numbers
            primes.append(i)
            # Loop over all of the other factors in the list
            for n in range(i**2, limit, i):
                # Make them not prime
                numbers[n] = False

    print(f"Nearest prime {primes[-1]}")

    return primes[-1]

start_time = perf_counter()
document_count = 1000000
documents = [
    {"_id": str(ObjectId()), "n#": i, "buy": [], "sell": []}
    for i in range(document_count)
]

seed = 0
episodes = 10
episode_length = 1000
total_episodes = document_count // episode_length

random.seed(seed)
selected_episodes = [
    episode_length * e for e in random.sample(range(total_episodes), episodes)
]

test_orderbooks = {}
for s in selected_episodes:
    print(f"{s=}")

    for x in documents[s : s + episode_length]:
        print(f"{x=}")
        test_orderbooks |= {x["_id"]: x}

# debug(test_orderbooks)
debug(len(test_orderbooks), total_episodes, selected_episodes)

key = -1

debug(
    list(test_orderbooks)[key],
    test_orderbooks[list(test_orderbooks)[key]],
    len(test_orderbooks),
    perf_counter() - start_time,
)
