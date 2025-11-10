# Taken from the Averill Law book, section 1.5

import simpy
import numpy as np
import matplotlib.pyplot as plt

RNG_SEED = 2129
DEMAND_MEAN_INTERARRIVAL = 0.1
INITIAL_INVENTORY = 100
POLICY_FLOOR = 20
POLICY_CEIL = 80

TRACE = True


def trace(content: str):
    if TRACE:
        print(content)


rng = np.random.default_rng(seed=RNG_SEED)


def demand(env: simpy.Environment, inventory: simpy.Container, inventory_levels: list[tuple[float, float]]):
    while True:
        interarrival_time = rng.exponential(DEMAND_MEAN_INTERARRIVAL)
        demand_size = rng.choice([1, 2, 3, 4], p=[1/6, 1/3, 1/3, 1/6])

        yield env.timeout(interarrival_time)
        yield inventory.get(demand_size)
        inventory_levels.append((env.now, inventory.level))
        trace(
            f'Time {env.now:.2f}: demand of {demand_size}. Current inventory: {inventory.level}')


def order(env: simpy.Environment, inventory: simpy.Container, order_amount: float, inventory_levels: list[tuple[float, float]], total_cost: float, accumulated_costs: list[tuple[float, float]]):
    lead_time = 0.5 + rng.random() * 0.5
    trace(
        f'Order of {order_amount} at time {env.now:.2f} with lead time of {lead_time:.2f}')
    order_cost = 32 + 3 * order_amount
    total_cost += order_cost
    accumulated_costs.append((env.now, total_cost))
    trace(f'Order cost: {order_cost:.2f}')
    yield env.timeout(lead_time)
    inventory.put(order_amount)
    inventory_levels.append((env.now, inventory.level))
    trace(f'Order of {order_amount} delivered at time {env.now:2f}')


def inventory_review(env: simpy.Environment, inventory: simpy.Container, floor: int, ceil: int, inventory_levels: list[tuple[float, float]], total_cost: float, accumulated_costs: list[tuple[float, float]]):
    while True:
        yield env.timeout(1)
        if inventory.level < floor:
            env.process(order(env, inventory, ceil -
                        inventory.level, inventory_levels, total_cost, accumulated_costs))


def main():
    env = simpy.Environment()
    inventory = simpy.Container(env, init=INITIAL_INVENTORY)
    inventory_levels = []
    accumulated_costs = []
    total_cost = 0
    inventory_levels.append((0, INITIAL_INVENTORY))
    accumulated_costs.append((0, 0))

    env.process(demand(env, inventory, inventory_levels))
    env.process(inventory_review(env, inventory,
                POLICY_FLOOR, POLICY_CEIL, inventory_levels, total_cost, accumulated_costs))

    env.run(15)

    inv_times, inv_levels = zip(*inventory_levels)
    cost_times, cost_levels = zip(*accumulated_costs)

    plt.step(x=inv_times, y=inv_levels, where='post', label='Inventory level')
    plt.step(x=cost_times, y=cost_levels,
             where='post', label='Cost', linestyle='--')
    plt.title('Inventory level over time')
    plt.xlabel('Time')
    plt.ylabel('Inventory level')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
