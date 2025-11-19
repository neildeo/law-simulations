import simpy
import numpy as np
import matplotlib.pyplot as plt

SERVICE_TIME = 8
MEAN_INTERARRIVAL_TIME = 3


def customer(
    env: simpy.Environment,
    name: str, counter: simpy.Resource,
    queue_lengths: list[tuple[float, int]],
    wait_times: list[float],
    active_servers: list[tuple[float, int]]
):
    arrival_time = env.now
    print(f'{name} arrives at time {arrival_time:.2f}')
    with counter.request() as req:
        queue_lengths.append((env.now, len(counter.queue)))
        active_servers.append((env.now, counter.count))
        yield req
        wait_times.append(env.now - arrival_time)
        print(f'{name} is being served at time {env.now:.2f}')
        yield env.timeout(SERVICE_TIME)
        print(f'{name} leaves at time {env.now:.2f}')


def customer_generator(
    env: simpy.Environment,
    counter: simpy.Resource,
    mean_interarrival_time: float,
    queue_lengths: list[tuple[float, int]],
    wait_times: list[float],
    active_servers: list[tuple[float, int]]
):
    for i in range(100):
        interarrival_time = np.random.exponential(mean_interarrival_time)
        yield env.timeout(interarrival_time)
        env.process(
            customer(env, f'Customer {i+1}', counter, queue_lengths, wait_times, active_servers))


def main():
    env = simpy.Environment()
    counter = simpy.Resource(env, capacity=3)
    queue_lengths = []
    wait_times = []
    active_servers = []

    env.process(customer_generator(
        env, counter, MEAN_INTERARRIVAL_TIME, queue_lengths, wait_times, active_servers))

    env.run(until=10000)

    times, queue_sizes = zip(*queue_lengths)
    times_servers, server_counts = zip(*active_servers)

    plt.plot(times, queue_sizes, label='Queue size')
    plt.plot(times_servers, server_counts,
             linestyle='--', label='Active servers')
    plt.title('Queue size and resource utilisation over time')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
