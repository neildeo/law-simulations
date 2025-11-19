import simpy


def car(env: simpy.Environment):
    while True:
        print(f'Car parks at {env.now}')
        yield env.timeout(5)
        print(f'Car drives at {env.now}')
        yield env.timeout(2)


def main():
    env = simpy.Environment()
    env.process(car(env))
    env.run(until=15)


if __name__ == "__main__":
    main()
