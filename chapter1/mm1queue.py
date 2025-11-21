import simpy
import numpy as np
import matplotlib.pyplot as plt

from utilities import integrate_trace


class ExperimentResults:
    """
    Container class for experiment metrics.
    """

    def __init__(self) -> None:
        self.metrics: dict[str, int | float] = {}
        self.lists: dict[str, list[int | float]] = {}
        self.traces: dict[str, list[tuple[float, int | float]]] = {}

        self.metrics["max_queue_length"] = 0

        self.lists["arrivals"] = []
        self.lists["delays"] = []
        self.lists["total_times"] = []

        self.traces["number_in_system"] = [(0, 0)]


class Experiment:
    """
    A simulation run with a given RNG.

    Either an RNG object can be provided, or a seed to feed into Numpy's default_rng.
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        self.results = ExperimentResults()

        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.server = simpy.Resource(self.env, 1)

        self.number_in_system = 0

    def run(self, until: float | None = None):
        print("Running simulation...")

        self.env.process(self.arrival_process())
        self.env.run(until)

        # Log final states of relevant traces
        final_number_in_system = self.results.traces["number_in_system"][-1][1]
        self.results.traces["number_in_system"].append(
            (self.env.now, final_number_in_system)
        )

        print("Simulation complete!")

    def get_results(self, plot_trace: bool = True):
        total_number_in_system = integrate_trace(
            self.results.traces["number_in_system"]
        )

        total_customers = len(self.results.lists["arrivals"])
        highly_delayed_customers = len(
            [x for x in self.results.lists["delays"] if x > 1]
        )

        print("\nResults of simulation")
        print("----------------------\n")
        print(
            f"Time-average number of customers in system: {total_number_in_system / self.env.now:.3f}"
        )
        print(
            f"Average total time in system: {np.array(self.results.lists['total_times']).mean():.3f}"
        )
        print(f"Maximum queue length: {self.results.metrics['max_queue_length']}")
        print(
            f"Maximum delay in queue: {np.array(self.results.lists['delays']).max():.3f}"
        )
        print(
            f"Maximum time in system: {np.array(self.results.lists['total_times']).max():.3f}"
        )
        print(
            f"Proportion delayed over 1 minute: {highly_delayed_customers / total_customers:.2%}"
        )

        if plot_trace:
            times, lengths = zip(*self.results.traces["number_in_system"])
            plt.step(times, lengths, where="post", lw=0.7)
            plt.xlabel("Time")
            plt.ylabel("Customers")
            plt.title("Number of customers in system")
            plt.show()

    def generate_interarrival_time(self) -> float:
        return self.rng.exponential(1 / self.arrival_rate)

    def generate_service_time(self) -> float:
        return self.rng.exponential(1 / self.service_rate)

    def arrival_process(self):
        while True:
            yield self.env.timeout(self.generate_interarrival_time())
            self.number_in_system += 1

            self.results.lists["arrivals"].append(self.env.now)
            self.results.traces["number_in_system"].append(
                (self.env.now, self.number_in_system)
            )
            self.env.process(self.customer())

    def customer(self):
        arrival_time = self.env.now
        # print(f"Customer arrived at time {arrival_time:.2f}")
        with self.server.request() as req:
            if len(self.server.queue) > self.results.metrics["max_queue_length"]:
                self.results.metrics["max_queue_length"] = len(self.server.queue)

            yield req
            service_start_time = self.env.now
            # print(f"Customer being served at {service_start_time:.2f}")
            yield self.env.timeout(self.generate_service_time())
            service_end_time = self.env.now
            # print(f"Customer leaving at {service_end_time:.2f}")
            self.number_in_system -= 1

            # Log results
            delay = service_start_time - arrival_time
            total_time = service_end_time - arrival_time

            self.results.lists["delays"].append(delay)
            self.results.lists["total_times"].append(total_time)
            self.results.traces["number_in_system"].append(
                (self.env.now, self.number_in_system)
            )


def main():
    exp = Experiment(1, 1.5, 1849)
    exp.run(1000)
    exp.get_results()


if __name__ == "__main__":
    main()
