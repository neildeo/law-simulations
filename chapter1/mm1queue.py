import simpy
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from utilities import integrate_trace


class Metric:
    """
    A scalar value tracked in a simulation
    """

    def __init__(
        self, name: str, initial_value: Any, display_name: str, format_spec: str
    ) -> None:
        self.name = name
        self.value = initial_value
        self.display_name = display_name
        self.format_spec = format_spec

    def assign(self, value: Any) -> None:
        """
        (Re-)assign the value of the metric
        """
        self.value = value

    def increment(self, amount: int | float) -> None:
        """
        Increment the value of the metric by the specified amount
        """
        self.value += amount


class Experiment:
    """
    A simulation run with a given RNG.

    Either an RNG object can be provided, or a seed to feed into Numpy's default_rng.

    Experiment keeps track of its own results and metrics.
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        # RNG and simulation environment
        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)
        self.env = simpy.Environment()

        # Simulation parameters
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.server = simpy.Resource(self.env, 1)

        self.number_in_system = 0

        # Results tracking
        self.metrics: dict[str, Metric] = {}
        self.lists: dict[str, list[int | float]] = {}
        self.traces: dict[str, list[tuple[float, int | float]]] = {}

        self.metrics["total_customers"] = Metric(
            "total_customers", 0, "Total customers seen", "d"
        )
        self.metrics["time_avg_number_in_system"] = Metric(
            "time_avg_number_in_system", 0, "Time-average number in system", ".3f"
        )
        self.metrics["max_queue_length"] = Metric(
            "max_queue_length", 0, "Max queue length", "d"
        )
        self.metrics["highly_delayed_customers"] = Metric(
            "highly_delayed_customers", 0, "Number of customers delayed >1min", "d"
        )
        self.metrics["prop_delayed_over_1_min"] = Metric(
            "prop_delayed_over_1_min",
            0,
            "Proportion of customers delayed >1min",
            ".1%",
        )
        self.metrics["avg_total_time_in_system"] = Metric(
            "avg_total_time_in_system", 0, "Avg total time in system", ".3f"
        )
        self.metrics["max_delay"] = Metric("max_delay", 0, "Maximum delay", ".3f")
        self.metrics["max_total_time_in_system"] = Metric(
            "max_total_time_in_system", 0, "Max total time in system", ".3f"
        )

        self.lists["arrivals"] = []
        self.lists["delays"] = []
        self.lists["total_times"] = []

        self.traces["number_in_system"] = [(0, 0)]

    def run(self, until: float | None = None, plot_trace: bool = True):
        print("Running simulation...")

        self.env.process(self.arrival_process())
        self.env.run(until)

        # Log final states of relevant traces
        final_number_in_system = self.traces["number_in_system"][-1][1]
        self.traces["number_in_system"].append((self.env.now, final_number_in_system))

        print("Simulation complete!")

        # Calculate metrics
        total_number_in_system = integrate_trace(self.traces["number_in_system"])
        self.metrics["time_avg_number_in_system"].assign(
            total_number_in_system / self.env.now
        )
        self.metrics["total_customers"].assign(len(self.lists["arrivals"]))
        self.metrics["highly_delayed_customers"].assign(
            len([x for x in self.lists["delays"] if x > 1])
        )
        self.metrics["prop_delayed_over_1_min"].assign(
            (
                self.metrics["highly_delayed_customers"].value
                / self.metrics["total_customers"].value
            )
        )
        self.metrics["avg_total_time_in_system"].assign(
            np.array(self.lists["total_times"]).mean()
        )
        self.metrics["max_delay"].assign(np.array(self.lists["delays"]).max())
        self.metrics["max_total_time_in_system"].assign(
            np.array(self.lists["total_times"]).max()
        )

        print("Results")
        print("----------------------")
        longest_name_length = max(
            [len(metric.display_name) for metric in self.metrics.values()]
        )
        for metric in self.metrics.values():
            print(
                f"{metric.display_name:<{longest_name_length + 6}} {metric.value:>7{metric.format_spec}}"
            )

        print("")

        if plot_trace:
            times, lengths = zip(*self.traces["number_in_system"])
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

            self.lists["arrivals"].append(self.env.now)
            self.traces["number_in_system"].append(
                (self.env.now, self.number_in_system)
            )
            self.env.process(self.customer())

    def customer(self):
        arrival_time = self.env.now
        # print(f"Customer arrived at time {arrival_time:.2f}")
        with self.server.request() as req:
            if len(self.server.queue) > self.metrics["max_queue_length"].value:
                self.metrics["max_queue_length"].assign(len(self.server.queue))

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

            self.lists["delays"].append(delay)
            self.lists["total_times"].append(total_time)
            self.traces["number_in_system"].append(
                (self.env.now, self.number_in_system)
            )


def main():
    exp = Experiment(1, 1.5, 1849)
    exp.run(1000)


if __name__ == "__main__":
    main()
