import numpy as np

from mm1queue import Experiment
import pandas as pd
import matplotlib.pyplot as plt


class ReplicatedExperiment:
    """
    A repeated simulation experiment
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        replications: int = 1,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.replications = replications

        if rng:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed)

        self.runs = [
            Experiment(arrival_rate, service_rate, rng=self.rng, experiment_id=i + 1)
            for i in range(replications)
        ]

    def gather_metrics(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {name: metric.value for name, metric in exp.metrics.items()}
                for exp in self.runs
            ],
            index=[exp.experiment_id for exp in self.runs],
        )

    def plot_traces(self) -> None:
        fig, ax = plt.subplots()
        for exp in self.runs:
            for trace in exp.traces.values():
                times, values = zip(*trace)
                ax.step(
                    times,
                    values,
                    where="post",
                    label=f"{exp.experiment_id}",
                    alpha=0.7,
                    lw=0.6,
                )

        ax.set_title("Number of customers in queue over time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Count")
        fig.legend(title="Replication", loc="center right")
        plt.show()

    def run(self, until: float | None = None, print_summaries: bool = False):
        for exp in self.runs:
            print(f"Replication {exp.experiment_id}")
            exp.run(until, print_summary=print_summaries, plot_trace=False)

        results = self.gather_metrics()
        print(results.describe())
        self.plot_traces()


def main():
    arrival_rate = 2
    service_rate = 1.5
    replications = 10

    exp = ReplicatedExperiment(arrival_rate, service_rate, replications, seed=1629)
    exp.run(1000)


if __name__ == "__main__":
    main()
