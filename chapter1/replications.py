import numpy as np

from mm1queue import Experiment, ExperimentResults


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
            Experiment(arrival_rate, service_rate, rng=self.rng)
            for _ in range(replications)
        ]

    def run(self, until: float | None = None):
        for exp in self.runs:
            exp.run(until)

    def get_results(self) -> list[ExperimentResults]:
        for exp in self.runs:
            exp.get_results(plot_trace=False)

        results: list[ExperimentResults] = [exp.results for exp in self.runs]

        return results


def main():
    arrival_rate = 2
    service_rate = 3
    replications = 10

    exp = ReplicatedExperiment(arrival_rate, service_rate, replications, seed=1432)
    exp.run(10000)
    exp.get_results()


if __name__ == "__main__":
    main()
