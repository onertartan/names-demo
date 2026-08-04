"""
Custom Genetic Algorithm clustering engine.
No pygad dependency — every operator is explicit and label-aware.
Compatible with BaseClustering interface.
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score

from clustering.base_clustering import BaseClustering
from clustering.evaluation.cvi_registry import dunn_index

# ---------------------------------------------------------------------------
# Selection strategy enum
# ---------------------------------------------------------------------------

class SelectionMethod(str, Enum):
    """
    Available parent-selection strategies.

    RANK        — probability proportional to fitness rank; robust against
                  fitness-scale outliers and negative values.
    ROULETTE    — probability proportional to raw fitness (fitness-proportionate
                  selection); fast but sensitive to scale and dominated by
                  high-fitness outliers.
    TOURNAMENT  — pick the best of t randomly sampled individuals; pressure
                  controlled by tournament_size.
    TRUNCATION  — sample uniformly from the top truncation_ratio fraction;
                  high pressure, fast convergence, low diversity.
    """
    RANK       = "rank"
    ROULETTE   = "roulette"
    TOURNAMENT = "tournament"
    TRUNCATION = "truncation"


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------

def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """
    Relabel so cluster IDs appear in order of first occurrence.

        [1, 0, 0, 1, 2] -> [0, 1, 1, 0, 2]

    Guarantees a unique canonical form for every equivalent partition,
    eliminating label-permutation duplicates from the population.
    """
    labels = np.asarray(labels, dtype=int)
    mapping: dict[int, int] = {}
    next_id = 0
    canonical = np.empty_like(labels)
    for i, lbl in enumerate(labels):
        if lbl not in mapping:
            mapping[lbl] = next_id
            next_id += 1
        canonical[i] = mapping[lbl]
    return canonical


def align_labels(reference: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    """
    Permute *target* label ids so they best match *reference* via the
    Hungarian algorithm on the k×k overlap (confusion) matrix.
    Both arrays must contain values in {0..k-1}.
    """
    overlap = np.zeros((k, k), dtype=int)
    for r, t in zip(reference, target):
        if 0 <= r < k and 0 <= t < k:
            overlap[r, t] += 1

    row_ind, col_ind = linear_sum_assignment(-overlap)

    mapping = np.arange(k)
    for ref_lbl, tgt_lbl in zip(row_ind, col_ind):
        mapping[tgt_lbl] = ref_lbl

    return mapping[target]


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

class Individual:
    """
    A single chromosome: a length-N array of cluster labels in {0..k-1}.
    Fitness is computed lazily and cached on the instance.
    """

    __slots__ = ("genes", "_fitness")

    def __init__(self, genes: np.ndarray):
        self.genes: np.ndarray = genes
        self._fitness: float | None = None

    def fitness(self, data: np.ndarray, metric_fun: Callable, maximize: bool) -> float:
        if self._fitness is not None:
            return self._fitness
        self._fitness = _evaluate(self.genes, data, metric_fun, maximize)
        return self._fitness

    def invalidate(self) -> None:
        """Call after genes are modified so fitness is recomputed."""
        self._fitness = None

    def crossover(self, other: "Individual", k: int, rng: np.random.Generator) -> "Individual":
        """
        Hungarian-aligned uniform crossover.

        1. Canonicalize both parents.
        2. Align *other* to *self* so mixed genes stay semantically consistent.
        3. Per-gene binary mask selects from self or aligned other.
        4. Re-canonicalize child.
        """
        p1 = canonicalize_labels(self.genes)
        p2 = align_labels(p1, canonicalize_labels(other.genes), k)
        mask = rng.integers(0, 2, size=len(p1), dtype=bool)
        return Individual(canonicalize_labels(np.where(mask, p1, p2)))

    def mutate(self, k: int, mutation_rate: float, rng: np.random.Generator) -> "Individual":
        """
        Smart reassignment mutation.

        Each selected position either shifts to an existing cluster (50 %)
        or introduces a new label if k is not yet reached (50 %).
        Result is canonicalized and repaired against degeneracy.
        """
        genes = self.genes.copy()
        n = len(genes)

        for i in rng.choice(n, size=max(1, int(n * mutation_rate)), replace=False):
            current = genes[i]
            unique_now = np.unique(genes)
            n_unique = len(unique_now)

            if (rng.random() < 0.5) or (n_unique >= k):
                choices = unique_now[unique_now != current]
                if len(choices):
                    genes[i] = rng.choice(choices)
            else:
                genes[i] = n_unique   # next unused label

        return Individual(_repair(canonicalize_labels(genes), k, rng))

    def clone(self) -> "Individual":
        c = Individual(self.genes.copy())
        c._fitness = self._fitness
        return c

    def __lt__(self, other: "Individual") -> bool:
        return (self._fitness or -np.inf) < (other._fitness or -np.inf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evaluate(
    genes: np.ndarray,
    data: np.ndarray,
    metric_fun: Callable,
    maximize: bool,
) -> float:
    """Compute fitness; return -inf for degenerate partitions."""
    n_unique = len(np.unique(genes))
    if n_unique < 2 or n_unique == len(genes):
        return -np.inf
    try:
        score = float(metric_fun(data, genes))
        return score if maximize else -score
    except Exception:
        return -np.inf


def _repair(genes: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Mutasyon sonrası bozulan kromozomları tam olarak k kümeye zorlar.

    n_unique < 2      — tek küme: rastgele yarısını yeni kümeye ayır.
    2 ≤ n_unique < k  — eksik kümeler var: en kalabalık kümelerden
                        rastgele noktalar kopararak yeni kümeler oluştur.
    n_unique > k      — fazla küme: etiketleri {0..k-1}'e kırp.
    n_unique == k     — dokunma.
    """
    genes = genes.copy()

    # --- Fazla küme: kırp ve erken çık ---
    if len(np.unique(genes)) > k:
        genes = np.clip(genes, 0, k - 1)
        return canonicalize_labels(genes)

    # --- Eksik küme: tam olarak k kümeye ulaşana kadar böl ---
    while len(np.unique(genes)) < k:
        # En kalabalık kümeyi bul — en güvenli bölme adayı
        unique, counts = np.unique(genes, return_counts=True)
        largest_label  = unique[np.argmax(counts)]
        largest_size   = counts.max()

        if largest_size < 2:
            # Bölecek nokta kalmadı — onarım mümkün değil, olduğu gibi bırak
            break

        # Yeni küme etiketi: mevcut maksimum + 1
        new_label = int(genes.max()) + 1

        # En kalabalık kümeden rastgele bir nokta kopar
        candidate_indices = np.where(genes == largest_label)[0]
        n_to_move = max(1, largest_size // 2)
        moved = rng.choice(candidate_indices, size=n_to_move, replace=False)
        genes[moved] = new_label

    return canonicalize_labels(genes)

def _random_individual(n: int, k: int, rng: np.random.Generator) -> Individual:
    """
    Generate a random valid chromosome in canonical form.
    Forces every cluster label to appear at least once.
    """
    genes = rng.integers(0, k, size=n, dtype=int)
    forced = rng.choice(n, size=k, replace=False)
    for label, pos in enumerate(forced):
        genes[pos] = label
    return Individual(canonicalize_labels(genes))


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

class _Selector:
    """
    Encapsulates all parent-selection strategies.

    select_pair() is called once per offspring — it draws exactly 2
    individuals from the population (or a pre-built truncation pool)
    according to the chosen strategy.
    """

    def __init__(
        self,
        method: SelectionMethod | str,
        rng: np.random.Generator,
        tournament_size: int = 3,
        truncation_ratio: float = 0.5,
    ):
        self.method = SelectionMethod(method)
        self.rng = rng
        self.tournament_size = tournament_size
        self.truncation_ratio = truncation_ratio

    def build_truncation_pool(self, population: list[Individual]) -> list[Individual]:
        """
        Pre-sort and slice the top fraction once per generation.
        Only used by TRUNCATION — called in fit() before the offspring loop
        so the O(N log N) sort is paid once, not once per offspring.
        """
        n_keep = max(2, int(len(population) * self.truncation_ratio))
        return sorted(population, reverse=True)[:n_keep]

    def select_pair(
        self,
        pool: list[Individual],
    ) -> tuple[Individual, Individual]:
        """
        Draw exactly 2 parents from *pool* using the configured strategy.
        For TRUNCATION pass the pre-built truncation pool; for all others
        pass the full population.
        """
        method_map = {
            SelectionMethod.RANK:       self._rank_pair,
            SelectionMethod.ROULETTE:   self._roulette_pair,
            SelectionMethod.TOURNAMENT: self._tournament_pair,
            SelectionMethod.TRUNCATION: self._truncation_pair,
        }
        return method_map[self.method](pool)

    # ------------------------------------------------------------------
    # Pair-selection implementations
    # ------------------------------------------------------------------

    def _rank_pair(self, population: list[Individual]) -> tuple[Individual, Individual]:
        """
        Probability ∝ rank position (1 = worst … N = best).
        Immune to fitness-scale distortions and negative values.
        """
        ranked = sorted(population, key=lambda ind: ind._fitness or -np.inf)
        ranks  = np.arange(1, len(ranked) + 1, dtype=float)
        probs  = ranks / ranks.sum()
        i, j   = self.rng.choice(len(ranked), size=2, replace=False, p=probs)
        return ranked[i], ranked[j]

    def _roulette_pair(self, population: list[Individual]) -> tuple[Individual, Individual]:
        """
        Probability ∝ fitness value (fitness-proportionate selection).
        Shifted so all probabilities are non-negative.
        """
        fitnesses = np.array([ind._fitness or -np.inf for ind in population])
        shifted   = fitnesses - fitnesses.min()
        total     = shifted.sum()
        probs     = (
            np.ones(len(population)) / len(population)   # uniform fallback
            if total == 0
            else shifted / total
        )
        i, j = self.rng.choice(len(population), size=2, replace=False, p=probs)
        return population[i], population[j]

    def _tournament_pair(self, population: list[Individual]) -> tuple[Individual, Individual]:
        """
        Each parent wins its own independent tournament of tournament_size
        randomly sampled competitors.
        Higher tournament_size → stronger selection pressure.
        """
        def _one_tournament() -> Individual:
            idx = self.rng.choice(len(population), size=self.tournament_size, replace=False)
            return max(population[i] for i in idx)

        p1 = _one_tournament()
        p2 = _one_tournament()
        # Re-draw p2 if identical object to p1 (very rare but possible)
        while p2 is p1:
            p2 = _one_tournament()
        return p1, p2

    def _truncation_pair(self, pool: list[Individual]) -> tuple[Individual, Individual]:
        """
        Sample uniformly from a pre-built top-fraction pool.
        pool is already sorted and sliced — no re-sorting here.
        """
        i, j = self.rng.choice(len(pool), size=2, replace=False)
        return pool[i], pool[j]


# ---------------------------------------------------------------------------
# GAEngine
# ---------------------------------------------------------------------------

class GAEngine(BaseClustering):
    """
    Genetic Algorithm clustering engine — fully custom, no pygad.

    Each chromosome is a length-N array of cluster labels in {0..k-1}.
    Label-permutation invariance is maintained at every stage:

      Initialisation  — all k labels forced present; canonical form set.
      Crossover       — Hungarian alignment before uniform gene mixing.
      Mutation        — smart reassignment; degenerate results repaired.
      Elitism         — top solutions copied verbatim, fitness preserved.
      Fitness cache   — keyed on canonical tuple; duplicate partitions
                        are never re-evaluated.

    Parameters
    ----------
    data              : array-like, shape (N, M)
    n_clusters        : number of target clusters k
    num_generations   : GA generations to run
    sol_per_pop       : population size
    elitism_count     : elite solutions carried forward unchanged
                        (default: sol_per_pop // 20)
    crossover_rate    : probability that two parents exchange genes;
                        otherwise the fitter parent is cloned  (default: 0.8)
    mutation_rate     : fraction of genes mutated per individual (default: 0.002)
    saturate_after    : early-stop after this many stagnant generations
                        (None = run all generations)
    selection_method  : SelectionMethod enum or string —
                        "rank" | "roulette" | "tournament" | "truncation"
    tournament_size   : competitors per tournament  (TOURNAMENT only)
    truncation_ratio  : top fraction kept as parents  (TRUNCATION only)
    metric_fun        : callable(data, labels) -> float
                        default: silhouette_score
    maximize          : True  — higher metric is better (silhouette, ARI …)
                        False — lower metric is better (Davies-Bouldin …)
    random_state      : int seed for reproducibility
    """

    def __init__(
        self,
        data=None,
        n_clusters: int = 2,
        num_generations: int = 1000,
        sol_per_pop: int = 100,
        elitism_count: int | None = None,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.001,
        saturate_after: int | None = 100,
        selection_method: SelectionMethod | str = SelectionMethod.RANK,
        tournament_size: int = 3,
        truncation_ratio: float = 0.3,
        metric_fun: Callable | None = None,
        maximize: bool = True,
        random_state: int | None = None,
    ):
        self.data = np.array(data)
        self.n_clusters = n_clusters
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.elitism_count = (
            elitism_count if elitism_count is not None else max(1, sol_per_pop // 20)
        )
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.saturate_after = saturate_after
        self.metric_fun = metric_fun or silhouette_score
        self.maximize = maximize
        self.random_state = random_state

        self.labels_: np.ndarray | None = None
        self.best_fitness_: float | None = None
        self.fitness_history_: list[float] = []

        self._rng = np.random.default_rng(random_state)
        self._cache: dict[tuple, float] = {}
        self._selector = _Selector(
            method=selection_method,
            rng=self._rng,
            tournament_size=tournament_size,
            truncation_ratio=truncation_ratio,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, data=None) -> "GAEngine":
        """Run the GA and store the best canonical labels in self.labels_."""
        n_samples = len(self.data)
        k = self.n_clusters
        rng = self._rng

        # --- Initialise population ---
        population: list[Individual] = [_random_individual(n_samples, k, rng) for _ in range(self.sol_per_pop)]

        best: Individual | None = None
        stagnant = 0

        for _generation in range(self.num_generations):

            # --- Evaluate + cache fitness ---
            for ind in population:
                key = tuple(ind.genes.tolist())
                if key in self._cache:
                    ind._fitness = self._cache[key]
                else:
                    f = ind.fitness(self.data, self.metric_fun, self.maximize)
                    self._cache[key] = f

            # --- Track global best ---
            gen_best = max(population)
            if best is None or gen_best._fitness > best._fitness:
                best = gen_best.clone()
                stagnant = 0
            else:
                stagnant += 1

            self.fitness_history_.append(best._fitness)

            # --- Early stopping ---
            if self.saturate_after and stagnant >= self.saturate_after:
                break

            # --- Elitism: carry top individuals unchanged ---
            elites = [
                e.clone()
                for e in sorted(population, reverse=True)[: self.elitism_count]
            ]

            # --- Build selection pool (truncation sorts once here) ---
            selection_pool = (
                self._selector.build_truncation_pool(population)
                if self._selector.method == SelectionMethod.TRUNCATION
                else population
            )

            # --- Crossover + mutation: one fresh pair drawn per offspring ---
            offspring: list[Individual] = []
            while len(offspring) < self.sol_per_pop - self.elitism_count:
                p1, p2 = self._selector.select_pair(selection_pool)

                if rng.random() < self.crossover_rate:
                    # Produce a genuinely crossed child
                    child = p1.crossover(p2, k, rng)
                else:
                    # Clone the fitter parent; let mutation diversify it
                    child = max(p1, p2).clone()
                    child.invalidate()

                offspring.append(child.mutate(k, self.mutation_rate, rng))

            population = elites + offspring

        self.labels_ = best.genes
        self.best_fitness_ = best._fitness
        return self

    def predict(self, data=None) -> np.ndarray:
        """Return cluster labels from the best solution found."""
        if self.labels_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self.labels_

    def plot_fitness(self) -> None:
        """Plot best-fitness progression across generations."""
        if not self.fitness_history_:
            raise RuntimeError("Call fit() before plot_fitness().")
        try:
            import matplotlib.pyplot as plt
        except ImportError as err:
            raise ImportError("matplotlib is required for plot_fitness().") from err

        plt.figure(figsize=(8, 4))
        plt.plot(self.fitness_history_, linewidth=2)
        plt.xlabel("Generation")
        metric_name = getattr(self.metric_fun, "__name__", "Fitness")
        selection_name = self._selector.method.value
        plt.ylabel(metric_name)
        plt.title(f"GA Clustering [{selection_name} selection] — {metric_name} per Generation")
        plt.tight_layout()
        plt.show()