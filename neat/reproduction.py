"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
from __future__ import division, annotations

import math
import random
from itertools import count
from typing import List, Dict, Optional, Set, Tuple, Type, TYPE_CHECKING

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean

if TYPE_CHECKING:
    from neat.config import DefaultClassConfig, Config
    from neat.reporting import ReporterSet
    from neat.stagnation import DefaultStagnation
    from neat.genome import DefaultGenome, DefaultGenomeConfig
    from neat.species import DefaultSpeciesSet, Species

    pass


# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict: Dict[str, str]) -> DefaultClassConfig:
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 2)])

    def __init__(self, config: DefaultClassConfig, reporters: ReporterSet, stagnation: DefaultStagnation):
        # pylint: disable=super-init-not-called
        self.reproduction_config: DefaultClassConfig = config
        self.reporters: ReporterSet = reporters
        self.genome_indexer: count = count(1)
        self.stagnation: DefaultStagnation = stagnation
        self.ancestors = {}

    def create_new(self, genome_type: Type[DefaultGenome], genome_config: DefaultGenomeConfig, num_genomes: int) -> Dict[int, DefaultGenome]:
        new_genomes: Dict[int, DefaultGenome] = {}
        for i in range(num_genomes):
            key: int = next(self.genome_indexer)
            g: DefaultGenome = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
            self.ancestors[key] = tuple()

        return new_genomes

    @staticmethod
    def compute_spawn(adjusted_fitness: List[float], previous_sizes: List[int], pop_size: int, min_species_size: int) -> List[int]:
        """Compute the proper number of offspring per species (proportional to fitness)."""
        af_sum: float = sum(adjusted_fitness)

        # おそらく新しく作られる種の各個体数
        spawn_amounts: List[int] = []
        for af, ps in zip(adjusted_fitness, previous_sizes):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * pop_size)
            else:
                s = min_species_size

            d: float = (s - ps) * 0.5
            c: int = int(round(d))
            spawn: int = ps
            if abs(c) > 0:
                spawn += c
            elif d > 0:
                spawn += 1
            elif d < 0:
                spawn -= 1

            spawn_amounts.append(spawn)

        # Normalize the spawn amounts so that the next generation is roughly
        # the population size requested by the user.
        total_spawn: int = sum(spawn_amounts)
        norm: float = pop_size / total_spawn
        spawn_amounts = [max(min_species_size, int(round(n * norm))) for n in spawn_amounts]

        return spawn_amounts

    def reproduce(self, config: Config, species_set: DefaultSpeciesSet, pop_size: int, generation: int) -> Dict[int, DefaultGenome]:
        """
        Handles creation of genomes, either from scratch or by sexual or
        asexual reproduction from parents.

        k世代目からk+1世代目のFlatな配列を作成する
        まだ種には入れていないっぽい
        """
        # TODO: I don't like this modification of the species and stagnation objects,
        # because it requires internal knowledge of the objects.

        # Filter out stagnated species, collect the set of non-stagnated
        # species members, and compute their average adjusted fitness.
        # The average adjusted fitness scheme (normalized to the interval
        # [0, 1]) allows the use of negative fitness values without
        # interfering with the shared fitness scheme.

        # 生き残る世代のすべての個体を配列にflat
        all_fitnesses: List[float] = []

        # 生き残る種
        remaining_species: List[Species] = []
        for stag_sid, stag_s, stagnant in self.stagnation.update(species_set, generation):
            if stagnant:
                # stagnantなら死んだことを伝える
                self.reporters.species_stagnant(stag_sid, stag_s)
            else:
                # 次の世代へ生き残らせる
                all_fitnesses.extend(m.fitness for m in stag_s.members.values())
                remaining_species.append(stag_s)
        # The above comment was not quite what was happening - now getting fitnesses
        # only from members of non-stagnated species.

        # No species left.　すべての種が同時に死んだ場合
        if not remaining_species:
            species_set.species = {}
            return {}  # was []

        # Find minimum/maximum fitness across the entire population, for use in
        # species adjusted fitness computation.
        min_fitness: float = min(all_fitnesses)
        max_fitness: float = max(all_fitnesses)
        # Do not allow the fitness range to be zero, as we divide by it below.
        # TODO: The ``1.0`` below is rather arbitrary, and should be configurable.
        # 生き残った種で最小と最大の平均適応度の幅を計算して正規化する
        fitness_range: float = max(1.0, max_fitness - min_fitness)
        for rs in remaining_species:
            # Compute adjusted fitness.
            msf: float = mean([m.fitness for m in rs.members.values()])
            af: float = (msf - min_fitness) / fitness_range
            rs.adjusted_fitness = af

        # 各種の調整された平均適応度
        adjusted_fitnesses: List[float] = [s.adjusted_fitness for s in remaining_species]
        avg_adjusted_fitness: float = mean(adjusted_fitnesses)
        self.reporters.info("Average adjusted fitness: {:.3f}".format(avg_adjusted_fitness))

        # Compute the number of new members for each species in the new generation.
        previous_sizes: List[int] = [len(s.members) for s in remaining_species]
        min_species_size: int = self.reproduction_config.min_species_size
        # Isn't the effective min_species_size going to be max(min_species_size,
        # self.reproduction_config.elitism)? That would probably produce more accurate tracking
        # of population sizes and relative fitnesses... doing. TODO: document.
        min_species_size: int = max(min_species_size, self.reproduction_config.elitism)

        # 新しい世代の各種の個体数を計算する（個体はまだ生成してない　数だけ）
        spawn_amounts: List[int] = self.compute_spawn(adjusted_fitnesses, previous_sizes,
                                                      pop_size, min_species_size)

        # k+1世代目のFlatな個体たち
        new_population: Dict[int, DefaultGenome] = {}

        # k + 1世代目の各種
        species_set.species = {}
        for spawn, s in zip(spawn_amounts, remaining_species):
            # If elitism is enabled, each species always at least gets to retain its elites.
            spawn: int = max(spawn, self.reproduction_config.elitism)

            assert spawn > 0

            # The species has at least one member for the next generation, so retain it.
            old_members: List[Tuple[int, DefaultGenome]] = list(s.members.items())
            s.members = {}
            species_set.species[s.key] = s

            # Sort members in order of descending fitness.
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # Transfer elites to new generation.
            # 強い個体はそのまま次の種に残す
            # またこのとき、まだ新しい種に直接は入れてないっぽい（flatな個体配列に入れているだけ）
            if self.reproduction_config.elitism > 0:
                for gid, genome in old_members[:self.reproduction_config.elitism]:
                    new_population[gid] = genome
                    spawn -= 1

            if spawn <= 0:
                continue

            # Only use the survival threshold fraction to use as parents for the next generation.
            # 交配で使う親の数
            repro_cutoff: int = int(math.ceil(self.reproduction_config.survival_threshold *
                                              len(old_members)))
            # Use at least two parents no matter what the threshold fraction result is.
            repro_cutoff: int = max(repro_cutoff, 2)

            # 上位repro_cutoff体だけ　個体を取り出して親候補にする
            old_members: List[Tuple[int, DefaultGenome]] = old_members[:repro_cutoff]

            # Randomly choose parents and produce the number of offspring allotted to the species.
            # 親を取り出してランダムに交配させる
            # ただここではまだ種に追加していないっぽい
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                # Note that if the parents are not distinct, crossover will produce a
                # genetically identical clone of the parent (but with a different ID).
                gid: int = next(self.genome_indexer)
                child: DefaultGenome = config.genome_type(gid)
                child.configure_crossover(parent1, parent2, config.genome_config)
                # 突然変異
                child.mutate(config.genome_config)
                new_population[gid] = child
                self.ancestors[gid] = (parent1_id, parent2_id)

        return new_population
