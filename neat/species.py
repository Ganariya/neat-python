"""Divides the population into species based on genomic distances."""
from __future__ import annotations

from itertools import count
from typing import List, Dict, Set, Tuple, Optional, TYPE_CHECKING

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean, stdev

if TYPE_CHECKING:
    from neat.config import DefaultClassConfig, Config
    from neat.reporting import ReporterSet
    from neat.genome import DefaultGenome, DefaultGenomeConfig


class Species(object):
    def __init__(self, key: int, generation: int):
        self.key: int = key
        self.created: int = generation
        self.last_improved: int = generation
        self.representative: Optional[DefaultGenome] = None
        self.members: Dict[int, DefaultGenome] = {}
        self.fitness: Optional[float] = None
        self.adjusted_fitness: Optional[float] = None
        self.fitness_history: List[float] = []

    def update(self, representative: DefaultGenome, members: Dict[int, DefaultGenome]):
        self.representative = representative
        self.members = members

    def get_fitnesses(self) -> List[float]:
        return [m.fitness for m in self.members.values()]


class GenomeDistanceCache(object):
    def __init__(self, config: DefaultGenomeConfig):
        self.distances: Dict[Tuple[int, int], float] = {}
        self.config: DefaultGenomeConfig = config
        self.hits: int = 0
        self.misses: int = 0

    def __call__(self, genome0, genome1) -> float:
        g0: int = genome0.key
        g1: int = genome1.key
        d: float = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            # 2つのgenomeの距離を計算する
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d


class DefaultSpeciesSet(DefaultClassConfig):
    """ Encapsulates the default speciation scheme. """

    def __init__(self, config: DefaultClassConfig, reporters: ReporterSet):
        # pylint: disable=super-init-not-called
        self.species_set_config: DefaultClassConfig = config
        self.reporters: ReporterSet = reporters
        self.indexer: count = count(1)
        self.species: Dict[int, Species] = {}
        self.genome_to_species: Dict[int, int] = {}

    @classmethod
    def parse_config(cls, param_dict: Dict[str, str]) -> DefaultClassConfig:
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('compatibility_threshold', float)])

    def speciate(self, config: Config, population: Dict[int, DefaultGenome], generation: int):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.

        ゲノムを遺伝的類似度によって種に配置する。
        この方法は、現在の種の代表者が旧世代のものであると仮定しており、種分化が行われた後、旧世代の代表者は削除され、新世代の代表者と入れ替わるべきであることに注意してください。
        この仮定に違反した場合は、コードの他の必要な部分が新しい動作を反映するように更新されていることを確認してください。
        """
        assert isinstance(population, dict)

        compatibility_threshold: float = self.species_set_config.compatibility_threshold

        # Find the best representatives for each existing species.
        # k+1世代の個体をこれから新しい種に振り分けていく
        unspeciated: Set[int] = set(population)
        distances: GenomeDistanceCache = GenomeDistanceCache(config.genome_config)

        # 種sidの新しい代表者ゲノムのID
        new_representatives: Dict[int, int] = {}

        # 種sidに含まれる個体たちのID
        new_members: Dict[int, List[int]] = {}

        # 1世代前であるk世代の種族ごと
        # このforでは１世代前の種族の代表に最も近い新しい代表者を取り出しているだけ（他のメンバは取り出していない）
        for sid, s in self.species.items():
            candidates: List[Tuple[float, DefaultGenome]] = []
            for gid in unspeciated:
                g = population[gid]

                # 前の世代のある種の代表s.representativeとgenome gの距離を計算
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            # sの代表者と最も距離が近いgenomeを取り出す rdistが距離 repが新しいk+1の代表者genome
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid: int = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        # まだ分類していないgenomeたちを類似度を用いて　代表者に近いものに割り振っていく
        # ただし似ていないものは新しいものにする
        while unspeciated:
            gid: int = unspeciated.pop()
            g: DefaultGenome = population[gid]

            # Find the species with the most similar representative.
            # 個体gと、新しい代表者たちと比較して　似ているものをcandidatesへ
            candidates: List[Tuple[float, int]] = []
            for sid, rid in new_representatives.items():
                rep: DefaultGenome = population[rid]
                d: float = distances(rep, g)
                if d < compatibility_threshold:
                    candidates.append((d, sid))

            # 最も似ている代表のグループに入れる
            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                # 新しく代表を作る
                sid: int = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        # 新しくできたSp
        self.genome_to_species: Dict[int, int] = {}
        for sid, rid in new_representatives.items():

            # 今のk+1世代の種族のID=sidとして、もし前k世代にもあるならその種族s=speciesとする
            # 各種族のSpeciesインスタンスを SpeciesSetのspecies辞書に入れる（単語が似ていて厄介）
            s: Optional[Species] = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            # k+1世代目の種sidに含まれる個体たち
            members: List[int] = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            # 種族s=sidの代表者とメンバーを更新する
            member_dict: Dict[int, DefaultGenome] = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean: float = mean(distances.distances.values())
        gdstdev: float = stdev(distances.distances.values())
        self.reporters.info(
            'Mean genetic distance {0:.3f}, standard deviation {1:.3f}'.format(gdmean, gdstdev))

    def get_species_id(self, individual_id: int) -> int:
        """
        個体iが所属する種のIDを返す
        """
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id: int):
        """
        個体iが所属する種を返す
        """
        sid = self.genome_to_species[individual_id]
        return self.species[sid]
