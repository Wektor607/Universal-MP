from torch_geometric.data import Data, Dataset, InMemoryDataset
from syn_random import init_pyg_random, RandomType
from syn_regulartilling import RegularTilling, init_regular_tilling
import os.path as osp
from typing import Union
from torch_geometric.utils import coalesce, to_undirected, from_networkx


class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        graphtype: Union[RegularTilling, RandomType],
        transform=None,
        N: int=10000,
    ):
        self.dataset_name = name
        self.N = N
        self.graphtype = graphtype
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'

    def process(self):
        if type(self.graphtype) in RegularTilling:
            from syn_regulartilling import RegularTilling, init_regular_tilling as generate_graph
        elif type(self.graphtype) in RandomType:
            from syn_random import init_pyg_random as generate_graph
        raise NotImplementedError(f"Graph type {self.graphtype} not implemented")
        graph_type_str = f"GraphType.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])