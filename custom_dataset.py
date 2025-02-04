from torch_geometric.data import InMemoryDataset
from syn_random import RandomType
import os.path as osp
from typing import Union
from torch_geometric.utils import from_networkx


class SyntheticRandom(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        graphtype: RandomType,
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
        from syn_random import init_pyg_random as generate_graph
        graph_type_str = f"RandomType.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])



class SyntheticRegularTilling(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        graphtype: RandomType,
        transform=None,
        N: int=10000,
    ):
        self.dataset_name = name
        self.N = N
        self.graphtype = graphtype
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    def process(self):
        from syn_regulartilling import init_regular_tilling as generate_graph
        graph_type_str = f"RegularTilling.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])





class SyntheticDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        graphtype: GraphType,
        transform=None,
        N: int=1000,
    ):
        self.dataset_name = name
        self.N = N
        super().__init__(root, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.__class__.__name__, 'processed')

    @property
    def processed_file_names(self) -> str:
        return f'{self.dataset_name}_{self.N}.pt'

    def process(self):
        graph_type_str = f"GraphType.{self.dataset_name}"
        nx_data = generate_graph(self.N, eval(graph_type_str), seed=0)
        data = from_networkx(nx_data)
        self.save([data], self.processed_paths[0])