from clustering.models.kmeans import KMeansEngine
from clustering.models.gmm import GMMEngine
from clustering.models.kmedoids import KMedoidsEngine
from clustering.models.spectral import SpectralClusteringEngine
from clustering.models.hierarchical import HierarchicalBaseClusteringEngine

def get_engine_class(algo: str):
    mapping = {
        "kmeans": KMeansEngine,
        "gmm": GMMEngine,
        "kmedoids": KMedoidsEngine,
        "spectral": SpectralClusteringEngine,
        "hierarchical": HierarchicalBaseClusteringEngine
    }
    if algo not in mapping:
        raise ValueError(f"Unsupported algorithm: {algo}")
    return mapping[algo]