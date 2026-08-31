import os
import json
import ijson
import torch
import numpy as np
import logging
import time  # Import time for logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data, Dataset
from torch.utils.data import random_split

# Import GraPhens for graph creation
from src.graphens import GraPhens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HPOGraphDataset(Dataset):
    """
    PyTorch Dataset for HPO graphs created from simulated patient data.
    
    This dataset:
    1. Loads a preprocessed index file
    2. On-demand loads individual graph objects
    3. Supports shuffling and random access
    """
    
    def __init__(self, root_dir: str, transform=None, pre_transform=None):
        """
        Initialize HPO Graph Dataset.
        
        Args:
            root_dir: Directory containing processed dataset
            transform: Transform to apply at load time
            pre_transform: Transform to apply during preprocessing
        """
        # Store our own copy of the paths that we know are Path objects
        self.root_path = Path(root_dir)  
        self.processed_path = self.root_path 
        self.index_path = self.processed_path / "index.pt"
        
        # Now initialize the Dataset parent class with the string version
        # This avoids path operation conflicts with the parent class implementation
        super().__init__(str(root_dir), transform, pre_transform)
        
        # Load index file
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file not found. Please run preprocessing first: {self.index_path}")
        
        self.index = torch.load(self.index_path)
        self.transform = transform
        self.pre_transform = pre_transform

        # Load metadata to locate the shared embedding memmap (data.x in each
        # saved graph is now an int64 index into this table, not the raw
        # embedding, see convert_to_dataset()).
        metadata_path = self.processed_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self._metadata = metadata
        self._embedding_memmap = np.memmap(
            metadata['embedding_memmap_path'],
            dtype='float32',
            mode='r',
            shape=(metadata['num_embeddings'], metadata['embedding_dim'])
        )

        # Preload every batch_N.pt once into RAM so get() never hits torch.load
        # again. Only the compact per-graph fields are kept (int64 index
        # tensors, not expanded (N, 768) embeddings), see get() below.
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor, str, List[str]]] = {}
        num_batches = metadata['num_batches']
        for file_idx in range(num_batches):
            batch_path = self.processed_path / f"batch_{file_idx}.pt"
            batch_data = torch.load(batch_path)
            for key, data in batch_data.items():
                self._cache[key] = (data.x, data.edge_index, data.gene, data.phenotype_ids)
            del batch_data
        rss_mb = self._current_rss_mb()
        logger.info(
            f"Preloaded {len(self._cache)} graphs from {num_batches} batch files into RAM"
            + (f" (RSS={rss_mb:.1f} MB)" if rss_mb is not None else "")
        )

        logger.info(f"Loaded HPO graph dataset with {len(self.index)} samples")

    @staticmethod
    def _current_rss_mb() -> Optional[float]:
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1]) / 1024
        except (FileNotFoundError, OSError):
            pass
        return None

    def len(self):
        return len(self.index)

    @property
    def metadata(self) -> Dict:
        """Returns the dataset metadata loaded from metadata.json."""
        return self._metadata

    @property
    def num_classes(self) -> Optional[int]:
        """Number of gene classes, inferred from metadata['genes']."""
        if self._metadata and 'genes' in self._metadata:
            return len(self._metadata['genes'])
        return None

    def get(self, idx):
        """Load a graph by index."""
        gene, case_id, file_idx = self.index[idx]

        # Read the raw graph fields from the in-RAM cache built in __init__
        # (no torch.load on the hot path).
        x_idx, edge_index, gene, phenotype_ids = self._cache[f"{gene}_{case_id}"]
        data = Data(x=x_idx, edge_index=edge_index, gene=gene, phenotype_ids=phenotype_ids)

        # Expand the stored node indices into real embeddings via the shared memmap
        embeddings = self._embedding_memmap[data.x.numpy()]
        data.x = torch.tensor(embeddings, dtype=torch.float)

        # Apply transform if provided
        if self.transform:
            data = self.transform(data)

        return data
    
    @staticmethod
    def get_gene_mask(dataset, gene):
        """Get a boolean mask for all samples from a specific gene."""
        mask = []
        for idx in range(len(dataset)):
            gene_name, _, _ = dataset.index[idx]
            mask.append(gene_name == gene)
        return torch.tensor(mask)


def convert_to_dataset(
    input_json: str,
    output_dir: str,
    batch_size: int = 1000,
    max_samples_per_gene: Optional[int] = None,
    embedding_lookup_path: str = "data/embeddings/hpo_embeddings_latest.pkl",
    include_ancestors: bool = True,
    include_reverse_edges: bool = True,
    clear_existing: bool = False,
    no_siblings: bool = False
):
    """
    Convert simulated patient data from JSON to PyTorch Geometric dataset.

    Args:
        input_json: Path to input JSON file with simulated patient data
        output_dir: Directory to save processed dataset
        batch_size: Number of samples to process at once
        max_samples_per_gene: Maximum samples to use per gene (useful for testing)
        embedding_lookup_path: Path to pre-computed HPO embeddings
        include_ancestors: Whether to augment with ancestor HPO terms
        include_reverse_edges: Whether to include reverse edges in the graph
        clear_existing: Whether to clear existing processed files
        no_siblings: If True, use ancestors-only augmentation (pre-150f639
            strategy), skipping the siblings step. For comparability against
            checkpoints trained before the siblings augmentation was added.
    """
    total_start_time = time.time()
    
    # Create output directories
    start_time = time.time()
    output_dir = Path(output_dir)
    processed_dir = output_dir 
    os.makedirs(processed_dir, exist_ok=True)
    logger.info(f"Directory setup took {time.time() - start_time:.2f} seconds")
    
    if clear_existing:
        start_time = time.time()
        logger.info(f"Clearing existing processed files in {processed_dir}")
        for f in processed_dir.glob("*.pt"):
            os.remove(f)
        logger.info(f"Clearing existing files took {time.time() - start_time:.2f} seconds")
            
    # Initialize GraPhens with configuration for our task
    start_time = time.time()
    logger.info("Initializing GraPhens...")
    if no_siblings:
        # Pre-150f639 strategy: ancestors only, no siblings. Kept for
        # comparability against checkpoints trained before siblings
        # augmentation existed (e.g. Gian's best_model.pt).
        augmentation_strategy = [
            {"type": "local", "include_ancestors": True, "include_descendants": False}
        ]
    else:
        augmentation_strategy = [
            {"type": "siblings"},
            {"type": "local", "include_ancestors": True, "include_descendants": False}
        ]

    graphens = (GraPhens()
                .with_lookup_embeddings(embedding_lookup_path)
                .with_augmentation(strategy=augmentation_strategy)
                .with_adjacency_settings(include_reverse_edges=include_reverse_edges))
    logger.info(f"GraPhens initialization took {time.time() - start_time:.2f} seconds")

    # Load the HPO -> global embedding row index, derived from the same path
    # already passed in embedding_lookup_path (generated by
    # convert_embedding_to_memmap in src/embedding/vector_db/memmap.py).
    # Each graph will store an int64 index into the shared memmap instead of
    # the full 768-float embedding per node.
    embedding_path = Path(embedding_lookup_path)
    embedding_memmap_path = str(embedding_path).replace('.pkl', '.mmap')
    hpo_index_path = Path(str(embedding_path).replace('.pkl', '.index.json'))
    if not hpo_index_path.exists():
        raise FileNotFoundError(
            f"No se encontro el indice de embeddings esperado en "
            f"{hpo_index_path}. Se esperaba junto a {embedding_lookup_path} "
            f"(generado por convert_embedding_to_memmap en "
            f"src/embedding/vector_db/memmap.py)."
        )
    with open(hpo_index_path, 'r') as f:
        hpo_to_global_idx = json.load(f)
    mmap_file_size = os.path.getsize(embedding_memmap_path)
    embedding_dim = mmap_file_size // (len(hpo_to_global_idx) * 4)  # 4 bytes por float32
    assert mmap_file_size == len(hpo_to_global_idx) * embedding_dim * 4, (
        f"Tamaño de {embedding_memmap_path} ({mmap_file_size} bytes) no es "
        f"consistente con num_embeddings={len(hpo_to_global_idx)} y "
        f"embedding_dim={embedding_dim} calculados."
    )
    logger.info(f"Cargado indice de {len(hpo_to_global_idx)} fenotipos desde {hpo_index_path}")

    # HPO ids encontrados durante esta corrida (incluyendo los agregados por
    # la augmentacion de siblings/ancestros) que no tenian entrada en
    # hpo_to_global_idx reciben aca un indice nuevo (el siguiente entero
    # disponible), en vez de descartar el caso. next_new_index se inicializa
    # UNA sola vez, a partir del maximo indice ya existente en el indice
    # cargado de disco.
    next_new_index = max(hpo_to_global_idx.values()) + 1 if hpo_to_global_idx else 0
    new_hpo_ids_found = set()

    # Build index and process batches
    # NOTE: gene_data ya no se carga completo en memoria (json.load reventaba
    # por OOM con el JSON de simulacion de ~2.8GB). Se streamea gen por gen
    # con ijson.kvitems, ya que la estructura confirmada del JSON es un dict
    # top-level {gene_symbol: [[hpo_id, ...], [hpo_id, ...], ...]}.
    # Como resultado, 'genes' y 'samples_per_gene' (usados mas abajo en
    # metadata) ya no se pueden reconstruir al final desde gene_data.keys()/
    # .items() -- el generador de ijson se agota en una sola pasada. Se
    # acumulan aca mismo, dentro del loop.
    start_processing_time = time.time()
    index = []
    batch_num = 0
    current_batch = {}  # Store current batch of graphs
    current_batch_count = 0
    genes_seen = []
    samples_per_gene = {}
    
    logger.info(f"Streaming JSON file gen por gen: {input_json}")
    with open(input_json, 'rb') as f:  # ijson requiere modo binario
        # Process each gene
        for gene, cases in tqdm(ijson.kvitems(f, ''), desc="Processing genes"):
            gene_start_time = time.time()

            genes_seen.append(gene)

            # Apply sample limit if specified
            if max_samples_per_gene:
                cases = cases[:max_samples_per_gene]

            samples_per_gene[gene] = len(cases)

            # Collect all phenotype lists for the current gene
            patient_data_for_gene = {}
            for case_idx, phenotype_ids in enumerate(cases):
                 # Check if phenotype_ids is not empty and contains valid HPO terms
                if phenotype_ids and all(isinstance(pid, str) and pid.startswith("HP:") for pid in phenotype_ids):
                    patient_data_for_gene[f"{gene}_{case_idx}"] = phenotype_ids
                else:
                    logger.warning(f"Skipping invalid or empty phenotype list for {gene} case {case_idx}: {phenotype_ids}")

            num_valid_cases = len(patient_data_for_gene)
            if num_valid_cases == 0:
                logger.warning(f"No valid cases found for gene {gene}. Skipping.")
                continue

            logger.info(f"Processing {num_valid_cases} valid cases for gene {gene}...")

            # Create graphs for all valid cases of the current gene in batch
            try:
                batch_graph_creation_start_time = time.time()
                graphs_for_gene = graphens.create_graphs_from_multiple_patients(
                    patient_data_for_gene, 
                    show_progress=False # Use outer tqdm progress
                )
                batch_graph_creation_time = time.time() - batch_graph_creation_start_time
                logger.info(f"Batch graph creation for {gene} ({num_valid_cases} cases) took {batch_graph_creation_time:.2f} seconds")

                # Iterate through the created graphs and add to batch/index
                for patient_key, graph in graphs_for_gene.items():
                    # Extract gene and case_idx from patient_key
                    _, case_id_str = patient_key.split('_', 1)
                    case_idx = int(case_id_str)

                    # Map each node (row of graph.node_features) to its HPO id,
                    # then to its row in the shared embedding memmap.
                    row_to_hpo = {row: hpo_id for hpo_id, row in graph.node_mapping.items()}
                    num_nodes = graph.node_features.shape[0]
                    node_idx_list = []
                    for i in range(num_nodes):
                        hpo_id = row_to_hpo[i]
                        if hpo_id not in hpo_to_global_idx:
                            hpo_to_global_idx[hpo_id] = next_new_index
                            new_hpo_ids_found.add(hpo_id)
                            next_new_index += 1
                        node_idx_list.append(hpo_to_global_idx[hpo_id])
                    node_indices = torch.tensor(node_idx_list, dtype=torch.long)

                    # Convert to PyTorch Geometric Data
                    data = Data(
                        x=node_indices,  # int64 [num_nodes] index into the shared embedding memmap
                        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
                        # Retrieve original phenotype IDs from the input data
                        phenotype_ids=patient_data_for_gene[patient_key],
                        gene=gene
                    )

                    # Add to current batch
                    current_batch[patient_key] = data
                    current_batch_count += 1

                    # Add to index
                    index.append((gene, case_idx, batch_num))

                    # Save the batch if it reaches the batch size
                    if current_batch_count >= batch_size:
                        batch_save_start_time = time.time()
                        batch_path = processed_dir / f"batch_{batch_num}.pt"
                        torch.save(current_batch, batch_path)
                        batch_save_time = time.time() - batch_save_start_time
                        logger.info(f"Saved batch {batch_num} with {current_batch_count} graphs. Took {batch_save_time:.2f} seconds")

                        # Reset for next batch
                        current_batch = {}
                        current_batch_count = 0
                        batch_num += 1

            except Exception as e:
                logger.error(f"Error processing gene {gene} with batch method: {e}")
                # Optionally, add fallback to single processing or skip the gene
                continue

            gene_processing_time = time.time() - gene_start_time
            logger.info(f"Processing gene {gene} took {gene_processing_time:.2f} seconds")
    
    processing_duration = time.time() - start_processing_time
    logger.info(f"Total data processing loop took {processing_duration:.2f} seconds")

    # Save any remaining graphs in the last batch
    if current_batch_count > 0:
        batch_save_start_time = time.time()
        batch_path = processed_dir / f"batch_{batch_num}.pt"
        torch.save(current_batch, batch_path)
        batch_save_time = time.time() - batch_save_start_time
        logger.info(f"Saved final batch {batch_num} with {current_batch_count} graphs. Took {batch_save_time:.2f} seconds")
        
    # Save the index
    start_time = time.time()
    index_path = processed_dir / "index.pt"
    torch.save(index, index_path)
    logger.info(f"Saved index with {len(index)} total samples. Took {time.time() - start_time:.2f} seconds")
    
    # Create metadata file
    start_time = time.time()
    metadata = {
        'num_samples': len(index),
        'num_batches': batch_num + 1,
        'batch_size': batch_size,
        'genes': genes_seen,
        'samples_per_gene': samples_per_gene,
        'embedding_lookup': embedding_lookup_path,
        'include_ancestors': include_ancestors,
        'include_reverse_edges': include_reverse_edges,
        'no_siblings': no_siblings,
        'augmentation_strategy': augmentation_strategy,
        'embedding_memmap_path': embedding_memmap_path,
        'embedding_dim': embedding_dim,
        'num_embeddings': len(hpo_to_global_idx),
        'new_hpo_ids_count': len(new_hpo_ids_found),
        'total_hpo_ids_in_index': len(hpo_to_global_idx)
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata. Took {time.time() - start_time:.2f} seconds")

    # Persist the extended HPO -> global embedding index back to disk,
    # overwriting the original .index.json. IMPORTANTE: este archivo es
    # compartido entre corridas de convert_to_dataset (p. ej. dataset A y
    # dataset B usando el mismo embedding_lookup_path). Como esta funcion
    # siempre carga el .index.json real de disco al principio, si A corre
    # y termina (incluida esta escritura) ANTES de que arranque B, entonces
    # B ya parte del indice extendido por A y reutiliza los mismos indices
    # nuevos para los HPO ids que A ya haya descubierto, en vez de asignarles
    # uno distinto. Esto exige ejecucion SECUENCIAL de A y B: si corrieran en
    # paralelo, ambos leerian el .index.json antes de que el otro terminara
    # de escribirlo, y podrian asignar indices nuevos duplicados/inconsistentes
    # para el mismo HPO id (race condition).
    with open(hpo_index_path, 'w') as f:
        json.dump(hpo_to_global_idx, f)
    logger.info(
        f"Indice de embeddings actualizado: {len(hpo_to_global_idx)} "
        f"entradas totales ({len(new_hpo_ids_found)} nuevas agregadas "
        f"sin embedding real todavia). Guardado en {hpo_index_path}."
    )

    # Traceability only (not used by the pipeline): HPO ids newly discovered
    # in THIS particular run (not the historical accumulation already
    # reflected in hpo_index_path).
    new_hpo_ids_path = processed_dir / "new_hpo_ids_sin_embedding.txt"
    with open(new_hpo_ids_path, 'w') as f:
        f.write('\n'.join(sorted(new_hpo_ids_found)))
    logger.info(f"{len(new_hpo_ids_found)} HPO IDs nuevos sin embedding real, listados en {new_hpo_ids_path}")

    logger.info(f"Dataset creation complete. Saved to {output_dir}")
    total_duration = time.time() - total_start_time
    logger.info(f"Total script execution time: {total_duration:.2f} seconds")
    
    return metadata


def create_train_val_test_splits(
    dataset_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create train/validation/test splits for the dataset.
    
    Args:
        dataset_dir: Directory containing the processed dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        seed: Random seed for reproducibility
    """
    total_split_start_time = time.time()
    logger.info("Starting dataset split creation...")

    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("train_ratio, val_ratio, and test_ratio must each be between 0 and 1.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total_ratio:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )
    
    dataset_dir = Path(dataset_dir)
    splits_dir = dataset_dir / "splits"
    os.makedirs(splits_dir, exist_ok=True)
    
    # Load the dataset
    start_time = time.time()
    dataset = HPOGraphDataset(root_dir=dataset_dir)
    logger.info(f"Dataset loading for splits took {time.time() - start_time:.2f} seconds")
    
    # Calculate split sizes
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = int(test_ratio * n_total)
    n_test += n_total - (n_train + n_val + n_test)
    
    # Create splits
    start_time = time.time()
    torch.manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test]
    )
    logger.info(f"Random split creation took {time.time() - start_time:.2f} seconds")
    
    # Save the split indices
    start_time = time.time()
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    torch.save(train_indices, splits_dir / "train_indices.pt")
    torch.save(val_indices, splits_dir / "val_indices.pt")
    torch.save(test_indices, splits_dir / "test_indices.pt")
    logger.info(f"Saving standard split indices took {time.time() - start_time:.2f} seconds")
    
    # Create a gene-wise stratified split too
    start_time = time.time()
    gene_splits = {}
    metadata_path = dataset_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    for gene in tqdm(metadata['genes'], desc="Creating gene-wise splits"):
        # Get indices for this gene
        gene_mask = HPOGraphDataset.get_gene_mask(dataset, gene)
        gene_indices_tensor = torch.nonzero(gene_mask).squeeze()
        
        # Handle case where gene might have 0 or 1 sample after filtering
        if gene_indices_tensor.numel() == 0:
            logger.warning(f"Gene {gene} has no samples in the dataset. Skipping for gene-wise split.")
            continue
        elif gene_indices_tensor.numel() == 1:
             # Convert scalar tensor to list
            gene_indices = [gene_indices_tensor.item()]
        else:
            gene_indices = gene_indices_tensor.tolist()

        # Ensure it's a list
        if not isinstance(gene_indices, list):
             gene_indices = [gene_indices] # Wrap scalar in list if needed

        if len(gene_indices) == 0: # Double check after potential conversion issues
            logger.warning(f"Gene {gene} resulted in zero indices. Skipping.")
            continue

        # Shuffle the indices
        np.random.seed(seed)
        np.random.shuffle(gene_indices)
        
        # Calculate split sizes
        n_gene_total = len(gene_indices)
        # Ensure at least one sample per split if possible, handle small counts
        n_gene_train = max(1, int(train_ratio * n_gene_total)) if n_gene_total > 2 else n_gene_total
        n_gene_val = max(1, int(val_ratio * n_gene_total)) if n_gene_total > n_gene_train else 0
        n_gene_test = max(0, n_gene_total - n_gene_train - n_gene_val)

        # Adjust if splits exceed total (due to min 1 requirement)
        if n_gene_train + n_gene_val + n_gene_test > n_gene_total:
             # Prioritize train, then val, then test
             if n_gene_train > n_gene_total: n_gene_train = n_gene_total
             n_gene_val = min(n_gene_val, n_gene_total - n_gene_train)
             n_gene_test = n_gene_total - n_gene_train - n_gene_val


        # Create the splits
        gene_train = gene_indices[:n_gene_train]
        gene_val = gene_indices[n_gene_train : n_gene_train + n_gene_val]
        gene_test = gene_indices[n_gene_train + n_gene_val :]
        
        gene_splits[gene] = {
            'train': gene_train,
            'val': gene_val,
            'test': gene_test
        }
    logger.info(f"Gene-wise split calculation took {time.time() - start_time:.2f} seconds")
    
    # Save gene-wise splits
    start_time = time.time()
    with open(splits_dir / "gene_splits.json", 'w') as f:
        # Convert numpy types to native types for JSON serialization
        serializable_splits = {}
        for gene, split in gene_splits.items():
            serializable_splits[gene] = {
                k: [int(i) for i in v] for k, v in split.items()
            }
        json.dump(serializable_splits, f, indent=2)
    logger.info(f"Saving gene-wise splits took {time.time() - start_time:.2f} seconds")
    
    logger.info(f"Created dataset splits in {splits_dir}")
    logger.info(f"Train: {n_train}, Validation: {n_val}, Test: {n_test}")
    total_split_duration = time.time() - total_split_start_time
    logger.info(f"Total split creation time: {total_split_duration:.2f} seconds")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert simulated patient data to PyTorch Geometric dataset")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with simulated patient data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for dataset")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of samples to process at once")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples per gene (for testing)")
    parser.add_argument("--embeddings", type=str, default="data/embeddings/hpo_embeddings_gsarti_biobert-nli_20250317_162424.pkl", 
                        help="Path to pre-computed HPO embeddings")
    parser.add_argument("--no-ancestors", action="store_true", help="Don't include ancestor terms")
    parser.add_argument("--no-siblings", action="store_true",
                        help="Use ancestors-only augmentation (pre-150f639 strategy), skipping "
                             "the siblings step. For comparability against checkpoints trained "
                             "before siblings augmentation existed.")
    parser.add_argument("--no-reverse-edges", action="store_true", help="Don't include reverse edges")
    parser.add_argument("--clear-existing", action="store_true", help="Clear existing processed files")
    parser.add_argument("--create-splits", action="store_true", help="Create train/val/test splits")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation data ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    
    args = parser.parse_args()
    
    # Convert dataset
    metadata = convert_to_dataset(
        input_json=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_samples_per_gene=args.max_samples,
        embedding_lookup_path=args.embeddings,
        include_ancestors=not args.no_ancestors,
        include_reverse_edges=not args.no_reverse_edges,
        clear_existing=args.clear_existing,
        no_siblings=args.no_siblings
    )
    
    # Create splits if requested
    if args.create_splits:
        create_train_val_test_splits(
            dataset_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=1.0 - args.train_ratio - args.val_ratio,
            seed=args.seed
        )
