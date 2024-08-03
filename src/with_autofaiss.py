from autofaiss import build_index
from PIL import Image
import os


class WithAutoFaiss:
    def __init__(self):
        ...

    def prepare_dataset(self, emb_dir, index_path, index_infos_path):
        build_index(embeddings=emb_dir, index_path=index_path,
                    index_infos_path=index_infos_path, max_index_memory_usage="4G",
                    current_memory_available="4G")
