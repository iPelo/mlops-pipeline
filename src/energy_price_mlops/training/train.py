from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    print("Training scaffold is ready. Implement the LightningDataModule and LightningModule next.")


if __name__ == "__main__":
    main()

