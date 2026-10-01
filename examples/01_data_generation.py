# -*- coding: UTF-8 -*-
"""
Example 01: Data Generation (DataGenerator)

Demonstrates the simulated data generation capability of federated_survival,
covering all 6 simulation types:
  - Accelerated failure time (AFT) models: Weibull, Lognormal
  - Proportional hazards (PH) models:       SDGM1, SDGM4
  - Non-proportional hazards (non-PH) models: SDGM2, SDGM3

Method path: federated_survival.data.generator.DataGenerator.generate
Run: python examples/01_data_generation.py
"""
import os
import sys

# Allow running directly from the project root without installing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from federated_survival.data.generator import DataGenerator, SimulationConfig


def summarize(name, df):
    censor_rate = 1.0 - df["status"].mean()
    print(
        "  [{:<9s}] shape={}, censoring rate={:.2%}, "
        "time range=[{:.2f}, {:.2f}], events={}".format(
            name, df.shape, censor_rate,
            df["time"].min(), df["time"].max(), int(df["status"].sum()),
        )
    )


def main():
    print("=== Example 01: Data Generation ===\n")

    sim_config = SimulationConfig(
        n_samples=500,
        n_features=10,
        random_state=42,
    )
    generator = DataGenerator(sim_config)
    print("Config: n_samples={}, n_features={}".format(
        sim_config.n_samples, sim_config.n_features))
    print("Supported simulation types: {}\n".format(generator.supported_types))

    # 1) AFT models: c_mean controls the censoring rate (larger = more censored)
    print("1) Weibull AFT model")
    data_weibull = generator.generate("weibull", c_mean=0.4)
    summarize("weibull", data_weibull)

    print("\n2) Lognormal AFT model")
    data_lognormal = generator.generate("lognormal", c_mean=0.4)
    summarize("lognormal", data_lognormal)

    # 2) Proportional hazards models
    print("\n3) SDGM1 standard proportional hazards model")
    data_sdgm1 = generator.generate("SDGM1", c_mean=0.4)
    summarize("SDGM1", data_sdgm1)

    print("\n4) SDGM4 proportional hazards + lognormal errors (c_step controls censoring)")
    data_sdgm4 = generator.generate("SDGM4", c_step=0.4)
    summarize("SDGM4", data_sdgm4)

    # 3) Non-proportional hazards models: u_max caps the censoring time
    print("\n5) SDGM2 mild non-proportional hazards (u_max controls censoring)")
    data_sdgm2 = generator.generate("SDGM2", u_max=4)
    summarize("SDGM2", data_sdgm2)

    print("\n6) SDGM3 strong non-proportional hazards (u_max controls censoring)")
    data_sdgm3 = generator.generate("SDGM3", u_max=7)
    summarize("SDGM3", data_sdgm3)

    # Data structure
    print("\nData structure (weibull as an example, first 3 rows):")
    print(data_weibull.head(3).to_string())
    print("\nColumns: {}".format(list(data_weibull.columns)))
    print("Note: x1..xp are features, time is the observed time, "
          "status is the event indicator (1=event, 0=censored)")

    print("\n=== Example 01 done ===")


if __name__ == "__main__":
    main()
