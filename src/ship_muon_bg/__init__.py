"""Pure-Python core for the SHiP adaptive muon-background project.

This package never imports FairShip or ROOT. Physics lives behind the
``simulation_backend`` boundary (see ``docs/contracts/fairship_adapter_contract_v0.md``).
"""

__all__ = ["data_contracts", "simulation"]
