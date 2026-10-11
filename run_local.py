"""Run the project outside Qollab with Qiskit's BasicSimulator as the backend.

    pip install qiskit numpy scipy matplotlib
    python run_local.py
"""
import builtins

from qiskit.providers.basic_provider import BasicSimulator

builtins.backend = BasicSimulator()  # Qollab pre-creates `backend`; do the same here

import main  # noqa: E402

main.main()
